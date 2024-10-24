/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cast_string.hpp"
#include "json_utils.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/json.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_memcpy.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/functional>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>

#include <limits>

namespace spark_rapids_jni {

namespace detail {

namespace {

constexpr bool not_whitespace(cudf::char_utf8 ch)
{
  return ch != ' ' && ch != '\r' && ch != '\n' && ch != '\t';
}

constexpr bool can_be_delimiter(char c)
{
  // The character list below is from `json_reader_options.set_delimiter`.
  switch (c) {
    case '{':
    case '[':
    case '}':
    case ']':
    case ',':
    case ':':
    case '"':
    case '\'':
    case '\\':
    case ' ':
    case '\t':
    case '\r': return false;
    default: return true;
  }
}

}  // namespace

std::tuple<std::unique_ptr<cudf::column>, std::unique_ptr<rmm::device_buffer>, char> concat_json(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const d_input_ptr = cudf::column_device_view::create(input.parent(), stream);
  auto const default_mr  = rmm::mr::get_current_device_resource();

  // Check if the input rows are either null, equal to `null` string literal, or empty.
  // This will be used for masking out the input when doing string concatenation.
  rmm::device_uvector<bool> is_valid_input(input.size(), stream, default_mr);

  // Check if the input rows are either null or empty.
  // This will be returned to the caller.
  rmm::device_uvector<bool> is_null_or_empty(input.size(), stream, mr);

  thrust::for_each(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0L),
    thrust::make_counting_iterator(input.size() * static_cast<int64_t>(cudf::detail::warp_size)),
    [input  = *d_input_ptr,
     output = thrust::make_zip_iterator(thrust::make_tuple(
       is_valid_input.begin(), is_null_or_empty.begin()))] __device__(int64_t tidx) {
      // Execute one warp per row to minimize thread divergence.
      if ((tidx % cudf::detail::warp_size) != 0) { return; }
      auto const idx = tidx / cudf::detail::warp_size;

      if (input.is_null(idx)) {
        output[idx] = thrust::make_tuple(false, true);
        return;
      }

      auto const d_str = input.element<cudf::string_view>(idx);
      auto const size  = d_str.size_bytes();
      int i            = 0;
      char ch;

      // Skip the very first whitespace characters.
      for (; i < size; ++i) {
        ch = d_str[i];
        if (not_whitespace(ch)) { break; }
      }

      if (i + 3 < size &&
          (d_str[i] == 'n' && d_str[i + 1] == 'u' && d_str[i + 2] == 'l' && d_str[i + 3] == 'l')) {
        i += 4;

        // Skip the very last whitespace characters.
        bool is_null_literal{true};
        for (; i < size; ++i) {
          ch = d_str[i];
          if (not_whitespace(ch)) {
            is_null_literal = false;
            break;
          }
        }

        // The current row contains only `null` string literal and not any other non-whitespace
        // characters. Such rows need to be masked out as null when doing concatenation.
        if (is_null_literal) {
          output[idx] = thrust::make_tuple(false, false);
          return;
        }
      }

      auto const not_eol = i < size;

      // If the current row is not null or empty, it should start with `{`. Otherwise, we need to
      // replace it by a null. This is necessary for libcudf's JSON reader to work.
      // Note that if we want to support ARRAY schema, we need to check for `[` instead.
      auto constexpr start_character = '{';
      if (not_eol && ch != start_character) {
        output[idx] = thrust::make_tuple(false, false);
        return;
      }

      output[idx] = thrust::make_tuple(not_eol, !not_eol);
    });

  auto constexpr num_levels  = 256;
  auto constexpr lower_level = std::numeric_limits<char>::min();
  auto constexpr upper_level = std::numeric_limits<char>::max();
  auto const num_chars       = input.chars_size(stream);

  rmm::device_uvector<uint32_t> histogram(num_levels, stream, default_mr);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), histogram.begin(), histogram.end(), 0);

  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(nullptr,
                                      temp_storage_bytes,
                                      input.chars_begin(stream),
                                      histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_chars,
                                      stream.value());
  rmm::device_buffer d_temp(temp_storage_bytes, stream);
  cub::DeviceHistogram::HistogramEven(d_temp.data(),
                                      temp_storage_bytes,
                                      input.chars_begin(stream),
                                      histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_chars,
                                      stream.value());

  auto const it             = thrust::make_counting_iterator(0);
  auto const zero_level_idx = -lower_level;  // the bin storing count for character `\0`
  auto const zero_level_it  = it + zero_level_idx;
  auto const end            = it + num_levels;

  auto const first_zero_count_pos =
    thrust::find_if(rmm::exec_policy_nosync(stream),
                    zero_level_it,  // ignore the negative characters
                    end,
                    [zero_level_idx, counts = histogram.begin()] __device__(auto idx) -> bool {
                      auto const count = counts[idx];
                      if (count > 0) { return false; }
                      auto const first_non_existing_char = static_cast<char>(idx - zero_level_idx);
                      return can_be_delimiter(first_non_existing_char);
                    });

  // This should never happen since the input should never cover the entire char range.
  if (first_zero_count_pos == end) {
    throw std::logic_error(
      "Cannot find any character suitable as delimiter during joining json strings.");
  }
  auto const delimiter = static_cast<char>(thrust::distance(zero_level_it, first_zero_count_pos));

  auto [null_mask, null_count] = cudf::detail::valid_if(
    is_valid_input.begin(), is_valid_input.end(), thrust::identity{}, stream, default_mr);
  // If the null count doesn't change, that mean we do not have any rows containing `null` string
  // literal or empty rows. In such cases, just use the input column for concatenation.
  auto const input_applied_null =
    null_count == input.null_count()
      ? cudf::column_view{}
      : cudf::column_view{cudf::data_type{cudf::type_id::STRING},
                          input.size(),
                          input.chars_begin(stream),
                          reinterpret_cast<cudf::bitmask_type const*>(null_mask.data()),
                          null_count,
                          0,
                          std::vector<cudf::column_view>{input.offsets()}};

  auto concat_strings = cudf::strings::detail::join_strings(
    null_count == input.null_count() ? input : cudf::strings_column_view{input_applied_null},
    cudf::string_scalar(std::string(1, delimiter), true, stream, default_mr),
    cudf::string_scalar("{}", true, stream, default_mr),
    stream,
    mr);

  return {std::make_unique<cudf::column>(std::move(is_null_or_empty), rmm::device_buffer{}, 0),
          std::move(concat_strings->release().data),
          delimiter};
}

std::unique_ptr<cudf::column> make_structs(std::vector<cudf::column_view> const& children,
                                           cudf::column_view const& is_null,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  if (children.size() == 0) { return nullptr; }

  auto const row_count = children.front().size();
  for (auto const& col : children) {
    CUDF_EXPECTS(col.size() == row_count, "All columns must have the same number of rows.");
  }

  auto const [null_mask, null_count] = cudf::detail::valid_if(
    is_null.begin<bool>(), is_null.end<bool>(), thrust::logical_not{}, stream, mr);

  auto const structs =
    cudf::column_view(cudf::data_type{cudf::type_id::STRUCT},
                      row_count,
                      nullptr,
                      reinterpret_cast<cudf::bitmask_type const*>(null_mask.data()),
                      null_count,
                      0,
                      children);
  return std::make_unique<cudf::column>(structs, stream, mr);
}

namespace {

using string_index_pair = thrust::pair<char const*, cudf::size_type>;

std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> cast_strings_to_booleans(
  cudf::column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const string_count = input.size();
  if (string_count == 0) {
    return {cudf::make_empty_column(cudf::data_type{cudf::type_id::BOOL8}),
            rmm::device_uvector<bool>(0, stream)};
  }

  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, string_count, cudf::mask_state::UNALLOCATED, stream, mr);
  auto validity = rmm::device_uvector<bool>(string_count, stream);

  auto const input_sv = cudf::strings_column_view{input};
  auto const offsets_it =
    cudf::detail::offsetalator_factory::make_input_iterator(input_sv.offsets());
  auto const d_input_ptr = cudf::column_device_view::create(input, stream);
  auto const is_valid_it = cudf::detail::make_validity_iterator<true>(*d_input_ptr);
  auto const output_it   = thrust::make_zip_iterator(
    thrust::make_tuple(output->mutable_view().begin<bool>(), validity.begin()));
  thrust::tabulate(
    rmm::exec_policy_nosync(stream),
    output_it,
    output_it + string_count,
    [chars = input_sv.chars_begin(stream), offsets = offsets_it, is_valid = is_valid_it] __device__(
      auto idx) -> thrust::tuple<bool, bool> {
      if (is_valid[idx]) {
        auto const start_offset = offsets[idx];
        auto const end_offset   = offsets[idx + 1];
        auto const size         = end_offset - start_offset;
        auto const str          = chars + start_offset;

        if (size == 4 && str[0] == 't' && str[1] == 'r' && str[2] == 'u' && str[3] == 'e') {
          return {true, true};
        }
        if (size == 5 && str[0] == 'f' && str[1] == 'a' && str[2] == 'l' && str[3] == 's' &&
            str[4] == 'e') {
          return {false, true};
        }
      }

      // Either null input, or the input string is neither `true` nor `false`.
      return {false, false};
    });

  // Reset null count, as it is invalidated after calling to `mutable_view()`.
  output->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);

  return {std::move(output), std::move(validity)};
}

std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> cast_strings_to_integers(
  cudf::column_view const& input,
  cudf::data_type output_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const string_count = input.size();
  if (string_count == 0) {
    return {cudf::make_empty_column(output_type), rmm::device_uvector<bool>(0, stream)};
  }

  auto const input_sv = cudf::strings_column_view{input};
  auto const input_offsets_it =
    cudf::detail::offsetalator_factory::make_input_iterator(input_sv.offsets());
  auto const d_input_ptr = cudf::column_device_view::create(input, stream);
  auto const is_valid_it = cudf::detail::make_validity_iterator<true>(*d_input_ptr);

  auto string_pairs = rmm::device_uvector<string_index_pair>(string_count, stream);
  // Since the strings store integer numbers, they should be very short.
  // As such, using one thread per string should be good.
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   string_pairs.begin(),
                   string_pairs.end(),
                   [chars    = input_sv.chars_begin(stream),
                    offsets  = input_offsets_it,
                    is_valid = is_valid_it] __device__(cudf::size_type idx) -> string_index_pair {
                     if (!is_valid[idx]) { return {nullptr, 0}; }

                     auto const start_offset = offsets[idx];
                     auto const end_offset   = offsets[idx + 1];

                     auto in_ptr = chars + start_offset;
                     auto in_end = chars + end_offset;
                     while (in_ptr != in_end) {
                       if (*in_ptr == '.' || *in_ptr == 'e' || *in_ptr == 'E') {
                         return {nullptr, 0};
                       }
                       ++in_ptr;
                     }

                     return {chars + start_offset, end_offset - start_offset};
                   });

  auto const size_it = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [string_pairs = string_pairs.begin()] __device__(cudf::size_type idx) -> cudf::size_type {
        return string_pairs[idx].second;
      }));
  auto [offsets_column, bytes] =
    cudf::strings::detail::make_offsets_child_column(size_it, size_it + string_count, stream, mr);
  auto chars_data = cudf::strings::detail::make_chars_buffer(
    offsets_column->view(), bytes, string_pairs.begin(), string_count, stream, mr);

  // Don't care about the null mask, as nulls imply empty strings, and will be nullified.
  auto const sanitized_input =
    cudf::make_strings_column(string_count, std::move(offsets_column), chars_data.release(), 0, {});

  auto output = string_to_integer(
    output_type, cudf::strings_column_view{sanitized_input->view()}, false, false, stream, mr);

  return {std::move(output), rmm::device_uvector<bool>(0, stream)};
}

std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> cast_strings_to_dates(
  cudf::column_view const& input,
  std::string const& date_regex,
  std::string const& date_format,
  bool error_if_invalid,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const string_count = input.size();
  if (string_count == 0) {
    return {cudf::make_empty_column(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}),
            rmm::device_uvector<bool>(0, stream)};
  }

  // TODO: mr
  auto const removed_quotes = remove_quotes(input, false, stream, mr);

  auto const input_sv   = cudf::strings_column_view{removed_quotes->view()};
  auto const regex_prog = cudf::strings::regex_program::create(
    date_regex, cudf::strings::regex_flags::DEFAULT, cudf::strings::capture_groups::NON_CAPTURE);
  auto const is_matched     = cudf::strings::matches_re(input_sv, *regex_prog, stream);
  auto const is_timestamp   = cudf::strings::is_timestamp(input_sv, date_format, stream);
  auto const d_is_matched   = is_matched->view().begin<bool>();
  auto const d_is_timestamp = is_timestamp->view().begin<bool>();

  auto const d_input_ptr   = cudf::column_device_view::create(removed_quotes->view(), stream);
  auto const is_valid_it   = cudf::detail::make_validity_iterator<true>(*d_input_ptr);
  auto const invalid_count = thrust::count_if(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(string_count),
    [is_valid = is_valid_it, is_matched = d_is_matched, is_timestamp = d_is_timestamp] __device__(
      auto idx) { return is_valid[idx] && (!is_matched[idx] || !is_timestamp[idx]); });

  if (invalid_count == 0) {
    auto output = cudf::strings::to_timestamps(
      input_sv, cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, date_format, stream, mr);
    return {std::move(output), rmm::device_uvector<bool>(0, stream)};
  }

  // From here we have invalid_count > 0.
  if (error_if_invalid) { return {nullptr, rmm::device_uvector<bool>(0, stream)}; }

  auto const input_offsets_it =
    cudf::detail::offsetalator_factory::make_input_iterator(input_sv.offsets());
  auto string_pairs = rmm::device_uvector<string_index_pair>(string_count, stream);

  thrust::tabulate(
    rmm::exec_policy_nosync(stream),
    string_pairs.begin(),
    string_pairs.end(),
    [chars        = input_sv.chars_begin(stream),
     offsets      = input_offsets_it,
     is_valid     = is_valid_it,
     is_matched   = d_is_matched,
     is_timestamp = d_is_timestamp] __device__(cudf::size_type idx) -> string_index_pair {
      if (!is_valid[idx] || !is_matched[idx] || !is_timestamp[idx]) { return {nullptr, 0}; }

      auto const start_offset = offsets[idx];
      auto const end_offset   = offsets[idx + 1];
      return {chars + start_offset, end_offset - start_offset};
    });

  auto const size_it = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [string_pairs = string_pairs.begin()] __device__(cudf::size_type idx) -> cudf::size_type {
        return string_pairs[idx].second;
      }));
  auto [offsets_column, bytes] =
    cudf::strings::detail::make_offsets_child_column(size_it, size_it + string_count, stream, mr);
  auto chars_data = cudf::strings::detail::make_chars_buffer(
    offsets_column->view(), bytes, string_pairs.begin(), string_count, stream, mr);

  // Don't care about the null mask, as nulls imply empty strings, and will be nullified.
  auto const sanitized_input =
    cudf::make_strings_column(string_count, std::move(offsets_column), chars_data.release(), 0, {});

  auto output = cudf::strings::to_timestamps(cudf::strings_column_view{sanitized_input->view()},
                                             cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                                             date_format,
                                             stream,
                                             mr);

  auto validity = rmm::device_uvector<bool>(string_count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    string_pairs.begin(),
                    string_pairs.end(),
                    validity.begin(),
                    [] __device__(string_index_pair const& pair) { return pair.first != nullptr; });

  // Null mask and null count will be updated later from the validity vector.
  return {std::move(output), std::move(validity)};
}

// TODO there is a bug here around 0 https://github.com/NVIDIA/spark-rapids/issues/10898
std::unique_ptr<cudf::column> cast_strings_to_decimals(cudf::column_view const& input,
                                                       int precision,
                                                       int scale,
                                                       bool is_us_locale,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  auto const string_count = input.size();
  if (string_count == 0) {
    auto const dtype = [precision, scale]() {
      if (precision <= std::numeric_limits<int32_t>::digits10) {
        return cudf::data_type(cudf::type_id::DECIMAL32, scale);
      } else if (precision <= std::numeric_limits<int64_t>::digits10) {
        return cudf::data_type(cudf::type_id::DECIMAL64, scale);
      } else if (precision <= std::numeric_limits<__int128_t>::digits10) {
        return cudf::data_type(cudf::type_id::DECIMAL128, scale);
      } else {
        CUDF_FAIL("Unable to support decimal with precision " + std::to_string(precision));
      }
    }();
    return cudf::make_empty_column(dtype);
  }

  CUDF_EXPECTS(is_us_locale, "String to decimal conversion is only supported in US locale.");

  auto const input_sv = cudf::strings_column_view{input};
  auto const in_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input_sv.offsets());

  // Count the number of characters `"`.
  rmm::device_uvector<int8_t> quote_counts(string_count, stream);
  // Count the number of characters `"` and `,` in each string.
  rmm::device_uvector<int8_t> remove_counts(string_count, stream);

  {
    using count_type    = thrust::tuple<int8_t, int8_t>;
    auto const check_it = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<count_type>(
        [chars = input_sv.chars_begin(stream)] __device__(auto idx) {
          auto const c             = chars[idx];
          auto const is_quote      = c == '"';
          auto const should_remove = is_quote || c == ',';
          return count_type{static_cast<int8_t>(is_quote), static_cast<int8_t>(should_remove)};
        }));
    auto const plus_op =
      cuda::proclaim_return_type<count_type>([] __device__(count_type lhs, count_type rhs) {
        return count_type{thrust::get<0>(lhs) + thrust::get<0>(rhs),
                          thrust::get<1>(lhs) + thrust::get<1>(rhs)};
      });

    auto const out_count_it =
      thrust::make_zip_iterator(quote_counts.begin(), remove_counts.begin());

    std::size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Reduce(nullptr,
                                       temp_storage_bytes,
                                       check_it,
                                       out_count_it,
                                       string_count,
                                       in_offsets,
                                       in_offsets + 1,
                                       plus_op,
                                       count_type{0, 0},
                                       stream.value());
    auto d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
    cub::DeviceSegmentedReduce::Reduce(d_temp_storage.data(),
                                       temp_storage_bytes,
                                       check_it,
                                       out_count_it,
                                       string_count,
                                       in_offsets,
                                       in_offsets + 1,
                                       plus_op,
                                       count_type{0, 0},
                                       stream.value());
  }

  auto const out_size_it = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [offsets       = in_offsets,
       quote_counts  = quote_counts.begin(),
       remove_counts = remove_counts.begin()] __device__(auto idx) {
        auto const input_size = offsets[idx + 1] - offsets[idx];
        // If the current row is a non-quoted string, just return the original string.
        if (quote_counts[idx] == 0) { return static_cast<cudf::size_type>(input_size); }
        // Otherwise, we will modify the string, removing characters '"' and ','.
        return static_cast<cudf::size_type>(input_size - remove_counts[idx]);
      }));
  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    out_size_it, out_size_it + string_count, stream, mr);

  // If the output strings column does not change in its total bytes, we know that it does not have
  // any '"' or ',' characters.
  if (bytes == input_sv.chars_size(stream)) {
    return string_to_decimal(precision, scale, input_sv, false, false, stream, mr);
  }

  auto const out_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());
  auto chars_data = rmm::device_uvector<char>(bytes, stream, mr);

  // Since the strings store decimal numbers, they should be very short.
  // As such, using one thread per string should be good.
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(string_count),
                   [in_offsets,
                    out_offsets,
                    input  = input_sv.chars_begin(stream),
                    output = chars_data.begin()] __device__(auto idx) {
                     auto const in_size  = in_offsets[idx + 1] - in_offsets[idx];
                     auto const out_size = out_offsets[idx + 1] - out_offsets[idx];
                     if (in_size == 0) { return; }

                     // If the output size is not changed, we are returning the original unquoted
                     // string. Such string may still contain other alphabet characters, but that
                     // should be handled in the conversion function later on.
                     if (in_size == out_size) {
                       memcpy(output + out_offsets[idx], input + in_offsets[idx], in_size);
                     } else {  // copy byte by byte, ignoring '"' and ',' characters.
                       auto in_ptr  = input + in_offsets[idx];
                       auto in_end  = input + in_offsets[idx + 1];
                       auto out_ptr = output + out_offsets[idx];
                       while (in_ptr != in_end) {
                         if (*in_ptr != '"' && *in_ptr != ',') {
                           *out_ptr = *in_ptr;
                           ++out_ptr;
                         }
                         ++in_ptr;
                       }
                     }
                   });

  auto const unquoted_strings = cudf::make_strings_column(string_count,
                                                          std::move(offsets_column),
                                                          chars_data.release(),
                                                          0,
                                                          rmm::device_buffer{0, stream, mr});
  return string_to_decimal(precision,
                           scale,
                           cudf::strings_column_view{unquoted_strings->view()},
                           false,
                           false,
                           stream,
                           mr);
}

std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> remove_quotes(
  cudf::column_view const& input,
  bool nullify_if_not_quoted,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const string_count = input.size();
  if (string_count == 0) {
    return {cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}),
            rmm::device_uvector<bool>(0, stream)};
  }

  auto const input_sv = cudf::strings_column_view{input};
  auto const input_offsets_it =
    cudf::detail::offsetalator_factory::make_input_iterator(input_sv.offsets());
  auto const d_input_ptr = cudf::column_device_view::create(input, stream);
  auto const is_valid_it = cudf::detail::make_validity_iterator<true>(*d_input_ptr);

  auto string_pairs = rmm::device_uvector<string_index_pair>(string_count, stream);
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   string_pairs.begin(),
                   string_pairs.end(),
                   [nullify_if_not_quoted,
                    chars    = input_sv.chars_begin(stream),
                    offsets  = input_offsets_it,
                    is_valid = is_valid_it] __device__(cudf::size_type idx) -> string_index_pair {
                     if (!is_valid[idx]) { return {nullptr, 0}; }

                     auto const start_offset = offsets[idx];
                     auto const end_offset   = offsets[idx + 1];
                     auto const size         = end_offset - start_offset;
                     auto const str          = chars + start_offset;

                     // Need to check for size, since the input string may contain just a single
                     // character `"`. Such input should not be considered as quoted.
                     auto const is_quoted = size > 1 && str[0] == '"' && str[size - 1] == '"';
                     if (nullify_if_not_quoted && !is_quoted) { return {nullptr, 0}; }

                     auto const output_size = is_quoted ? size - 2 : size;
                     return {chars + start_offset + (is_quoted ? 1 : 0), output_size};
                   });

  auto const size_it = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [string_pairs = string_pairs.begin()] __device__(cudf::size_type idx) -> cudf::size_type {
        return string_pairs[idx].second;
      }));
  auto [offsets_column, bytes] =
    cudf::strings::detail::make_offsets_child_column(size_it, size_it + string_count, stream, mr);
  auto chars_data = cudf::strings::detail::make_chars_buffer(
    offsets_column->view(), bytes, string_pairs.begin(), string_count, stream, mr);

  if (nullify_if_not_quoted) {
    auto validity = rmm::device_uvector<bool>(string_count, stream);
    thrust::transform(
      rmm::exec_policy_nosync(stream),
      string_pairs.begin(),
      string_pairs.end(),
      validity.begin(),
      [] __device__(string_index_pair const& pair) { return pair.first != nullptr; });

    // Null mask and null count will be updated later from the validity vector.
    auto output = cudf::make_strings_column(string_count,
                                            std::move(offsets_column),
                                            chars_data.release(),
                                            0,
                                            rmm::device_buffer{0, stream, mr});

    return {std::move(output), std::move(validity)};
  } else {
    auto output = cudf::make_strings_column(string_count,
                                            std::move(offsets_column),
                                            chars_data.release(),
                                            input.null_count(),
                                            cudf::detail::copy_bitmask(input, stream, mr));

    return {std::move(output), rmm::device_uvector<bool>(0, stream)};
  }
}

// TODO: extract commond code for this and `remove_quotes`.
std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> remove_quotes_for_floats(
  cudf::column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const string_count = input.size();
  if (string_count == 0) {
    return {cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING}),
            rmm::device_uvector<bool>(0, stream)};
  }

  auto const input_sv = cudf::strings_column_view{input};
  auto const input_offsets_it =
    cudf::detail::offsetalator_factory::make_input_iterator(input_sv.offsets());
  auto const d_input_ptr = cudf::column_device_view::create(input, stream);
  auto const is_valid_it = cudf::detail::make_validity_iterator<true>(*d_input_ptr);

  auto string_pairs = rmm::device_uvector<string_index_pair>(string_count, stream);
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   string_pairs.begin(),
                   string_pairs.end(),
                   [chars    = input_sv.chars_begin(stream),
                    offsets  = input_offsets_it,
                    is_valid = is_valid_it] __device__(cudf::size_type idx) -> string_index_pair {
                     if (!is_valid[idx]) { return {nullptr, 0}; }

                     auto const start_offset = offsets[idx];
                     auto const end_offset   = offsets[idx + 1];
                     auto const size         = end_offset - start_offset;
                     auto const str          = chars + start_offset;

                     // Need to check for size, since the input string may contain just a single
                     // character `"`. Such input should not be considered as quoted.
                     auto const is_quoted = size > 1 && str[0] == '"' && str[size - 1] == '"';

                     // We check and remove quotes only for the special cases (non-numeric numbers
                     // wrapped in double quotes) that are accepted in `from_json`.
                     // They are "NaN", "+INF", "-INF", "+Infinity", "Infinity", "-Infinity".
                     if (is_quoted) {
                       // "NaN"
                       auto accepted = size == 5 && str[1] == 'N' && str[2] == 'a' && str[3] == 'N';

                       // "+INF" and "-INF"
                       accepted = accepted || (size == 6 && (str[1] == '+' || str[1] == '-') &&
                                               str[2] == 'I' && str[3] == 'N' && str[4] == 'F');

                       // "Infinity"
                       accepted = accepted || (size == 10 && str[1] == 'I' && str[2] == 'n' &&
                                               str[3] == 'f' && str[4] == 'i' && str[5] == 'n' &&
                                               str[6] == 'i' && str[7] == 't' && str[8] == 'y');

                       // "+Infinity" and "-Infinity"
                       accepted = accepted || (size == 11 && (str[1] == '+' || str[1] == '-') &&
                                               str[2] == 'I' && str[3] == 'n' && str[4] == 'f' &&
                                               str[5] == 'i' && str[6] == 'n' && str[7] == 'i' &&
                                               str[8] == 't' && str[9] == 'y');

                       if (accepted) { return {str + 1, size - 2}; }
                     }

                     return {str, size};
                   });

  auto const size_it = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [string_pairs = string_pairs.begin()] __device__(cudf::size_type idx) -> cudf::size_type {
        return string_pairs[idx].second;
      }));
  auto [offsets_column, bytes] =
    cudf::strings::detail::make_offsets_child_column(size_it, size_it + string_count, stream, mr);
  auto chars_data = cudf::strings::detail::make_chars_buffer(
    offsets_column->view(), bytes, string_pairs.begin(), string_count, stream, mr);

  auto output = cudf::make_strings_column(string_count,
                                          std::move(offsets_column),
                                          chars_data.release(),
                                          input.null_count(),
                                          cudf::detail::copy_bitmask(input, stream, mr));

  return {std::move(output), rmm::device_uvector<bool>(0, stream)};
}

std::unique_ptr<cudf::column> convert_column_type(std::unique_ptr<cudf::column>& input,
                                                  json_schema_element const& schema,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  if (schema.type.id() == cudf::type_id::BOOL8) {
    CUDF_EXPECTS(input->type().id() == cudf::type_id::STRING, "Input column should be STRING.");
    return ::spark_rapids_jni::cast_strings_to_booleans(input->view(), stream, mr);
  }
  if (cudf::is_integral(schema.type)) {
    CUDF_EXPECTS(input->type().id() == cudf::type_id::STRING, "Input column should be STRING.");
    return ::spark_rapids_jni::cast_strings_to_integers(input->view(), schema.type, stream, mr);
  }
  // if (cudf::is_floating_point(schema.type)) {
  //   CUDF_EXPECTS(input->type().id() == cudf::type_id::STRING, "Input column should be STRING.");
  //   return ::cast_strings_to_floats(input->view(), schema.type, stream, mr);
  // }
  // if (cudf::is_fixed_point(schema.type)) {
  //   CUDF_EXPECTS(input->type().id() == cudf::type_id::STRING, "Input column should be STRING.");
  //   return ::cast_strings_to_decimals(input->view(), schema.type, stream, mr);
  // }
  if (schema.type.id() == cudf::type_id::STRING) {
    CUDF_EXPECTS(input->type().id() == cudf::type_id::STRING, "Input column should be STRING.");
    return ::spark_rapids_jni::remove_quotes(input->view(), false, stream, mr);
  }

  auto const num_rows   = input->size();
  auto const null_count = input->null_count();

  if (schema.type.id() == cudf::type_id::LIST) {
    CUDF_EXPECTS(input->type().id() == cudf::type_id::LIST, "Input column should be LIST.");

    auto input_content = input->release();
    auto new_child =
      convert_column_type(input_content.children[cudf::lists_column_view::child_column_index],
                          schema.child_types.front().second,
                          stream,
                          mr);

    return cudf::make_lists_column(
      num_rows,
      std::move(input_content.children[cudf::lists_column_view::offsets_column_index]),
      std::move(new_child),
      null_count,
      std::move(*input_content.null_mask),
      stream,
      mr);
  }

  if (schema.type.id() == cudf::type_id::STRUCT) {
    CUDF_EXPECTS(input->type().id() == cudf::type_id::STRUCT, "Input column should be LIST.");

    auto const num_children = input->num_children();
    std::vector<std::unique_ptr<cudf::column>> new_children;
    new_children.reserve(num_children);
    auto input_content = input->release();
    for (cudf::size_type i = 0; i < input->num_children(); ++i) {
      new_children.emplace_back(
        convert_column_type(input_content.children[i], schema.child_types[i].second, stream, mr));
    }

    return cudf::make_structs_column(num_rows,
                                     std::move(new_children),
                                     null_count,
                                     std::move(*input_content.null_mask),
                                     stream,
                                     mr);
  }

  CUDF_FAIL("Unexpected column type for conversion.");
  return nullptr;
}

void check_schema(cudf::io::column_name_info const& read_info,
                  std::pair<std::string, json_schema_element> const& column_schema)
{
  CUDF_EXPECTS(read_info.name == column_schema.first, "Mismatched column name.");
  CUDF_EXPECTS(read_info.children.size() == column_schema.second.child_types.size(),
               "Mismatched number of children.");
  for (std::size_t i = 0; i < read_info.children.size(); ++i) {
    // auto const find_it = column_schema.second.child_types.find(read_info.children[i].name);
    // check_schema(read_info.children[i],
    //              find_it != column_schema.child_types.end() ? *find_it : {"", {}});
    check_schema(read_info.children[i], column_schema.second.child_types[i]);
  }
}

}  // namespace

std::unique_ptr<cudf::column> from_json_to_structs(
  cudf::strings_column_view const& input,
  std::vector<std::pair<std::string, json_schema_element>> const& schema,
  cudf::io::json_reader_options const& json_options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const [is_null_or_empty, concat_input, delimiter] = concat_json(input, stream, mr);
  auto opts_builder =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::device_span<std::byte const>{
        static_cast<std::byte const*>(concat_input->data()), concat_input->size()}})
      .dayfirst(json_options.is_enabled_dayfirst())
      .lines(json_options.is_enabled_lines())
      .recovery_mode(json_options.recovery_mode())
      .normalize_single_quotes(json_options.is_enabled_normalize_single_quotes())
      .normalize_whitespace(json_options.is_enabled_normalize_whitespace())
      .mixed_types_as_string(json_options.is_enabled_mixed_types_as_string())
      .delimiter(json_options.get_delimiter())
      .strict_validation(json_options.is_strict_validation())
      .keep_quotes(json_options.is_enabled_keep_quotes())
      .prune_columns(json_options.is_enabled_prune_columns())
      .experimental(json_options.is_enabled_experimental());
  if (json_options.is_strict_validation()) {
    opts_builder.numeric_leading_zeros(json_options.is_allowed_numeric_leading_zeros())
      .nonnumeric_numbers(json_options.is_allowed_nonnumeric_numbers())
      .unquoted_control_chars(json_options.is_allowed_unquoted_control_chars());
  }
  auto const parsed_table_with_meta = cudf::io::read_json(opts_builder.build());
  auto const& parsed_meta           = parsed_table_with_meta.metadata;
  auto parsed_columns               = parsed_table_with_meta.tbl->release();

  CUDF_EXPECTS(parsed_columns.size() == schema.size(),
               "Numbers of columns in the input table is different from schema size.");

  std::vector<std::unique_ptr<cudf::column>> converted_cols(parsed_columns.size());
  for (std::size_t i = 0; i < parsed_columns.size(); ++i) {
    check_schema(parsed_meta.schema_info[i], schema[i]);
    converted_cols[i] = convert_column_type(parsed_columns[i], schema[i].second, stream, mr);
  }

  auto const valid_it          = is_null_or_empty->view().begin<bool>();
  auto [null_mask, null_count] = cudf::detail::valid_if(
    valid_it, valid_it + is_null_or_empty->size(), thrust::logical_not{}, stream, mr);

  return cudf::make_structs_column(
    input.size(),
    std::move(converted_cols),
    null_count,
    null_count > 0 ? std::move(null_mask) : rmm::device_buffer{0, stream, mr},
    stream,
    mr);
}

}  // namespace detail

std::tuple<std::unique_ptr<cudf::column>, std::unique_ptr<rmm::device_buffer>, char> concat_json(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::concat_json(input, stream, mr);
}

std::unique_ptr<cudf::column> make_structs(std::vector<cudf::column_view> const& children,
                                           cudf::column_view const& is_null,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::make_structs(children, is_null, stream, mr);
}

std::unique_ptr<cudf::column> cast_strings_to_booleans(cudf::column_view const& input,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto [output, validity] = detail::cast_strings_to_booleans(input, stream, mr);
  auto [null_mask, null_count] =
    cudf::detail::valid_if(validity.begin(), validity.end(), thrust::identity{}, stream, mr);
  if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }
  return std::move(output);
}

std::unique_ptr<cudf::column> cast_strings_to_integers(cudf::column_view const& input,
                                                       cudf::data_type output_type,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto [output, validity] = detail::cast_strings_to_integers(input, output_type, stream, mr);
  return std::move(output);
}

std::unique_ptr<cudf::column> cast_strings_to_dates(cudf::column_view const& input,
                                                    std::string const& date_regex,
                                                    std::string const& date_format,
                                                    bool error_if_invalid,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto [output, validity] =
    detail::cast_strings_to_dates(input, date_regex, date_format, error_if_invalid, stream, mr);

  if (output == nullptr) { return nullptr; }
  auto [null_mask, null_count] =
    cudf::detail::valid_if(validity.begin(), validity.end(), thrust::identity{}, stream, mr);
  if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }
  return std::move(output);
}

std::unique_ptr<cudf::column> cast_strings_to_decimals(cudf::column_view const& input,
                                                       int precision,
                                                       int scale,
                                                       bool is_us_locale,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return detail::cast_strings_to_decimals(input, precision, scale, is_us_locale, stream, mr);
}

std::unique_ptr<cudf::column> remove_quotes(cudf::column_view const& input,
                                            bool nullify_if_not_quoted,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto [output, validity] = detail::remove_quotes(input, nullify_if_not_quoted, stream, mr);
  if (validity.size() > 0) {
    auto [null_mask, null_count] =
      cudf::detail::valid_if(validity.begin(), validity.end(), thrust::identity{}, stream, mr);
    if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }
  }
  return std::move(output);
}

std::unique_ptr<cudf::column> cast_strings_to_floats(cudf::column_view const& input,
                                                     cudf::data_type output_type,
                                                     bool allow_nonnumeric_numbers,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (allow_nonnumeric_numbers) {
    auto [removed_quotes, validity] = detail::remove_quotes_for_floats(input, stream, mr);
    return string_to_float(
      output_type, cudf::strings_column_view{removed_quotes->view()}, false, stream, mr);
  }
  return string_to_float(output_type, cudf::strings_column_view{input}, false, stream, mr);
}

}  // namespace spark_rapids_jni
