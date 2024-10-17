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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_histogram.cuh>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>

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

}  // namespace spark_rapids_jni
