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

using string_index_pair = thrust::pair<char const*, cudf::size_type>;

std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> cast_strings_to_booleans(
  cudf::column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

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
  CUDF_FUNC_RANGE();

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

// TODO: extract commond code for this and `remove_quotes`.
// This function always return zero size validity array.
std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> remove_quotes_for_floats(
  cudf::column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

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

std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> cast_strings_to_floats(
  cudf::column_view const& input,
  cudf::data_type output_type,
  bool allow_nonnumeric_numbers,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (allow_nonnumeric_numbers) {
    auto [removed_quotes, validity] = remove_quotes_for_floats(input, stream, mr);
    return {::spark_rapids_jni::string_to_float(
              output_type, cudf::strings_column_view{removed_quotes->view()}, false, stream, mr),
            rmm::device_uvector<bool>{0, stream, mr}};
  }
  return {::spark_rapids_jni::string_to_float(
            output_type, cudf::strings_column_view{input}, false, stream, mr),
          rmm::device_uvector<bool>{0, stream, mr}};
}

// TODO there is a bug here around 0 https://github.com/NVIDIA/spark-rapids/issues/10898
std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> cast_strings_to_decimals(
  cudf::column_view const& input,
  cudf::data_type output_type,
  int precision,
  bool is_us_locale,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const string_count = input.size();
  if (string_count == 0) {
    return {cudf::make_empty_column(output_type), rmm::device_uvector<bool>{0, stream, mr}};
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
    return {string_to_decimal(precision, output_type.scale(), input_sv, false, false, stream, mr),
            rmm::device_uvector<bool>{0, stream, mr}};
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
  return {string_to_decimal(precision,
                            output_type.scale(),
                            cudf::strings_column_view{unquoted_strings->view()},
                            false,
                            false,
                            stream,
                            mr),
          rmm::device_uvector<bool>{0, stream, mr}};
}

std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> remove_quotes(
  cudf::column_view const& input,
  bool nullify_if_not_quoted,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

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

/**
 * @brief The struct similar to `cudf::io::schema_element` with adding decimal precision and
 * preserving column order.
 */
struct schema_element_with_precision {
  cudf::data_type type;
  int precision;
  std::vector<std::pair<std::string, schema_element_with_precision>> child_types;
};

std::pair<cudf::io::schema_element, schema_element_with_precision> parse_schema_element(
  std::size_t& index,
  std::vector<std::string> const& col_names,
  std::vector<int> const& num_children,
  std::vector<int> const& types,
  std::vector<int> const& scales,
  std::vector<int> const& precisions)
{
  // Get data for the current column.
  auto const d_type    = cudf::data_type{static_cast<cudf::type_id>(types[index]), scales[index]};
  auto const precision = precisions[index];
  auto const col_num_children = num_children[index];
  index++;

  std::map<std::string, cudf::io::schema_element> children;
  std::vector<std::pair<std::string, schema_element_with_precision>> children_with_precisions;
  std::vector<std::string> child_names(col_num_children);

  if (d_type.id() == cudf::type_id::STRUCT || d_type.id() == cudf::type_id::LIST) {
    for (int i = 0; i < col_num_children; ++i) {
      auto const& name = col_names[index];
      auto [child, child_with_precision] =
        parse_schema_element(index, col_names, num_children, types, scales, precisions);
      children.emplace(name, std::move(child));
      children_with_precisions.emplace_back(name, std::move(child_with_precision));
      child_names[i] = name;
    }
  } else {
    CUDF_EXPECTS(col_num_children == 0,
                 "Found children for a type that should have none.",
                 std::invalid_argument);
  }

  // Note that if the first schema element does not has type STRUCT/LIST then it always has type
  // STRING, since we intentionally parse JSON into strings column for later post-processing.
  auto const schema_dtype =
    d_type.id() == cudf::type_id::STRUCT || d_type.id() == cudf::type_id::LIST
      ? d_type
      : cudf::data_type{cudf::type_id::STRING};
  return {cudf::io::schema_element{schema_dtype, std::move(children), {std::move(child_names)}},
          schema_element_with_precision{d_type, precision, std::move(children_with_precisions)}};
}

// Travel the schema data by depth-first search order.
// Two separate schema is generated:
// - The first one is used as input to `cudf::read_json`, in which the data types of all columns
//   are specified as STRING type. As such, the table returned by `cudf::read_json` will contain
//   only strings columns.
// - The second schema is used for converting from STRING type to the desired types for the final
//   output.
std::pair<cudf::io::schema_element, schema_element_with_precision> generate_struct_schema(
  std::vector<std::string> const& col_names,
  std::vector<int> const& num_children,
  std::vector<int> const& types,
  std::vector<int> const& scales,
  std::vector<int> const& precisions)
{
  std::map<std::string, cudf::io::schema_element> schema_cols;
  std::vector<std::pair<std::string, schema_element_with_precision>> schema_cols_with_precisions;
  std::vector<std::string> name_order;

  std::size_t index = 0;
  while (index < types.size()) {
    auto const& name = col_names[index];
    auto [child, child_with_precision] =
      parse_schema_element(index, col_names, num_children, types, scales, precisions);
    schema_cols.emplace(name, std::move(child));
    schema_cols_with_precisions.emplace_back(name, std::move(child_with_precision));
    name_order.push_back(name);
  }
  return {
    cudf::io::schema_element{
      cudf::data_type{cudf::type_id::STRUCT}, std::move(schema_cols), {std::move(name_order)}},
    schema_element_with_precision{
      cudf::data_type{cudf::type_id::STRUCT}, -1, std::move(schema_cols_with_precisions)}};
}

std::unique_ptr<cudf::column> make_column_from_pair(
  std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>>&& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto& [output, validity] = input;
  if (validity.size() > 0) {
    auto [null_mask, null_count] =
      cudf::detail::valid_if(validity.begin(), validity.end(), thrust::identity{}, stream, mr);
    if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }
  }
  return std::move(output);
}

std::vector<std::unique_ptr<cudf::column>> make_column_array_from_pairs(
  std::vector<std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>>>& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_columns = input.size();
  std::vector<rmm::device_buffer> null_masks;
  null_masks.reserve(num_columns);

  rmm::device_uvector<cudf::size_type> d_valid_counts(num_columns, stream, mr);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), d_valid_counts.begin(), d_valid_counts.end(), 0);

  for (std::size_t idx = 0; idx < num_columns; ++idx) {
    auto const col_size = input[idx].first->size();
    if (col_size == 0) {
      null_masks.emplace_back(rmm::device_buffer{});  // placeholder
      continue;
    }

    null_masks.emplace_back(
      cudf::create_null_mask(col_size, cudf::mask_state::UNINITIALIZED, stream, mr));
    constexpr cudf::size_type block_size{256};
    auto const grid =
      cudf::detail::grid_1d{static_cast<cudf::thread_index_type>(col_size), block_size};
    cudf::detail::valid_if_kernel<block_size>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        reinterpret_cast<cudf::bitmask_type*>(null_masks.back().data()),
        input[idx].second.data(),
        col_size,
        thrust::identity{},
        d_valid_counts.data() + idx);
  }

  auto const valid_counts = cudf::detail::make_std_vector_sync(d_valid_counts, stream);
  std::vector<std::unique_ptr<cudf::column>> output(num_columns);

  for (std::size_t idx = 0; idx < num_columns; ++idx) {
    auto const col_size    = input[idx].first->size();
    auto const valid_count = valid_counts[idx];
    auto const null_count  = col_size - valid_count;
    output[idx]            = std::move(input[idx].first);
    if (null_count > 0) { output[idx]->set_null_mask(std::move(null_masks[idx]), null_count); }
  }

  return output;
}

template <typename InputType>
std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> convert_data_type(
  InputType&& input,
  schema_element_with_precision const& schema,
  bool allow_nonnumeric_numbers,
  bool is_us_locale,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  using DecayInputT                  = std::decay_t<InputType>;
  auto constexpr input_is_const_cv   = std::is_same_v<DecayInputT, cudf::column_view>;
  auto constexpr input_is_column_ptr = std::is_same_v<DecayInputT, std::unique_ptr<cudf::column>>;
  static_assert(input_is_const_cv ^ input_is_column_ptr);

  if (cudf::is_chrono(schema.type)) {
    // Date/time is not processed here - it should be handled separately in spark-rapids.
    if constexpr (input_is_column_ptr) {
      return {std::move(input), rmm::device_uvector<bool>{0, stream, mr}};
    } else {
      CUDF_FAIL("Cannot convert data type to a chrono (date/time) type.");
      return {nullptr, rmm::device_uvector<bool>{0, stream, mr}};
    }
  }

  if (schema.type.id() == cudf::type_id::BOOL8) {
    if constexpr (input_is_column_ptr) {
      return cast_strings_to_booleans(input->view(), stream, mr);
    } else {
      return cast_strings_to_booleans(input, stream, mr);
    }
  }

  if (cudf::is_integral(schema.type)) {
    if constexpr (input_is_column_ptr) {
      return cast_strings_to_integers(input->view(), schema.type, stream, mr);
    } else {
      return cast_strings_to_integers(input, schema.type, stream, mr);
    }
  }

  if (cudf::is_floating_point(schema.type)) {
    if constexpr (input_is_column_ptr) {
      return cast_strings_to_floats(
        input->view(), schema.type, allow_nonnumeric_numbers, stream, mr);
    } else {
      return cast_strings_to_floats(input, schema.type, allow_nonnumeric_numbers, stream, mr);
    }
  }

  if (cudf::is_fixed_point(schema.type)) {
    if constexpr (input_is_column_ptr) {
      return cast_strings_to_decimals(
        input->view(), schema.type, schema.precision, is_us_locale, stream, mr);
    } else {
      return cast_strings_to_decimals(
        input, schema.type, schema.precision, is_us_locale, stream, mr);
    }
  }

  if (schema.type.id() == cudf::type_id::STRING) {
    if constexpr (input_is_column_ptr) {
      return remove_quotes(input->view(), /*nullify_if_not_quoted*/ false, stream, mr);
    } else {
      return remove_quotes(input, /*nullify_if_not_quoted*/ false, stream, mr);
    }
  }

  if constexpr (input_is_column_ptr) {
    auto const d_type       = input->type().id();
    auto const num_rows     = input->size();
    auto const null_count   = input->null_count();
    auto const num_children = input->num_children();
    auto input_content      = input->release();

    if (schema.type.id() == cudf::type_id::LIST) {
      CUDF_EXPECTS(d_type == cudf::type_id::LIST, "Input column should be LIST.");
      std::vector<std::unique_ptr<cudf::column>> new_children;
      new_children.emplace_back(
        std::move(input_content.children[cudf::lists_column_view::offsets_column_index]));
      new_children.emplace_back(make_column_from_pair(
        convert_data_type(
          std::move(input_content.children[cudf::lists_column_view::child_column_index]),
          schema.child_types.front().second,
          allow_nonnumeric_numbers,
          is_us_locale,
          stream,
          mr),
        stream,
        mr));
      return {std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                             num_rows,
                                             rmm::device_buffer{},
                                             std::move(*input_content.null_mask),
                                             null_count,
                                             std::move(new_children)),
              rmm::device_uvector<bool>{0, stream, mr}};
    }

    if (schema.type.id() == cudf::type_id::STRUCT) {
      CUDF_EXPECTS(d_type == cudf::type_id::STRUCT, "Input column should be STRUCT.");
      std::vector<std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>>>
        new_children_with_validity;
      new_children_with_validity.reserve(num_children);
      for (cudf::size_type i = 0; i < num_children; ++i) {
        new_children_with_validity.emplace_back(
          convert_data_type(std::move(input_content.children[i]),
                            schema.child_types[i].second,
                            allow_nonnumeric_numbers,
                            is_us_locale,
                            stream,
                            mr));
      }

      return {std::make_unique<cudf::column>(
                cudf::data_type{cudf::type_id::STRUCT},
                num_rows,
                rmm::device_buffer{},
                std::move(*input_content.null_mask),
                null_count,
                make_column_array_from_pairs(new_children_with_validity, stream, mr)),
              rmm::device_uvector<bool>{0, stream, mr}};
    }
  } else {  // input_is_const_cv
    auto const d_type       = input.type().id();
    auto const num_rows     = input.size();
    auto const null_count   = input.null_count();
    auto const num_children = input.num_children();

    if (schema.type.id() == cudf::type_id::LIST) {
      CUDF_EXPECTS(d_type == cudf::type_id::LIST, "Input column should be LIST.");
      std::vector<std::unique_ptr<cudf::column>> new_children;
      new_children.emplace_back(
        std::make_unique<cudf::column>(input.child(cudf::lists_column_view::offsets_column_index)));
      new_children.emplace_back(make_column_from_pair(
        convert_data_type(input.child(cudf::lists_column_view::child_column_index),
                          schema.child_types.front().second,
                          allow_nonnumeric_numbers,
                          is_us_locale,
                          stream,
                          mr),
        stream,
        mr));
      return {std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::LIST},
                                             num_rows,
                                             rmm::device_buffer{},
                                             cudf::detail::copy_bitmask(input, stream, mr),
                                             null_count,
                                             std::move(new_children)),
              rmm::device_uvector<bool>{0, stream, mr}};
    }

    if (schema.type.id() == cudf::type_id::STRUCT) {
      CUDF_EXPECTS(d_type == cudf::type_id::STRUCT, "Input column should be STRUCT.");
      std::vector<std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>>>
        new_children_with_validity;
      new_children_with_validity.reserve(num_children);
      for (cudf::size_type i = 0; i < num_children; ++i) {
        new_children_with_validity.emplace_back(convert_data_type(input.child(i),
                                                                  schema.child_types[i].second,
                                                                  allow_nonnumeric_numbers,
                                                                  is_us_locale,
                                                                  stream,
                                                                  mr));
      }
      return {std::make_unique<cudf::column>(
                cudf::data_type{cudf::type_id::STRUCT},
                num_rows,
                rmm::device_buffer{},
                cudf::detail::copy_bitmask(input, stream, mr),
                null_count,
                make_column_array_from_pairs(new_children_with_validity, stream, mr)),
              rmm::device_uvector<bool>{0, stream, mr}};
    }
  }

  CUDF_FAIL("Unexpected column type for conversion.");
  return {nullptr, rmm::device_uvector<bool>{0, stream, mr}};
}

}  // namespace

std::unique_ptr<cudf::column> from_json_to_structs(cudf::strings_column_view const& input,
                                                   std::vector<std::string> const& col_names,
                                                   std::vector<int> const& num_children,
                                                   std::vector<int> const& types,
                                                   std::vector<int> const& scales,
                                                   std::vector<int> const& precisions,
                                                   bool normalize_single_quotes,
                                                   bool allow_leading_zeros,
                                                   bool allow_nonnumeric_numbers,
                                                   bool allow_unquoted_control,
                                                   bool is_us_locale,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto const [concat_input, delimiter, is_invalid_or_empty] =
    concat_json(input, false, stream, cudf::get_current_device_resource());
  auto const [schema, schema_with_precision] =
    generate_struct_schema(col_names, num_children, types, scales, precisions);

  auto opts_builder =
    cudf::io::json_reader_options::builder(
      cudf::io::source_info{cudf::device_span<std::byte const>{
        static_cast<std::byte const*>(concat_input->data()), concat_input->size()}})
      // fixed options
      .lines(true)
      .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
      .normalize_whitespace(true)
      .mixed_types_as_string(true)
      .keep_quotes(true)
      .experimental(true)
      .normalize_single_quotes(normalize_single_quotes)
      .strict_validation(true)
      // specifying parameters
      .delimiter(delimiter)
      .numeric_leading_zeros(allow_leading_zeros)
      .nonnumeric_numbers(allow_nonnumeric_numbers)
      .unquoted_control_chars(allow_unquoted_control)
      .dtypes(schema)
      .prune_columns(schema.child_types.size() != 0);

  auto const parsed_table_with_meta = cudf::io::read_json(opts_builder.build());
  auto const& parsed_meta           = parsed_table_with_meta.metadata;
  auto parsed_columns               = parsed_table_with_meta.tbl->release();

  CUDF_EXPECTS(parsed_columns.size() == schema.child_types.size(),
               "Numbers of output columns is different from schema size.");

  std::vector<std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>>>
    converted_cols_with_validity;
  converted_cols_with_validity.reserve(parsed_columns.size());
  for (std::size_t i = 0; i < parsed_columns.size(); ++i) {
    auto const d_type = parsed_columns[i]->type().id();
    CUDF_EXPECTS(d_type == cudf::type_id::LIST || d_type == cudf::type_id::STRUCT ||
                   d_type == cudf::type_id::STRING,
                 "Input column should be STRING or nested.");

    auto const& [col_name, col_schema] = schema_with_precision.child_types[i];
    CUDF_EXPECTS(parsed_meta.schema_info[i].name == col_name, "Mismatched column name.");
    converted_cols_with_validity.emplace_back(convert_data_type(std::move(parsed_columns[i]),
                                                                col_schema,
                                                                allow_nonnumeric_numbers,
                                                                is_us_locale,
                                                                stream,
                                                                mr));
  }

  auto const valid_it          = is_invalid_or_empty->view().begin<bool>();
  auto [null_mask, null_count] = cudf::detail::valid_if(
    valid_it, valid_it + is_invalid_or_empty->size(), thrust::logical_not{}, stream, mr);

  return cudf::make_structs_column(
    input.size(),
    make_column_array_from_pairs(converted_cols_with_validity, stream, mr),
    null_count,
    null_count > 0 ? std::move(null_mask) : rmm::device_buffer{0, stream, mr},
    stream,
    mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> from_json_to_structs(cudf::strings_column_view const& input,
                                                   std::vector<std::string> const& col_names,
                                                   std::vector<int> const& num_children,
                                                   std::vector<int> const& types,
                                                   std::vector<int> const& scales,
                                                   std::vector<int> const& precisions,
                                                   bool normalize_single_quotes,
                                                   bool allow_leading_zeros,
                                                   bool allow_nonnumeric_numbers,
                                                   bool allow_unquoted_control,
                                                   bool is_us_locale,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return detail::from_json_to_structs(input,
                                      col_names,
                                      num_children,
                                      types,
                                      scales,
                                      precisions,
                                      normalize_single_quotes,
                                      allow_leading_zeros,
                                      allow_nonnumeric_numbers,
                                      allow_unquoted_control,
                                      is_us_locale,
                                      stream,
                                      mr);
}

std::unique_ptr<cudf::column> convert_data_type(cudf::column_view const& input,
                                                std::vector<int> const& num_children,
                                                std::vector<int> const& types,
                                                std::vector<int> const& scales,
                                                std::vector<int> const& precisions,
                                                bool allow_nonnumeric_numbers,
                                                bool is_us_locale,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  [[maybe_unused]] auto const [schema, schema_with_precision] = detail::generate_struct_schema(
    /*dummy col_names*/ std::vector<std::string>(num_children.size(), std::string{}),
    num_children,
    types,
    scales,
    precisions);
  CUDF_EXPECTS(schema_with_precision.child_types.size() == 1,
               "The input schema must have exactly one column.");

  return detail::make_column_from_pair(
    detail::convert_data_type(input,
                              schema_with_precision.child_types.front().second,
                              allow_nonnumeric_numbers,
                              is_us_locale,
                              stream,
                              mr),
    stream,
    mr);
}

}  // namespace spark_rapids_jni
