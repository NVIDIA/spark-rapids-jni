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
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/combine.hpp>
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

  thrust::transform(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(input.size()),
    thrust::make_zip_iterator(thrust::make_tuple(is_valid_input.begin(), is_null_or_empty.begin())),
    [input = *d_input_ptr] __device__(cudf::size_type idx) -> thrust::tuple<bool, bool> {
      if (input.is_null(idx)) { return {false, false}; }

      // Currently, only check for empty character.
      constexpr char empty_char{' '};

      auto const d_str  = input.element<cudf::string_view>(idx);
      cudf::size_type i = 0;
      for (; i < d_str.size_bytes(); ++i) {
        if (d_str[i] != empty_char) { break; }
      }

      bool is_null_literal{false};
      if (i + 3 < d_str.size_bytes() &&
          (d_str[i] == 'n' && d_str[i + 1] == 'u' && d_str[i + 2] == 'l' && d_str[i + 3] == 'l')) {
        is_null_literal = true;
        i += 4;
      }

      for (; i < d_str.size_bytes(); ++i) {
        if (d_str[i] != empty_char) {
          is_null_literal = false;
          break;
        }
      }

      // The current row contains only `null` string literal and not any other non-empty characters.
      // Such rows need to be masked out as null when doing concatenation.
      if (is_null_literal) { return {false, true}; }

      auto const not_eol = i < d_str.size_bytes();

      // If the first row is not null or empty, it should start with `{`.
      // Otherwise, we need to replace it by a null.
      // This is necessary for libcudf's JSON reader to work.
      // Note that if we want to support ARRAY schema, we need to check for either `{` or `[`.
      auto constexpr start_character = '{';
      if (not_eol && d_str[i] != start_character) { return {false, true}; }

      return {not_eol, not_eol};
    });

  auto constexpr num_levels  = 256;
  auto constexpr lower_level = std::numeric_limits<char>::min();
  auto constexpr upper_level = std::numeric_limits<char>::max();
  auto const num_chars       = input.chars_size(stream);

  rmm::device_uvector<uint32_t> d_histogram(num_levels, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), d_histogram.begin(), d_histogram.end(), 0);

  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(nullptr,
                                      temp_storage_bytes,
                                      input.chars_begin(stream),
                                      d_histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_chars,
                                      stream.value());
  rmm::device_buffer d_temp(temp_storage_bytes, stream);
  cub::DeviceHistogram::HistogramEven(d_temp.data(),
                                      temp_storage_bytes,
                                      input.chars_begin(stream),
                                      d_histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_chars,
                                      stream.value());

  auto const zero_level = d_histogram.begin() - lower_level;

  // Firstly, search from the `\n` character to the end of the histogram,
  // so the delimiter will be `\n` if it doesn't exist in the input.
  auto first_zero_count_pos =
    thrust::find(rmm::exec_policy_nosync(stream), zero_level + '\n', d_histogram.end(), 0);
  if (first_zero_count_pos == d_histogram.end()) {
    // Try again, but search from the beginning of the histogram to the last begin position.
    first_zero_count_pos =
      thrust::find(rmm::exec_policy_nosync(stream), d_histogram.begin(), zero_level + '\n', 0);

    // This should never happen, since we are searching even with the characters starting from `\0`.
    if (first_zero_count_pos == d_histogram.end()) {
      throw std::logic_error(
        "Cannot find any character suitable as delimiter during joining json strings.");
    }
  }
  auto const first_non_existing_char = static_cast<char>(first_zero_count_pos - zero_level);

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
    cudf::string_scalar(std::string(1, first_non_existing_char), true, stream, default_mr),
    cudf::string_scalar("{}", true, stream, default_mr),
    stream,
    mr);

  return {std::make_unique<cudf::column>(std::move(is_null_or_empty), rmm::device_buffer{}, 0),
          std::move(concat_strings->release().data),
          first_non_existing_char};
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

}  // namespace spark_rapids_jni