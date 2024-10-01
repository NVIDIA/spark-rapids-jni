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
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>

namespace spark_rapids_jni {

namespace detail {

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

      auto const d_str = input.element<cudf::string_view>(idx);
      auto itr         = d_str.data();
      auto const end   = itr + d_str.size_bytes();

      while (itr < end) {
        cudf::char_utf8 ch   = 0;
        auto const chr_width = cudf::strings::detail::to_char_utf8(itr, ch);
        if (not_whitespace(ch)) { break; }
        itr += chr_width;
      }

      bool is_null_literal{false};
      if (itr + 3 < end &&
          (*itr == 'n' && *(itr + 1) == 'u' && *(itr + 2) == 'l' && *(itr + 3) == 'l')) {
        is_null_literal = true;
        itr += 4;
      }

      while (itr < end) {
        cudf::char_utf8 ch   = 0;
        auto const chr_width = cudf::strings::detail::to_char_utf8(itr, ch);
        if (not_whitespace(ch)) {
          is_null_literal = false;
          break;
        }
        itr += chr_width;
      }

      // The current row contains only `null` string literal and not any other non-empty characters.
      // Such rows need to be masked out as null when doing concatenation.
      if (is_null_literal) { return {false, true}; }

      auto const not_eol = itr < end;

      // If the current row is not null or empty, it should start with `{`. Otherwise, we need to
      // replace it by a null. This is necessary for libcudf's JSON reader to work.
      // Note that if we want to support ARRAY schema, we need to check for `[` instead.
      auto constexpr start_character = '{';
      if (not_eol && *itr != start_character) { return {false, true}; }

      return {not_eol, not_eol};
    });

  auto constexpr min_value  = std::numeric_limits<char>::min();
  auto constexpr max_value  = std::numeric_limits<char>::max();
  auto constexpr num_values = max_value - min_value;
  auto const num_chars      = input.chars_size(stream);

  rmm::device_uvector<bool> existence_map(num_values, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), existence_map.begin(), existence_map.end(), false);
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   input.chars_begin(stream),
                   input.chars_end(stream),
                   [min_value, existence = existence_map.begin()] __device__(char ch) {
                     auto const idx = static_cast<int>(ch) - min_value;
                     existence[idx] = true;
                   });

  auto const it            = thrust::make_counting_iterator(0);
  auto const zero_char_idx = -min_value;  // the bin storing count for character `\0`
  auto const zero_char_it  = it + zero_char_idx;

  auto const first_zero_count_pos = thrust::find_if(
    rmm::exec_policy_nosync(stream),
    it,
    it + num_values,
    [zero_char_idx, existence = existence_map.begin()] __device__(auto idx) -> bool {
      if (existence[idx]) { return false; }
      auto const first_non_existing_char = static_cast<char>(idx - zero_char_idx);
      return can_be_delimiter(first_non_existing_char);
    });

  // In theory, this should never happen, since we are searching even with the characters starting
  // from `\0`.
  if (thrust::distance(it, first_zero_count_pos) == num_values) {
    throw std::logic_error(
      "Cannot find any character suitable as delimiter during joining json strings.");
  }
  auto const delimiter = static_cast<char>(thrust::distance(zero_char_it, first_zero_count_pos));

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
