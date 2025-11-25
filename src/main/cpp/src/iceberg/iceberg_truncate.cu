/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "iceberg_truncate.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cstdint>

namespace spark_rapids_jni {

namespace {

/**
 * @brief Truncate towards negative infinity direction for types: int32, int64 or int128
 *
 * Example, width = 10:
 * - truncate(10, 5) = 0
 * - truncate(10, 15) = 10
 * - truncate(10, -5) = -10
 */
template <typename T>
struct truncate_integral_fn {
  cudf::column_device_view input;
  int32_t width;

  __device__ T operator()(int row_index) const
  {
    if (input.is_null(row_index)) {
      return T{};  // null value
    }

    T value = input.element<T>(row_index);
    return value - (((value % width) + width) % width);
  }
};

template <typename T>
constexpr T min_decimal_value();

// decimal32 precision: 9
template <>
constexpr int32_t min_decimal_value<int32_t>()
{
  return -999'999'999;  // it's 9 digits of '9'
}

// decimal64 precision: 18
template <>
constexpr int64_t min_decimal_value<int64_t>()
{
  return -999'999'999'999'999'999L;  // it's 18 digits of '9'
}

// decimal128 precision: 38
// return -99'999'999'999'999'999'999'999'999'999'999'999'999;  // it's 38 digits of '9'
template <>
constexpr __int128_t min_decimal_value<__int128_t>()
{
  constexpr int64_t high = 0x4B3B4CA85A86C47ALL;
  constexpr int64_t low  = 0x098A223FFFFFFFFFLL;
  return -(((__int128_t)high << 64) | (static_cast<__int128_t>(low) & 0xFFFFFFFFFFFFFFFF));
}

/**
 * @brief Check if truncation for decimal types will cause overflow
 *
 * T is the underlying integral type for decimal type: int32, int64 or int128
 */
template <typename T>
struct is_truncate_decimal_overflow_fn {
  constexpr static T MIN = min_decimal_value<T>();
  cudf::column_device_view input;
  int32_t width;

  __device__ bool operator()(int row_index) const
  {
    if (input.is_null(row_index)) { return false; }

    T value = input.element<T>(row_index);
    if (value < 0) {
      T positive_diff = ((value % width) + width) % width;
      if (value < MIN + positive_diff) { return true; }
    }

    return false;
  }
};

/**
 * @brief Truncate by character for string values in UTF-8 encoding.
 */
struct truncate_string_fn {
  // Input data
  char const* input_chars;
  cudf::size_type const* input_offsets;
  int32_t truncate_length;

  // Output buffers
  cudf::size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(cudf::size_type idx)
  {
    cudf::string_view str(input_chars + input_offsets[idx],
                          input_offsets[idx + 1] - input_offsets[idx]);
    if (!d_chars) {
      // first phase
      // Note: one character can be multiple(1-4) bytes for UTF8 encoding
      if (str.length() < truncate_length) {
        d_sizes[idx] = str.size_bytes();
      } else {
        auto tmp     = str.substr(0, truncate_length);
        d_sizes[idx] = tmp.size_bytes();
      }
    } else {
      // second phase
      int out_len = d_offsets[idx + 1] - d_offsets[idx];
      memcpy(d_chars + d_offsets[idx], str.data(), out_len);
    }
  }
};

struct truncate_binary_fn {
  // Input data
  char const* input_chars;
  cudf::size_type const* input_offsets;
  int32_t truncate_length;

  // Output buffers
  cudf::size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(cudf::size_type idx)
  {
    if (!d_chars) {
      // first phase
      int binary_len = input_offsets[idx + 1] - input_offsets[idx];
      d_sizes[idx]   = std::min(binary_len, truncate_length);
    } else {
      // second phase
      auto binary_ptr = input_chars + input_offsets[idx];
      int out_len     = d_offsets[idx + 1] - d_offsets[idx];
      memcpy(d_chars + d_offsets[idx], binary_ptr, out_len);
    }
  }
};

template <typename RepT>
void truncate_integral_and_fill(std::unique_ptr<cudf::column>& output,
                                cudf::column_device_view d_input,
                                int32_t width,
                                rmm::cuda_stream_view stream)
{
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   output->mutable_view().begin<RepT>(),
                   output->mutable_view().end<RepT>(),
                   truncate_integral_fn<RepT>{d_input, width});
}

template <typename T>
bool is_truncate_decimal_overflow(cudf::column_device_view input,
                                  int32_t width,
                                  rmm::cuda_stream_view stream)
{
  return thrust::any_of(rmm::exec_policy_nosync(stream),
                        thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(input.size()),
                        is_truncate_decimal_overflow_fn<T>{input, width});
}

std::unique_ptr<cudf::column> truncate_integral_impl(cudf::column_view const& input,
                                                     int32_t width,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(input.type().id()); }
  auto input_type_id       = input.type().id();
  cudf::size_type num_rows = input.size();
  auto output              = cudf::make_fixed_width_column(input.type(),
                                              num_rows,
                                              cudf::detail::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);
  auto d_input             = cudf::column_device_view::create(input, stream);

  if (input_type_id == cudf::type_id::INT32) {
    truncate_integral_and_fill<int32_t>(output, *d_input, width, stream);
  } else if (input_type_id == cudf::type_id::INT64) {
    truncate_integral_and_fill<int64_t>(output, *d_input, width, stream);
  } else {
    CUDF_FAIL("Unsupported type for truncate_integral_impl");
  }
  return output;
}

std::unique_ptr<cudf::column> truncate_decimal_impl(cudf::column_view const& input,
                                                    int32_t width,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(input.type().id()); }
  auto input_type_id       = input.type().id();
  cudf::size_type num_rows = input.size();
  auto d_input             = cudf::column_device_view::create(input, stream);

  if (input_type_id == cudf::type_id::DECIMAL32) {
    // treat DECIMAL32 column as int32 column
    if (!is_truncate_decimal_overflow<int32_t>(*d_input, width, stream)) {
      auto output = cudf::make_fixed_width_column(input.type(),
                                                  num_rows,
                                                  cudf::detail::copy_bitmask(input, stream, mr),
                                                  input.null_count(),
                                                  stream,
                                                  mr);
      truncate_integral_and_fill<int32_t>(output, *d_input, width, stream);
      return output;
    } else {
      // promote DECIMAL32 to DECIMAL64 to avoid overflow
      auto promote_type     = cudf::data_type{cudf::type_id::DECIMAL64, input.type().scale()};
      auto promoted_input   = cudf::cast(input, promote_type, stream, mr);
      auto output           = cudf::make_fixed_width_column(promote_type,
                                                  num_rows,
                                                  cudf::detail::copy_bitmask(input, stream, mr),
                                                  input.null_count(),
                                                  stream,
                                                  mr);
      auto d_promoted_input = cudf::column_device_view::create(*promoted_input, stream);
      truncate_integral_and_fill<int64_t>(output, *d_promoted_input, width, stream);
      return output;
    }
  } else if (input_type_id == cudf::type_id::DECIMAL64) {
    // treat DECIMAL64 column as int64 column
    if (!is_truncate_decimal_overflow<int64_t>(*d_input, width, stream)) {
      auto output = cudf::make_fixed_width_column(input.type(),
                                                  num_rows,
                                                  cudf::detail::copy_bitmask(input, stream, mr),
                                                  input.null_count(),
                                                  stream,
                                                  mr);
      truncate_integral_and_fill<int64_t>(output, *d_input, width, stream);
      return output;
    } else {
      // promote DECIMAL64 to DECIMAL128 to avoid overflow
      auto promote_type     = cudf::data_type{cudf::type_id::DECIMAL128, input.type().scale()};
      auto promoted_input   = cudf::cast(input, promote_type, stream, mr);
      auto output           = cudf::make_fixed_width_column(promote_type,
                                                  num_rows,
                                                  cudf::detail::copy_bitmask(input, stream, mr),
                                                  input.null_count(),
                                                  stream,
                                                  mr);
      auto d_promoted_input = cudf::column_device_view::create(*promoted_input, stream);
      truncate_integral_and_fill<__int128_t>(output, *d_promoted_input, width, stream);
      return output;
    }
  } else if (input_type_id == cudf::type_id::DECIMAL128) {
    // treat DECIMAL128 column as int128 column
    if (!is_truncate_decimal_overflow<__int128_t>(*d_input, width, stream)) {
      auto output = cudf::make_fixed_width_column(input.type(),
                                                  num_rows,
                                                  cudf::detail::copy_bitmask(input, stream, mr),
                                                  input.null_count(),
                                                  stream,
                                                  mr);
      truncate_integral_and_fill<__int128_t>(output, *d_input, width, stream);
      return output;
    } else {
      // can not promote, throw error
      // Note: Spark can handle, but cuDF can not, should throw runtime error here
      CUDF_FAIL("Truncation causes overflow for DECIMAL128 type");
    }
  } else {
    CUDF_FAIL("Unsupported type for truncate_decimal_impl");
  }
}

std::unique_ptr<cudf::column> truncate_string_impl(cudf::column_view const& input,
                                                   int32_t truncate_length,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRING, "Input must be STRING");
  CUDF_EXPECTS(truncate_length > 0, "Length must be positive");

  auto const num_rows = input.size();
  if (num_rows == 0) { return cudf::make_empty_column(cudf::type_id::STRING); }

  cudf::strings_column_view input_strings_view(input);
  // Build the output strings column using the computed lengths
  auto [offsets, chars] = cudf::strings::detail::make_strings_children(
    truncate_string_fn{input_strings_view.chars_begin(stream),
                       input_strings_view.offsets().begin<cudf::size_type>(),
                       truncate_length},
    num_rows,
    stream,
    mr);

  return cudf::make_strings_column(num_rows,
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input, stream, mr));
}

std::unique_ptr<cudf::column> truncate_binary_impl(cudf::column_view const& input,
                                                   int32_t truncate_length,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(truncate_length > 0, "Length must be positive");
  CUDF_EXPECTS(input.type().id() == cudf::type_id::LIST, "Input must be LIST");
  cudf::lists_column_view list_col(input);
  auto const binary_col_child   = list_col.child();
  auto const binary_col_offsets = list_col.offsets();
  CUDF_EXPECTS(binary_col_child.type().id() == cudf::type_id::UINT8, "Input must be LIST(UINT8)");

  auto const num_rows = input.size();
  if (num_rows == 0) {
    return cudf::lists::detail::make_empty_lists_column(
      cudf::data_type{cudf::type_id::UINT8}, stream, mr);
  }
  CUDF_EXPECTS(!binary_col_child.nullable(), "Child column of binary column must be non-nullable");

  // Build the output binary column using the computed lengths
  // Binary type is list(uint8), it's similar to strings type
  // Here we reuse the strings children building function
  auto [new_offsets, new_chars] = cudf::strings::detail::make_strings_children(
    truncate_binary_fn{
      binary_col_child.begin<char>(), binary_col_offsets.begin<cudf::size_type>(), truncate_length},
    num_rows,
    stream,
    mr);
  auto new_chars_size = new_chars.size();

  auto new_child =
    std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},  // Data type
                                   new_chars_size,                         // Number of elements
                                   new_chars.release(),   // Transfer ownership of the buffer
                                   rmm::device_buffer{},  // no nulls in child
                                   0);

  return cudf::make_lists_column(num_rows,
                                 std::move(new_offsets),
                                 std::move(new_child),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input, stream, mr));
}

}  // anonymous namespace

std::unique_ptr<cudf::column> truncate_integral(cudf::column_view const& input,
                                                int32_t width,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(width > 0, "Width must be positive");
  if (input.type().id() == cudf::type_id::INT32 || input.type().id() == cudf::type_id::INT64) {
    return truncate_integral_impl(input, width, stream, mr);
  }
  return truncate_decimal_impl(input, width, stream, mr);
}

std::unique_ptr<cudf::column> truncate_string(cudf::column_view const& input,
                                              int32_t truncate_length,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return truncate_string_impl(input, truncate_length, stream, mr);
}

std::unique_ptr<cudf::column> truncate_binary(cudf::column_view const& input,
                                              int32_t truncate_length,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return truncate_binary_impl(input, truncate_length, stream, mr);
}

}  // namespace spark_rapids_jni
