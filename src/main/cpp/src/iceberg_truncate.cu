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
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

namespace spark_rapids_jni {

namespace {

/**
 * @brief Truncate an integer (int32/int64) value.
 *
 * For positive values, Iceberg truncation is: value - (value % width)
 * For negative values, this uses a floored modulo approach:
 * value - (((value % width) + width) % width)
 *
 * Example:
 * - truncate_integer(10, 5) = 0
 * - truncate_integer(10, 15) = 10
 * - truncate_integer(10, -5) = -10
 *
 * @param width Truncation width (must be positive)
 * @param value Integer value to truncate
 * @return Truncated integer value
 */
template <typename T>
struct truncate_integer_fn {
  cudf::column_device_view input;
  int32_t width;

  __device__ T operator()(int row_index) const
  {
    // do not handle nulls here, because 0 input also results in 0 output
    T value = input.element<T>(row_index);
    return value - (((value % width) + width) % width);
  }
};

struct truncate_string_fn {
  // Input data
  cudf::column_device_view d_strings;
  int32_t truncate_length;

  // Output buffers
  cudf::size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(cudf::size_type idx)
  {
    auto str = d_strings.element<cudf::string_view>(idx);
    if (!d_chars) {
      // first phase
      if (str.length() < truncate_length) {
        d_sizes[idx] = str.size_bytes();
      } else {
        auto tmp     = str.substr(0, truncate_length);
        d_sizes[idx] = tmp.size_bytes();
      }
    } else {
      // second phase
      int len = d_offsets[idx + 1] - d_offsets[idx];
      memcpy(d_chars + d_offsets[idx], str.data(), len);
    }
  }
};

std::unique_ptr<cudf::column> truncate_integral_impl(cudf::column_view const& input,
                                                     int32_t width,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    input.type().id() == cudf::type_id::INT32 || input.type().id() == cudf::type_id::INT64,
    "Input must be INT32 or INT64");
  CUDF_EXPECTS(width > 0, "Width must be positive");

  if (input.is_empty()) { return cudf::make_empty_column(input.type().id()); }

  auto output  = cudf::make_fixed_width_column(cudf::data_type{input.type().id()},
                                              input.size(),
                                              cudf::detail::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);
  auto d_input = cudf::column_device_view::create(input, stream);

  if (input.type().id() == cudf::type_id::INT32) {
    using T = int32_t;
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     output->mutable_view().begin<T>(),
                     output->mutable_view().end<T>(),
                     truncate_integer_fn<T>{*d_input, width});
  } else {
    using T = int64_t;
    thrust::tabulate(rmm::exec_policy_nosync(stream),
                     output->mutable_view().begin<T>(),
                     output->mutable_view().end<T>(),
                     truncate_integer_fn<T>{*d_input, width});
  }
  return output;
}

std::unique_ptr<cudf::column> truncate_string_impl(cudf::column_view const& input,
                                                   int32_t length,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRING, "Input must be STRING");
  CUDF_EXPECTS(length > 0, "Length must be positive");

  auto const strings_count = input.size();
  if (strings_count == 0) { return cudf::make_empty_column(cudf::type_id::STRING); }

  auto const strings_col = cudf::strings_column_view(input);
  auto const d_strings   = cudf::column_device_view::create(strings_col.parent(), stream);

  // Build the output strings column using the computed lengths
  auto [offsets, chars] = cudf::strings::detail::make_strings_children(
    truncate_string_fn{*d_strings, length}, strings_count, stream, mr);

  return cudf::make_strings_column(strings_count,
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input, stream, mr));
}

std::unique_ptr<cudf::column> truncate_binary_impl(cudf::column_view const& input,
                                                   int32_t length,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  return nullptr;
}

std::unique_ptr<cudf::column> truncate_decimal32_impl(cudf::column_view const& input,
                                                      int32_t width,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  return nullptr;
}

std::unique_ptr<cudf::column> truncate_decimal64_impl(cudf::column_view const& input,
                                                      int32_t width,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  return nullptr;
}

std::unique_ptr<cudf::column> truncate_decimal128_impl(cudf::column_view const& input,
                                                       int32_t width,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  return nullptr;
}

}  // anonymous namespace

std::unique_ptr<cudf::column> truncate_integral(cudf::column_view const& input,
                                                int32_t width,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return truncate_integral_impl(input, width, stream, mr);
}

std::unique_ptr<cudf::column> truncate_string(cudf::column_view const& input,
                                              int32_t length,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return truncate_string_impl(input, length, stream, mr);
}

std::unique_ptr<cudf::column> truncate_binary(cudf::column_view const& input,
                                              int32_t length,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return truncate_binary_impl(input, length, stream, mr);
}

std::unique_ptr<cudf::column> truncate_decimal32(cudf::column_view const& input,
                                                 int32_t width,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return truncate_decimal32_impl(input, width, stream, mr);
}

std::unique_ptr<cudf::column> truncate_decimal64(cudf::column_view const& input,
                                                 int32_t width,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return truncate_decimal64_impl(input, width, stream, mr);
}

std::unique_ptr<cudf::column> truncate_decimal128(cudf::column_view const& input,
                                                  int32_t width,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return truncate_decimal128_impl(input, width, stream, mr);
}

}  // namespace spark_rapids_jni
