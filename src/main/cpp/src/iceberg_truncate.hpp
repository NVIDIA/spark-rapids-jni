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

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace spark_rapids_jni {

/**
 * @brief Truncate int32 or int64 values for Iceberg partitioning.
 *
 * For integer types, Iceberg truncation is: value - (value % width)
 * where width is the truncation parameter.
 *
 * @param input Integer column to truncate
 * @param width Truncation width
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned column
 * @return Truncated integer column
 */
std::unique_ptr<cudf::column> truncate_integral(
  cudf::column_view const& input,
  int32_t width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Truncate string values for Iceberg partitioning with proper UTF-8 handling.
 *
 * This function truncates strings to the specified byte length while ensuring:
 * 1. UTF-8 multi-byte characters are not split
 * 2. Surrogate code points (U+D800 to U+DFFF) are properly handled and excluded
 * 3. The resulting string is valid UTF-8
 *
 * If a truncation point falls in the middle of a multi-byte UTF-8 character,
 * the truncation backs up to the start of that character. If a character at
 * the truncation boundary encodes a surrogate code point, it is also excluded.
 *
 * @param input String column to truncate
 * @param length Maximum byte length for truncated strings
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned column
 * @return Truncated string column with valid UTF-8 encoding
 */
std::unique_ptr<cudf::column> truncate_string(
  cudf::column_view const& input,
  int32_t length,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Truncate binary values for Iceberg partitioning.
 *
 * This function truncates binary data to the specified byte length.
 * Unlike string truncation, binary truncation simply cuts at the byte boundary
 * without any special handling for character encoding.
 *
 * @param input Binary (list of bytes) column to truncate
 * @param length Maximum byte length for truncated binary data
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned column
 * @return Truncated binary column
 */
std::unique_ptr<cudf::column> truncate_binary(
  cudf::column_view const& input,
  int32_t length,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Truncate decimal values for Iceberg partitioning.
 *
 * This function truncates decimal values using the formula:
 * result = value - (((unscaled_value % unscaled_width) + unscaled_width) % unscaled_width)
 *
 * Supports decimal32, decimal64, and decimal128 types. The scale of the input
 * values is preserved in the output.
 *
 * For decimal types, the width parameter represents the unscaled truncation width.
 * The floored modulo approach ensures proper handling of negative decimal values.
 *
 * @param input Decimal column to truncate (decimal32, decimal64, or decimal128)
 * @param unscaled_width Unscaled truncation width as string (must be positive)
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource used to allocate the returned column
 * @return Truncated decimal column with the same type and scale as input
 */
std::unique_ptr<cudf::column> truncate_decimal32(
  cudf::column_view const& input,
  int32_t width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::column> truncate_decimal64(
  cudf::column_view const& input,
  int32_t width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::column> truncate_decimal128(
  cudf::column_view const& input,
  int32_t width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
