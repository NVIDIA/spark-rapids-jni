/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>

namespace spark_rapids_jni {

/**
 * @brief Supported character set types for decoding.
 */
enum class charset_type : int32_t {
  GBK = 0,
};

/**
 * @brief Decode a binary column from the specified charset encoding to a UTF-8 strings column.
 *
 * Each row of the input column contains bytes in the source charset encoding.
 * The output is a strings column with the same number of rows, where each row
 * contains the UTF-8 encoded string.
 *
 * Invalid byte sequences in the source encoding are replaced with the Unicode
 * replacement character U+FFFD.
 *
 * @param input The input column of type LIST<UINT8> (Spark BinaryType)
 * @param charset The source charset encoding
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for allocating output column
 * @return A new strings column containing the decoded UTF-8 strings
 */
std::unique_ptr<cudf::column> decode_charset(cudf::column_view const& input,
                                             charset_type charset,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr);

}  // namespace spark_rapids_jni
