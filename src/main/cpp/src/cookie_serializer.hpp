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

#include <cudf/io/types.hpp>
#include <cudf/utilities/span.hpp>

#include <vector>

namespace spark_rapids_jni {

/*
 * Cookie serialization format is a custom serialization format used by Spark RAPIDS that targets
 * efficiency for both data serialization/deserialization and disk IO. Regardless of the number of
 * input buffers, the output serialized data is always one byte vector for efficient disk IO.
 *
 * The name "Cookie" was chosen as the serialized data look like a stack of "cookies" that stores
 * a series of serialized data buffers one after another, similar to how cookies are stacked in a
 * cookie bag.
 *
 * TODO: Add checksum support to the serialized data for integrity verification.
 * TODO: Currently the checksum values are stored as dummy values (0) and is unused.
 *
 * The layout of the serialized data (byte array) is defined as follows:
 *  - First 6 bytes: must be `0x43 0x4f 0x4f 0x4b 0x49 0x45` which equals to string "COOKIE".
 *  - Next 8 bytes: a pair of 32-bit unsigned integers representing the format versions
 *    (major and minor). This is used to make sure the serialized data is compatible with the
 *    current version of the deserializer. The format versions are compatible if their major
 *    versions are the same.
 *  - Next 4 bytes: a 32-bit unsigned integer representing the number of serialized buffers `N`.
 *  - Next 8*(N+1) bytes: a series of 64-bit unsigned integers representing the offsets (in bytes)
 *    of the serialized data belonging to each individual buffer. These offsets are always relative
 *    to the position after the last offset values in the input byte array. The last offset value
 *    should be the total size of the serialized data of all buffers.
 *  - Within the serialized data for each buffer:
 *    - First 4 bytes: a 32-bit unsigned integer representing the checksum of the original data.
 *    - Next 4 bytes: a 32-bit SIGNED integer representing the compression type used for
 *      compressing the original data, which is one of the `cudf::io::compression_type` enum values.
 *      This is stored per buffer as for some buffers, the compressed data may have large size than
 *      then uncompressed one thus we will store the uncompressed data instead.
 *    - The rest bytes: the compressed buffer data if it has smaller size than the original data, or
 *      the original data otherwise.
 */

/**
 * @brief Serialize an array of host buffers into one byte vector using Cookie serialization format.
 *
 * @param compression The compression type to use for compressing the input data for serialization
 * @param inputs An array of host buffers to be serialized
 * @return A vector of bytes representing the serialized data
 */
[[nodiscard]] std::vector<uint8_t> serialize_cookie(
  cudf::host_span<cudf::host_span<uint8_t const> const> inputs,
  cudf::io::compression_type compression = cudf::io::compression_type::SNAPPY);

/**
 * @brief Deserialize a byte vector into an array of byte vectors using Cookie serialization format.
 *
 * @param input A byte vector representing the serialized data in Cookie format
 * @return A vector of byte vectors, each representing a deserialized data buffer
 */
[[nodiscard]] std::vector<std::vector<uint8_t>> deserialize_cookie(
  cudf::host_span<uint8_t const> input);

}  // namespace spark_rapids_jni
