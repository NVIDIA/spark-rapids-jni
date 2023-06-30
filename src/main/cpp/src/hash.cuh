/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/reverse.h>

namespace spark_rapids_jni {

/**
 * @brief Converts a cudf decimal128 value to a java bigdecimal value.
 *
 * @param key The cudf decimal value
 *
 * @returns A 128 bit value containing the converted decimal bits and a length
 *          representing the relevant number of bytes in the value.
 *
 */
__device__ __inline__ std::pair<__int128_t, cudf::size_type> to_java_bigdecimal(
  numeric::decimal128 key)
{
  // java.math.BigDecimal.valueOf(unscaled_value, _scale).unscaledValue().toByteArray()
  // https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/hash.scala#L381
  __int128_t const val               = key.value();
  constexpr cudf::size_type key_size = sizeof(__int128_t);
  std::byte const* data              = reinterpret_cast<std::byte const*>(&val);

  // Small negative values start with 0xff..., small positive values start with 0x00...
  bool const is_negative     = val < 0;
  std::byte const zero_value = is_negative ? std::byte{0xff} : std::byte{0x00};

  // If the value can be represented with a shorter than 16-byte integer, the
  // leading bytes of the little-endian value are truncated and are not hashed.
  auto const reverse_begin = thrust::reverse_iterator(data + key_size);
  auto const reverse_end   = thrust::reverse_iterator(data);
  auto const first_nonzero_byte =
    thrust::find_if_not(thrust::seq, reverse_begin, reverse_end, [zero_value](std::byte const& v) {
      return v == zero_value;
    }).base();
  // Max handles special case of 0 and -1 which would shorten to 0 length otherwise
  cudf::size_type length =
    std::max(1, static_cast<cudf::size_type>(thrust::distance(data, first_nonzero_byte)));

  // Preserve the 2's complement sign bit by adding a byte back on if necessary.
  // e.g. 0x0000ff would shorten to 0x00ff. The 0x00 byte is retained to
  // preserve the sign bit, rather than leaving an "f" at the front which would
  // change the sign bit. However, 0x00007f would shorten to 0x7f. No extra byte
  // is needed because the leftmost bit matches the sign bit. Similarly for
  // negative values: 0xffff00 --> 0xff00 and 0xffff80 --> 0x80.
  if ((length < key_size) && (is_negative ^ bool(data[length - 1] & std::byte{0x80}))) { ++length; }

  // Convert to big endian by reversing the range of nonzero bytes. Only those bytes are hashed.
  __int128_t big_endian_value = 0;
  auto big_endian_data        = reinterpret_cast<std::byte*>(&big_endian_value);
  thrust::reverse_copy(thrust::seq, data, data + length, big_endian_data);

  return {big_endian_value, length};
}

/**
 * @brief Computes the murmur32 hash value of each row in the input set of columns.
 *
 * @param input The table of columns to hash
 * @param seed Optional seed value to use for the hash function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input.
 */
std::unique_ptr<cudf::column> murmur_hash3_32(
  cudf::table_view const& input,
  uint32_t seed                       = 0,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni