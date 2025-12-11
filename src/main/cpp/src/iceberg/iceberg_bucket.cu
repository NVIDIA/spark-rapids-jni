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

#include "iceberg_bucket.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/tabulate.h>

#include <cstdint>
#include <limits>

namespace spark_rapids_jni {

namespace {

// Constants for bucket computation
constexpr int32_t INT_MAX_VALUE = std::numeric_limits<int32_t>::max();

/**
 * @brief Iceberg MurmurHash3_32 implementation that matches Java Guava's Murmur3_32HashFunction.
 *
 * This implementation is specifically for Iceberg bucket transform and matches the behavior of:
 * - com.google.common.hash.Murmur3_32HashFunction (Guava)
 * - org.apache.iceberg.transforms.Bucket
 *
 * Key differences from Spark's MurmurHash3:
 * - Tail bytes are processed differently: accumulated into k1, then h1 ^= mix_k1(k1) (no mix_h1)
 * - Uses unsigned byte interpretation for tail bytes
 */
class iceberg_murmur_hash3_32 {
 public:
  static constexpr int32_t C1   = 0xcc9e2d51;
  static constexpr int32_t C2   = 0x1b873593;
  static constexpr int32_t SEED = 0;

  /**
   * @brief Rotate left operation - same signature as Java Integer.rotateLeft(int i, int distance)
   * Java: return (i << distance) | (i >>> -distance);
   */
  __device__ static inline int32_t rotate_left(int32_t i, int distance)
  {
    uint32_t u = static_cast<uint32_t>(i);
    return static_cast<int32_t>((u << distance) | (u >> (-distance & 31)));
  }

  /**
   * @brief Mix k1 value - matches Java exactly
   * Java: k1 *= C1; k1 = Integer.rotateLeft(k1, 15); k1 *= C2;
   */
  __device__ static inline int32_t mix_k1(int32_t k1)
  {
    k1 *= C1;
    k1 = rotate_left(k1, 15);
    k1 *= C2;
    return k1;
  }

  /**
   * @brief Mix h1 value - matches Java exactly
   * Java: h1 ^= k1; h1 = Integer.rotateLeft(h1, 13); h1 = h1 * 5 + 0xe6546b64;
   */
  __device__ static inline int32_t mix_h1(int32_t h1, int32_t k1)
  {
    h1 ^= k1;
    h1 = rotate_left(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
    return h1;
  }

  /**
   * @brief Finalization mix - force all bits of a hash block to avalanche
   * Java uses >>> (unsigned right shift), we simulate with cast to uint32_t
   */
  __device__ static inline int32_t fmix(int32_t h1, int32_t length)
  {
    h1 ^= length;
    h1 ^= static_cast<uint32_t>(h1) >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= static_cast<uint32_t>(h1) >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= static_cast<uint32_t>(h1) >> 16;
    return h1;
  }

  /**
   * @brief Read 4 bytes as little-endian int32
   */
  __device__ static inline int32_t get_int_little_endian(uint8_t const* data, int offset)
  {
    return static_cast<int32_t>(static_cast<uint32_t>(data[offset]) |
                                (static_cast<uint32_t>(data[offset + 1]) << 8) |
                                (static_cast<uint32_t>(data[offset + 2]) << 16) |
                                (static_cast<uint32_t>(data[offset + 3]) << 24));
  }

  /**
   * @brief Hash a 64-bit integer
   * Direct translation of Java hashLong
   */
  __device__ static inline int32_t hash_long(int64_t input)
  {
    int32_t low  = static_cast<int32_t>(input);
    int32_t high = static_cast<int32_t>(static_cast<uint64_t>(input) >> 32);

    int32_t k1 = mix_k1(low);
    int32_t h1 = mix_h1(SEED, k1);

    k1 = mix_k1(high);
    h1 = mix_h1(h1, k1);

    return fmix(h1, 8);
  }

  /**
   * @brief Hash a 32-bit integer
   * IMPORTANT: Iceberg uses hashLong for integers! See BucketUtil.java line 38-39:
   *   return MURMUR3.hashLong(value).asInt();
   */
  __device__ static inline int32_t hash_int(int32_t input)
  {
    // Iceberg promotes int to long and uses hashLong
    return hash_long(static_cast<int64_t>(input));
  }

  /**
   * @brief Hash a byte array
   * Matches Java: hashBytes(byte[] input, int off, int len)
   *
   * For strings (UTF-8 encoded), this is the correct method to use since
   * C++ strings are already UTF-8 encoded bytes.
   */
  __device__ static inline int32_t hash_bytes(uint8_t const* data, int32_t len)
  {
    int32_t h1 = SEED;
    int32_t i  = 0;

    // Process 4-byte chunks
    for (; i + 4 <= len; i += 4) {
      int32_t k1 = mix_k1(get_int_little_endian(data, i));
      h1         = mix_h1(h1, k1);
    }

    // Process remaining bytes (tail)
    // Key difference from Spark: accumulate all tail bytes into k1, then h1 ^= mix_k1(k1)
    int32_t k1 = 0;
    for (int shift = 0; i < len; i++, shift += 8) {
      // Use unsigned byte interpretation (toInt in Java's UnsignedBytes)
      k1 ^= static_cast<int32_t>(data[i]) << shift;
    }
    h1 ^= mix_k1(k1);

    return fmix(h1, len);
  }

  /**
   * @brief Hash a string view (UTF-8 encoded)
   */
  __device__ static inline int32_t hash_string(cudf::string_view const& str)
  {
    return hash_bytes(reinterpret_cast<uint8_t const*>(str.data()), str.size_bytes());
  }

  /**
   * @brief Hash a decimal32 value
   * Iceberg hashes decimals as their unscaled BigInteger value in minimal two's complement bytes
   */
  __device__ static inline int32_t hash_decimal32(numeric::decimal32 const& value)
  {
    // For decimal32, the unscaled value is an int32
    // We need to convert to minimal two's complement byte representation
    int32_t unscaled = value.value();
    return hash_decimal_unscaled(unscaled);
  }

  /**
   * @brief Hash a decimal64 value
   */
  __device__ static inline int32_t hash_decimal64(numeric::decimal64 const& value)
  {
    int64_t unscaled = value.value();
    return hash_decimal_unscaled(unscaled);
  }

  /**
   * @brief Hash a decimal128 value
   */
  __device__ static inline int32_t hash_decimal128(numeric::decimal128 const& value)
  {
    __int128_t unscaled = value.value();
    return hash_decimal_unscaled_128(unscaled);
  }

 private:
  /**
   * @brief Get the number of bytes needed for minimal two's complement representation
   */
  template <typename T>
  __device__ static inline int32_t minimal_bytes_needed(T value)
  {
    if (value == 0) return 1;

    // For negative numbers, find the position of the first 0 bit from the left
    // For positive numbers, find the position of the first 1 bit from the left
    int leading;
    if constexpr (sizeof(T) == 4) {
      if (value < 0) {
        leading = __clz(~static_cast<uint32_t>(value));
      } else {
        leading = __clz(static_cast<uint32_t>(value));
      }
    } else if constexpr (sizeof(T) == 8) {
      if (value < 0) {
        leading = __clzll(~static_cast<uint64_t>(value));
      } else {
        leading = __clzll(static_cast<uint64_t>(value));
      }
    }

    int significant_bits = sizeof(T) * 8 - leading;
    // Add 1 for sign bit, then round up to bytes
    return (significant_bits + 8) / 8;
  }

  /**
   * @brief Hash an unscaled decimal value (int32 or int64)
   * Converts to minimal two's complement big-endian bytes and hashes
   */
  template <typename T>
  __device__ static inline int32_t hash_decimal_unscaled(T value)
  {
    // Get minimal byte representation (big-endian, two's complement)
    int32_t num_bytes = minimal_bytes_needed(value);

    // Convert to big-endian bytes
    uint8_t bytes[8];
    for (int i = 0; i < num_bytes; i++) {
      bytes[num_bytes - 1 - i] = static_cast<uint8_t>(value >> (i * 8));
    }

    return hash_bytes(bytes, num_bytes);
  }

  /**
   * @brief Hash an unscaled decimal128 value
   */
  __device__ static inline int32_t hash_decimal_unscaled_128(__int128_t value)
  {
    if (value == 0) {
      uint8_t zero = 0;
      return hash_bytes(&zero, 1);
    }

    // Determine number of bytes needed
    int32_t num_bytes;
    if (value > 0) {
      // Count leading zeros
      uint64_t high = static_cast<uint64_t>(static_cast<unsigned __int128>(value) >> 64);
      uint64_t low  = static_cast<uint64_t>(value);
      int leading;
      if (high != 0) {
        leading = __clzll(high);
      } else {
        leading = 64 + __clzll(low);
      }
      int significant_bits = 128 - leading;
      num_bytes            = (significant_bits + 8) / 8;
    } else {
      // For negative, count leading ones (find first 0)
      uint64_t high = static_cast<uint64_t>(static_cast<unsigned __int128>(value) >> 64);
      uint64_t low  = static_cast<uint64_t>(value);
      int leading;
      if (~high != 0) {
        leading = __clzll(~high);
      } else {
        leading = 64 + __clzll(~low);
      }
      int significant_bits = 128 - leading;
      num_bytes            = (significant_bits + 8) / 8;
    }

    // Convert to big-endian bytes
    uint8_t bytes[16];
    unsigned __int128 uvalue = static_cast<unsigned __int128>(value);
    for (int i = 0; i < num_bytes; i++) {
      bytes[num_bytes - 1 - i] = static_cast<uint8_t>(uvalue >> (i * 8));
    }

    return hash_bytes(bytes, num_bytes);
  }
};

// Bucket functors using iceberg_murmur_hash3_32

/**
 * @brief Functor to compute bucket for int32
 */
struct bucket_int32_fn {
  cudf::column_device_view input;
  int32_t num_buckets;

  __device__ int32_t operator()(cudf::size_type row_idx) const
  {
    if (input.is_null(row_idx)) { return 0; }
    int32_t value  = input.element<int32_t>(row_idx);
    int32_t hash   = iceberg_murmur_hash3_32::hash_int(value);
    int32_t bucket = (hash & INT_MAX_VALUE) % num_buckets;
    return bucket;
  }
};

/**
 * @brief Functor to compute bucket for int64
 */
struct bucket_int64_fn {
  cudf::column_device_view input;
  int32_t num_buckets;

  __device__ int32_t operator()(cudf::size_type row_idx) const
  {
    if (input.is_null(row_idx)) { return 0; }
    int64_t value  = input.element<int64_t>(row_idx);
    int32_t hash   = iceberg_murmur_hash3_32::hash_long(value);
    int32_t bucket = (hash & INT_MAX_VALUE) % num_buckets;
    return bucket;
  }
};

/**
 * @brief Functor to compute bucket for string type (UTF-8 encoded)
 */
struct bucket_string_fn {
  cudf::column_device_view input;
  int32_t num_buckets;

  __device__ int32_t operator()(cudf::size_type row_idx) const
  {
    if (input.is_null(row_idx)) { return 0; }
    cudf::string_view str = input.element<cudf::string_view>(row_idx);
    int32_t hash          = iceberg_murmur_hash3_32::hash_string(str);
    int32_t bucket        = (hash & INT_MAX_VALUE) % num_buckets;
    return bucket;
  }
};

/**
 * @brief Functor to compute bucket for binary type (LIST of UINT8)
 */
struct bucket_binary_fn {
  uint8_t const* chars;
  cudf::size_type const* offsets;
  cudf::bitmask_type const* null_mask;
  int32_t num_buckets;

  __device__ int32_t operator()(cudf::size_type row_idx) const
  {
    if (null_mask != nullptr && !cudf::bit_is_set(null_mask, row_idx)) { return 0; }

    cudf::size_type start = offsets[row_idx];
    cudf::size_type end   = offsets[row_idx + 1];
    cudf::size_type len   = end - start;

    int32_t hash   = iceberg_murmur_hash3_32::hash_bytes(chars + start, len);
    int32_t bucket = (hash & INT_MAX_VALUE) % num_buckets;
    return bucket;
  }
};

/**
 * @brief Functor to compute bucket for date type (TIMESTAMP_DAYS)
 * Iceberg stores dates as days since epoch (int32)
 */
struct bucket_date_fn {
  cudf::column_device_view input;
  int32_t num_buckets;

  __device__ int32_t operator()(cudf::size_type row_idx) const
  {
    if (input.is_null(row_idx)) { return 0; }
    // Date is stored as int32 days since epoch
    int32_t value  = input.element<int32_t>(row_idx);
    int32_t hash   = iceberg_murmur_hash3_32::hash_int(value);
    int32_t bucket = (hash & INT_MAX_VALUE) % num_buckets;
    return bucket;
  }
};

/**
 * @brief Functor to compute bucket for timestamp type (TIMESTAMP_MICROSECONDS)
 * Iceberg stores timestamps as microseconds since epoch (int64)
 */
struct bucket_timestamp_fn {
  cudf::column_device_view input;
  int32_t num_buckets;

  __device__ int32_t operator()(cudf::size_type row_idx) const
  {
    if (input.is_null(row_idx)) { return 0; }
    // Timestamp is stored as int64 microseconds since epoch
    int64_t value  = input.element<int64_t>(row_idx);
    int32_t hash   = iceberg_murmur_hash3_32::hash_long(value);
    int32_t bucket = (hash & INT_MAX_VALUE) % num_buckets;
    return bucket;
  }
};

/**
 * @brief Functor to compute bucket for decimal32
 */
struct bucket_decimal32_fn {
  cudf::column_device_view input;
  int32_t num_buckets;

  __device__ int32_t operator()(cudf::size_type row_idx) const
  {
    if (input.is_null(row_idx)) { return 0; }
    numeric::decimal32 value = input.element<numeric::decimal32>(row_idx);
    int32_t hash             = iceberg_murmur_hash3_32::hash_decimal32(value);
    int32_t bucket           = (hash & INT_MAX_VALUE) % num_buckets;
    return bucket;
  }
};

/**
 * @brief Functor to compute bucket for decimal64
 */
struct bucket_decimal64_fn {
  cudf::column_device_view input;
  int32_t num_buckets;

  __device__ int32_t operator()(cudf::size_type row_idx) const
  {
    if (input.is_null(row_idx)) { return 0; }
    numeric::decimal64 value = input.element<numeric::decimal64>(row_idx);
    int32_t hash             = iceberg_murmur_hash3_32::hash_decimal64(value);
    int32_t bucket           = (hash & INT_MAX_VALUE) % num_buckets;
    return bucket;
  }
};

/**
 * @brief Functor to compute bucket for decimal128
 */
struct bucket_decimal128_fn {
  cudf::column_device_view input;
  int32_t num_buckets;

  __device__ int32_t operator()(cudf::size_type row_idx) const
  {
    if (input.is_null(row_idx)) { return 0; }
    numeric::decimal128 value = input.element<numeric::decimal128>(row_idx);
    int32_t hash              = iceberg_murmur_hash3_32::hash_decimal128(value);
    int32_t bucket            = (hash & INT_MAX_VALUE) % num_buckets;
    return bucket;
  }
};

std::unique_ptr<cudf::column> compute_bucket_impl(cudf::column_view const& input,
                                                  int32_t num_buckets,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(num_buckets > 0, "num_buckets must be positive");

  cudf::size_type num_rows = input.size();
  if (num_rows == 0) { return cudf::make_empty_column(cudf::type_id::INT32); }

  // Create output column with INT32 type
  auto output = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                              num_rows,
                                              cudf::detail::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);

  auto d_input     = cudf::column_device_view::create(input, stream);
  auto output_view = output->mutable_view();

  auto type_id = input.type().id();

  switch (type_id) {
    case cudf::type_id::INT32: {
      thrust::tabulate(rmm::exec_policy_nosync(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       bucket_int32_fn{*d_input, num_buckets});
      break;
    }
    case cudf::type_id::INT64: {
      thrust::tabulate(rmm::exec_policy_nosync(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       bucket_int64_fn{*d_input, num_buckets});
      break;
    }
    case cudf::type_id::DECIMAL32: {
      thrust::tabulate(rmm::exec_policy_nosync(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       bucket_decimal32_fn{*d_input, num_buckets});
      break;
    }
    case cudf::type_id::DECIMAL64: {
      thrust::tabulate(rmm::exec_policy_nosync(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       bucket_decimal64_fn{*d_input, num_buckets});
      break;
    }
    case cudf::type_id::DECIMAL128: {
      thrust::tabulate(rmm::exec_policy_nosync(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       bucket_decimal128_fn{*d_input, num_buckets});
      break;
    }
    case cudf::type_id::TIMESTAMP_DAYS: {
      thrust::tabulate(rmm::exec_policy_nosync(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       bucket_date_fn{*d_input, num_buckets});
      break;
    }
    case cudf::type_id::TIMESTAMP_MICROSECONDS: {
      thrust::tabulate(rmm::exec_policy_nosync(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       bucket_timestamp_fn{*d_input, num_buckets});
      break;
    }
    case cudf::type_id::STRING: {
      thrust::tabulate(rmm::exec_policy_nosync(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       bucket_string_fn{*d_input, num_buckets});
      break;
    }
    case cudf::type_id::LIST: {
      // Binary is represented as LIST of UINT8
      cudf::lists_column_view list_col(input);
      auto const child = list_col.child();
      CUDF_EXPECTS(child.type().id() == cudf::type_id::UINT8, "Binary type must be LIST of UINT8");

      auto const offsets_view = list_col.offsets();
      thrust::tabulate(rmm::exec_policy_nosync(stream),
                       output_view.begin<int32_t>(),
                       output_view.end<int32_t>(),
                       bucket_binary_fn{child.begin<uint8_t>(),
                                        offsets_view.begin<cudf::size_type>(),
                                        input.null_mask(),
                                        num_buckets});
      break;
    }
    default:
      CUDF_FAIL("Unsupported type for bucket transform: " +
                std::to_string(static_cast<int>(type_id)));
  }

  return output;
}

}  // anonymous namespace

std::unique_ptr<cudf::column> compute_bucket(cudf::column_view const& input,
                                             int32_t num_buckets,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return compute_bucket_impl(input, num_buckets, stream, mr);
}

}  // namespace spark_rapids_jni
