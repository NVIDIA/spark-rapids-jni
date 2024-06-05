/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "hash.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/algorithm.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/tabulate.h>

namespace spark_rapids_jni {

namespace {

using hash_value_type = int64_t;
using half_size_type  = int32_t;

constexpr __device__ inline int64_t rotate_bits_left_signed(hash_value_type h, int8_t r)
{
  return (h << r) | (h >> (64 - r)) & ~(-1 << r);
}

template <typename Key>
struct XXHash_64 {
  using result_type = hash_value_type;

  constexpr XXHash_64() = delete;
  constexpr XXHash_64(hash_value_type seed) : m_seed(seed) {}

  template <typename T>
  __device__ inline T getblock32(std::byte const* data, cudf::size_type offset) const
  {
    // Read a 4-byte value from the data pointer as individual bytes for safe
    // unaligned access (very likely for string types).
    auto block      = reinterpret_cast<std::uint8_t const*>(data + offset);
    uint32_t result = static_cast<uint32_t>(block[0]) | (static_cast<uint32_t>(block[1]) << 8) |
                      (static_cast<uint32_t>(block[2]) << 16) |
                      (static_cast<uint32_t>(block[3]) << 24);
    return reinterpret_cast<T const*>(&result)[0];
  }

  __device__ inline hash_value_type getblock64(std::byte const* data, cudf::size_type offset) const
  {
    uint64_t result = static_cast<uint64_t>(getblock32<uint32_t>(data, offset)) |
                      static_cast<uint64_t>(getblock32<uint32_t>(data, offset + 4)) << 32;
    return reinterpret_cast<hash_value_type const*>(&result)[0];
  }

  result_type __device__ inline operator()(Key const& key) const { return compute(key); }

  template <typename T>
  result_type __device__ inline compute(T const& key) const
  {
    return compute_bytes(reinterpret_cast<std::byte const*>(&key), sizeof(T));
  }

  result_type __device__ inline compute_remaining_bytes(std::byte const* data,
                                                        cudf::size_type const nbytes,
                                                        cudf::size_type offset,
                                                        result_type h64) const
  {
    // remaining data can be processed in 8-byte chunks
    if ((nbytes % 32) >= 8) {
      for (; offset <= nbytes - 8; offset += 8) {
        hash_value_type k1 = getblock64(data, offset) * prime2;
        k1                 = rotate_bits_left_signed(k1, 31) * prime1;
        h64 ^= k1;
        h64 = rotate_bits_left_signed(h64, 27) * prime1 + prime4;
      }
    }

    // remaining data can be processed in 4-byte chunks
    if (((nbytes % 32) % 8) >= 4) {
      for (; offset <= nbytes - 4; offset += 4) {
        h64 ^= (getblock32<half_size_type>(data, offset) & 0xffffffffL) * prime1;
        h64 = rotate_bits_left_signed(h64, 23) * prime2 + prime3;
      }
    }

    // and the rest
    if (nbytes % 4) {
      while (offset < nbytes) {
        h64 ^= (static_cast<uint8_t>(data[offset]) & 0xff) * prime5;
        h64 = rotate_bits_left_signed(h64, 11) * prime1;
        ++offset;
      }
    }
    return h64;
  }

  result_type __device__ compute_bytes(std::byte const* data, cudf::size_type const nbytes) const
  {
    uint64_t offset = 0;
    hash_value_type h64;
    // data can be processed in 32-byte chunks
    if (nbytes >= 32) {
      auto limit         = nbytes - 32;
      hash_value_type v1 = m_seed + prime1 + prime2;
      hash_value_type v2 = m_seed + prime2;
      hash_value_type v3 = m_seed;
      hash_value_type v4 = m_seed - prime1;

      do {
        // pipeline 4*8byte computations
        v1 += getblock64(data, offset) * prime2;
        v1 = rotate_bits_left_signed(v1, 31);
        v1 *= prime1;
        offset += 8;
        v2 += getblock64(data, offset) * prime2;
        v2 = rotate_bits_left_signed(v2, 31);
        v2 *= prime1;
        offset += 8;
        v3 += getblock64(data, offset) * prime2;
        v3 = rotate_bits_left_signed(v3, 31);
        v3 *= prime1;
        offset += 8;
        v4 += getblock64(data, offset) * prime2;
        v4 = rotate_bits_left_signed(v4, 31);
        v4 *= prime1;
        offset += 8;
      } while (offset <= limit);

      h64 = rotate_bits_left_signed(v1, 1) + rotate_bits_left_signed(v2, 7) +
            rotate_bits_left_signed(v3, 12) + rotate_bits_left_signed(v4, 18);

      v1 *= prime2;
      v1 = rotate_bits_left_signed(v1, 31);
      v1 *= prime1;
      h64 ^= v1;
      h64 = h64 * prime1 + prime4;

      v2 *= prime2;
      v2 = rotate_bits_left_signed(v2, 31);
      v2 *= prime1;
      h64 ^= v2;
      h64 = h64 * prime1 + prime4;

      v3 *= prime2;
      v3 = rotate_bits_left_signed(v3, 31);
      v3 *= prime1;
      h64 ^= v3;
      h64 = h64 * prime1 + prime4;

      v4 *= prime2;
      v4 = rotate_bits_left_signed(v4, 31);
      v4 *= prime1;
      h64 ^= v4;
      h64 = h64 * prime1 + prime4;
    } else {
      h64 = m_seed + prime5;
    }

    h64 += nbytes;
    h64 = compute_remaining_bytes(data, nbytes, offset, h64);

    return finalize(h64);
  }

  constexpr __host__ __device__ hash_value_type finalize(hash_value_type h) const noexcept
  {
    h ^= static_cast<hash_value_type>(static_cast<uint64_t>(h) >> 33);
    h *= prime2;
    h ^= static_cast<hash_value_type>(static_cast<uint64_t>(h) >> 29);
    h *= prime3;
    h ^= static_cast<hash_value_type>(static_cast<uint64_t>(h) >> 32);
    return h;
  }

 private:
  hash_value_type m_seed{};

  static constexpr hash_value_type prime1 = 0x9E3779B185EBCA87L;
  static constexpr hash_value_type prime2 = 0xC2B2AE3D27D4EB4FL;
  static constexpr hash_value_type prime3 = 0x165667B19E3779F9L;
  static constexpr hash_value_type prime4 = 0x85EBCA77C2B2AE63L;
  static constexpr hash_value_type prime5 = 0x27D4EB2F165667C5L;
};

template <>
hash_value_type __device__ inline XXHash_64<bool>::operator()(bool const& key) const
{
  return compute<uint32_t>(key);
}

template <>
hash_value_type __device__ inline XXHash_64<int8_t>::operator()(int8_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
hash_value_type __device__ inline XXHash_64<uint8_t>::operator()(uint8_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
hash_value_type __device__ inline XXHash_64<int16_t>::operator()(int16_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
hash_value_type __device__ inline XXHash_64<uint16_t>::operator()(uint16_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
hash_value_type __device__ inline XXHash_64<float>::operator()(float const& key) const
{
  return compute<float>(spark_rapids_jni::normalize_nans_and_zeros(key));
}

template <>
hash_value_type __device__ inline XXHash_64<double>::operator()(double const& key) const
{
  return compute<double>(spark_rapids_jni::normalize_nans_and_zeros(key));
}

template <>
hash_value_type __device__ inline XXHash_64<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<std::byte const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
hash_value_type __device__ inline XXHash_64<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{
  return compute<uint64_t>(key.value());
}

template <>
hash_value_type __device__ inline XXHash_64<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{
  return compute<uint64_t>(key.value());
}

template <>
hash_value_type __device__ inline XXHash_64<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const
{
  auto [java_d, length] = to_java_bigdecimal(key);
  auto bytes            = reinterpret_cast<std::byte*>(&java_d);
  return compute_bytes(bytes, length);
}

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <typename Nullate>
class device_row_hasher {
 public:
  device_row_hasher(Nullate nulls, cudf::table_device_view const& t, hash_value_type seed)
    : _check_nulls(nulls), _table(t), _seed(seed)
  {
  }

  __device__ auto operator()(cudf::size_type row_index) const noexcept
  {
    return cudf::detail::accumulate(
      _table.begin(),
      _table.end(),
      _seed,
      cuda::proclaim_return_type<hash_value_type>(
        [row_index, nulls = _check_nulls] __device__(auto hash, auto column) {
          return cudf::type_dispatcher(
            column.type(), element_hasher_adapter{}, column, row_index, nulls, hash);
        }));
  }

  /**
   * @brief Computes the hash value of an element in the given column.
   */
  class element_hasher_adapter {
   public:
    template <typename T, CUDF_ENABLE_IF(cudf::column_device_view::has_element_accessor<T>())>
    __device__ hash_value_type operator()(cudf::column_device_view const& col,
                                          cudf::size_type row_index,
                                          Nullate const _check_nulls,
                                          hash_value_type const _seed) const noexcept
    {
      if (_check_nulls && col.is_null(row_index)) { return _seed; }
      auto const hasher = XXHash_64<T>{_seed};
      return hasher(col.element<T>(row_index));
    }

    template <typename T, CUDF_ENABLE_IF(not cudf::column_device_view::has_element_accessor<T>())>
    __device__ hash_value_type operator()(cudf::column_device_view const&,
                                          cudf::size_type,
                                          Nullate const,
                                          hash_value_type const) const noexcept
    {
      CUDF_UNREACHABLE("Unsupported type for xxhash64");
    }
  };

  Nullate const _check_nulls;
  cudf::table_device_view const _table;
  hash_value_type const _seed;
};

}  // namespace

std::unique_ptr<cudf::column> xxhash64(cudf::table_view const& input,
                                       int64_t _seed,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  hash_value_type seed = static_cast<hash_value_type>(_seed);

  auto output = cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<hash_value_type>()),
                                          input.num_rows(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  bool const nullable   = has_nulls(input);
  auto const input_view = cudf::table_device_view::create(input, stream);
  auto output_view      = output->mutable_view();

  // Compute the hash value for each row
  thrust::tabulate(rmm::exec_policy(stream),
                   output_view.begin<hash_value_type>(),
                   output_view.end<hash_value_type>(),
                   device_row_hasher(nullable, *input_view, seed));

  return output;
}

}  // namespace spark_rapids_jni
