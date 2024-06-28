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

#include "hash.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

namespace spark_rapids_jni {

namespace {

using hive_hash_value_t = int32_t;

constexpr hive_hash_value_t HIVE_HASH_FACTOR = 31;
constexpr hive_hash_value_t HIVE_INIT_HASH   = 0;

hive_hash_value_t __device__ inline compute_int(int32_t key) { return key; }

hive_hash_value_t __device__ inline compute_long(int64_t key)
{
  return (static_cast<uint64_t>(key) >> 32) ^ key;
}

hive_hash_value_t __device__ inline compute_bytes(int8_t const* data, cudf::size_type const len)
{
  hive_hash_value_t ret = HIVE_INIT_HASH;
  for (auto i = 0; i < len; i++) {
    ret = ret * HIVE_HASH_FACTOR + static_cast<int32_t>(data[i]);
  }
  return ret;
}

template <typename Key>
struct hive_hash_function {
  // 'seed' is not used in 'hive_hash_function', but required by 'element_hasher'.
  constexpr hive_hash_function(uint32_t) {}

  [[nodiscard]] hive_hash_value_t __device__ inline operator()(Key const& key) const
  {
    CUDF_UNREACHABLE("Unsupported type for hive hash");
  }
};  // struct hive_hash_function

template <>
hive_hash_value_t __device__ inline hive_hash_function<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<int8_t const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<bool>::operator()(bool const& key) const
{
  return compute_int(static_cast<int32_t>(key));
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int8_t>::operator()(int8_t const& key) const
{
  return compute_int(static_cast<int32_t>(key));
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int16_t>::operator()(
  int16_t const& key) const
{
  return compute_int(static_cast<int32_t>(key));
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int32_t>::operator()(
  int32_t const& key) const
{
  return compute_int(key);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<int64_t>::operator()(
  int64_t const& key) const
{
  return compute_long(key);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<float>::operator()(float const& key) const
{
  auto normalized = spark_rapids_jni::normalize_nans(key);
  auto* p_int     = reinterpret_cast<int32_t const*>(&normalized);
  return compute_int(*p_int);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<double>::operator()(double const& key) const
{
  auto normalized = spark_rapids_jni::normalize_nans(key);
  auto* p_long    = reinterpret_cast<int64_t const*>(&normalized);
  return compute_long(*p_long);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<cudf::timestamp_D>::operator()(
  cudf::timestamp_D const& key) const
{
  auto* p_int = reinterpret_cast<int32_t const*>(&key);
  return compute_int(*p_int);
}

template <>
hive_hash_value_t __device__ inline hive_hash_function<cudf::timestamp_us>::operator()(
  cudf::timestamp_us const& key) const
{
  auto time_as_long            = *reinterpret_cast<int64_t const*>(&key);
  constexpr int MICRO_PER_SEC  = 1000000;
  constexpr int NANO_PER_MICRO = 1000;

  int64_t ts  = time_as_long / MICRO_PER_SEC;
  int64_t tns = (time_as_long % MICRO_PER_SEC) * NANO_PER_MICRO;

  int64_t result = ts;
  result <<= 30;
  result |= tns;

  result = (static_cast<uint64_t>(result) >> 32) ^ result;
  return static_cast<hive_hash_value_t>(result);
}

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * This functor produces the same result as "HiveHash" in Spark for supported types.
 *
 * @tparam hash_function Hash functor to use for hashing elements. Must be hive_hash_function.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
class hive_device_row_hasher {
 public:
  CUDF_HOST_DEVICE hive_device_row_hasher(Nullate check_nulls, cudf::table_device_view t) noexcept
    : _check_nulls{check_nulls}, _table{t}
  {
    // Error out if passed an unsupported hash_function
    static_assert(std::is_base_of_v<hive_hash_function<int>, hash_function<int>>,
                  "hive_device_row_hasher only supports the 'hive_hash_function' hash function");
  }

  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ auto operator()(cudf::size_type row_index) const noexcept
  {
    return cudf::detail::accumulate(
      _table.begin(),
      _table.end(),
      HIVE_INIT_HASH,
      cuda::proclaim_return_type<hive_hash_value_t>(
        [row_index, nulls = this->_check_nulls] __device__(auto hash, auto const& column) {
          auto cur_hash = cudf::type_dispatcher<cudf::experimental::dispatch_void_if_nested>(
            column.type(), element_hasher_adapter{nulls}, column, row_index);
          return HIVE_HASH_FACTOR * hash + cur_hash;
        }));
  }

 private:
  /**
   * @brief Computes the hash value of an element in the given column.
   *
   * Only supported non nested types now
   */
  class element_hasher_adapter {
   public:
    using hash_functor_t = cudf::experimental::row::hash::element_hasher<hash_function, Nullate>;

    __device__ element_hasher_adapter(Nullate check_nulls) noexcept
      : hash_functor{check_nulls, HIVE_INIT_HASH, HIVE_INIT_HASH}
    {
    }

    template <typename T, CUDF_ENABLE_IF(not cudf::is_nested<T>())>
    __device__ hive_hash_value_t operator()(cudf::column_device_view const& col,
                                            cudf::size_type row_index) const noexcept
    {
      return this->hash_functor.template operator()<T>(col, row_index);
    }

    template <typename T, CUDF_ENABLE_IF(cudf::is_nested<T>())>
    __device__ hive_hash_value_t operator()(cudf::column_device_view const& col,
                                            cudf::size_type row_index) const noexcept
    {
      CUDF_UNREACHABLE("Nested type is not supported");
    }

   private:
    hash_functor_t const hash_functor;
  };

  Nullate const _check_nulls;
  cudf::table_device_view const _table;
};
}  // namespace

std::unique_ptr<cudf::column> hive_hash(cudf::table_view const& input,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  auto output = cudf::make_numeric_column(cudf::data_type(cudf::type_to_id<hive_hash_value_t>()),
                                          input.num_rows(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  bool const nullable   = has_nested_nulls(input);
  auto const input_view = cudf::table_device_view::create(input, stream);
  auto output_view      = output->mutable_view();

  // Compute the hash value for each row
  thrust::tabulate(rmm::exec_policy(stream),
                   output_view.begin<hive_hash_value_t>(),
                   output_view.end<hive_hash_value_t>(),
                   hive_device_row_hasher<hive_hash_function, bool>(nullable, *input_view));

  return output;
}

}  // namespace spark_rapids_jni
