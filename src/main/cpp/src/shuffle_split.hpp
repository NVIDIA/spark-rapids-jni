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

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <vector>

namespace spark_rapids_jni {

// TODO: this is duplicated from cudf because the cudf function is not marked as constexpr, so it
// cannot be called on the gpu. there is an issue filed against cudf to make
// is_fixed_point(cudf::data_type type) constexpr. Once that is done, we can remove this
template <typename T>
constexpr inline bool is_fixed_point()
{
  return std::is_same_v<numeric::decimal32, T> || std::is_same_v<numeric::decimal64, T> ||
         std::is_same_v<numeric::decimal128, T> ||
         std::is_same_v<numeric::fixed_point<int32_t, numeric::Radix::BASE_2>, T> ||
         std::is_same_v<numeric::fixed_point<int64_t, numeric::Radix::BASE_2>, T> ||
         std::is_same_v<numeric::fixed_point<__int128_t, numeric::Radix::BASE_2>, T>;
}
struct is_fixed_point_impl {
  template <typename T>
  constexpr bool operator()()
  {
    return is_fixed_point<T>();
  }
};
constexpr bool is_fixed_point(cudf::data_type type)
{
  return cudf::type_dispatcher(type, is_fixed_point_impl{});
}

// per-column information, stored in the metadata header
struct shuffle_split_col_data {
  cudf::type_id type;

  // for kud0 size purposes, we store the num_children/scale parameters as a union, since
  // the only types that use scale are decimal and decimal columns do not have children.
  shuffle_split_col_data(cudf::type_id _type, cudf::size_type _param) : type(_type), param{_param}
  {
  }

  constexpr cudf::size_type num_children() const
  {
    return spark_rapids_jni::is_fixed_point(cudf::data_type{type}) ? 0 : param.num_children;
  }
  cudf::size_type scale() const
  {
    CUDF_EXPECTS(cudf::is_fixed_point(cudf::data_type{type}),
                 "Unexpected call to scale() on a non-fixed-point type.");
    return param.scale;
  }

 private:
  union {
    cudf::size_type num_children;
    cudf::size_type scale;
  } param;
};

// generated by Spark, independent of any cudf data
struct shuffle_split_metadata {
  // depth-first traversal of the input table, by children.
  std::vector<shuffle_split_col_data> col_info;
};

struct shuffle_split_result {
  // packed partition buffers.
  // - it is one big buffer where all of the partitions are glued together instead
  //   of one buffer per-partition
  // - each partition is prepended with a metadata buffer
  //   the metadata is of the format:
  //   - 4 byte row count for the partition
  //   - for each string column in shuffle_split_metadata.col_info
  //     - 4 byte char count
  //   - for each entry in shuffle_split_metadata.col_info
  //     - 1 bit per column indicating whether or not validity info is
  //       included, rounded up the nearest bitmask_type number of elements
  //   - pad to partition_data_align bytes
  //   - the contiguous-split style buffer of column data (which is also padded to
  //   partition_data_align bytes)
  std::unique_ptr<rmm::device_buffer> partitions{};

  // offsets into the partition buffer for each partition. offsets.size() will be
  // num partitions + 1
  rmm::device_uvector<size_t> offsets{0, cudf::get_default_stream()};
};

/**
 * @brief Performs a split operation on a cudf table, returning a buffer of data containing
 * all of the sub-tables as a contiguous buffer of anonymous bytes.
 *
 * Performs a split on the input table similar to the cudf::split function.
 * @code{.pseudo}
 * Example:
 * input:   [{10, 12, 14, 16, 18, 20, 22, 24, 26, 28},
 *           {50, 52, 54, 56, 58, 60, 62, 64, 66, 68}]
 * splits:  {2, 5, 9}
 * output:  [{{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}},
 *           {{50, 52}, {54, 56, 58}, {60, 62, 64, 66}, {68}}]
 * @endcode
 *
 * The result is returned as a blob of bytes representing the individual partitions resulting from
 * the splits, and a set of offsets indicating the beginning of each resulting partition in the
 * result. The function also returns a shuffle_split_metadata struct which contains additional
 * information needed to reconstruct the buffer during shuffle_assemble.
 *
 * @param input The input table
 * @param splits The set of splits to split the table with
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return A shuffle_split_result struct containing the resulting buffer and offsets to each
 * partition, and a shuffle_split_metadata struct which contains the metadata needed to reconstruct
 * a table using shuffle_assemble.
 */
std::pair<shuffle_split_result, shuffle_split_metadata> shuffle_split(
  cudf::table_view const& input,
  std::vector<cudf::size_type> const& splits,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Reassembles a set of partitions generated by shuffle_split into a complete cudf table.
 *
 * @param metadata Metadata describing the contents of the partitions
 * @param partitions A buffer of anonymous bytes representing multiple partitions of data to be
 * merged
 * @param partition_offsets Offsets into the partitions buffer indicating where each individual
 * partition begins. The number of partitions is partition_offsets.size() - 1
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return A cudf table
 */
std::unique_ptr<cudf::table> shuffle_assemble(shuffle_split_metadata const& metadata,
                                              cudf::device_span<uint8_t const> partitions,
                                              cudf::device_span<size_t const> partition_offsets,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr);

}  // namespace spark_rapids_jni
