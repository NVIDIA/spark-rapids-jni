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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cstddef>
#include <numeric>

#include "shuffle_split.hpp"
#include "shuffle_split_detail.hpp"

namespace spark_rapids_jni {

using namespace cudf;
using namespace spark_rapids_jni::detail;

/*
 * Code block for computing assemble_column_info structures for columns and column instances. 
 * Key function: assemble_build_column_info
 *
 */
namespace {

#define OUTPUT_ITERATOR(__name, __T, __field_name)                                                  \
  template<typename __T>                                                                            \
  struct __name##generic_output_iter {                                                              \
    __T* c;                                                                                         \
    using value_type        = decltype(__T::__field_name);                                          \
    using difference_type   = size_t;                                                               \
    using pointer           = decltype(__T::__field_name)*;                                         \
    using reference         = decltype(__T::__field_name)&;                                         \
    using iterator_category = thrust::output_device_iterator_tag;                                   \
                                                                                                    \
    __name##generic_output_iter operator+ __host__ __device__(int i) { return {c + i}; }            \
                                                                                                    \
    __name##generic_output_iter& operator++ __host__ __device__()                                   \
    {                                                                                               \
      c++;                                                                                          \
      return *this;                                                                                 \
    }                                                                                               \
                                                                                                    \
    reference operator[] __device__(int i) { return dereference(c + i); }                           \
    reference operator* __device__() { return dereference(c); }                                     \
                                                                                                    \
  private:                                                                                          \
    reference __device__ dereference(__T* c) { return c->__field_name; }                            \
  };                                                                                                \
  using __name = __name##generic_output_iter<__T>

/**
 * @brief Struct which contains information about columns and column instances.
 *
 * Used for both columns (one per output column) and column instances (one column per
 * output column * the number of partitions)
 */
struct assemble_column_info {
  cudf::type_id         type;
  bool                  has_validity;
  size_type             num_rows, num_chars;
  size_type             valid_count;
  size_type             num_children;

  // only valid for column instances
  size_type             row_index;
  size_type             char_index;
};
OUTPUT_ITERATOR(assemble_column_info_has_validity_output_iter, assemble_column_info, has_validity);

/**
 * @brief Helper function for computing the max depth of a column 
 */
template <typename ColumnIter>
ColumnIter compute_max_depth_traverse(ColumnIter col, int depth, int& max_depth)
{ 
  auto start = col;
  col++;
  max_depth = max(max_depth, depth);
  for(int idx=0; idx<start->num_children(); idx++){
    col = compute_max_depth_traverse(col, depth+1, max_depth);
  }
  return col;
}

/**
 * @brief Computes the set of column indices for all root columns that contain offsets, and the
 * max depth of their hierarchies.
 *
 * We need to do some extra processing on all columns that contain offsets, so we build a list
 * of all the 'root' columns which contain offsets and generate the max depth of all the hierarchies
 * within them. 
 * 
 *
 * @param metadata Metadata for the input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr User provided resource used for allocating the returned device memory
 *
 * @returns A vector containing the start index and number of child columns for a root offset column, and the max depth of
 * any hierarchy.
 */
std::pair<rmm::device_uvector<thrust::pair<int, int>>, int> compute_root_offset_columns_and_max_depth(shuffle_split_metadata const& metadata,
                                                                                                      rmm::cuda_stream_view stream,
                                                                                                      rmm::device_async_resource_ref mr)
{  
  int max_depth = 0;
  std::vector<thrust::pair<int, int>> root_offset_columns;
  root_offset_columns.reserve(metadata.col_info.size());  // worst case
  auto col = metadata.col_info.begin();
  while(col != metadata.col_info.end()){
    auto end = compute_max_depth_traverse(col, 0, max_depth);
    if(col->type == cudf::type_id::STRING || col->type == cudf::type_id::LIST){
      root_offset_columns.push_back({static_cast<int>(std::distance(metadata.col_info.begin(), col)), static_cast<int>(std::distance(col, end))});
    }
    col = end;
  }
  return {cudf::detail::make_device_uvector_async(root_offset_columns, stream, mr), max_depth};
}

/**
 * @brief Kernel which computes the row counts for all column instances that are the children
 * of one or more offset columns.
 * 
 * The number of rows in the partition header is only applicable to the root columns of the
 * input. As soon as we have a column that contains offsets (strings or lists), the row counts 
 * change for all children. This kernel computes these row counts for all child column instances.
 *
 * @param offset_columns Offset column indices and child counts
 * @param column_metadata Input column metadata
 * @param column_instances The column instance structs to be filled in with row counts.
 * @param partitions The input buffer
 * @param partition_offsets Per-partition offsets into the input buffer
 * @param per_partition_metadata_size Size of the header for each partition
 * 
 */
__global__ void compute_offset_child_row_counts(cudf::device_span<thrust::pair<int, int> const> offset_columns,
                                                cudf::device_span<shuffle_split_col_data const> column_metadata,
                                                cudf::device_span<assemble_column_info> column_instances,
                                                cudf::device_span<uint8_t const> partitions,
                                                cudf::device_span<size_t const> partition_offsets,
                                                size_t per_partition_metadata_size)
{
  if(threadIdx.x != 0){
    return;
  }
  auto const partition_index = blockIdx.x;
  partition_header const*const pheader = reinterpret_cast<partition_header const*>(partitions.begin() + partition_offsets[partition_index]);
  size_t const offsets_begin = cudf::util::round_up_safe(partition_offsets[partition_index] + per_partition_metadata_size + pheader->validity_size, validity_pad);
  size_type const* offsets = reinterpret_cast<size_type const*>(partitions.begin() + offsets_begin);
  
  auto base_col_index = column_metadata.size() * partition_index;
  for(auto idx=0; idx<offset_columns.size(); idx++){
    // root column starts with the number of rows in the partition
    auto num_rows = pheader->num_rows;
    auto col_index = offset_columns[idx].first;
    auto const num_cols = offset_columns[idx].second;
    auto col_instance_index = col_index + base_col_index;
    
    int depth = 0;
    for(auto c_idx=0; c_idx<num_cols; c_idx++){    
      auto const& meta = column_metadata[col_index];
      auto& col_inst = column_instances[col_instance_index];
      col_inst.num_rows = num_rows;
      switch(meta.type){
      case cudf::type_id::STRING:
        col_inst.num_chars = offsets[num_rows] - offsets[0];
        offsets += (num_rows + 1);
        depth--;
        break;
      case cudf::type_id::LIST: {
          auto const last_num_rows = num_rows + 1;
          num_rows = offsets[num_rows] - offsets[0];
          offsets += last_num_rows;
          depth++; }
        break;
      default:
        depth--;
        break;
      }
      col_index++;
      col_instance_index++;
    }
  }
}

// returns:
// - a vector of assemble_column_info structs representing the destination column data.
//   the vector is of length global_metadata.col_info.size()  that is, the flattened list of columns in the table.
//
// - the same vector as above, but in host memory. 
//
// - a vector of assemble_column_info structs, representing the source column data.
//   the vector is of length global_metadata.col_info.size() * the # of partitions. 
//
/**
 * @brief Generate assemble_column_info structures for columns and column_instances
 *
 * We need various pieces of information at the column level (corresponding to the
 * actual input schema) and the column instance level (per-column, per-partition), such
 * as type, number of children, row counts, etc. This function generates that data from
 * the input buffer.
 * 
 * @param shuffle_split_metadata Global metadata about the partition buffer
 * @param partitions The partition buffer. One large blob of bytes.
 * @param partition_offsets Per-partition offets into the buffer
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr User provided resource used for allocating the returned device memory
 * 
 * @return per-column info, per-column-instance info and the total per-partition metadata size
 */
std::tuple<rmm::device_uvector<assemble_column_info>,
           rmm::device_uvector<assemble_column_info>,
           size_t>
assemble_build_column_info(shuffle_split_metadata const& h_global_metadata,
                           cudf::device_span<uint8_t const> partitions,
                           cudf::device_span<size_t const> partition_offsets,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<shuffle_split_col_data> global_metadata = cudf::detail::make_device_uvector_async(h_global_metadata.col_info, stream, temp_mr);

  // "columns" here means the number of flattened columns in the entire source table, not just the
  // number of columns at the top level
  auto const num_columns = global_metadata.size();
  size_type const num_partitions = partition_offsets.size() - 1;
  auto const num_column_instances = num_columns * num_partitions;

  // return values
  rmm::device_uvector<assemble_column_info> column_info(num_columns, stream, mr);
  rmm::device_uvector<assemble_column_info> column_instance_info(num_column_instances, stream, mr);

  // compute per-partition metadata size
  auto const per_partition_metadata_size = compute_per_partition_metadata_size(h_global_metadata.col_info.size());

  // generate per-column data ------------------------------------------------------    

  // compute has-validity
  // note that we are iterating vertically -> horizontally here, with each column's individual piece per partition first.
  auto column_keys = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_type>([num_partitions] __device__ (size_type i){
    return i / num_partitions;
  }));
  auto has_validity_values = cudf::detail::make_counting_transform_iterator(0, 
    cuda::proclaim_return_type<bool>([num_partitions,
                                      partitions = partitions.data(),
                                      partition_offsets = partition_offsets.begin()]
                                      __device__ (int i) -> bool {
      auto const partition_index = i % num_partitions;
      bitmask_type const*const has_validity_buf = reinterpret_cast<bitmask_type const*>(partitions + partition_offsets[partition_index] + sizeof(partition_header));
      auto const col_index = i / num_partitions;
      
      return has_validity_buf[col_index / 32] & (1 << (col_index % 32)) ? 1 : 0;
    })
  );
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                        column_keys,
                        column_keys + num_column_instances,
                        has_validity_values,
                        thrust::make_discard_iterator(),
                        assemble_column_info_has_validity_output_iter{column_info.begin()},
                        thrust::equal_to<size_type>{},
                        thrust::logical_or<bool>{});
  
  // compute everything else except row count (which will be done later after we have computed column instance information)
  auto iter = thrust::make_counting_iterator(0);
  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr), iter, iter + num_columns, [column_info = column_info.begin(),
                                                                                        global_metadata = global_metadata.begin()]
                                                                                        __device__ (size_type col_index){
    auto const& metadata = global_metadata[col_index];
    auto& cinfo = column_info[col_index];
    
    cinfo.type = metadata.type;
    cinfo.valid_count = 0;
    cinfo.num_children = metadata.num_children();
  });

  // generate per-column-instance data ------------------------------------------------------

  // has-validity, type, # of children, row count for non-offset child columns
  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr), iter, iter + num_column_instances, [column_instance_info = column_instance_info.begin(),
                                                                                                 global_metadata = global_metadata.begin(),
                                                                                                 partitions = partitions.data(),
                                                                                                 partition_offsets = partition_offsets.begin(),
                                                                                                 num_columns,
                                                                                                 per_partition_metadata_size]
                                                                                                 __device__ (size_type i){
    auto const partition_index = i / num_columns;
    auto const col_index = i % num_columns;
    auto const col_instance_index = (partition_index * num_columns) + col_index;

    auto const& metadata = global_metadata[col_index];
    auto& cinstance_info = column_instance_info[col_instance_index];

    uint8_t const*const buf_start = reinterpret_cast<uint8_t const*>(partitions + partition_offsets[partition_index]);
    partition_header const*const pheader = reinterpret_cast<partition_header const*>(buf_start);

    bitmask_type const*const has_validity_buf = reinterpret_cast<bitmask_type const*>(buf_start + sizeof(partition_header));
    cinstance_info.has_validity = has_validity_buf[col_index / 32] & (1 << (col_index % 32)) ? 1 : 0;
    
    cinstance_info.type = metadata.type;
    cinstance_info.valid_count = 0;
    cinstance_info.num_chars = 0;
    cinstance_info.num_children = metadata.num_children();

    // note that this will be incorrect for any columns that are children of offset columns. those values will be fixed up below.
    cinstance_info.num_rows = pheader->num_rows;
  });
  
  // reconstruct row counts for columns and columns instances  ------------------------------

  // compute row counts for offset-based column instances.
  // TODO: the kudo format forces us to be less parallel here than we could be. maybe find a way around that
  // which doesn't grow size very much.

  // get the indices of root offset columns and the max depth of any hierarchy
  auto [root_offset_columns, max_depth] = compute_root_offset_columns_and_max_depth(h_global_metadata, stream, temp_mr);

  // parallelize by partition.
  // unfortunately, there's no way to parallelize this at the column level. we don't know where the offsets start in the partition buffer for any given column, 
  // so we have to march through each partition linearly. to fix this, we'd have to change the kudo format in a way that would increase it's size.
  // I'm doing this as a kernel instead of through thrust so that I can guarantee each partition is being marched by a seperate block to 
  // avoid thread divergence.
  compute_offset_child_row_counts<<<num_partitions, 32, 0, stream.value()>>>(root_offset_columns,
                                                                             global_metadata,
                                                                             column_instance_info,
                                                                             partitions,
                                                                             partition_offsets,
                                                                             per_partition_metadata_size);

  // returns column instance from index, in the order of 0->num_partitions, 0->num_columns
  auto col_instance_vertical = cuda::proclaim_return_type<assemble_column_info&>([num_partitions, 
                                                                                  num_columns,
                                                                                  column_instance_info = column_instance_info.begin()]
                                                                                  __device__ (int i) -> assemble_column_info& {
    auto const partition_index = i % num_partitions;
    auto const col_inst_index = (partition_index * num_columns) + (i / num_partitions);
    return column_instance_info[col_inst_index];
  });

  // compute row indices per column instance  
  auto col_inst_row_index = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<int&>([col_instance_vertical] __device__ (int i) -> int& {
    return col_instance_vertical(i).row_index;
  }));
  auto col_inst_row_count = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_type>([col_instance_vertical] __device__ (int i){
    return col_instance_vertical(i).num_rows;
  }));
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                                column_keys,
                                column_keys + num_column_instances,
                                col_inst_row_count,
                                col_inst_row_index);

  // compute char indices per column instance  
  auto col_inst_char_index = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<int&>([col_instance_vertical] __device__ (int i) -> int& {
    return col_instance_vertical(i).char_index;
  }));  
  auto col_inst_char_count = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_type>([col_instance_vertical] __device__ (int i){
    return col_instance_vertical(i).num_chars;
  }));
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(stream, temp_mr),
                                column_keys,
                                column_keys + num_column_instances,
                                col_inst_char_count,
                                col_inst_char_index);

  // compute row counts and char counts. because we already know per-instance indices and counts, this can be done without a reduction
  thrust::for_each(rmm::exec_policy_nosync(stream, temp_mr),
                   iter,
                   iter + num_columns,
                   [col_inst_begin = (num_partitions - 1) * num_columns,
                    column_info = column_info.begin(),
                    column_instance_info = column_instance_info.begin()] __device__ (int i){
                      auto const& last_col_inst = column_instance_info[col_inst_begin + i];
                      column_info[i].num_rows = last_col_inst.row_index + last_col_inst.num_rows;
                      column_info[i].num_chars = last_col_inst.char_index + last_col_inst.num_chars;
                   });

  return {std::move(column_info), std::move(column_instance_info), per_partition_metadata_size};
}

} // anonymous namespace for assemble_build_column_info

/*
 * Code block for generating output buffers and the associated copy batch information 
 * needed to fill them
 * Key function: assemble_build_buffers
 *
 */
namespace {

/**
 * @brief Compute size of a bitmask in bytes, including padding
 * 
 * @param number_of_bits Number of bits in the bitmask
 * @param pad Padding for the final buffer size
 * 
 * @return Size of the required allocation in bytes
 */
constexpr size_t bitmask_allocation_size_bytes(size_type number_of_bits, int pad)
{
  return cudf::util::round_up_safe((number_of_bits + 7) / 8, pad);
}

/**
 * @brief Functor that fills in buffer sizes (validity, offsets, data) per column.
 * 
 * This returns the size of the buffer -without- padding. Just the size of
 * the raw bytes containing the actual data.
 *
 */
struct assemble_buffer_size_functor {
  template <typename T, typename OutputIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  __device__ void operator()(assemble_column_info const& col, OutputIter validity_out, OutputIter offsets_out, OutputIter data_out)
  {
    // validity
    *validity_out = col.has_validity ? bitmask_allocation_size_bytes(col.num_rows, 1) : 0;

    // no offsets for fixed width types
    *offsets_out = 0;

    // data
    *data_out = cudf::type_dispatcher(data_type{col.type}, size_of_helper{}) * col.num_rows;
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter validity_out, OutputIter offsets_out, OutputIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? bitmask_allocation_size_bytes(col.num_rows, 1) : 0;

    // offsets
    *offsets_out = sizeof(size_type) * (col.num_rows + 1);

    // no data for lists
    *data_out = 0;
  } 

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter validity_out, OutputIter offsets_out, OutputIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? bitmask_allocation_size_bytes(col.num_rows, 1) : 0;

    // no offsets or data for structs
    *offsets_out = 0;
    *data_out = 0;
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  __device__ void operator()(assemble_column_info const& col, OutputIter validity_out, OutputIter offsets_out, OutputIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? bitmask_allocation_size_bytes(col.num_rows, 1) : 0;

    // chars
    *data_out = sizeof(int8_t) * col.num_chars;

    // offsets
    *offsets_out = sizeof(size_type) * (col.num_rows + 1);
  }

  template <typename T, typename OutputIter, CUDF_ENABLE_IF(!std::is_same_v<T, cudf::struct_view> && 
                                                            !std::is_same_v<T, cudf::list_view> && 
                                                            !std::is_same_v<T, cudf::string_view> && 
                                                            !cudf::is_fixed_width<T>())>
  __device__ void operator()(assemble_column_info const& col, OutputIter validity_out, OutputIter offsets_out, OutputIter data_out)
  {
  }
};

/**
 * @brief Utility function which expands a range of sizes, and invokes a function on 
 * each element in all of the groups, providing group and subgroup indices, returning
 * the generated list of values.
 *
 * As an example, imagine we had the input
 * [2, 5, 1]
 * 
 * transform_expand will invoke `op` 8 times, with the values
 * 
 * (0, 0) (0, 1),  (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),  (2, 0)
 * 
 * @param first Beginning of the input range
 * @param end End of the input range
 * @param op Function to be invoked on each expanded value. This function should return
 * a single value that gets collected into the overall result returned from transform_expand
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr User provided resource used for allocating the returned device memory
 *
 * @return The resulting device_uvector containing the output from `op`
 */
template<typename SizeIterator, typename GroupFunction>
rmm::device_uvector<std::invoke_result_t<GroupFunction>> transform_expand(SizeIterator first,
                                                                          SizeIterator last,
                                                                          GroupFunction op,
                                                                          rmm::cuda_stream_view stream,
                                                                          rmm::device_async_resource_ref mr)
{ 
  auto temp_mr = cudf::get_current_device_resource_ref();

  auto value_count = std::distance(first, last);
  auto size_wrapper = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([value_count, first] __device__ (size_t i){
    return i >= value_count ? 0 : first[i];
  }));
  rmm::device_uvector<size_t> group_offsets(value_count + 1, stream, temp_mr);
  thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                         size_wrapper,
                         size_wrapper + group_offsets.size(),
                         group_offsets.begin());
  size_t total_size = group_offsets.back_element(stream); // note memcpy and device sync
  
  using OutputType = std::invoke_result_t<GroupFunction>;
  rmm::device_uvector<OutputType> result(total_size, stream, mr);
  auto iter = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + total_size,
                    result.begin(),
                    cuda::proclaim_return_type<OutputType>([op, group_offsets_begin = group_offsets.begin(), group_offsets_end = group_offsets.end()] __device__ (size_t i){
                      auto const group_index = thrust::lower_bound(thrust::seq, group_offsets_begin, group_offsets_end, i) - group_offsets_begin;
                      auto const intra_group_index = i - group_offsets_begin[group_index];
                      return op(group_index, intra_group_index);
                    }));

  return result;
}

/**
 * @brief Functor that fills in buffer sizes (validity, offsets, data) per column.
 * 
 * This returns the size of the buffer -without- padding. Just the size of
 * the raw bytes containing the actual data.
 *
 */
struct assemble_buffer_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, typename BufIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  void operator()(assemble_column_info const& col, BufIter validity_out, BufIter offsets_out, BufIter data_out)
  {
    // validity
    *validity_out = col.has_validity ? alloc_validity(col.num_rows) : rmm::device_buffer(0, stream, mr);

    // no offsets for fixed width types
    
    // data
    auto const data_size = cudf::util::round_up_safe(cudf::type_dispatcher(data_type{col.type}, size_of_helper{}) * col.num_rows, split_align);
    *data_out = rmm::device_buffer(data_size, stream, mr);
  }

  template <typename T, typename BufIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  void operator()(assemble_column_info const& col, BufIter validity_out, BufIter offsets_out, BufIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? alloc_validity(col.num_rows) : rmm::device_buffer(0, stream, mr);    

    // offsets
    auto const offsets_size = cudf::util::round_up_safe(sizeof(size_type) * (col.num_rows + 1), split_align);
    *offsets_out = rmm::device_buffer(offsets_size, stream, mr);

    // no data for lists
  } 

  template <typename T, typename BufIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  void operator()(assemble_column_info const& col, BufIter validity_out, BufIter offsets_out, BufIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? alloc_validity(col.num_rows) : rmm::device_buffer(0, stream, mr);

    // no offsets or data for structs
  }

  template <typename T, typename BufIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  void operator()(assemble_column_info const& col, BufIter validity_out, BufIter offsets_out, BufIter data_out)
  { 
    // validity
    *validity_out = col.has_validity ? alloc_validity(col.num_rows) : rmm::device_buffer(0, stream, mr);

    // chars
    *data_out = rmm::device_buffer(col.num_chars, stream, mr);

    // offsets
    auto const offsets_size = cudf::util::round_up_safe(sizeof(size_type) * (col.num_rows + 1), split_align);
    *offsets_out = rmm::device_buffer(offsets_size, stream, mr);    
  }

  template <typename T, typename BufIter, CUDF_ENABLE_IF(!std::is_same_v<T, cudf::struct_view> && 
                                                         !std::is_same_v<T, cudf::list_view> && 
                                                         !std::is_same_v<T, cudf::string_view> && 
                                                         !cudf::is_fixed_width<T>())>
  void operator()(assemble_column_info const& col, BufIter& validity_out, BufIter& offsets_out, BufIter& data_out)
  { 
    CUDF_FAIL("Unsupported type in assemble_buffer_functor");
  }
 
private:
  rmm::device_buffer alloc_validity(size_type num_rows)
  {
    auto res = rmm::device_buffer(bitmask_allocation_size_bytes(num_rows, split_align), stream, mr);
    // necessary because of the way the validity copy step works (use of atomicOr instead of stores for copy endpoints). 
    // TODO: think of a way to eliminate.
    cudaMemsetAsync(res.data(), 0, res.size(), stream);
    return res;
  }
};

// The size that shuffle_assemble uses internally as the GPU unit of work.
// For the validity and offset copy steps, the work is broken up into
// (bytes / desired_assemble_batch_size) kernel blocks
constexpr std::size_t desired_assemble_batch_size = 1 * 1024 * 1024;

/**
 * @brief Return the number of batches the specified number of bytes should be split into.
 */
constexpr size_t size_to_batch_count(size_t bytes)
{
  return std::max(std::size_t{1}, util::round_up_unsafe(bytes, desired_assemble_batch_size) / desired_assemble_batch_size);
}

/**
 * @brief Information on a copy batch.
 */
struct assemble_batch {
  __device__ assemble_batch(uint8_t const* _src, uint8_t* _dst, size_t _size, buffer_type _btype, int _value_shift, int _src_bit_shift, int _dst_bit_shift, size_type _validity_row_count, size_type* _valid_count):
    src(_src), dst(_dst), size(_size), btype(_btype), value_shift(_value_shift), src_bit_shift(_src_bit_shift), dst_bit_shift(_dst_bit_shift), validity_row_count(_validity_row_count), valid_count(_valid_count){}

  uint8_t const* src;
  uint8_t* dst;
  size_t              size;     // bytes
  buffer_type         btype;
  int value_shift;              // amount to shift values down by (for offset buffers)
  int src_bit_shift;            // source bit (right) shift. easy way to think about this is
                                // 'the number of rows at the beginning of the buffer to ignore'.
                                // we need to ignore them because the split-copy happens at 
                                // byte boundaries, not bit/row boundaries. so we may have 
                                // irrelevant rows at the very beginning.
  int dst_bit_shift;            // dest bit (left) shift
  size_type validity_row_count; // only valid for validity buffers
  size_type* valid_count;       // (output) validity count for this block of work
};

/**
 * @brief Generate the final (but still empty) output device memory buffers for the reassembled columns
 * and the set of copy batches needed to fill them.
 * 
 * @param column_info Per-column information
 * @param h_column_info Host memory per-column information
 * @param column_instance_info Per-column-instance information
 * a single value that gets collected into the overall result returned from transform_expand
 * @param partitions The partition buffer
 * @param partition_offsets Per-partition offsets into the partition buffer
 * @param per_partition_metadata_size Per-partition metadata header size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr User provided resource used for allocating the returned device memory
 *
 * @return The output device buffers and the copy batches needed to fill them
 */
std::pair<std::vector<rmm::device_buffer>, rmm::device_uvector<assemble_batch>> assemble_build_buffers(cudf::device_span<assemble_column_info> column_info,
                                                                                                       cudf::host_span<assemble_column_info const> h_column_info,
                                                                                                       cudf::device_span<assemble_column_info const> const& column_instance_info,
                                                                                                       cudf::device_span<uint8_t const> partitions,
                                                                                                       cudf::device_span<size_t const> partition_offsets,
                                                                                                       size_t per_partition_metadata_size,
                                                                                                       rmm::cuda_stream_view stream,
                                                                                                       rmm::device_async_resource_ref mr)
{  
  auto temp_mr = cudf::get_current_device_resource_ref();
  auto const num_columns = column_info.size();
  auto const num_column_instances = column_instance_info.size();
  auto const num_partitions = partition_offsets.size() - 1;

  // to simplify things, we will reserve 3 buffers for each column. validity, data, offsets. not every column will use all of them, so those
  // buffers will remain unallocated/zero size. 
  size_t const num_dst_buffers = num_columns * 3;
  size_t const num_src_buffers = num_dst_buffers * num_partitions;
  size_t const buffers_per_partition = num_dst_buffers;

  // allocate output buffers. ordered in the array as (validity, offsets, data) per column.
  // so for 4 columns and 2 partitions, the ordering would be:
  // vod vod vod vod | vod vod vod vod
  std::vector<rmm::device_buffer> assemble_buffers(num_dst_buffers);
  auto dst_validity_iter = assemble_buffers.begin();
  auto dst_offsets_iter = assemble_buffers.begin() + 1;
  auto dst_data_iter = assemble_buffers.begin() + 2;
  for(size_t idx=0; idx<column_info.size(); idx++){
    cudf::type_dispatcher(cudf::data_type{h_column_info[idx].type},
                          assemble_buffer_functor{stream, mr},
                          h_column_info[idx],
                          dst_validity_iter,
                          dst_offsets_iter,
                          dst_data_iter);
    dst_validity_iter += 3;
    dst_offsets_iter += 3;
    dst_data_iter += 3;
  }
  std::vector<uint8_t*> h_dst_buffers(assemble_buffers.size());
  std::transform(assemble_buffers.begin(), assemble_buffers.end(), h_dst_buffers.begin(), [](rmm::device_buffer& buf){
    return reinterpret_cast<uint8_t*>(buf.data());
  });
  auto dst_buffers = cudf::detail::make_device_uvector_async(h_dst_buffers, stream, temp_mr);  

  // compute:
  // - unpadded sizes of the source buffers
  // - offsets into the incoming partition data where each source buffer starts
  // - offsets into the output buffer data where each source buffer is copied into
  
  // ordered by the same as the incoming partition buffers (all validity buffers, all offset buffers, all data buffers)
  // so for 4 columns and 2 partitions, the ordering would be:
  // vvvv oooo dddd | vvvv oooo dddd
  rmm::device_uvector<size_t> src_sizes_unpadded(num_src_buffers, stream, mr);
  // these are per-source-buffer absolute offsets into the incoming partition buffer
  rmm::device_uvector<size_t> src_offsets(num_src_buffers, stream, mr);
  // arranged in destination buffer order, by partition
  // so for 4 columns and 2 partitions, the ordering would be:
  // vv oo dd vv oo dd vv oo dd vv oo dd  vp0/vp1, op0/op1, dp0/dp1, etc
  // 0  0  0  1  1  1  2  2  2  3  3  3   col_index
  rmm::device_uvector<size_t> dst_offsets(num_src_buffers, stream, mr);
  
  // mapping functions to try and manage the slipperiness of the various orderings.
  //
  // for 4 columns, 2 partitions
  // source buffer ordering: vvvv oooo dddd    | vvvv oooo dddd
  // dst buffer ordering:    vod vod vod vod   (the dst buffers are not partitioned. they are the final column output buffers)
  // dst offset ordering:    vv oo dd vv oo dd | vv oo dd vv oo dd
  auto src_buf_to_type = cuda::proclaim_return_type<size_t>([buffers_per_partition, num_columns] __device__ (size_t src_buf_index){
    return (src_buf_index % buffers_per_partition) / num_columns; // 0, 1, 2 (validity, offsets, data)
  });
  auto src_buf_to_dst_buf = cuda::proclaim_return_type<size_t>([buffers_per_partition, num_columns, src_buf_to_type] __device__ (size_t src_buf_index){
    auto const col_index = src_buf_index % num_columns;
    auto const buffer_index = src_buf_to_type(src_buf_index);
    return (col_index * 3) + buffer_index;
  });
  auto col_buffer_inst_to_dst_offset = cuda::proclaim_return_type<size_t>([num_columns, num_partitions] __device__ (size_t partition_index, size_t col_index, size_t buffer_index){
    return (col_index * num_partitions * 3) + (buffer_index * num_partitions) + partition_index;
  });
  auto src_buf_to_dst_offset = cuda::proclaim_return_type<size_t>([num_columns, buffers_per_partition, src_buf_to_type, col_buffer_inst_to_dst_offset] __device__ (size_t src_buf_index){
    auto const partition_index = src_buf_index / buffers_per_partition;
    auto const col_index = src_buf_index % num_columns;
    auto const buffer_index = src_buf_to_type(src_buf_index);
    return col_buffer_inst_to_dst_offset(partition_index, col_index, buffer_index);
  });
  auto dst_offset_to_src_buf = cuda::proclaim_return_type<size_t>([num_partitions, num_columns, buffers_per_partition] __device__ (size_t dst_offset_index){
    auto const partition_index = dst_offset_index % num_partitions;
    auto const col_index = dst_offset_index / (num_partitions * 3);
    auto const buffer_index = (dst_offset_index / num_partitions) % 3;
    return (partition_index * buffers_per_partition) + col_index + (buffer_index * num_columns);
  });

  {
    // generate unpadded sizes of the source buffers
    auto const num_column_instances = column_instance_info.size();
    auto iter = thrust::make_counting_iterator(0);
    thrust::for_each(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + num_column_instances,
                    [buffers_per_partition,
                     num_columns,
                     column_instance_info = column_instance_info.begin(),
                     src_sizes_unpadded = src_sizes_unpadded.begin()] __device__ (size_type i){

      auto const partition_index = i / num_columns;
      auto const col_index = i % num_columns;
      auto const col_instance_index = (partition_index * num_columns) + col_index;

      auto const& cinfo_instance = column_instance_info[col_instance_index];
      auto const validity_buf_index = (partition_index * buffers_per_partition) + col_index;
      auto const offset_buf_index = (partition_index * buffers_per_partition) + num_columns + col_index;
      auto const data_buf_index = (partition_index * buffers_per_partition) + (num_columns * 2) + col_index;
      cudf::type_dispatcher(cudf::data_type{cinfo_instance.type},
                            assemble_buffer_size_functor{},
                            cinfo_instance,
                            &src_sizes_unpadded[validity_buf_index],
                            &src_sizes_unpadded[offset_buf_index],
                            &src_sizes_unpadded[data_buf_index]);
    });

    // scan to source offsets, by partition
    auto partition_keys = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([buffers_per_partition] __device__ (size_t i){
      return i / buffers_per_partition;
    }));
    thrust::exclusive_scan_by_key(rmm::exec_policy(stream, temp_mr),
                                  partition_keys,
                                  partition_keys + num_src_buffers,
                                  src_sizes_unpadded.begin(),
                                  src_offsets.begin());
    
    // adjust the source offsets:
    // - add metadata offset
    // - take padding into account
    // - add partition offset
    thrust::for_each(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + num_column_instances,
                    [num_columns,
                     buffers_per_partition,
                     column_instance_info = column_instance_info.begin(),
                     src_offsets = src_offsets.begin(),
                     partition_offsets = partition_offsets.begin(),
                     partitions = partitions.data(),
                     per_partition_metadata_size] __device__ (size_type i){

      auto const partition_index = i / num_columns;
      auto const partition_offset = partition_offsets[partition_index];
      auto const col_index = i % num_columns;
      auto const col_instance_index = (partition_index * num_columns) + col_index;
      
      partition_header const*const pheader = reinterpret_cast<partition_header const*>(partitions + partition_offset);

      auto const validity_buf_index = (partition_index * buffers_per_partition) + col_index;
      auto const offset_buf_index = validity_buf_index + num_columns;
      auto const data_buf_index = offset_buf_index + num_columns;

      auto const validity_section_offset = partition_offset + per_partition_metadata_size;
      src_offsets[validity_buf_index] += validity_section_offset;
    
      auto const offset_section_offset = cudf::util::round_up_safe(validity_section_offset + pheader->validity_size, validity_pad);
      src_offsets[offset_buf_index] = (src_offsets[offset_buf_index] - pheader->validity_size) + offset_section_offset;
      
      auto const data_section_offset = cudf::util::round_up_safe(offset_section_offset + pheader->offset_size, offset_pad);
      src_offsets[data_buf_index] = (src_offsets[data_buf_index] - (pheader->validity_size + pheader->offset_size)) + data_section_offset;
    });

    // compute: destination buffer offsets. see note above about ordering of dst_offsets.
    // Note: we're wasting a little work here as the work for validity has to be redone later.
    {
      auto dst_buf_key = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([num_partitions] __device__ (size_t i){
        return i / num_partitions;
      }));
      auto size_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([src_sizes_unpadded = src_sizes_unpadded.begin(), num_partitions, dst_offset_to_src_buf] __device__ (size_t i){
        auto const src_buf_index = dst_offset_to_src_buf(i);
        auto const buffer_index = (i / num_partitions) % 3;
        bool const is_offsets_buffer = buffer_index == 1;

        // there is a mismatch between input and output sizes when it comes to offset buffers. Each partition contains num_rows+1 offsets, however as we 
        // reassembly them, we only consume num_rows offsets from each partition (except for the last one). So adjust our side accordingly
        return src_sizes_unpadded[src_buf_index] - ((is_offsets_buffer && src_sizes_unpadded[src_buf_index] > 0) ? 4 : 0);
      }));
      thrust::exclusive_scan_by_key(rmm::exec_policy(stream, temp_mr),
                                    dst_buf_key,
                                    dst_buf_key + num_src_buffers,
                                    size_iter,
                                    dst_offsets.begin());
    }

    // for validity, we need to do a little more work. our destination positions are defined by bit position,
    // not byte position. so round down into the nearest starting bitmask word. note that this implies we will
    // potentially be writing our leading bits into the same word as another copy is writing it's trailing bits, so atomics
    // will be necessary.
    thrust::for_each(rmm::exec_policy(stream, temp_mr),
                     iter,
                     iter + num_column_instances,
                     [column_info = column_info.begin(),
                      column_instance_info = column_instance_info.begin(),
                      num_columns,
                      num_partitions,
                      col_buffer_inst_to_dst_offset,
                      dst_offsets = dst_offsets.begin()] __device__ (size_t i){
      auto const partition_index = i / num_columns;
      auto const col_index = i % num_columns;
      auto const& cinfo = column_info[col_index];
      if(cinfo.has_validity){
        // for 4 columns and 2 partitions, the ordering of offsets is:
        // vv oo dd vv oo dd vv oo dd vv oo dd  vp0/vp1, op0/op1, dp0/dp1, etc
        auto const dst_offset_index = col_buffer_inst_to_dst_offset(partition_index, col_index, static_cast<int>(buffer_type::VALIDITY));
        auto const col_instance_index = (partition_index * num_columns) + col_index;
        dst_offsets[dst_offset_index] = (column_instance_info[col_instance_index].row_index / 32) * sizeof(bitmask_type);
      }
    });
  }
  
  // generate copy batches ------------------------------------
  //
  // - validity and offsets will be copied by custom kernels, so we will subdivide them them into batches of 1 MB
  // - data is copied by cub, so we will do no subdivision of the batches under the assumption that cub will make it's
  //   own smart internal decisions
  auto batch_count_iter = cudf::detail::make_counting_transform_iterator(0, 
                                                                         cuda::proclaim_return_type<size_t>([src_sizes_unpadded = src_sizes_unpadded.begin(), src_buf_to_type] __device__ (size_t src_buf_index){
                                                                          // data buffers use cub batched memcpy, so we won't pre-batch them at all. just use the raw size.
                                                                           return static_cast<buffer_type>(src_buf_to_type(src_buf_index)) == buffer_type::DATA ? 1 : size_to_batch_count(src_sizes_unpadded[src_buf_index]);
                                                                         }));
  auto copy_batches = transform_expand(batch_count_iter, 
                                       batch_count_iter + src_sizes_unpadded.size(),
                                       cuda::proclaim_return_type<assemble_batch>([dst_buffers = dst_buffers.begin(),
                                                                                   dst_offsets = dst_offsets.begin(),
                                                                                   partitions = partitions.data(),
                                                                                   partition_offsets = partition_offsets.data(),
                                                                                   buffers_per_partition,
                                                                                   num_partitions,
                                                                                   src_sizes_unpadded = src_sizes_unpadded.begin(),
                                                                                   src_offsets = src_offsets.begin(),
                                                                                   desired_assemble_batch_size = desired_assemble_batch_size,
                                                                                   column_info = column_info.begin(),
                                                                                   column_instance_info = column_instance_info.begin(),
                                                                                   num_columns,
                                                                                   src_buf_to_dst_buf,
                                                                                   src_buf_to_dst_offset,
                                                                                   src_buf_to_type] __device__ (size_t src_buf_index, size_t batch_index){
                                         auto const batch_offset = batch_index * desired_assemble_batch_size;
                                         auto const partition_index = src_buf_index / buffers_per_partition;
                                         auto const col_index = src_buf_index % num_columns;
                                         auto const col_instance_index = (partition_index * num_columns) + col_index;
                                         
                                         auto const src_offset = src_offsets[src_buf_index];

                                         buffer_type const btype = static_cast<buffer_type>(src_buf_to_type(src_buf_index));
                                                                                  
                                         auto const dst_buf_index = src_buf_to_dst_buf(src_buf_index);
                                         auto const dst_offset_index = src_buf_to_dst_offset(src_buf_index);
                                         auto const dst_offset = dst_offsets[dst_offset_index];

                                         auto const bytes = [&] __device__ (){
                                           switch(btype){
                                           // validity gets batched
                                           case buffer_type::VALIDITY:
                                             return std::min(src_sizes_unpadded[src_buf_index] - batch_offset, desired_assemble_batch_size);

                                           // for offsets, all source buffers have an extra offset per partition (the terminating offset for that partition)
                                           // that we need to ignore, except in the case of the final partition.
                                           case buffer_type::OFFSETS:
                                             {
                                              if(src_sizes_unpadded[src_buf_index] == 0){
                                                return size_t{0};
                                              }
                                              bool const end_of_buffer = (batch_offset + desired_assemble_batch_size) >= src_sizes_unpadded[src_buf_index];
                                              if(!end_of_buffer){
                                                return desired_assemble_batch_size;
                                              } else {
                                                auto const size = std::min(src_sizes_unpadded[src_buf_index] - batch_offset, desired_assemble_batch_size);
                                                return partition_index == num_partitions - 1 ? size : size - 4;
                                              }
                                             }
                                           default:
                                            break;
                                           }

                                           // data copies go through the cub batched memcopy, so just do the whole thing in one shot.
                                           return src_sizes_unpadded[src_buf_index];
                                         }();

                                         partition_header const*const pheader = reinterpret_cast<partition_header const*>(partitions + partition_offsets[partition_index]);
                                         auto const validity_rows_per_batch = desired_assemble_batch_size * 8;
                                         auto const validity_batch_row_index = (batch_index * validity_rows_per_batch);
                                         auto const validity_row_count = min(pheader->num_rows - validity_batch_row_index, validity_rows_per_batch);
                                         // since the initial split copy is done on simple byte boundaries, the first bit we want to copy may not
                                         // be the first bit in the source buffer. so we need to shift right by these leading bits.
                                         // for example, the partition may start at row 3. but in that case, we will have started copying from 
                                         // byte 0. so we have to shift right 3 rows.
                                         auto const row_index = column_instance_info[col_instance_index].row_index;
                                         auto const src_bit_shift = row_index % 8;
                                         auto const dst_bit_shift = row_index % 32;
                                      
                                         // to transform the incoming unadjusted offsets into final offsets
                                         int const offset_shift = [&] __device__ (){
                                           if(btype != buffer_type::OFFSETS){
                                             return 0;
                                           }

                                           auto const& col = column_info[col_index];
                                           // subtract the first offset value in the buffer, then add the row/char index
                                           if(col.type == cudf::type_id::STRING){
                                             return column_instance_info[col_instance_index].char_index - (reinterpret_cast<size_type const*>(partitions + src_offset))[0];
                                           } else if(col.type == cudf::type_id::LIST){
                                             auto const& child_inst = column_instance_info[col_instance_index + 1];
                                             return child_inst.row_index - (reinterpret_cast<size_type const*>(partitions + src_offset))[0];
                                           }
                                           // not an offset based column
                                           return 0;
                                         }();

                                         auto& cinfo = column_info[col_index];
                                         return assemble_batch {
                                          partitions + src_offset + batch_offset,
                                          dst_buffers[dst_buf_index] + dst_offset + batch_offset,
                                          bytes,
                                          btype,
                                          offset_shift,
                                          src_bit_shift,
                                          dst_bit_shift,
                                          static_cast<size_type>(validity_row_count),
                                          &cinfo.valid_count};
                                         }),
                                       stream,
                                       mr);

  return {std::move(assemble_buffers), std::move(copy_batches)};
}

} // anonymous namespace for assemble_build_buffers

/*
 * Code block for copying validity, offset and data buffers.
 * Key function: assemble_copy_data
 *
 */
namespace {

/**
 * @brief Copy validity batches.
 * 
 * This kernel assumes misaligned source buffers and an aligned (4 byte) destination buffer.
 */
template<int block_size>
__global__ void copy_validity(cudf::device_span<assemble_batch> batches)
{
  int batch_index = blockIdx.x;
  auto& batch = batches[batch_index];
  if(batch.size <= 0 || (batch.btype != buffer_type::VALIDITY)){
    return;
  }
  
  __shared__ bitmask_type prev_word[block_size];

  // note that in several cases here, we are reading past the end of the actual input buffer. but this is safe
  // because we are always guaranteed final padding of partitions out to 8 bytes.
  
  // how many leading misaligned bytes we have
  int const leading_bytes = (4 - (reinterpret_cast<uint64_t>(batch.src) % 4)) % 4;
  int remaining_rows = batch.validity_row_count;

  // - if the address is misaligned, load byte-by-byte and only store up to that many bits/rows off instead of a full 32  
  // handle the leading (misaligned) bytes in the source
  // Note: src_bit_shift effectively means "how many rows at the beginning should we ignore", so we subtract that from
  // the amount of rows that are actually in the bytes that we've read.
  int rows_in_batch = min(remaining_rows, leading_bytes != 0 ? (leading_bytes * 8) - batch.src_bit_shift 
                                                             : 32 - batch.src_bit_shift);
  remaining_rows -= rows_in_batch;

  size_type valid_count = 0;
  
  // thread 0 does all the work for the leading (unaligned) bytes in the source
  if(threadIdx.x == 0){
    bitmask_type word = leading_bytes != 0 ? (batch.src[0] | (batch.src[1] << 8) | (batch.src[2] << 16) | (batch.src[3] << 24))
                                            : (reinterpret_cast<bitmask_type const*>(batch.src))[0];
    bitmask_type const relevant_row_mask = ((1 << rows_in_batch) - 1);

    // shift and mask the incoming word so that bit 0 is the first row we're going to store.
    word = (word >> batch.src_bit_shift) & relevant_row_mask;
    // any bits that are not being stored in the current dest word get overflowed to the next copy
    prev_word[0] = word >> (32 - batch.dst_bit_shift);
    // shift to the final destination bit position.
    word <<= batch.dst_bit_shift;
    // count and store
    valid_count += __popc(word);
    // use an atomic because we could be overlapping with another copy
    atomicOr(reinterpret_cast<bitmask_type*>(batch.dst), word);
  }
  if(remaining_rows == 0){
    if(threadIdx.x == 0){
      atomicAdd(batch.valid_count, valid_count);
    }
    return;
  }

  // src and dst pointers. src will be word-aligned now
  auto src = reinterpret_cast<bitmask_type const*>(batch.src + leading_bytes);
  auto dst = reinterpret_cast<bitmask_type*>(batch.dst);
  
  // compute a new bit_shift.
  // - src_bit_shift is now irrelevant because we have skipped past any leading irrelevant bits in the input 
  // - the amount we have to dst shift is simply incremented by the number of rows we've processed.
  auto const bit_shift = (batch.dst_bit_shift + rows_in_batch) % 32;
  auto remaining_words = (remaining_rows + 31) / 32;

  // we can still be anywhere in the destination buffer, so any stores to either the first or last
  // destination words (where other copies may be happening at the same time) need to use atomicOr.
  auto store_word = [last_word_index = remaining_words - 1, dst] __device__ (int i, uint32_t val){
    if(i == 0 || i == last_word_index){
      atomicOr(dst + i, val);
    } else {
      dst[i] = val;
    }
  };

  // copy the remaining words
  auto const rows_per_batch = blockDim.x * 32;  
  int src_word_index = threadIdx.x;
  auto const num_leading_words = ((rows_in_batch + batch.dst_bit_shift) / 32);
  int dst_word_index = threadIdx.x + num_leading_words; // NOTE: we may still be at destination word 0
  int words_in_batch;
  bitmask_type cur, prev;
  do {
    words_in_batch = min(block_size, remaining_words);
    rows_in_batch = min(remaining_rows, rows_per_batch);

    __syncthreads();
    if(threadIdx.x < words_in_batch){
      // load current word, strip down to exactly the number of rows this thread is dealing with
      auto const thread_num_rows = min(remaining_rows - (threadIdx.x * 32), 32);
      bitmask_type const relevant_row_mask = ((1 << thread_num_rows) - 1);
      cur = (src[src_word_index] & relevant_row_mask);
      valid_count += __popc(cur);

      // bounce our trailing bits off shared memory. for example, if bit_shift is
      // 27, we are only storing the first 5 bits at the top of the current destination. The
      // trailing 27 bits become the -leading- 27 bits for the next word.
      //
      // dst_bit_shift = 27;
      // src:      00000000000000000000000000011111
      // 
      // dst[0] =  11111xxxxxxxxxxxxxxxxxxxxxxxxxxx
      // dst[1] =  xxxx0000000000000000000000000000
      //
      prev = cur >> (32 - bit_shift);
      // the trailing bits the last thread goes to the 0th thread for the next iteration of the loop, so don't
      // write it until the end of the loop, otherwise we'll inadvertently blow away the 0th thread's correct value
      if(threadIdx.x < words_in_batch - 1){
        prev_word[threadIdx.x + 1] = prev;
      }
    }
    __syncthreads();
    if(threadIdx.x < words_in_batch){
      // construct final word from cur leading bits and prev trailing bits
      auto const word = (cur << bit_shift) | prev_word[threadIdx.x];
      store_word(dst_word_index, word);
    }
    __syncthreads();
    // store the final trailing bits at the beginning for the next iteration
    if(threadIdx.x == words_in_batch - 1){
      prev_word[0] = prev;
    }
    src_word_index += words_in_batch;
    dst_word_index += words_in_batch;
    remaining_words -= words_in_batch;
    remaining_rows -= rows_in_batch;
  } while (remaining_words > 0);

  // final trailing bits.
  if(threadIdx.x == 0){
    store_word(dst_word_index, prev_word[0]);
  }

  // add the valid count for the entire block to the count for the entire buffer.
  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  valid_count = block_reduce(temp_storage).Sum(valid_count);
  if(threadIdx.x == 0){
    atomicAdd(batch.valid_count, valid_count);
  }
}

/**
 * @brief Copy offset batches.
 * 
 * This kernel assumes aligned source buffer and destination buffers. It adjusts offsets
 * as necessary to handle stitching together disparate partitions
 */
template<int block_size>
__global__ void copy_offsets(cudf::device_span<assemble_batch> batches)
{
  int batch_index = blockIdx.x;
  auto& batch = batches[batch_index];
  if(batch.size <= 0 || (batch.btype != buffer_type::OFFSETS)){
    return;
  }

  auto const offset_shift = batch.value_shift;
  auto num_offsets = batch.size / sizeof(cudf::size_type); // TODO, someday handle long string columns
  auto offset_index = threadIdx.x;
  size_type const*const src = reinterpret_cast<size_type const*>(batch.src);
  size_type* dst = reinterpret_cast<size_type*>(batch.dst);
  while(offset_index < num_offsets){
    dst[offset_index] = src[offset_index] + offset_shift;
    offset_index += blockDim.x;
  }
}

/**
 * @brief Copy all batches into the final destination buffers.
 * 
 * Invokes three separate copy kernels. Cub batched memcpy for data buffers, and custom kernels for validity
 * and offset copying. The validity and offset kernels both need to manipulate the data during the copy.
 * 
 * @param batches The set of all copy batches to be performed
 * @param column_info Per-column information
 * @param h_column_info Host-side per-column information. The valid_count field of this struct is updated
 * by this function
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void assemble_copy(cudf::device_span<assemble_batch> batches, cudf::device_span<assemble_column_info const> column_info, cudf::host_span<assemble_column_info> h_column_info, rmm::cuda_stream_view stream)
{
  // TODO: it might make sense to launch these three copies on separate streams. It is likely that the validity and offset copies will be 
  // much smaller than the data copies and could theoretically not cause the gpu to be saturated.

  // main data copy. everything except validity and offsets
  {
    auto input_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<void*>([batches = batches.begin(), num_columns = column_info.size()] __device__ (size_t i){
      return batches[i].btype == buffer_type::DATA ? reinterpret_cast<void*>(const_cast<uint8_t*>(batches[i].src)) : nullptr;
    }));
    auto output_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<void*>([batches = batches.begin(), num_columns = column_info.size()] __device__ (size_t i){
      return batches[i].btype == buffer_type::DATA ? reinterpret_cast<void*>(const_cast<uint8_t*>(batches[i].dst)) : nullptr;
    }));
    auto size_iter = cudf::detail::make_counting_transform_iterator(0, cuda::proclaim_return_type<size_t>([batches = batches.begin(), num_columns = column_info.size()] __device__ (size_t i){
      return batches[i].btype == buffer_type::DATA ? batches[i].size : 0;
    }));

    size_t temp_storage_bytes;
    cub::DeviceMemcpy::Batched(nullptr, temp_storage_bytes, input_iter, output_iter, size_iter, batches.size(), stream);
    rmm::device_buffer temp_storage(temp_storage_bytes, stream, cudf::get_current_device_resource_ref());
    cub::DeviceMemcpy::Batched(temp_storage.data(), temp_storage_bytes, input_iter, output_iter, size_iter, batches.size(), stream);
  }

  // copy validity
  constexpr int copy_validity_block_size = 128;
  copy_validity<copy_validity_block_size><<<batches.size(), copy_validity_block_size, 0, stream.value()>>>(batches);

  // copy offsets
  constexpr int copy_offsets_block_size = 128;
  copy_offsets<copy_offsets_block_size><<<batches.size(), copy_offsets_block_size, 0, stream.value()>>>(batches);

  // we have to sync because the build_table step will need the cpu-side valid_count in h_column_info when constructing the columns.
  cudaMemcpyAsync(h_column_info.data(), column_info.data(), column_info.size() * sizeof(assemble_column_info), cudaMemcpyDeviceToHost, stream);
  stream.synchronize();
}

} // anonymous namespace for assemble_copy_data

/*
 * Code block for assembling the final cudf output table
 * Key function: build_table
 *
 */
namespace {

/**
 * @brief Functor that generates final cudf columns assembled from provided input buffers
 */
struct assemble_column_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, typename ColumnIter, typename BufferIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  std::pair<ColumnIter, BufferIter> operator()(ColumnIter col, BufferIter buffer, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    auto const validity = buffer;
    // no offsets
    auto const data = buffer + 2;

    out.push_back(std::make_unique<cudf::column>(cudf::data_type{col->type},
                  col->num_rows,
                  std::move(*data),
                  col->has_validity ? std::move(*validity) : rmm::device_buffer{},
                  col->has_validity ? col->num_rows - col->valid_count : 0));
    
    return {col + 1, buffer + 3};
  }

  template <typename T, typename ColumnIter, typename BufferIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  std::pair<ColumnIter, BufferIter> operator()(ColumnIter col, BufferIter buffer, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    auto const validity = buffer;
    buffer += 3;

    // build children
    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(col->num_children);
    auto next = col + 1;
    for(size_type i=0; i<col->num_children; i++){
      std::tie(next, buffer) = cudf::type_dispatcher(cudf::data_type{next->type},
                                                     assemble_column_functor{stream, mr},
                                                     next,
                                                     buffer,
                                                     children);
    }

    out.push_back(cudf::make_structs_column(col->num_rows,
                                            std::move(children),
                                            col->has_validity ? col->num_rows - col->valid_count : 0,
                                            col->has_validity ? std::move(*validity) : rmm::device_buffer{},
                                            stream,
                                            mr));
    return {next, buffer};
  }

  template <typename T, typename ColumnIter, typename BufferIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::string_view>)>
  std::pair<ColumnIter, BufferIter> operator()(ColumnIter col, BufferIter buffer, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    auto const validity = buffer;
    auto const offsets = buffer + 1;
    auto const chars = buffer + 2;

    out.push_back(cudf::make_strings_column(col->num_rows,
                                            std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                                           col->num_rows + 1,
                                                                           std::move(*offsets),
                                                                           rmm::device_buffer{},
                                                                           0),
                                            std::move(*chars),
                                            col->has_validity ? col->num_rows - col->valid_count : 0,
                                            col->has_validity ? std::move(*validity) : rmm::device_buffer{}));
    return {col + 1, buffer + 3};
  }

  template <typename T, typename ColumnIter, typename BufferIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::list_view>)>
  std::pair<ColumnIter, BufferIter> operator()(ColumnIter col, BufferIter buffer, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    auto const validity = buffer;
    auto const offsets = buffer + 1;
    buffer += 3;

    // build the child
    std::vector<std::unique_ptr<cudf::column>> child_col;
    auto next = col + 1;
    std::tie(next, buffer) = cudf::type_dispatcher(cudf::data_type{next->type},
                                                   assemble_column_functor{stream, mr},
                                                   next,
                                                   buffer,
                                                   child_col);
    
    // build the final column
    out.push_back(cudf::make_lists_column(col->num_rows,
                                          std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                                         col->num_rows + 1,
                                                                         std::move(*offsets),
                                                                         rmm::device_buffer{},
                                                                         0),
                                          std::move(child_col.back()),
                                          col->has_validity ? col->num_rows - col->valid_count : 0,
                                          col->has_validity ? std::move(*validity) : rmm::device_buffer{},
                                          stream,
                                          mr));
    return {next, buffer};
  }

  template <typename T, typename ColumnIter, typename BufferIter, CUDF_ENABLE_IF(!cudf::is_fixed_width<T>() and !std::is_same_v<T, cudf::struct_view> and !std::is_same_v<T, cudf::string_view> and !std::is_same_v<T, cudf::list_view>)>
  std::pair<ColumnIter, BufferIter> operator()(ColumnIter col, BufferIter buffer, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    CUDF_FAIL("Unsupported type in shuffle_assemble");
  }
};

// assemble all the columns and the final table from the intermediate buffers
std::unique_ptr<cudf::table> build_table(cudf::host_span<assemble_column_info const> assemble_data,
                                         cudf::host_span<rmm::device_buffer> assemble_buffers,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto column = assemble_data.begin();
  auto buffer = assemble_buffers.begin();
  while(column != assemble_data.end()){
    std::tie(column, buffer) = cudf::type_dispatcher(cudf::data_type{column->type},
                                                     assemble_column_functor{stream, mr},
                                                     column,
                                                     buffer,
                                                     columns);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Functor that generates final empty cudf columns from the provided metadata
 * schema.
 */
struct assemble_empty_column_functor {
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  template <typename T, typename ColumnIter, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
  ColumnIter operator()(ColumnIter col, std::vector<std::unique_ptr<cudf::column>>& out)
  {    
    out.push_back(std::make_unique<cudf::column>(cudf::data_type{col->type},
                  0,
                  rmm::device_buffer{},
                  rmm::device_buffer{},
                  0));
    
    return {col + 1};
  }

  template <typename T, typename ColumnIter, CUDF_ENABLE_IF(std::is_same_v<T, cudf::struct_view>)>
  ColumnIter operator()(ColumnIter col, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    // build children
    std::vector<std::unique_ptr<cudf::column>> children;
    children.reserve(col->num_children());
    auto next = col + 1;
    for(size_type i=0; i<col->num_children(); i++){
      next = cudf::type_dispatcher(cudf::data_type{next->type},
                                   assemble_empty_column_functor{stream, mr},
                                   next,
                                   children);
    }    

    out.push_back(cudf::make_structs_column(0,
                                            std::move(children),
                                            0,
                                            rmm::device_buffer{},
                                            stream,
                                            mr));
    return next;
  }

  // template <typename T, CUDF_ENABLE_IF(!cudf::is_fixed_width<T>() and !std::is_same_v<T, cudf::list_view> and !std::is_same_v<T, cudf::struct_view>)>
  template <typename T, typename ColumnIter, CUDF_ENABLE_IF(!cudf::is_fixed_width<T>() and !std::is_same_v<T, cudf::struct_view>)>
  ColumnIter operator()(ColumnIter col, std::vector<std::unique_ptr<cudf::column>>& out)
  {
    CUDF_FAIL("Unsupported type in shuffle_assemble");
  }
};

// assemble all the columns and the final table from the intermediate buffers
std::unique_ptr<cudf::table> build_empty_table(cudf::host_span<shuffle_split_col_data const> col_info,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  auto column = col_info.begin();
  while(column != col_info.end()){
    column = cudf::type_dispatcher(cudf::data_type{column->type},
                                   assemble_empty_column_functor{stream, mr},
                                   column,
                                   columns);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

} // anonymous namespace for build_table

/**
 * @copydoc spark_rapids_jni::shuffle_assemble
 */
std::unique_ptr<cudf::table> shuffle_assemble(shuffle_split_metadata const& metadata,
                                              cudf::device_span<uint8_t const> partitions,
                                              cudf::device_span<size_t const> partition_offsets,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  // if the input is empty, just generate an empty table
  if(partition_offsets.size() == 1){
    return build_empty_table(metadata.col_info, stream, mr);
  }

  // generate the info structs representing the flattened column hierarchy. the total number of assembled rows, null counts, etc
  auto [column_info,  column_instance_info, per_partition_metadata_size] = assemble_build_column_info(metadata, partitions, partition_offsets, stream, cudf::get_current_device_resource_ref());
  auto h_column_info = cudf::detail::make_std_vector_sync(column_info, stream);

  // generate the (empty) output buffers based on the column info. 
  // generate the copy batches to be performed to copy data to the output buffers
  auto [dst_buffers, batches] = assemble_build_buffers(column_info, h_column_info, column_instance_info, partitions, partition_offsets, per_partition_metadata_size, stream, mr);  

  // copy the data. also updates valid_count in column_info
  assemble_copy(batches, column_info, h_column_info, stream);

  // the gpu is synchronized at this point. a hypothetical optimization here would be to 
  // not synchronize at the end of assemble_copy, which would mean the valid_count field
  // in h_column_info would not be updated yet, and poke those values specifically after
  // build_table completes.  If we are dealing with very large schemas, it is possible that
  // there's a decent chunk of time spend on the cpu in build_table that could be overlapped
  // with the the gpu work being done in assemble_copy.
  
  // build the final table while the gpu is performing the copy
  auto ret = build_table(h_column_info, dst_buffers, stream, mr);
  return ret;
}

}  // namespace spark_rapids_jni