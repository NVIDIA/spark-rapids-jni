#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <thrust/scan.h>
#include <cudf/reduction.hpp>
#include <cudf/lists/set_operations.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/lists/gather.hpp>
#include "map_zip_with_utils.hpp"

using namespace cudf;

namespace spark_rapids_jni {

[[maybe_unused]] std::unique_ptr<column> generate_labels(lists_column_view const& input,
                                        size_type n_elements,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  auto labels = make_numeric_column(
    data_type(type_to_id<size_type>()), n_elements, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const labels_begin = labels->mutable_view().template begin<size_type>();
  cudf::detail::label_segments(
    input.offsets_begin(), input.offsets_end(), labels_begin, labels_begin + n_elements, stream);
  return labels;
}

[[maybe_unused]] std::unique_ptr<column> indices_of(
  lists_column_view const& search_keys,    // Column containing lists of keys to search with
  lists_column_view const& search_values,  // Column containing lists of values to search for
  rmm::cuda_stream_view stream,            // CUDA stream for asynchronous execution
  rmm::device_async_resource_ref mr) {     // Memory resource for allocations
    
    CUDF_EXPECTS(search_keys.size() == search_values.size(),
               "Number of search keys lists must match search values lists.");

    // Get the number of lists (rows) in the input columns
    auto const num_lists = search_values.size();

    // Extract the child columns containing the actual key and value data
    auto const all_keys = search_keys.child();
    auto const num_keys = all_keys.size();
    auto const all_values = search_values.child();

    // Calculate the number of elements in each list for both keys and values
    auto list_sizes = cudf::lists::count_elements(search_values, stream, mr);
    auto keys_sizes = cudf::lists::count_elements(search_keys, stream, mr);

    // Generate labels to map each key back to its corresponding list index
    // This helps us know which list each key belongs to
    auto keys_labels = generate_labels(search_keys, num_keys, stream, mr);

    // For each key, get the size of the associated values list
    // This tells us how many values we need to compare against for each key
    auto associated_list_sizes =                                      
        thrust::make_transform_iterator(keys_labels->view().begin<size_type>(),
                      [list_sizes_ptr = list_sizes->view().begin<size_type>(), num_lists] __device__(auto const offset_val) {
                        return offset_val >= num_lists ? 0 : *(list_sizes_ptr + offset_val);
                      });

    // Calculate cumulative offsets for the associated list sizes
    // This gives us the starting position for each key's value comparisons
    auto list_sizes_offsets = make_numeric_column(data_type(type_to_id<size_type>()),
                                          num_keys+1,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
    auto d_list_sizes_offsets = list_sizes_offsets->mutable_view().template data<size_type>();
    thrust::exclusive_scan(rmm::exec_policy(stream), associated_list_sizes, associated_list_sizes + num_keys+1, d_list_sizes_offsets);
      
    // Calculate the total number of comparisons needed for each row
    // For each list, we need: number_of_keys * number_of_values comparisons
    auto total_compares_per_row =  
        thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                      [keys_sizes = keys_sizes->mutable_view().template data<size_type>(), list_sizes = list_sizes->mutable_view().template data<size_type>(), num_lists] __device__(auto const offset_val) {
                        return offset_val >= num_lists ? 0 : keys_sizes[offset_val] * list_sizes[offset_val];
                      });
    // Sum up all comparisons across all rows
    auto total_compares = thrust::reduce(rmm::exec_policy(stream), total_compares_per_row, total_compares_per_row + num_lists);

    // Create an index array that maps each comparison to its corresponding key
    // This tells us which key we're comparing in each comparison operation
    auto key_index = make_numeric_column(data_type(type_to_id<size_type>()),
                                          total_compares,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
    auto d_key_index = key_index->mutable_view().template data<size_type>();

    // Use label_segments to expand the key indices based on the list size offsets
    cudf::detail::label_segments(list_sizes_offsets->view().begin<size_type>(),
                  list_sizes_offsets->view().end<size_type>(),
                  d_key_index,
                  d_key_index + total_compares,
                  stream);

    // Calculate offsets for the total comparisons per row
    // This gives us the starting position for each row's comparisons
    auto total_compares_offsets = make_numeric_column(data_type(type_to_id<size_type>()),
                                          num_lists+1,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
    auto d_total_compares_offsets = total_compares_offsets->mutable_view().template data<size_type>();
    thrust::exclusive_scan(
        rmm::exec_policy(stream), total_compares_per_row, total_compares_per_row + num_lists+1, d_total_compares_offsets);

    // Create an array that maps each comparison to its corresponding value list
    // This tells us which value list we're comparing against
    auto val_offsets = make_numeric_column(data_type(type_to_id<size_type>()),
                                          total_compares,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
    auto d_val_offsets = val_offsets->mutable_view().template data<size_type>();
    cudf::detail::label_segments(d_total_compares_offsets,
                  d_total_compares_offsets + num_lists+1,
                  d_val_offsets,
                  d_val_offsets + total_compares,
                  stream);

    // Calculate the index of the value within its list for each comparison
    // This gives us the specific value position we're comparing against
    auto values_idx = thrust::make_transform_iterator (
      thrust::make_counting_iterator(0),
      cuda::proclaim_return_type<size_type>(
          [d_key_index, d_list_sizes_offsets, d_val_offsets, d_list_sizes_val_offsets = search_values.offsets_begin()] __device__(auto const idx) {
            return d_list_sizes_offsets[d_key_index[idx]] > 0 ? idx % d_list_sizes_offsets[d_key_index[idx]] + d_list_sizes_val_offsets[d_val_offsets[idx]] : idx;
          }) 
    );

    // Perform the actual key-value comparisons
    // For each comparison: if key == value, return the value index; otherwise return 100
    auto compare_results = make_numeric_column(data_type(type_to_id<size_type>()),
                                          total_compares,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
    thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),  
                    thrust::make_counting_iterator(total_compares),
                    compare_results->mutable_view().template begin<size_type>(),
                    cuda::proclaim_return_type<size_type>(
                                  [values_idx, d_key_index, all_keys = all_keys.begin<size_type>(), all_values = all_values.begin<size_type>(), d_val_offsets, d_list_sizes_val_offsets = search_values.offsets_begin()] __device__(auto const idx) {
                                    return all_keys[d_key_index[idx]] == all_values[values_idx[idx]] ? values_idx[idx] - d_list_sizes_val_offsets[d_val_offsets[idx]] : 100;
                                  })); 
    
    // Use segmented reduction to find the minimum value (first match) for each list
    // This gives us the first matching value index for each row, or 100 if no match found
    auto offsets_span   = cudf::device_span<cudf::size_type const>(list_sizes_offsets->view());
    auto results =
      cudf::segmented_reduce(compare_results->view(),
                            offsets_span,
                            *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                            cudf::data_type{cudf::type_to_id<size_type>()},
                            cudf::null_policy::EXCLUDE);
    return results;
  }

[[maybe_unused]] std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> map_zip(
  cudf::lists_column_view const& col1,    // First map column containing key-value pairs
  cudf::lists_column_view const& col2,    // Second map column containing key-value pairs
  rmm::cuda_stream_view stream,           // CUDA stream for asynchronous execution
  rmm::device_async_resource_ref mr) {    // Memory resource for allocations


  // Extract keys and values from the first map column (col1)
  // Create a column view that represents the keys and values part of col1
  auto map1_keys = column_view{data_type{type_id::LIST},
                     col1.size(),
                     nullptr,
                     col1.null_mask(),
                     col1.null_count(),
                     col1.offset(),
                     {col1.offsets(), col1.child().child(0)}};
  
  auto map1_values = column_view{data_type{type_id::LIST},
                     col1.size(),
                     nullptr,
                     col1.null_mask(),
                     col1.null_count(),
                     col1.offset(),
                     {col1.offsets(), col1.child().child(1)}};

  // Extract keys and values from the second map column (col2)
  // Create a column view that represents the keys and values part of col2
  auto map2_keys = column_view{data_type{type_id::LIST},
                     col2.size(),
                     nullptr,
                     col2.null_mask(),
                     col2.null_count(),
                     col2.offset(),
                     {col2.offsets(), col2.child().child(0)}};
  
  auto map2_values = column_view{data_type{type_id::LIST},
                     col2.size(),
                     nullptr,
                     col2.null_mask(),
                     col2.null_count(),
                     col2.offset(),
                     {col2.offsets(), col2.child().child(1)}};

  // Find the union of all unique keys from both maps
  // This creates a combined set of keys that will be used for the final result
  auto search_keys = cudf::lists::union_distinct(cudf::lists_column_view(map1_keys), cudf::lists_column_view(map2_keys));
  auto search_keys_list = cudf::lists_column_view(*search_keys);
  
  // Find the indices of each key in the first map (map1)
  // This tells us where each key appears in map1, or if it doesn't exist
  auto map1_indices = indices_of(search_keys_list, cudf::lists_column_view(map1_keys), stream, mr);
  auto map1_indices_list = make_lists_column(search_keys_list.size(),
                             std::make_unique<column>(search_keys_list.offsets()),
                             std::move(map1_indices),
                             0,
                             rmm::device_buffer{0, stream, mr});
  
  // Find the indices of each key in the second map (map2)
  // This tells us where each key appears in map2, or if it doesn't exist
  auto map2_indices = indices_of(search_keys_list, cudf::lists_column_view(map2_keys), stream, mr);
  auto map2_indices_list = make_lists_column(search_keys_list.size(),
                             std::make_unique<column>(search_keys_list.offsets()),
                             std::move(map2_indices),
                             0,
                             rmm::device_buffer{0, stream, mr});

  // Gather the values from map1 and map2 using the calculated indices
  // This extracts the values corresponding to each key in the union
  auto map1_values_list_gather =  cudf::lists::segmented_gather(cudf::lists_column_view(map1_values), cudf::lists_column_view(*map1_indices_list), out_of_bounds_policy::NULLIFY);
  auto map2_values_list_gather =  cudf::lists::segmented_gather(cudf::lists_column_view(map2_values), cudf::lists_column_view(*map2_indices_list), out_of_bounds_policy::NULLIFY);

  // Return the gathered values from map1
  return std::pair(std::move(map1_values_list_gather), std::move(map2_values_list_gather));
}

} // namespace spark_rapids_jni


/*        keys        vals
// m00 = [1, 2, 3] | [4, 5, 6]
// m01 = [5, 6, 7] | [4, 5, 6]

// m10 = [2, 3, 4] | [7, 8, 9]
// m11 = [6, 7, 8] | [7, 8, 9]


// searchkeys0 = [(1, true, false), (2, true, true), (3, true, true), (4, false, true)] from union_distinct
// searchkeys1 = [5, 6, 7, 8]

// Map labels and keys
// map column 0 label, key => [(0,1), (0,2), (0,3), (1,5), (1,6), (1,7)]
// map column 1 label, key => [(0,2), (0,3), (0,4), (1,6), (1,7), (1,8)]

// search_keys label and keys
// label, key => [(0,1), (0,2), (0,3), (0,4)]
// label, key => [(1,5), (1,6), (1,7), (1,8)]


// // For both map insert label,key into hash table as shown in set insert async

// // Generate output list size from the static set size
// // For each map, check for existence of search key in map using static set ref contains
// // If key exists, insert index into output list, else Out of Bounds
// // Reduce by key where key is the list label to count of the number of values in the output list, giving us the size of the output map row
// // Then compute offsets using exclusive scan



[1,2,3]   | [45,2,3]
[4, 5]    | [4,7,3,5]
[6, 7, 8] | [10,8]


3 3 3 4 4 2 2 2
0 3 6 9 13 17 19 21
0 0 0 1 1 1 2 2 2 3 3 3 3 4 4 4 4 5 5 6 6 7 7

0 3 6 9 13 17 19 21
0 0 0 1 1 1 2 2 2 3 3 3 3 4 4 4 4 5 5 6 6 7 7
0 1 2 0 1 2 0 1 2 0 1 2 3 0 1 2 3 0 1 0 1 0 1
0 3 7
0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 2 2
0 0 0 0 0 0 0 0 0 3 3 3 3 3 3 3 3 7 7 7 7 7 7
0 1 2 0 1 2 0 1 2 3 4 5 6 3 4 5 6 7 8 7 8 7 8

*/