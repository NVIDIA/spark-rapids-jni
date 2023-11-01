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

#include "map_utils_debug.cuh"

//
#include <limits>

//
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/algorithm.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>

//
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

//
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

//
#include <cub/device/device_radix_sort.cuh>

namespace spark_rapids_jni {

using namespace cudf::io::json;

namespace {

// Unify the input json strings by:
// 1. Append one comma character (',') to the end of each input string, except the last one.
// 2. Concatenate all input strings into one string.
// 3. Add a pair of bracket characters ('[' and ']') to the beginning and the end of the output.
rmm::device_uvector<char> unify_json_strings(cudf::column_view const &input,
                                             rmm::cuda_stream_view stream) {
  if (input.is_empty()) {
    return cudf::detail::make_device_uvector_async<char>(std::vector<char>{'[', ']'}, stream,
                                                         rmm::mr::get_current_device_resource());
  }

  auto const d_strings = cudf::column_device_view::create(input, stream);
  auto const chars_size = input.child(cudf::strings_column_view::chars_column_index).size();
  auto const output_size =
      2l + // two extra bracket characters '[' and ']'
      static_cast<int64_t>(chars_size) +
      static_cast<int64_t>(input.size() - 1) +       // append `,` character between input rows
      static_cast<int64_t>(input.null_count()) * 2l; // replace null with "{}"
  CUDF_EXPECTS(output_size <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
               "The input json column is too large and causes overflow.");

  auto const joined_input = cudf::strings::detail::join_strings(
      cudf::strings_column_view{input},
      cudf::string_scalar(","),  // append `,` character between the input rows
      cudf::string_scalar("{}"), // replacement for null rows
      stream, rmm::mr::get_current_device_resource());
  auto const joined_input_child =
      joined_input->child(cudf::strings_column_view::chars_column_index);
  auto const joined_input_size_bytes = joined_input_child.size();
  CUDF_EXPECTS(joined_input_size_bytes + 2 == output_size, "Incorrect output size computation.");

  // We want to concatenate 3 strings: "[" + joined_input + "]".
  // For efficiency, let's use memcpy instead of `cudf::strings::detail::concatenate`.
  auto output = rmm::device_uvector<char>(joined_input_size_bytes + 2, stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(output.data(), static_cast<int>('['), 1, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(output.data() + 1, joined_input_child.view().data<char>(),
                                joined_input_size_bytes, cudaMemcpyDefault, stream.value()));
  CUDF_CUDA_TRY(cudaMemsetAsync(output.data() + joined_input_size_bytes + 1, static_cast<int>(']'),
                                1, stream.value()));

#ifdef DEBUG_FROM_JSON
  print_debug<char, char>(output, "Processed json string", "", stream);
#endif
  return output;
}

// Check and throw exception if there is any parsing error.
void throw_if_error(rmm::device_uvector<char> const &input_json,
                    rmm::device_uvector<PdaTokenT> const &tokens,
                    rmm::device_uvector<SymbolOffsetT> const &token_indices,
                    rmm::cuda_stream_view stream) {
  auto const error_count =
      thrust::count(rmm::exec_policy(stream), tokens.begin(), tokens.end(), token_t::ErrorBegin);

  if (error_count > 0) {
    auto const error_location =
        thrust::find(rmm::exec_policy(stream), tokens.begin(), tokens.end(), token_t::ErrorBegin);
    SymbolOffsetT error_index;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        &error_index, token_indices.data() + thrust::distance(tokens.begin(), error_location),
        sizeof(SymbolOffsetT), cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();

    constexpr SymbolOffsetT extension = 10u;

    // Warning: SymbolOffsetT is unsigned type thus we need to be careful with subtractions.
    auto const begin_print_idx =
        error_index > extension ? error_index - extension : SymbolOffsetT{0};
    auto const end_print_idx =
        std::min(error_index + extension, static_cast<SymbolOffsetT>(input_json.size()));
    auto const print_size = end_print_idx - begin_print_idx;
    auto const h_input_json = cudf::detail::make_host_vector_sync(
        cudf::device_span<char const>{input_json.data() + begin_print_idx, print_size}, stream);

    std::cerr << "Substring in the range [" + std::to_string(begin_print_idx) + ", " +
                     std::to_string(end_print_idx) + "]" + " of the input (invalid) json:\n";
    std::cerr << std::string(h_input_json.data(), h_input_json.size()) << std::endl;

    CUDF_FAIL("JSON Parser encountered an invalid format at location " +
              std::to_string(error_index));
  }
}

// Check if a token is a json node.
struct is_node {
  __host__ __device__ bool operator()(PdaTokenT const token) const {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin:
      case token_t::ErrorBegin: return true;
      default: return false;
    };
  }
};

// Compute the level of each token node.
// The top json node (top json object level) has level 0.
// Each row in the input column should have levels starting from 1.
// This is copied from cudf's `json_tree.cu`.
rmm::device_uvector<TreeDepthT> compute_node_levels(int64_t num_nodes,
                                                    rmm::device_uvector<PdaTokenT> const &tokens,
                                                    rmm::cuda_stream_view stream) {
  auto token_levels = rmm::device_uvector<TreeDepthT>(tokens.size(), stream);

  // Whether the token pops from the parent node stack.
  auto const does_pop = [] __device__(PdaTokenT const token) -> bool {
    switch (token) {
      case token_t::StructMemberEnd:
      case token_t::StructEnd:
      case token_t::ListEnd: return true;
      default: return false;
    };
  };

  // Whether the token pushes onto the parent node stack.
  auto const does_push = [] __device__(PdaTokenT const token) -> bool {
    switch (token) {
      case token_t::FieldNameBegin:
      case token_t::StructBegin:
      case token_t::ListBegin: return true;
      default: return false;
    };
  };

  auto const push_pop_it = thrust::make_transform_iterator(
      tokens.begin(), [does_push, does_pop] __device__(PdaTokenT const token) -> cudf::size_type {
        return does_push(token) - does_pop(token);
      });
  thrust::exclusive_scan(rmm::exec_policy(stream), push_pop_it, push_pop_it + tokens.size(),
                         token_levels.begin());

  auto node_levels = rmm::device_uvector<TreeDepthT>(num_nodes, stream);
  auto const copy_end =
      cudf::detail::copy_if_safe(token_levels.begin(), token_levels.end(), tokens.begin(),
                                 node_levels.begin(), is_node{}, stream);
  CUDF_EXPECTS(thrust::distance(node_levels.begin(), copy_end) == num_nodes,
               "Node level count mismatch");

#ifdef DEBUG_FROM_JSON
  print_debug(node_levels, "Node levels", ", ", stream);
#endif
  return node_levels;
}

// Compute the map from nodes to their indices in the list of all tokens.
rmm::device_uvector<NodeIndexT>
compute_node_to_token_index_map(int64_t num_nodes, rmm::device_uvector<PdaTokenT> const &tokens,
                                rmm::cuda_stream_view stream) {
  auto node_token_ids = rmm::device_uvector<NodeIndexT>(num_nodes, stream);
  auto const node_id_it = thrust::counting_iterator<NodeIndexT>(0);
  auto const copy_end =
      cudf::detail::copy_if_safe(node_id_it, node_id_it + tokens.size(), tokens.begin(),
                                 node_token_ids.begin(), is_node{}, stream);
  CUDF_EXPECTS(thrust::distance(node_token_ids.begin(), copy_end) == num_nodes,
               "Invalid computation for node-to-token-index map");

#ifdef DEBUG_FROM_JSON
  print_map_debug(node_token_ids, "Node-to-token-index map", stream);
#endif
  return node_token_ids;
}

// This is copied from cudf's `json_tree.cu`.
template <typename KeyType, typename IndexType = cudf::size_type>
std::pair<rmm::device_uvector<KeyType>, rmm::device_uvector<IndexType>>
stable_sorted_key_order(rmm::device_uvector<KeyType> const &keys, rmm::cuda_stream_view stream) {
  // Buffers used for storing intermediate results during sorting.
  rmm::device_uvector<KeyType> keys_buffer1(keys.size(), stream);
  rmm::device_uvector<KeyType> keys_buffer2(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer1(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer2(keys.size(), stream);
  cub::DoubleBuffer<KeyType> keys_buffer(keys_buffer1.data(), keys_buffer2.data());
  cub::DoubleBuffer<IndexType> order_buffer(order_buffer1.data(), order_buffer2.data());

  thrust::copy(rmm::exec_policy(stream), keys.begin(), keys.end(), keys_buffer1.begin());
  thrust::sequence(rmm::exec_policy(stream), order_buffer1.begin(), order_buffer1.end());

  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, keys_buffer, order_buffer,
                                  keys.size());
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);
  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(), temp_storage_bytes, keys_buffer,
                                  order_buffer, keys.size(), 0, sizeof(KeyType) * 8,
                                  stream.value());

  return std::pair{keys_buffer.Current() == keys_buffer1.data() ? std::move(keys_buffer1) :
                                                                  std::move(keys_buffer2),
                   order_buffer.Current() == order_buffer1.data() ? std::move(order_buffer1) :
                                                                    std::move(order_buffer2)};
}

// This is copied from cudf's `json_tree.cu`.
void propagate_parent_to_siblings(rmm::device_uvector<TreeDepthT> const &node_levels,
                                  rmm::device_uvector<NodeIndexT> &parent_node_ids,
                                  rmm::cuda_stream_view stream) {
  auto const [sorted_node_levels, sorted_order] = stable_sorted_key_order(node_levels, stream);

  // Instead of gather, using permutation_iterator, which is ~17% faster.
  thrust::inclusive_scan_by_key(
      rmm::exec_policy(stream), sorted_node_levels.begin(), sorted_node_levels.end(),
      thrust::make_permutation_iterator(parent_node_ids.begin(), sorted_order.begin()),
      thrust::make_permutation_iterator(parent_node_ids.begin(), sorted_order.begin()),
      thrust::equal_to<TreeDepthT>{}, thrust::maximum<NodeIndexT>{});
}

// This is copied from cudf's `json_tree.cu`.
rmm::device_uvector<NodeIndexT>
compute_parent_node_ids(int64_t num_nodes, rmm::device_uvector<PdaTokenT> const &tokens,
                        rmm::device_uvector<NodeIndexT> const &node_token_ids,
                        rmm::cuda_stream_view stream) {
  auto const first_childs_parent_token_id = [tokens =
                                                 tokens.begin()] __device__(auto i) -> NodeIndexT {
    if (i <= 0) {
      return -1;
    }
    if (tokens[i - 1] == token_t::StructBegin || tokens[i - 1] == token_t::ListBegin) {
      return i - 1;
    } else if (tokens[i - 1] == token_t::FieldNameEnd) {
      return i - 2;
    } else if (tokens[i - 1] == token_t::StructMemberBegin &&
               (tokens[i - 2] == token_t::StructBegin || tokens[i - 2] == token_t::ListBegin)) {
      return i - 2;
    } else {
      return -1;
    }
  };

  auto parent_node_ids = rmm::device_uvector<NodeIndexT>(num_nodes, stream);
  thrust::transform(rmm::exec_policy(stream), node_token_ids.begin(), node_token_ids.end(),
                    parent_node_ids.begin(),
                    [node_ids_gpu = node_token_ids.begin(), num_nodes,
                     first_childs_parent_token_id] __device__(NodeIndexT const tid) -> NodeIndexT {
                      auto const pid = first_childs_parent_token_id(tid);
                      return pid < 0 ? cudf::io::json::parent_node_sentinel :
                                       thrust::lower_bound(thrust::seq, node_ids_gpu,
                                                           node_ids_gpu + num_nodes, pid) -
                                           node_ids_gpu;
                    });

  // Propagate parent node to siblings from first sibling - inplace.
  auto const node_levels = compute_node_levels(num_nodes, tokens, stream);
  propagate_parent_to_siblings(node_levels, parent_node_ids, stream);

#ifdef DEBUG_FROM_JSON
  print_debug(parent_node_ids, "Parent node ids", ", ", stream);
#endif
  return parent_node_ids;
}

constexpr int8_t key_sentinel{1};
constexpr int8_t value_sentinel{2};

// Check for each node if it is a key or a value field.
rmm::device_uvector<int8_t>
check_key_or_value_nodes(rmm::device_uvector<NodeIndexT> const &parent_node_ids,
                         rmm::cuda_stream_view stream) {
  auto key_or_value = rmm::device_uvector<int8_t>(parent_node_ids.size(), stream);
  auto const transform_it = thrust::counting_iterator<int>(0);
  thrust::transform(
      rmm::exec_policy(stream), transform_it, transform_it + parent_node_ids.size(),
      key_or_value.begin(),
      [key_sentinel = key_sentinel, value_sentinel = value_sentinel,
       parent_ids = parent_node_ids.begin()] __device__(auto const node_id) -> int8_t {
        if (parent_ids[node_id] > 0) {
          auto const grand_parent = parent_ids[parent_ids[node_id]];
          if (grand_parent == 0) {
            return key_sentinel;
          } else if (parent_ids[grand_parent] == 0) {
            return value_sentinel;
          }
        }

        return 0;
      });

#ifdef DEBUG_FROM_JSON
  print_debug(key_or_value, "Nodes are key/value (1==key, 2==value)", ", ", stream);
#endif
  return key_or_value;
}

// Convert token indices to node ranges for each valid node.
struct node_ranges_fn {
  cudf::device_span<PdaTokenT const> tokens;
  cudf::device_span<SymbolOffsetT const> token_indices;
  cudf::device_span<NodeIndexT const> node_token_ids;
  cudf::device_span<NodeIndexT const> parent_node_ids;
  cudf::device_span<int8_t const> key_or_value;

  // Whether the extracted string values from json map will have the quote character.
  static const bool include_quote_char{false};

  __device__ thrust::pair<SymbolOffsetT, SymbolOffsetT> operator()(cudf::size_type node_id) const {
    [[maybe_unused]] auto const is_begin_of_section = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StructBegin:
        case token_t::ListBegin:
        case token_t::StringBegin:
        case token_t::ValueBegin:
        case token_t::FieldNameBegin: return true;
        default: return false;
      };
    };

    // The end-of-* partner token for a given beginning-of-* token
    auto const end_of_partner = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StructBegin: return token_t::StructEnd;
        case token_t::ListBegin: return token_t::ListEnd;
        case token_t::StringBegin: return token_t::StringEnd;
        case token_t::ValueBegin: return token_t::ValueEnd;
        case token_t::FieldNameBegin: return token_t::FieldNameEnd;
        default: return token_t::ErrorBegin;
      };
    };

    // Encode a fixed value for nested node types (list+struct).
    auto const nested_node_to_value = [] __device__(PdaTokenT const token) -> int32_t {
      switch (token) {
        case token_t::StructBegin: return 1;
        case token_t::StructEnd: return -1;
        case token_t::ListBegin: return 1 << 8;
        case token_t::ListEnd: return -(1 << 8);
        default: return 0;
      };
    };

    auto const get_token_index = [include_quote_char = include_quote_char] __device__(
                                     PdaTokenT const token, SymbolOffsetT const token_index) {
      constexpr SymbolOffsetT quote_char_size = 1;
      switch (token) {
        // Strip off quote char included for StringBegin
        case token_t::StringBegin: return token_index + (include_quote_char ? 0 : quote_char_size);
        // Strip off or Include trailing quote char for string values for StringEnd
        case token_t::StringEnd: return token_index + (include_quote_char ? quote_char_size : 0);
        // Strip off quote char included for FieldNameBegin
        case token_t::FieldNameBegin: return token_index + quote_char_size;
        default: return token_index;
      };
    };

    if (key_or_value[node_id] != key_sentinel && key_or_value[node_id] != value_sentinel) {
      return thrust::make_pair(0, 0);
    }

    auto const token_idx = node_token_ids[node_id];
    auto const token = tokens[token_idx];
    cudf_assert(is_begin_of_section(token) && "Invalid node category.");

    // The section from the original JSON input that this token demarcates.
    auto const range_begin = get_token_index(token, token_indices[token_idx]);
    auto range_end = range_begin + 1; // non-leaf, non-field nodes ignore this value.
    if ((token_idx + 1) < tokens.size() && end_of_partner(token) == tokens[token_idx + 1]) {
      // Update the range_end for this pair of tokens
      range_end = get_token_index(tokens[token_idx + 1], token_indices[token_idx + 1]);
    } else {
      auto nested_range_value = nested_node_to_value(token); // iterate until this is zero
      auto end_idx = token_idx + 1;
      while (end_idx < tokens.size()) {
        nested_range_value += nested_node_to_value(tokens[end_idx]);
        if (nested_range_value == 0) {
          range_end = get_token_index(tokens[end_idx], token_indices[end_idx]) + 1;
          break;
        }
        ++end_idx;
      }
      cudf_assert(nested_range_value == 0 && "Invalid range computation.");
      cudf_assert((end_idx + 1 < tokens.size()) && "Invalid range computation.");
    }
    return thrust::make_pair(range_begin, range_end);
  }
};

// Compute index range for each node.
// These ranges identify positions to extract nodes from the unified json string.
rmm::device_uvector<thrust::pair<SymbolOffsetT, SymbolOffsetT>>
compute_node_ranges(int64_t num_nodes, rmm::device_uvector<PdaTokenT> const &tokens,
                    rmm::device_uvector<SymbolOffsetT> const &token_indices,
                    rmm::device_uvector<NodeIndexT> const &node_token_ids,
                    rmm::device_uvector<NodeIndexT> const &parent_node_ids,
                    rmm::device_uvector<int8_t> const &key_or_value, rmm::cuda_stream_view stream) {
  auto node_ranges =
      rmm::device_uvector<thrust::pair<SymbolOffsetT, SymbolOffsetT>>(num_nodes, stream);
  auto const transform_it = thrust::counting_iterator<int>(0);
  thrust::transform(
      rmm::exec_policy(stream), transform_it, transform_it + num_nodes, node_ranges.begin(),
      node_ranges_fn{tokens, token_indices, node_token_ids, parent_node_ids, key_or_value});

#ifdef DEBUG_FROM_JSON
  print_pair_debug(node_ranges, "Node ranges", stream);
#endif
  return node_ranges;
}

// Function logic for substring API.
// This both calculates the output size and executes the substring.
// No bound check is performed, assuming that the substring bounds are all valid.
struct substring_fn {
  cudf::device_span<char const> const d_string;
  cudf::device_span<thrust::pair<SymbolOffsetT, SymbolOffsetT> const> const d_ranges;

  cudf::size_type *d_offsets{};
  char *d_chars{};

  __device__ void operator()(cudf::size_type const idx) {
    auto const range = d_ranges[idx];
    auto const size = range.second - range.first;
    if (d_chars) {
      memcpy(d_chars + d_offsets[idx], d_string.data() + range.first, size);
    } else {
      d_offsets[idx] = size;
    }
  }
};

// Extract key-value string pairs from the input json string.
std::unique_ptr<cudf::column> extract_keys_or_values(
    bool extract_key, int64_t num_nodes,
    rmm::device_uvector<thrust::pair<SymbolOffsetT, SymbolOffsetT>> const &node_ranges,
    rmm::device_uvector<int8_t> const &key_or_value,
    rmm::device_uvector<char> const &unified_json_buff, rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr) {
  auto const is_key = [key_or_value = key_or_value.begin()] __device__(auto const node_id) {
    return key_or_value[node_id] == key_sentinel;
  };

  auto const is_value = [key_or_value = key_or_value.begin()] __device__(auto const node_id) {
    return key_or_value[node_id] == value_sentinel;
  };

  auto extract_ranges =
      rmm::device_uvector<thrust::pair<SymbolOffsetT, SymbolOffsetT>>(num_nodes, stream, mr);
  auto const stencil_it = thrust::make_counting_iterator(0);
  auto const range_end =
      extract_key ? cudf::detail::copy_if_safe(node_ranges.begin(), node_ranges.end(), stencil_it,
                                               extract_ranges.begin(), is_key, stream) :
                    cudf::detail::copy_if_safe(node_ranges.begin(), node_ranges.end(), stencil_it,
                                               extract_ranges.begin(), is_value, stream);
  auto const num_extract = thrust::distance(extract_ranges.begin(), range_end);

  auto children = cudf::strings::detail::make_strings_children(
      substring_fn{unified_json_buff, extract_ranges}, num_extract, stream, mr);
  return cudf::make_strings_column(num_extract, std::move(children.first),
                                   std::move(children.second), 0, rmm::device_buffer{});
}

// Compute the offsets for the final lists of Struct<String,String>.
rmm::device_uvector<cudf::size_type>
compute_list_offsets(cudf::size_type n_lists,
                     rmm::device_uvector<NodeIndexT> const &parent_node_ids,
                     rmm::device_uvector<int8_t> const &key_or_value, rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource *mr) {
  // Count the number of children nodes for the json object nodes.
  // These object nodes are given as one row of the input json strings column.
  auto node_child_counts = rmm::device_uvector<NodeIndexT>(parent_node_ids.size(), stream);

  // For the nodes having parent_id == 0 (they are json object given by one input row), set their
  // child counts to zero. Otherwise, set child counts to `-1` (a sentinel number).
  thrust::transform(rmm::exec_policy(stream), parent_node_ids.begin(), parent_node_ids.end(),
                    node_child_counts.begin(), [] __device__(auto const parent_id) -> NodeIndexT {
                      return parent_id == 0 ? 0 : std::numeric_limits<NodeIndexT>::lowest();
                    });

  auto const is_key = [key_or_value = key_or_value.begin()] __device__(auto const node_id) {
    return key_or_value[node_id] == key_sentinel;
  };

  // Count the number of keys for each json object using `atomicAdd`.
  auto const transform_it = thrust::counting_iterator<int>(0);
  thrust::for_each(rmm::exec_policy(stream), transform_it, transform_it + parent_node_ids.size(),
                   [is_key, child_counts = node_child_counts.begin(),
                    parent_ids = parent_node_ids.begin()] __device__(auto const node_id) {
                     if (is_key(node_id)) {
                       auto const parent_id = parent_ids[node_id];
                       atomicAdd(&child_counts[parent_id], 1);
                     }
                   });
#ifdef DEBUG_FROM_JSON
  print_debug(node_child_counts, "Nodes' child keys counts", ", ", stream);
#endif

  auto list_offsets = rmm::device_uvector<cudf::size_type>(n_lists + 1, stream, mr);
  auto const copy_end = cudf::detail::copy_if_safe(
      node_child_counts.begin(), node_child_counts.end(), list_offsets.begin(),
      [] __device__(auto const count) { return count >= 0; }, stream);
  CUDF_EXPECTS(thrust::distance(list_offsets.begin(), copy_end) == static_cast<int64_t>(n_lists),
               "Invalid list size computation.");
#ifdef DEBUG_FROM_JSON
  print_debug(list_offsets, "Output list sizes (except the last one)", ", ", stream);
#endif

  thrust::exclusive_scan(rmm::exec_policy(stream), list_offsets.begin(), list_offsets.end(),
                         list_offsets.begin());
#ifdef DEBUG_FROM_JSON
  print_debug(list_offsets, "Output list offsets", ", ", stream);
#endif
  return list_offsets;
}

} // namespace

std::unique_ptr<cudf::column> from_json(cudf::column_view const &input,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource *mr) {
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRING, "Invalid input format");

  // Firstly, concatenate all the input json strings into one giant input json string.
  // When testing/debugging, the output can be validated using
  // https://jsonformatter.curiousconcept.com.
  auto const unified_json_buff = unify_json_strings(input, stream);

  // Tokenize the input json strings.
  static_assert(sizeof(SymbolT) == sizeof(char),
                "Invalid internal data for nested json tokenizer.");
  auto const [tokens, token_indices] = cudf::io::json::detail::get_token_stream(
      cudf::device_span<char const>{unified_json_buff.data(), unified_json_buff.size()},
      cudf::io::json_reader_options{}, stream, rmm::mr::get_current_device_resource());

#ifdef DEBUG_FROM_JSON
  print_debug(tokens, "Tokens", ", ", stream);
  print_debug(token_indices, "Token indices", ", ", stream);
#endif

  // Make sure there is no error during parsing.
  throw_if_error(unified_json_buff, tokens, token_indices, stream);

  auto const num_nodes =
      thrust::count_if(rmm::exec_policy(stream), tokens.begin(), tokens.end(), is_node{});

  // Compute the map from nodes to their indices in the list of all tokens.
  auto const node_token_ids = compute_node_to_token_index_map(num_nodes, tokens, stream);

  // A map from each node to the index of its parent node.
  auto const parent_node_ids = compute_parent_node_ids(num_nodes, tokens, node_token_ids, stream);

  // Check for each node if it is a map key or a map value to extract.
  auto const key_or_value_node = check_key_or_value_nodes(parent_node_ids, stream);

  // Compute index range for each node.
  // These ranges identify positions to extract nodes from the unified json string.
  auto const node_ranges = compute_node_ranges(num_nodes, tokens, token_indices, node_token_ids,
                                               parent_node_ids, key_or_value_node, stream);

  //
  // From below are variables for returning output.
  //

  auto extracted_keys = extract_keys_or_values(true /*key*/, num_nodes, node_ranges,
                                               key_or_value_node, unified_json_buff, stream, mr);
  auto extracted_values = extract_keys_or_values(false /*value*/, num_nodes, node_ranges,
                                                 key_or_value_node, unified_json_buff, stream, mr);
  CUDF_EXPECTS(extracted_keys->size() == extracted_values->size(),
               "Invalid key-value pair extraction.");

  // Compute the offsets of the final output lists column.
  auto list_offsets =
      compute_list_offsets(input.size(), parent_node_ids, key_or_value_node, stream, mr);

#ifdef DEBUG_FROM_JSON
  print_output_spark_map(list_offsets, extracted_keys, extracted_values, stream);
#endif

  auto const num_pairs = extracted_keys->size();
  std::vector<std::unique_ptr<cudf::column>> out_keys_vals;
  out_keys_vals.emplace_back(std::move(extracted_keys));
  out_keys_vals.emplace_back(std::move(extracted_values));
  auto structs_col = cudf::make_structs_column(num_pairs, std::move(out_keys_vals), 0,
                                               rmm::device_buffer{}, stream, mr);

  auto offsets = std::make_unique<cudf::column>(std::move(list_offsets), rmm::device_buffer{}, 0);

  return cudf::make_lists_column(input.size(), std::move(offsets), std::move(structs_col),
                                 input.null_count(), cudf::detail::copy_bitmask(input, stream, mr),
                                 stream, mr);
}

} // namespace spark_rapids_jni
