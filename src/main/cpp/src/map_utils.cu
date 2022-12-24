/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "map_utils.hpp"

//
#include <iomanip>
#include <limits>
#include <sstream>

#include <string_view>

//
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/detail/nested_json.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/column_utilities.hpp>
#include <rmm/cuda_stream_view.hpp>

//
#include <rmm/exec_policy.hpp>

//
#include <cub/device/device_radix_sort.cuh>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

//
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#define DEBUG_FROM_JSON

namespace spark_rapids_jni {

using namespace cudf::io::json;

namespace {

// Convert token indices to node range for each valid node.
struct node_ranges {
  cudf::device_span<PdaTokenT const> tokens;
  cudf::device_span<SymbolOffsetT const> token_indices;
  cudf::device_span<SymbolOffsetT const> node_to_token_indices;
  cudf::device_span<NodeIndexT const> parent_node_ids;
  static const bool include_quote_char{false};
  __device__ auto operator()(cudf::size_type node_idx)
      -> thrust::tuple<SymbolOffsetT, SymbolOffsetT> {
    // Whether a token expects to be followed by its respective end-of-* token partner
    auto const is_begin_of_section = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StringBegin:
        case token_t::ValueBegin:
        case token_t::FieldNameBegin: return true;
        default: return false;
      };
    };
    // The end-of-* partner token for a given beginning-of-* token
    auto const end_of_partner = [] __device__(PdaTokenT const token) {
      switch (token) {
        case token_t::StringBegin: return token_t::StringEnd;
        case token_t::ValueBegin: return token_t::ValueEnd;
        case token_t::FieldNameBegin: return token_t::FieldNameEnd;
        default: return token_t::ErrorBegin;
      };
    };
    // Includes quote char for end-of-string token or Skips the quote char for
    // beginning-of-field-name token
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

    // root json object
    if (parent_node_ids[node_idx] <= 0) {
      return thrust::make_tuple(0, 0);
    }

    if (parent_node_ids[parent_node_ids[node_idx]] == 0 // key
        || (parent_node_ids[parent_node_ids[node_idx]] > 0 &&
            parent_node_ids[parent_node_ids[parent_node_ids[node_idx]]] == 0) // value
    ) {
      auto const token_idx = node_to_token_indices[node_idx];
      PdaTokenT const token = tokens[token_idx];
      // The section from the original JSON input that this token demarcates
      SymbolOffsetT range_begin = get_token_index(token, token_indices[token_idx]);
      SymbolOffsetT range_end = range_begin + 1; // non-leaf, non-field nodes ignore this value.
      if (is_begin_of_section(token)) {
        if ((token_idx + 1) < tokens.size() && end_of_partner(token) == tokens[token_idx + 1]) {
          // Update the range_end for this pair of tokens
          range_end = get_token_index(tokens[token_idx + 1], token_indices[token_idx + 1]);
        }
      }
      return thrust::make_tuple(range_begin, range_end);
    }

    return thrust::make_tuple(0, 0);
  }
};

template <typename IndexType = size_t, typename KeyType>
std::pair<rmm::device_uvector<KeyType>, rmm::device_uvector<IndexType>>
stable_sorted_key_order(cudf::device_span<KeyType const> keys, rmm::cuda_stream_view stream) {

  // Determine temporary device storage requirements
  rmm::device_uvector<KeyType> keys_buffer1(keys.size(), stream);
  rmm::device_uvector<KeyType> keys_buffer2(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer1(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer2(keys.size(), stream);
  cub::DoubleBuffer<IndexType> order_buffer(order_buffer1.data(), order_buffer2.data());
  cub::DoubleBuffer<KeyType> keys_buffer(keys_buffer1.data(), keys_buffer2.data());
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, keys_buffer, order_buffer,
                                  keys.size());
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);

  thrust::copy(rmm::exec_policy(stream), keys.begin(), keys.end(), keys_buffer1.begin());
  thrust::sequence(rmm::exec_policy(stream), order_buffer1.begin(), order_buffer1.end());

  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(), temp_storage_bytes, keys_buffer,
                                  order_buffer, keys.size(), 0, sizeof(KeyType) * 8,
                                  stream.value());

  return std::pair{keys_buffer.Current() == keys_buffer1.data() ? std::move(keys_buffer1) :
                                                                  std::move(keys_buffer2),
                   order_buffer.Current() == order_buffer1.data() ? std::move(order_buffer1) :
                                                                    std::move(order_buffer2)};
}
void propagate_parent_to_siblings(cudf::device_span<TreeDepthT const> node_levels,
                                  cudf::device_span<NodeIndexT> parent_node_ids,
                                  rmm::cuda_stream_view stream) {
  auto [sorted_node_levels, sorted_order] =
      stable_sorted_key_order<cudf::size_type>(node_levels, stream);
  // instead of gather, using permutation_iterator, which is ~17% faster

  thrust::inclusive_scan_by_key(
      rmm::exec_policy(stream), sorted_node_levels.begin(), sorted_node_levels.end(),
      thrust::make_permutation_iterator(parent_node_ids.begin(), sorted_order.begin()),
      thrust::make_permutation_iterator(parent_node_ids.begin(), sorted_order.begin()),
      thrust::equal_to<TreeDepthT>{}, thrust::maximum<NodeIndexT>{});
}

#ifdef DEBUG_FROM_JSON
std::string token_to_string(PdaTokenT token_type) {
  switch (token_type) {
    case token_t::StructBegin: return "StructBegin";
    case token_t::StructEnd: return "StructEnd";
    case token_t::ListBegin: return "ListBegin";
    case token_t::ListEnd: return "ListEnd";
    case token_t::StructMemberBegin: return "StructMemberBegin";
    case token_t::StructMemberEnd: return "StructMemberEnd";
    case token_t::FieldNameBegin: return "FieldNameBegin";
    case token_t::FieldNameEnd: return "FieldNameEnd";
    case token_t::StringBegin: return "StringBegin";
    case token_t::StringEnd: return "StringEnd";
    case token_t::ValueBegin: return "ValueBegin";
    case token_t::ValueEnd: return "ValueEnd";
    case token_t::ErrorBegin: return "ErrorBegin";
    default: return "Unknown";
  }
}

#endif

auto make_empty_json_string_buffer(rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource *mr) {
  return cudf::detail::make_device_uvector_async<char>(std::vector<char>{'[', ']'}, stream, mr);
}

auto unify_json_strings(cudf::column_view const &input, rmm::cuda_stream_view stream,
                        rmm::mr::device_memory_resource *mr) {
  if (input.is_empty()) {
    return make_empty_json_string_buffer(stream, mr);
  }

  // We append one comma character (',') to the end of all input strings before concatenating them.
  // Note that we also need to concatenate the opening and closing brackets ('[' and ']') to the
  // beginning and the end of the output.
  auto const output_size =
      static_cast<std::size_t>(input.child(cudf::strings_column_view::chars_column_index).size()) +
      static_cast<std::size_t>(input.size()) -
      1 + // one ',' character for each input row, except the last row
      2;  // two extra bracket characters '[' and ']'
  CUDF_EXPECTS(output_size <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
               "The input json column is too large and causes overflow.");

  auto const input_append_commas = cudf::strings::detail::join_strings(
      cudf::strings_column_view{input},
      cudf::string_scalar(","),       // append `,` character between the input rows
      cudf::string_scalar("", false), // ignore null rows
      stream, mr);
  auto const input_append_commas_size_bytes =
      cudf::strings_column_view{input_append_commas->view()}.chars_size();

  // `input_append_commas` has exactly one row and should not have null.
  if (input_append_commas->has_nulls() || input_append_commas_size_bytes == 0) {
    return make_empty_json_string_buffer(stream, mr);
  }

  // We want to concatenate 3 strings: "["+input_append_commas+"]".
  // For efficiency, let's use memcpy instead of `cudf::strings::detail::concatenate`.
  auto output = rmm::device_uvector<char>(input_append_commas_size_bytes + 2, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(output.data(), static_cast<int>('['), 1, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
      output.data() + 1,
      input_append_commas->child(cudf::strings_column_view::chars_column_index).view().data<char>(),
      input_append_commas_size_bytes, cudaMemcpyDefault, stream.value()));
  CUDF_CUDA_TRY(cudaMemsetAsync(output.data() + input_append_commas_size_bytes + 1,
                                static_cast<int>(']'), 1, stream.value()));

  return output;
}

template <typename InputIterator, typename StencilIterator, typename OutputIterator,
          typename Predicate>
OutputIterator thrust_copy_if(rmm::exec_policy policy, InputIterator first, InputIterator last,
                              StencilIterator stencil, OutputIterator result, Predicate pred) {
  auto const copy_size = std::min(static_cast<std::size_t>(std::distance(first, last)),
                                  static_cast<std::size_t>(std::numeric_limits<int>::max()));

  auto itr = first;
  while (itr != last) {
    auto const copy_end =
        static_cast<std::size_t>(std::distance(itr, last)) <= copy_size ? last : itr + copy_size;
    result = thrust::copy_if(policy, itr, copy_end, stencil, result, pred);
    stencil += std::distance(itr, copy_end);
    itr = copy_end;
  }
  return result;
}

template <typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator thrust_copy_if(rmm::exec_policy policy, InputIterator first, InputIterator last,
                              OutputIterator result, Predicate pred) {
  return thrust_copy_if(policy, first, last, first, result, pred);
}

} // namespace

std::unique_ptr<cudf::column> from_json(cudf::column_view const &input,
                                        bool throw_if_keys_duplicate, rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource *mr) {
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRING, "Invalid input format");

  auto const default_mr = rmm::mr::get_current_device_resource();

  // Firstly, concatenate all the input json strings into one giant input json string.
  // The output can be validated using https://jsonformatter.curiousconcept.com/.
  auto const unified_json_buff = unify_json_strings(input, stream, default_mr);
  auto const d_unified_json =
      cudf::device_span<char const>{unified_json_buff.data(), unified_json_buff.size()};

#ifdef DEBUG_FROM_JSON
  {
    auto const h_json = cudf::detail::make_host_vector_sync(d_unified_json, stream);
    std::stringstream ss;
    ss << "Processed json string:\n";
    ss << std::string_view{h_json.data(), h_json.size()};
    std::cerr << ss.str() << std::endl;
  }
#endif

  // Tokenize the input json strings.
  static_assert(sizeof(SymbolT) == sizeof(char),
                "Invalid internal data for nested json tokenizer.");
  auto const [tokens, token_indices] =
      detail::get_token_stream(d_unified_json, cudf::io::json_reader_options{}, stream, default_mr);

#ifdef DEBUG_FROM_JSON
  {
    auto const h_tokens = cudf::detail::make_host_vector_sync(
        cudf::device_span<PdaTokenT const>{tokens.data(), tokens.size()}, stream);
    auto const h_token_indices = cudf::detail::make_host_vector_sync(
        cudf::device_span<SymbolOffsetT const>{token_indices.data(), token_indices.size()}, stream);

    std::stringstream ss;
    ss << "Tokens:\n";
    for (auto const token : h_tokens) {
      ss << static_cast<int>(token) << ", ";
    }
    ss << "\nToken indices:\n";
    for (auto const token_idx : h_token_indices) {
      ss << static_cast<int>(token_idx) << ", ";
    }
    std::cerr << ss.str() << std::endl;
  }
#endif

  // Check for error.
  CUDF_EXPECTS(thrust::count(rmm::exec_policy(stream), tokens.begin(), tokens.end(),
                             token_t::ErrorBegin) == 0,
               "There is error during parsing the input json string(s).");

  // Whether a token does represent a node in the tree representation
  auto const is_node = [] __host__ __device__(PdaTokenT const token) -> bool {
    switch (token) {
      case token_t::StructBegin:
      case token_t::ListBegin:
      case token_t::StringBegin:
      case token_t::ValueBegin:
      case token_t::FieldNameBegin:
      case token_t::ErrorBegin: return true;
      default: return false;
    };
  };

  // Whether the token pops from the parent node stack
  auto const does_pop = [] __device__(PdaTokenT const token) -> bool {
    switch (token) {
      case token_t::StructMemberEnd:
      case token_t::StructEnd:
      case token_t::ListEnd: return true;
      default: return false;
    };
  };

  // Whether the token pushes onto the parent node stack
  auto const does_push = [] __device__(PdaTokenT const token) -> bool {
    switch (token) {
      case token_t::FieldNameBegin:
      case token_t::StructBegin:
      case token_t::ListBegin: return true;
      default: return false;
    };
  };

  auto const num_tokens = tokens.size();
  auto const num_nodes =
      thrust::count_if(rmm::exec_policy(stream), tokens.begin(), tokens.end(), is_node);

  // Node levels: transform_exclusive_scan, copy_if.
  rmm::device_uvector<TreeDepthT> node_levels(num_nodes, stream, default_mr);
  {
    rmm::device_uvector<TreeDepthT> token_levels(num_tokens, stream, default_mr);
    auto const push_pop_it = thrust::make_transform_iterator(
        tokens.begin(), [does_push, does_pop] __device__(PdaTokenT const token) -> cudf::size_type {
          return does_push(token) - does_pop(token);
        });
    thrust::exclusive_scan(rmm::exec_policy(stream), push_pop_it, push_pop_it + num_tokens,
                           token_levels.begin());
#ifdef DEBUG_FROM_JSON
    {
      auto const h_json = cudf::detail::make_host_vector_sync(d_unified_json, stream);
      auto const h_tokens = cudf::detail::make_host_vector_sync(
          cudf::device_span<PdaTokenT const>{tokens.data(), tokens.size()}, stream);
      auto const h_token_indices = cudf::detail::make_host_vector_sync(
          cudf::device_span<SymbolOffsetT const>{token_indices.data(), token_indices.size()},
          stream);
      auto const h_token_levels = cudf::detail::make_host_vector_sync(
          cudf::device_span<TreeDepthT const>{token_levels.data(), token_levels.size()}, stream);

      std::stringstream ss;
      ss << "Token levels:\n";
      SymbolOffsetT print_idx{0};
      NodeT node_id{0};
      for (size_t i = 0; i < h_token_levels.size(); ++i) {
        auto const token_idx = h_token_indices[i];
        while (print_idx < token_idx) {
          ss << std::setw(5) << print_idx << ": " << h_json[print_idx] << "\n";
          ++print_idx;
        }
        print_idx = token_idx + 1;

        auto const c = h_json[token_idx];
        auto const level = h_token_levels[i];
        ss << std::setw(5) << token_idx << ": " << c << " | " << std::left << std::setw(17)
           << token_to_string(h_tokens[i]) << " | level = " << static_cast<int>(level)
           << (is_node(h_tokens[i]) ? " (node, id = " + std::to_string(node_id++) + ")" : "")
           << "\n";
      }
      std::cerr << ss.str() << std::endl;
    }
#endif

    auto const node_levels_end =
        thrust_copy_if(rmm::exec_policy(stream), token_levels.begin(), token_levels.end(),
                       tokens.begin(), node_levels.begin(), is_node);
    CUDF_EXPECTS(thrust::distance(node_levels.begin(), node_levels_end) == num_nodes,
                 "Node level count mismatch");
  }

#ifdef DEBUG_FROM_JSON
  {
    auto const h_node_levels = cudf::detail::make_host_vector_sync(
        cudf::device_span<TreeDepthT const>{node_levels.data(), node_levels.size()}, stream);
    std::stringstream ss;
    ss << "Node levels:\n";
    for (auto const level : h_node_levels) {
      ss << static_cast<int>(level) << ", ";
    }
    std::cerr << ss.str() << std::endl;
  }
#endif

  // Node parent ids:
  // previous push node_id transform, stable sort by level, segmented scan with Max, reorder.
  rmm::device_uvector<NodeIndexT> parent_node_ids(num_nodes, stream, mr);
  // This block of code is generalized logical stack algorithm. TODO: make this a separate function.
  {
    rmm::device_uvector<NodeIndexT> node_token_ids(num_nodes, stream);
    thrust_copy_if(rmm::exec_policy(stream), thrust::make_counting_iterator<NodeIndexT>(0),
                   thrust::make_counting_iterator<NodeIndexT>(0) + num_tokens, tokens.begin(),
                   node_token_ids.begin(), is_node);

    // previous push node_id
    // if previous node is a push, then i-1
    // if previous node is FE, then i-2 (returns FB's index)
    // if previous node is SMB and its previous node is a push, then i-2
    // eg. `{ SMB FB FE VB VE SME` -> `{` index as FB's parent.
    // else -1
    auto const first_childs_parent_token_id =
        [tokens_gpu = tokens.begin()] __device__(auto i) -> NodeIndexT {
      if (i <= 0) {
        return -1;
      }
      if (tokens_gpu[i - 1] == token_t::StructBegin or tokens_gpu[i - 1] == token_t::ListBegin) {
        return i - 1;
      } else if (tokens_gpu[i - 1] == token_t::FieldNameEnd) {
        return i - 2;
      } else if (tokens_gpu[i - 1] == token_t::StructMemberBegin and
                 (tokens_gpu[i - 2] == token_t::StructBegin ||
                  tokens_gpu[i - 2] == token_t::ListBegin)) {
        return i - 2;
      } else {
        return -1;
      }
    };

    thrust::transform(
        rmm::exec_policy(stream), node_token_ids.begin(), node_token_ids.end(),
        parent_node_ids.begin(),
        [node_ids_gpu = node_token_ids.begin(), num_nodes,
         first_childs_parent_token_id] __device__(NodeIndexT const tid) -> NodeIndexT {
          auto const pid = first_childs_parent_token_id(tid);
          return pid < 0 ?
                     parent_node_sentinel :
                     thrust::lower_bound(thrust::seq, node_ids_gpu, node_ids_gpu + num_nodes, pid) -
                         node_ids_gpu;
          // parent_node_sentinel is -1, useful for segmented max operation below
        });
  }
  // Propagate parent node to siblings from first sibling - inplace.
  propagate_parent_to_siblings(
      cudf::device_span<TreeDepthT const>{node_levels.data(), node_levels.size()}, parent_node_ids,
      stream);

#ifdef DEBUG_FROM_JSON
  {
    auto const h_parent_node_ids = cudf::detail::make_host_vector_sync(
        cudf::device_span<NodeIndexT const>{parent_node_ids.data(), parent_node_ids.size()},
        stream);
    std::stringstream ss;
    ss << "Parent node id:\n";
    for (auto const id : h_parent_node_ids) {
      ss << static_cast<int>(id) << ", ";
    }
    std::cerr << ss.str() << std::endl;
  }
#endif

  rmm::device_uvector<SymbolOffsetT> node_to_token_indices(num_nodes, stream, mr);
  auto const node_to_token_indices_end =
      thrust_copy_if(rmm::exec_policy(stream), thrust::make_counting_iterator<cudf::size_type>(0),
                     thrust::make_counting_iterator<cudf::size_type>(0) + num_tokens,
                     node_to_token_indices.begin(),
                     [is_node, tokens_gpu = tokens.begin()] __device__(cudf::size_type i) -> bool {
                       return is_node(tokens_gpu[i]);
                     });
  CUDF_EXPECTS(thrust::distance(node_to_token_indices.begin(), node_to_token_indices_end) ==
                   num_nodes,
               "node to token index map count mismatch");
#ifdef DEBUG_FROM_JSON
  {
    auto const h_node_to_token_indices = cudf::detail::make_host_vector_sync(
        cudf::device_span<SymbolOffsetT const>{node_to_token_indices.data(),
                                               node_to_token_indices.size()},
        stream);
    std::stringstream ss;
    ss << "Node-to-token-index map:\n";
    for (size_t i = 0; i < h_node_to_token_indices.size(); ++i) {
      ss << i << " => " << static_cast<int>(h_node_to_token_indices[i]) << "\n";
    }
    std::cerr << ss.str() << std::endl;
  }
#endif

  // Node ranges: copy_if with transform.
  rmm::device_uvector<SymbolOffsetT> node_range_begin(num_nodes, stream, mr);
  rmm::device_uvector<SymbolOffsetT> node_range_end(num_nodes, stream, mr);
  auto const node_range_tuple_it =
      thrust::make_zip_iterator(node_range_begin.begin(), node_range_end.begin());
  // Whether the tokenizer stage should keep quote characters for string values
  // If the tokenizer keeps the quote characters, they may be stripped during type casting
  auto const node_range_out_it = thrust::make_transform_output_iterator(
      node_range_tuple_it,
      node_ranges{tokens, token_indices, node_to_token_indices, parent_node_ids});

  thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(0) + num_nodes,
                    node_range_out_it, thrust::identity{});

#ifdef DEBUG_FROM_JSON
  {
    auto const h_node_range_begin = cudf::detail::make_host_vector_sync(
        cudf::device_span<SymbolOffsetT const>{node_range_begin.data(), node_range_begin.size()},
        stream);
    auto const h_node_range_end = cudf::detail::make_host_vector_sync(
        cudf::device_span<SymbolOffsetT const>{node_range_end.data(), node_range_end.size()},
        stream);
    std::stringstream ss;
    ss << "Node range:\n";
    for (size_t i = 0; i < h_node_range_begin.size(); ++i) {
      ss << "[ " << static_cast<int>(h_node_range_begin[i]) << ", "
         << static_cast<int>(h_node_range_end[i]) << " ]\n";
    }
    std::cerr << ss.str() << std::endl;
  }
#endif

  // Identify the key-value tokens.
  // Keys: Nodes with parent_idx[parent_idx] == 0.
  // Values: The nodes that are direct children of the key nodes.
#if 0

  // Substring the input to extract out keys.

  // Substring the input to extract out values.

// Compute the offsets of the output lists column.
// Firstly, extract the key nodes.
// Compute the numbers of keys having the same parent using reduce_by_key.
// These numbers will also be sizes of the output lists.

#else
  return cudf::make_strings_column(
      1, cudf::detail::make_device_uvector_async<int>(std::vector<int>{0, 1}, stream, default_mr),
      unify_json_strings(input, stream, default_mr));
#endif
}

} // namespace spark_rapids_jni
