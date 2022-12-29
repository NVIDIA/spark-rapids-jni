
#include <iomanip>
#include <limits>
#include <sstream>

#include <string_view>

//
#include <cudf/io/detail/nested_json.hpp>
//
#include <cudf/strings/detail/concatenate.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/strings/substring.hpp>

//
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
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
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

//
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
//
//
//
//
//
//
//
//
//
//
//
//

#define DEBUG_FROM_JSON

namespace spark_rapids_jni {

using namespace cudf::io::json;

namespace {

template <typename T, typename U = int>
void print_debug(rmm::device_uvector<T> const &input, std::string const &name,
                 std::string const &separator, rmm::cuda_stream_view stream) {
  auto const h_input = cudf::detail::make_host_vector_sync(
      cudf::device_span<T const>{input.data(), input.size()}, stream);
  std::stringstream ss;
  ss << name << ":\n";
  for (size_t i = 0; i < h_input.size(); ++i) {
    ss << static_cast<U>(h_input[i]);
    if (separator.size() > 0 && i + 1 < h_input.size()) {
      ss << separator;
    }
  }
  std::cerr << ss.str() << std::endl;
}

rmm::device_uvector<char> unify_json_strings(cudf::column_view const &input,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource *mr) {
  if (input.is_empty()) {
    return cudf::detail::make_device_uvector_async<char>(std::vector<char>{'[', ']'}, stream, mr);
  }

  // Unify the input json strings by:
  // 1. Append one comma character (',') to the end of each input string, except the last one.
  // 2. Concatenate all input strings into one string.
  // 3. Add a pair of bracket characters ('[' and ']') to the beginning and the end of the output.
  auto const d_strings = cudf::column_device_view::create(input, stream);
  auto const output_size =
      2 + // two extra bracket characters '[' and ']'
      thrust::transform_reduce(
          rmm::exec_policy(stream), thrust::make_counting_iterator(0),
          thrust::make_counting_iterator(input.size()),
          [d_strings = *d_strings,
           n_rows = input.size()] __device__(cudf::size_type idx) -> int64_t {
            auto bytes = d_strings.is_null(idx) ?
                             2 // replace null with "{}"
                             :
                             d_strings.element<cudf::string_view>(idx).size_bytes();
            if (idx + 1 < n_rows) {
              bytes += 1; // append `,` character to each input rows, except the last one
            }
            return static_cast<int64_t>(bytes);
          },
          int64_t{0}, thrust::plus{});

  CUDF_EXPECTS(output_size <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
               "The input json column is too large and causes overflow.");

  auto const joined_input = cudf::strings::detail::join_strings(
      cudf::strings_column_view{input},
      cudf::string_scalar(","),  // append `,` character between the input rows
      cudf::string_scalar("{}"), // replacement for null rows
      stream, mr);
  auto const joined_input_child =
      joined_input->child(cudf::strings_column_view::chars_column_index);
  auto const joined_input_size_bytes = joined_input_child.size();
  CUDF_EXPECTS(joined_input_size_bytes + 2 == output_size, "Incorrect output size computation.");

  // We want to concatenate 3 strings: "[" + joined_input + "]".
  // For efficiency, let's use memcpy instead of `cudf::strings::detail::concatenate`.
  auto output = rmm::device_uvector<char>(joined_input_size_bytes + 2, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(output.data(), static_cast<int>('['), 1, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(output.data() + 1, joined_input_child.view().data<char>(),
                                joined_input_size_bytes, cudaMemcpyDefault, stream.value()));
  CUDF_CUDA_TRY(cudaMemsetAsync(output.data() + joined_input_size_bytes + 1, static_cast<int>(']'),
                                1, stream.value()));

  return output;
}

void throw_if_error(rmm::device_uvector<PdaTokenT> const &tokens,
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
    CUDF_FAIL("JSON Parser encountered an invalid format at location " +
              std::to_string(error_index));
  }
}

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

rmm::device_uvector<TreeDepthT> compute_node_levels(int64_t num_nodes,
                                                    rmm::device_uvector<PdaTokenT> const &tokens,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource *mr) {
  rmm::device_uvector<TreeDepthT> node_levels(num_nodes, stream, mr);
  rmm::device_uvector<TreeDepthT> token_levels(tokens.size(), stream, mr);

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

  auto const push_pop_it = thrust::make_transform_iterator(
      tokens.begin(), [does_push, does_pop] __device__(PdaTokenT const token) -> cudf::size_type {
        return does_push(token) - does_pop(token);
      });
  thrust::exclusive_scan(rmm::exec_policy(stream), push_pop_it, push_pop_it + tokens.size(),
                         token_levels.begin());
#ifdef DEBUG_FROM_JSONX
  {
    auto const h_json = cudf::detail::make_host_vector_sync(d_unified_json, stream);
    auto const h_tokens = cudf::detail::make_host_vector_sync(
        cudf::device_span<PdaTokenT const>{tokens.data(), tokens.size()}, stream);
    auto const h_token_indices = cudf::detail::make_host_vector_sync(
        cudf::device_span<SymbolOffsetT const>{token_indices.data(), token_indices.size()}, stream);
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
         << (is_node{}(h_tokens[i]) ? " (node, id = " + std::to_string(node_id++) + ")" : "")
         << "\n";
    }
    std::cerr << ss.str() << std::endl;
  }
#endif

  auto const node_levels_end =
      thrust_copy_if(rmm::exec_policy(stream), token_levels.begin(), token_levels.end(),
                     tokens.begin(), node_levels.begin(), is_node{});
  CUDF_EXPECTS(thrust::distance(node_levels.begin(), node_levels_end) == num_nodes,
               "Node level count mismatch");

  return node_levels;
}

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

rmm::device_uvector<NodeIndexT>
compute_parent_node_ids(int64_t num_nodes, rmm::device_uvector<PdaTokenT> const &tokens,
                        rmm::device_uvector<TreeDepthT> const &node_levels,
                        rmm::cuda_stream_view stream, rmm::mr::device_memory_resource *mr) {
  // previous push node_id transform, stable sort by level, segmented scan with Max, reorder.
  rmm::device_uvector<NodeIndexT> parent_node_ids(num_nodes, stream, mr);
  // This block of code is generalized logical stack algorithm. TODO: make this a separate
  // function.
  rmm::device_uvector<NodeIndexT> node_token_ids(num_nodes, stream);
  thrust_copy_if(rmm::exec_policy(stream), thrust::make_counting_iterator<NodeIndexT>(0),
                 thrust::make_counting_iterator<NodeIndexT>(0) + tokens.size(), tokens.begin(),
                 node_token_ids.begin(), is_node{});

  // previous push node_id
  // if previous node is a push, then i-1
  // if previous node is FE, then i-2 (returns FB's index)
  // if previous node is SMB and its previous node is a push, then i-2
  // eg. `{ SMB FB FE VB VE SME` -> `{` index as FB's parent.
  // else -1
  auto const first_childs_parent_token_id = [tokens_gpu =
                                                 tokens.begin()] __device__(auto i) -> NodeIndexT {
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

  thrust::transform(rmm::exec_policy(stream), node_token_ids.begin(), node_token_ids.end(),
                    parent_node_ids.begin(),
                    [node_ids_gpu = node_token_ids.begin(), num_nodes,
                     first_childs_parent_token_id] __device__(NodeIndexT const tid) -> NodeIndexT {
                      auto const pid = first_childs_parent_token_id(tid);
                      return pid < 0 ? parent_node_sentinel :
                                       thrust::lower_bound(thrust::seq, node_ids_gpu,
                                                           node_ids_gpu + num_nodes, pid) -
                                           node_ids_gpu;
                      // parent_node_sentinel is -1, useful for segmented max operation below
                    });

  // Propagate parent node to siblings from first sibling - inplace.
  propagate_parent_to_siblings(
      cudf::device_span<TreeDepthT const>{node_levels.data(), node_levels.size()}, parent_node_ids,
      stream);

  return parent_node_ids;
}

struct is_key {
  cudf::device_span<NodeIndexT const> parent_node_ids;
  __device__ auto operator()(cudf::size_type node_idx) const {
    return parent_node_ids[node_idx] > 0 && parent_node_ids[parent_node_ids[node_idx]] == 0;
  }
};

struct is_value {
  cudf::device_span<NodeIndexT const> parent_node_ids;
  __device__ auto operator()(cudf::size_type node_idx) const {
    return parent_node_ids[node_idx] > 0 && parent_node_ids[parent_node_ids[node_idx]] > 0 &&
           parent_node_ids[parent_node_ids[parent_node_ids[node_idx]]] == 0;
  }
};

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
    //    auto const is_begin_of_section = [] __device__(PdaTokenT const token) {
    //      switch (token) {
    //        case token_t::StructBegin:
    //        case token_t::ListBegin:
    //        case token_t::StringBegin:
    //        case token_t::ValueBegin:
    //        case token_t::FieldNameBegin: return true;
    //        default: return false;
    //      };
    //    };
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

    auto const nested_node_to_value = [] __device__(PdaTokenT const token) -> int32_t {
      switch (token) {
        case token_t::StructBegin: return 1;
        case token_t::StructEnd: return -1;
        case token_t::ListBegin: return 1 << 8;
        case token_t::ListEnd: return -(1 << 8);
        default: return 0;
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
      cudf_assert(is_begin_of_section(token) && "Invalid node category.");

      // The section from the original JSON input that this token demarcates
      SymbolOffsetT range_begin = get_token_index(token, token_indices[token_idx]);
      SymbolOffsetT range_end = range_begin + 1; // non-leaf, non-field nodes ignore this value.
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
      }
      return thrust::make_tuple(range_begin, range_end);
    }

    return thrust::make_tuple(0, 0);
  }
};

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

#define NO_STRING_VIEW

/**
 * @brief Function logic for substring_from API.
 *
 * This both calculates the output size and executes the substring.
 * No bound check is performed, assuming that the substring bounds are all valid.
 */
struct substring_fn {
#ifdef NO_STRING_VIEW
  cudf::device_span<char const> const d_string;
#else
  cudf::string_view const d_string;
#endif
  cudf::device_span<SymbolOffsetT const> const d_begins;
  cudf::device_span<SymbolOffsetT const> const d_ends;

  cudf::offset_type *d_offsets{};
  char *d_chars{};

  __device__ void operator()(cudf::size_type idx)
#ifdef NO_STRING_VIEW
  {
    auto const length = static_cast<cudf::size_type>(d_string.size());
    auto const begin = d_begins[idx];
    auto const end = d_ends[idx];

    if (d_chars) {
      memcpy(d_chars + d_offsets[idx], d_string.data() + begin, end - begin);
    } else {
      d_offsets[idx] = end - begin;
    }
  }
#else
  {
    auto const length = d_string.length();
    auto const start = std::max(d_starts[idx], SymbolOffsetT{0});
    if (start >= length) {
      if (!d_chars) {
        d_offsets[idx] = 0;
      }
      return;
    }
    auto const stop = d_stops[idx];
    auto const end = stop > length ? length : stop;

    auto const d_substr = d_string.substr(start, end - start);
    if (d_chars) {
      memcpy(d_chars + d_offsets[idx], d_substr.data(), d_substr.size_bytes());
    } else {
      d_offsets[idx] = d_substr.size_bytes();
    }
  }
#endif
};

} // namespace

std::unique_ptr<cudf::column> from_json(cudf::column_view const &input,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource *mr) {
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRING, "Invalid input format");

  auto const default_mr = rmm::mr::get_current_device_resource();

  // Firstly, concatenate all the input json strings into one giant input json string.
  // When testing/debugging, the output can be validated using
  // https://jsonformatter.curiousconcept.com.
  auto const unified_json_buff = unify_json_strings(input, stream, default_mr);

#ifdef DEBUG_FROM_JSON
  { print_debug<char, char>(unified_json_buff, "Processed json string", "", stream); }
#endif

  // Tokenize the input json strings.
  static_assert(sizeof(SymbolT) == sizeof(char),
                "Invalid internal data for nested json tokenizer.");
  auto const [tokens, token_indices] = cudf::io::json::detail::get_token_stream(
      cudf::device_span<char const>{unified_json_buff.data(), unified_json_buff.size()},
      cudf::io::json_reader_options{}, stream, default_mr);
#ifdef DEBUG_FROM_JSON
  {
    print_debug(tokens, "Tokens", ", ", stream);
    print_debug(token_indices, "Token indices", ", ", stream);
  }
#endif

  // Make sure there is not any error during parsing.
  throw_if_error(tokens, token_indices, stream);

  auto const num_tokens = tokens.size();
  auto const num_nodes =
      thrust::count_if(rmm::exec_policy(stream), tokens.begin(), tokens.end(), is_node{});

  auto const node_levels = compute_node_levels(num_nodes, tokens, stream, default_mr);

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

  auto const parent_node_ids =
      compute_parent_node_ids(num_nodes, tokens, node_levels, stream, default_mr);

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
                     [is_node = is_node{}, tokens_gpu = tokens.begin()] __device__(
                         cudf::size_type i) -> bool { return is_node(tokens_gpu[i]); });
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

#if 0
  // Temporary do this for testing: it is not efficient.
  auto const json_col =
    cudf::make_strings_column(1,
                              cudf::detail::make_device_uvector_async<int>(
                                std::vector<int>{0, (int)unified_json_buff.size()}, stream, mr),
                              unify_json_strings(input, stream, default_mr));
  std::vector<cudf::column_view> cols;
  for (int i = 0; i < num_nodes; ++i) {
    cols.push_back(json_col->view());
  }
  auto const duplicates_json = cudf::strings::detail::concatenate(cols, stream, mr);

  auto const extracted_json = cudf::strings::slice_strings(
    duplicates_json->view(),
    cudf::column_view{
      cudf::data_type{cudf::type_id::INT32}, (int)node_range_begin.size(), node_range_begin.data()},
    cudf::column_view{
      cudf::data_type{cudf::type_id::INT32}, (int)node_range_end.size(), node_range_end.data()});

  // Fix this
  //  CUDF_EXPECTS(child.size() % 2 == 0, "Invalid key-value pair extraction.");

#endif

#ifdef DEBUG_FROM_JSONx
  {
    auto const child = extracted_json->child(cudf::strings_column_view::chars_column_index);
    auto const offsets = extracted_json->child(cudf::strings_column_view::offsets_column_index);

    auto const h_extracted_json = cudf::detail::make_host_vector_sync(
        cudf::device_span<char const>{child.view().data<char>(), (size_t)child.size()}, stream);

    auto const h_extracted_offsets = cudf::detail::make_host_vector_sync(
        cudf::device_span<int const>{offsets.view().data<int>(), (size_t)offsets.size()}, stream);

    std::stringstream ss;
    ss << "Extract json:\n";
    bool is_key{true};

    for (size_t i = 0; i + 1 < h_extracted_offsets.size(); ++i) {
      auto const ptr = &h_extracted_json[h_extracted_offsets[i]];
      auto const size = h_extracted_offsets[i + 1] - h_extracted_offsets[i];
      if (size > 0) {
        if (is_key) {
          ss << "\"" << std::string(ptr, size) << "\" : ";
        } else {
          ss << "\"" << std::string(ptr, size) << "\"\n";
        }
        is_key = !is_key;
      }
    }
    //    ss << std::string(h_extracted_json.data(), h_extracted_json.size()) << "\n";
    std::cerr << ss.str() << std::endl;
  }
#endif

  // Identify the key-value tokens.
  // Keys: Nodes with parent_idx[parent_idx] == 0.
  // Values: The nodes that are direct children of the key nodes.

  rmm::device_uvector<NodeIndexT> key_parent_node_ids(num_nodes, stream, mr);
  {
    auto const copy_end = thrust_copy_if(
        rmm::exec_policy(stream), parent_node_ids.begin(), parent_node_ids.end(),
        thrust::make_counting_iterator<cudf::size_type>(0), key_parent_node_ids.begin(),
        [parent_node_ids = parent_node_ids.begin()] __device__(auto const node_idx) {
          return parent_node_ids[node_idx] > 0 && parent_node_ids[parent_node_ids[node_idx]] == 0;
        });
    key_parent_node_ids.resize(thrust::distance(key_parent_node_ids.begin(), copy_end), stream);
  }

#ifdef DEBUG_FROM_JSON
  {
    auto const h_key_parent_node_ids = cudf::detail::make_host_vector_sync(
        cudf::device_span<NodeIndexT const>{key_parent_node_ids.data(), key_parent_node_ids.size()},
        stream);

    std::stringstream ss;
    ss << "Keys's parent node ids:\n";
    for (auto const id : h_key_parent_node_ids) {
      ss << static_cast<int>(id) << ", ";
    }
    std::cerr << ss.str() << std::endl;
  }
#endif

  rmm::device_uvector<cudf::size_type> list_sizes(num_nodes, stream, mr);
  {
    auto const last =
        thrust::reduce_by_key(rmm::exec_policy(stream), key_parent_node_ids.begin(),
                              key_parent_node_ids.end(), thrust::make_constant_iterator(1),
                              thrust::make_discard_iterator(), list_sizes.begin())
            .second;
    list_sizes.resize(thrust::distance(list_sizes.begin(), last), stream);
    CUDF_EXPECTS(list_sizes.size() == static_cast<std::size_t>(input.size()),
                 "Output size mismatch");
  }

#ifdef DEBUG_FROM_JSON
  {
    auto const h_list_sizes = cudf::detail::make_host_vector_sync(
        cudf::device_span<cudf::size_type const>{list_sizes.data(), list_sizes.size()}, stream);

    std::stringstream ss;
    ss << "Output list sizes:\n";
    for (auto const size : h_list_sizes) {
      ss << static_cast<int>(size) << ", ";
    }
    std::cerr << ss.str() << std::endl;
  }
#endif

  auto list_offsets =
      cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::offset_type>()},
                                input.size() + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const d_offsets = list_offsets->mutable_view().begin<cudf::offset_type>();
  CUDF_CUDA_TRY(cudaMemsetAsync(d_offsets, 0, sizeof(cudf::offset_type), stream.value()));
  thrust::inclusive_scan(rmm::exec_policy(stream), list_sizes.begin(), list_sizes.end(),
                         d_offsets + 1);

#ifdef DEBUG_FROM_JSON
  {
    auto const h_list_offsets = cudf::detail::make_host_vector_sync(
        cudf::device_span<cudf::size_type const>{d_offsets, (size_t)input.size() + 1}, stream);

    std::stringstream ss;
    ss << "Output list offsets:\n";
    for (auto const offset : h_list_offsets) {
      ss << static_cast<int>(offset) << ", ";
    }
    std::cerr << ss.str() << std::endl;
  }
#endif

  //
  //
  rmm::device_uvector<SymbolOffsetT> key_range_begin(num_nodes, stream, mr);
  rmm::device_uvector<SymbolOffsetT> key_range_end(num_nodes, stream, mr);
  rmm::device_uvector<SymbolOffsetT> value_range_begin(num_nodes, stream, mr);
  rmm::device_uvector<SymbolOffsetT> value_range_end(num_nodes, stream, mr);
  auto const key_range_tuple_it =
      thrust::make_zip_iterator(key_range_begin.begin(), key_range_end.begin());
  auto const value_range_tuple_it =
      thrust::make_zip_iterator(value_range_begin.begin(), value_range_end.begin());

  auto const key_end =
      thrust_copy_if(rmm::exec_policy(stream), node_range_tuple_it, node_range_tuple_it + num_nodes,
                     thrust::make_counting_iterator<cudf::size_type>(0), key_range_tuple_it,
                     is_key{parent_node_ids});
  auto const num_keys = thrust::distance(key_range_tuple_it, key_end);

  auto const value_end =
      thrust_copy_if(rmm::exec_policy(stream), node_range_tuple_it, node_range_tuple_it + num_nodes,
                     thrust::make_counting_iterator<cudf::size_type>(0), value_range_tuple_it,
                     is_value{parent_node_ids});
  auto const num_values = thrust::distance(value_range_tuple_it, value_end);

  CUDF_EXPECTS(num_keys == num_values, "Invalid key-value pair extraction.");

  //  printf("num values: %zu\n", num_values);

  //  if (key_range_begin.size() > 0) {
  //    return nullptr;
  //  }

  auto extracted_keys = [&] {
#if 0
    std::vector<cudf::column_view> cols;
    for (int i = 0; i < num_keys; ++i) {
      cols.push_back(json_col->view());
    }
    auto const duplicates_json = cudf::strings::detail::concatenate(cols, stream, mr);

    return cudf::strings::slice_strings(duplicates_json->view(),
                                        cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                                          (int)num_keys, key_range_begin.data()},
                                        cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                                          (int)num_keys, key_range_end.data()});
#else
    auto children = cudf::strings::detail::make_strings_children(
        substring_fn{
#ifdef NO_STRING_VIEW
            cudf::device_span<char const>(unified_json_buff.data(), unified_json_buff.size()),
#else
            cudf::string_view{unified_json_buff.data(),
                              static_cast<cudf::size_type>(unified_json_buff.size())},
#endif
            cudf::device_span<SymbolOffsetT const>{key_range_begin.data(), key_range_begin.size()},
            cudf::device_span<SymbolOffsetT const>{key_range_end.data(), key_range_end.size()}},
        num_keys, stream, mr);

    return cudf::make_strings_column(num_keys, std::move(children.first),
                                     std::move(children.second), 0, rmm::device_buffer{});
#endif
  }();

  auto extracted_values = [&] {
#if 0
    std::vector<cudf::column_view> cols;
    for (int i = 0; i < num_values; ++i) {
      cols.push_back(json_col->view());
    }
    auto const duplicates_json = cudf::strings::detail::concatenate(cols, stream, mr);

    return cudf::strings::slice_strings(duplicates_json->view(),
                                        cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                                          (int)num_values,
                                                          value_range_begin.data()},
                                        cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                                          (int)num_values, value_range_end.data()});
#else
    auto children = cudf::strings::detail::make_strings_children(
        substring_fn{
#ifdef NO_STRING_VIEW
            cudf::device_span<char const>(unified_json_buff.data(), unified_json_buff.size()),
#else
            cudf::string_view{unified_json_buff.data(),
                              static_cast<cudf::size_type>(unified_json_buff.size())},
#endif
            cudf::device_span<SymbolOffsetT const>{value_range_begin.data(),
                                                   value_range_begin.size()},
            cudf::device_span<SymbolOffsetT const>{value_range_end.data(), value_range_end.size()}},
        num_values, stream, mr);

    return cudf::make_strings_column(num_values, std::move(children.first),
                                     std::move(children.second), 0, rmm::device_buffer{});
#endif
  }();

  //  printf("num values: %zu", num_values);

  //  if (extracted_values->size() > 0) {
  //    //    printf("return\n");
  //    return nullptr;
  //  }

#ifdef DEBUG_FROM_JSON
  {
    auto const keys_child = extracted_keys->child(cudf::strings_column_view::chars_column_index);
    auto const keys_offsets =
        extracted_keys->child(cudf::strings_column_view::offsets_column_index);
    auto const values_child =
        extracted_values->child(cudf::strings_column_view::chars_column_index);
    auto const values_offsets =
        extracted_values->child(cudf::strings_column_view::offsets_column_index);

    auto const h_extracted_keys_child = cudf::detail::make_host_vector_sync(
        cudf::device_span<char const>{keys_child.view().data<char>(), (size_t)keys_child.size()},
        stream);
    auto const h_extracted_keys_offsets = cudf::detail::make_host_vector_sync(
        cudf::device_span<int const>{keys_offsets.view().data<int>(), (size_t)keys_offsets.size()},
        stream);

    auto const h_extracted_values_child = cudf::detail::make_host_vector_sync(
        cudf::device_span<char const>{values_child.view().data<char>(),
                                      (size_t)values_child.size()},
        stream);
    auto const h_extracted_values_offsets = cudf::detail::make_host_vector_sync(
        cudf::device_span<int const>{values_offsets.view().data<int>(),
                                     (size_t)values_offsets.size()},
        stream);

    auto const h_list_offsets = cudf::detail::make_host_vector_sync(
        cudf::device_span<cudf::offset_type const>{list_offsets->view().data<cudf::offset_type>(),
                                                   (size_t)list_offsets->size()},
        stream);

    CUDF_EXPECTS(h_list_offsets.back() == extracted_keys->size() &&
                     h_list_offsets.back() == extracted_values->size(),
                 "Invalid ...");

    std::stringstream ss;
    ss << "Extract keys-values:\n";

    for (size_t i = 0; i + 1 < h_list_offsets.size(); ++i) {
      ss << "List " << i << ": [" << h_list_offsets[i] << ", " << h_list_offsets[i + 1] << "]\n";
      for (cudf::size_type string_idx = h_list_offsets[i]; string_idx < h_list_offsets[i + 1];
           ++string_idx) {
        {
          auto const string_begin = h_extracted_keys_offsets[string_idx];
          auto const string_end = h_extracted_keys_offsets[string_idx + 1];
          auto const size = string_end - string_begin;
          auto const ptr = &h_extracted_keys_child[string_begin];
          ss << "\t\"" << std::string(ptr, size) << "\" : ";
        }
        {
          auto const string_begin = h_extracted_values_offsets[string_idx];
          auto const string_end = h_extracted_values_offsets[string_idx + 1];
          auto const size = string_end - string_begin;
          auto const ptr = &h_extracted_values_child[string_begin];
          ss << "\"" << std::string(ptr, size) << "\"\n";
        }
      }
    }
    //    ss << std::string(h_extracted_json.data(), h_extracted_json.size()) << "\n";
    std::cerr << ss.str() << std::endl;
  }
#endif

  std::vector<std::unique_ptr<cudf::column>> out_keys_vals;
  out_keys_vals.emplace_back(std::move(extracted_keys));
  out_keys_vals.emplace_back(std::move(extracted_values));
  auto structs_col = cudf::make_structs_column(num_keys, std::move(out_keys_vals), 0,
                                               rmm::device_buffer{}, stream, mr);

#if 0

  // Substring the input to extract out keys.

  // Substring the input to extract out values.

// Compute the offsets of the output lists column.
// Firstly, extract the key nodes.
// Compute the numbers of keys having the same parent using reduce_by_key.
// These numbers will also be sizes of the output lists.

#else
  // TODO: copy null mask.
  return cudf::make_lists_column(input.size(), std::move(list_offsets), std::move(structs_col), 0,
                                 rmm::device_buffer{}, stream, mr);
#endif
}

} // namespace spark_rapids_jni
