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
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#define DEBUG_FROM_JSON

namespace spark_rapids_jni {

using namespace cudf::io::json;

namespace {

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
           << (is_node(h_tokens[i]) ? " (node)" : "") << "\n";
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

  // Identify the key-value tokens.
#if 0

  // Substring the input to extract out keys.

  // Substring the input to extract out values.

#else
  return cudf::make_strings_column(
      1, cudf::detail::make_device_uvector_async<int>(std::vector<int>{0, 1}, stream, default_mr),
      unify_json_strings(input, stream, default_mr));
#endif
}

} // namespace spark_rapids_jni
