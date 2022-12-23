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
#include <limits>

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
#include <rmm/exec_policy.hpp>

#define PRINT_DEBUG_JSON

namespace {

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

} // namespace

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> from_json(cudf::column_view const &input,
                                        bool throw_if_keys_duplicate, rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource *mr) {
  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRING, "Invalid input format");

  auto const default_mr = rmm::mr::get_current_device_resource();

  // Firstly, concatenate all the input json strings into one giant input json string.
  // The output can be validated using https://jsonformatter.curiousconcept.com/.
  auto const processed_input = unify_json_strings(input, stream, default_mr);

#ifdef PRINT_DEBUG_JSON
  auto h_json = cudf::detail::make_host_vector_sync(
      cudf::device_span<char const>{processed_input.data(), processed_input.size()}, stream);
  std::cout << "Processed json string:\n";
  std::cout << std::string_view{h_json.data(), h_json.size()} << std::endl;
#endif

#if 0
  // Tokenize the input json strings.
  auto const d_input =
      cudf::device_span<SymbolT const>{processed_input.data(), processed_input.size()};
  auto const [tokens, token_indices] =
      cudf::io::json::detail::get_token_stream(d_input, default_options, stream);
  // Identify the key-value tokens.

  // Substring the input to extract out keys.

  // Substring the input to extract out values.

#else
  return cudf::make_strings_column(
      1, cudf::detail::make_device_uvector_async<int>(std::vector<int>{0, 1}, stream, default_mr),
      unify_json_strings(input, stream, default_mr));
#endif
}

} // namespace spark_rapids_jni
