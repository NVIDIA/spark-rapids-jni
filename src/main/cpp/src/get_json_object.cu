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

#include "get_json_object.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/optional.h>

#include <memory>

namespace spark_rapids_jni {

namespace detail {

/**
 * write JSON style
 */
enum class write_style { raw_style, quoted_style, flatten_style };

/**
 * path instruction type
 */
enum class path_instruction_type { subscript, wildcard, key, index, named };

/**
 * path instruction
 */
struct path_instruction {
  CUDF_HOST_DEVICE inline path_instruction(path_instruction_type _type) : type(_type) {}

  path_instruction_type type;

  // used when type is named type
  cudf::string_view name;

  // used when type is index
  int index{-1};
};

// TODO parse JSON path
thrust::optional<rmm::device_uvector<path_instruction>> parse_path(
  cudf::string_scalar const& json_path)
{
  return thrust::nullopt;
}

/**
 * TODO: JSON generator
 *
 */
template <int max_json_nesting_depth>
class json_generator {
 public:
  json_generator(char* output, cudf::size_type len) {}

  // Spark creates new generators in `evaluatePath`,
  // should support nested generator
  json_generator<max_json_nesting_depth> new_generator() {}

  json_generator<max_json_nesting_depth> delete_generator()
  {
    // logically delete current top generator by set pointer to previous
  }

  /**
   * get buffer for the current top generator
  */
  thrust::pair<char const*, cudf::size_type> get_buffer()
  {
    // delete_generator after get_buffer is used.
    return thrust::make_pair(nullptr, 0);
  }

  void write_start_array() {}

  void write_end_array() {}

  void copy_current_structure(json_parser<max_json_nesting_depth>& parser) {}

  /**
   * write a char
  */
  void write_char(char c) {
  }

  /**
   * Get text from JSON parser and then write the text
   * Note: Because JSON strings contains '\' for escaping,
   * JSON parser should do unescape to remove '\' and JSON parser
   * then can not return a pointer and length pair (char *, len),
   * For number token, JSON parser can return a pair (char *, len)
  */
  void write_raw(json_parser<max_json_nesting_depth>& parser) {
    // impl will like:
    // parser.write_current_text(g);
    // def parser.write_text {
    // call g.write_char(c)
    // }
  }

  void write_raw_value(char const*, cudf::size_type len) {}

  /**
   * Because nested generator, should record the max length.
  */
  cudf::size_type get_max_len() {
    return 0;
  }

 private:
  // records generator start pointers
  int start_stack[max_json_nesting_depth]{0};

  // records current generator length
  int length = 0;

  // records generator stack size
  int stack_size = 0;
};

}  // namespace detail

std::unique_ptr<cudf::column> get_json_object(cudf::strings_column_view const& col,
                                              cudf::string_scalar const& json_path,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  // TODO: main logic
  return cudf::make_empty_column(cudf::type_to_id<cudf::size_type>());
}

}  // namespace spark_rapids_jni
