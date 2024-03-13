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

#include "json_parser.hpp"

#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <memory>

namespace spark_rapids_jni {

namespace detail {
/**
 * JSON generator
 */
template <int max_json_nesting_depth = curr_max_json_nesting_depth>
class json_generator {
 public:
  CUDF_HOST_DEVICE json_generator(char* _output, size_t _output_len)
    : output(_output), output_len(_output_len)
  {
  }
  CUDF_HOST_DEVICE json_generator() : output(nullptr), output_len(0) {}

  enum class item { OBJECT, ARRAY, EMPTY };
  // create a nested child generator based on this parent generator
  // child generator is a view
  CUDF_HOST_DEVICE json_generator new_child_generator()
  {
    if (nullptr == output) {
      return json_generator();
    } else {
      return json_generator(output + output_len, 0);
    }
  }

  CUDF_HOST_DEVICE json_generator finish_child_generator(json_generator const& child_generator)
  {
    // logically delete child generator
    output_len += child_generator.get_output_len();
  }

  CUDF_HOST_DEVICE inline size_t get_output_len() const { return output_len; }

  CUDF_HOST_DEVICE inline char* get_output_start_position() const { return output; }

  CUDF_HOST_DEVICE inline char* get_current_output_position() const
  {
    return output + get_output_len();
  }

  CUDF_HOST_DEVICE void write_output(const char* str, size_t len)
  {
    if (output != nullptr) {
      std::memcpy(output + output_len, str, len);
      output_len = output_len + len;
    }
  }

  CUDF_HOST_DEVICE void write_start_array()
  {
    initialize_new_context(item::ARRAY);
    add_start_array();
  }

  CUDF_HOST_DEVICE void write_end_array()
  {
    add_end_array();
    pop_curr_context();
  }

  /**
   * Get current text from JSON parser and then write the text
   * Note: Because JSON strings contains '\' to do escape,
   * JSON parser should do unescape to remove '\' and JSON parser
   * then can not return a pointer and length pair (char *, len),
   * For number token, JSON parser can return a pair (char *, len)
   */
  CUDF_HOST_DEVICE void write_raw(json_parser<max_json_nesting_depth>& parser)
  {
    if (output != nullptr) {
      auto copied = parser.write_unescaped_text(get_current_output_position());
      output_len += copied;
    }
  }

  CUDF_HOST_DEVICE void write_raw_value(json_parser<max_json_nesting_depth>& parser)
  {
    // check if current is a list and add comma if not first member
    if (!is_context_stack_empty() && is_array_context() && !is_first_member()) { add_comma(); }
    // increment count
    register_member();
    // write to output
    write_raw(parser);
  }

  CUDF_HOST_DEVICE void write_escaped_text(json_parser<max_json_nesting_depth>& parser)
  {
    if (output != nullptr) {
      auto copied = parser.write_escaped_text(get_current_output_position());
      output_len += copied;
    }
  }

  CUDF_HOST_DEVICE void write_integer(json_parser<max_json_nesting_depth>& parser)
  {
    auto [start_position, length] = parser.get_current_number_text();
    write_output(start_position, length);
  }

  CUDF_HOST_DEVICE void copy_current_structure(json_parser<max_json_nesting_depth>& parser)
  {
    json_token token = json_token::INIT;
    do {
      token = parser.next_token();
      // std::cout << debug_enumToString(token) << std::endl;  // DEBUG
      consume_token(token, parser);
      // std::cout << debug_getOutputString() << std::endl;    // DEBUG
    } while (token != json_token::ERROR && token != json_token::SUCCESS);

    CUDF_EXPECTS(current == -1, "ERROR:copy_current_structure:INVALID JSON STRUCTURE");
  }

  CUDF_HOST_DEVICE void consume_token(json_token next_token,
                                      json_parser<max_json_nesting_depth>& parser)
  {
    if (next_token == json_token::INIT || next_token == json_token::SUCCESS) { return; }
    if (next_token != json_token::FIELD_NAME && next_token != json_token::END_OBJECT &&
        next_token != json_token::END_ARRAY) {
      consume_value_token(next_token, parser);
      return;
    }
    // if object context
    if (!is_context_stack_empty()) {
      // OBJECT Context
      if (is_object_context()) {
        if (next_token == json_token::FIELD_NAME) {
          if (!is_first_member()) { add_comma(); }
          // write key
          write_escaped_text(parser);
          // add :
          write_output(":", 1);
        } else if (next_token == json_token::END_OBJECT) {
          // add }
          add_end_object();
          pop_curr_context();
        }
      }
      // ARRAY Context
      else {
        if (next_token == json_token::END_ARRAY) {
          // add ]
          add_end_array();
          pop_curr_context();
        }
      }
    } else {
      CUDF_FAIL("ERROR:consume_token:INVALID JSON STRUCTURE");
    }
    return;
  }

  CUDF_HOST_DEVICE void consume_value_token(json_token next_token,
                                            json_parser<max_json_nesting_depth>& parser)
  {
    // check if current is a list and add comma if not first member
    if (!is_context_stack_empty() && is_array_context() && !is_first_member()) { add_comma(); }
    // make true
    register_member();

    switch (next_token) {
      case json_token::START_OBJECT:
        initialize_new_context(item::OBJECT);
        add_start_object();
        break;
      case json_token::START_ARRAY:
        initialize_new_context(item::ARRAY);
        add_start_array();
        break;
      case json_token::VALUE_TRUE: add_true(); break;
      case json_token::VALUE_FALSE: add_false(); break;
      case json_token::VALUE_NULL: add_null(); break;
      case json_token::VALUE_NUMBER_INT: write_integer(parser); break;
      case json_token::VALUE_NUMBER_FLOAT: write_integer(parser); break;
      case json_token::VALUE_STRING: write_escaped_text(parser); break;
      default: CUDF_FAIL("ERROR:consume_value_token:INVALID JSON STRUCTURE"); break;
    }
  }

  std::string debug_enumToString(json_token token)
  {
    switch (token) {
      case json_token::INIT: return "INIT";
      case json_token::SUCCESS: return "SUCCESS";
      case json_token::ERROR: return "ERROR";
      case json_token::START_OBJECT: return "START_OBJECT";
      case json_token::END_OBJECT: return "END_OBJECT";
      case json_token::START_ARRAY: return "START_ARRAY";
      case json_token::END_ARRAY: return "END_ARRAY";
      case json_token::FIELD_NAME: return "FIELD_NAME";
      case json_token::VALUE_STRING: return "VALUE_STRING";
      case json_token::VALUE_NUMBER_INT: return "VALUE_NUMBER_INT";
      case json_token::VALUE_NUMBER_FLOAT: return "VALUE_NUMBER_FLOAT";
      case json_token::VALUE_TRUE: return "VALUE_TRUE";
      case json_token::VALUE_FALSE: return "VALUE_FALSE";
      case json_token::VALUE_NULL: return "VALUE_NULL";
    }
    return "UNKNOWN_TOKEN";
  }

  std::string debug_getOutputString()
  {
    if (get_output_start_position() != nullptr) {
      return std::string(get_output_start_position(), get_output_len());
    } else {
      return "";
    }
  }

 private:
  bool has_members[max_json_nesting_depth] = {false};
  item type[max_json_nesting_depth]        = {item::EMPTY};
  int current                              = -1;

  char* const output;
  size_t output_len;

  CUDF_HOST_DEVICE inline bool is_context_stack_empty() { return current == -1; }

  CUDF_HOST_DEVICE inline bool is_object_context() { return type[current] == item::OBJECT; }

  CUDF_HOST_DEVICE inline bool is_array_context() { return type[current] == item::ARRAY; }

  CUDF_HOST_DEVICE inline void pop_curr_context()
  {
    // Restore default values
    has_members[current] = false;
    type[current]        = item::EMPTY;
    current--;
  }

  CUDF_HOST_DEVICE inline bool is_first_member() { return has_members[current] == false; }

  CUDF_HOST_DEVICE inline void register_member() { has_members[current] = true; }

  CUDF_HOST_DEVICE inline void initialize_new_context(item _item)
  {
    current++;
    type[current]        = _item;
    has_members[current] = false;
  }

  CUDF_HOST_DEVICE void add_start_array() { write_output("[", 1); }

  CUDF_HOST_DEVICE void add_end_array() { write_output("]", 1); }

  CUDF_HOST_DEVICE void add_start_object() { write_output("{", 1); }

  CUDF_HOST_DEVICE void add_end_object() { write_output("}", 1); }

  CUDF_HOST_DEVICE void add_true() { write_output("true", 4); }

  CUDF_HOST_DEVICE void add_false() { write_output("false", 4); }

  CUDF_HOST_DEVICE void add_null() { write_output("null", 4); }

  CUDF_HOST_DEVICE void add_comma() { write_output(",", 1); }
};
}  // namespace detail

/**
 * Extracts json object from a json string based on json path specified, and
 * returns json string of the extracted json object. It will return null if the
 * input json string is invalid.
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  cudf::string_scalar const& json_path,
  spark_rapids_jni::json_parser_options options,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
