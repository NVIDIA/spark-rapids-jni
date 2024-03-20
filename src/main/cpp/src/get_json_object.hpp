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

#include <thrust/optional.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>

#include <memory>
#include <string>
#include <vector>

namespace spark_rapids_jni {

/**
 * path instruction type
 */
enum class path_instruction_type { subscript, wildcard, key, index, named };

namespace detail {

/**
 * write JSON style
 */
enum class write_style { raw_style, quoted_style, flatten_style };

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

/**
 * JSON generator is used to write out JSON string.
 * It's not a full featured JSON generator, because get json object
 * outputs an array or single item. JSON object is wroten as a whole item.
 */
template <int max_json_nesting_depth = curr_max_json_nesting_depth>
class json_generator {
 public:
  CUDF_HOST_DEVICE json_generator(char* _output)
    : output(_output), output_len(0), hide_outer_array_tokens(false)
  {
  }
  CUDF_HOST_DEVICE json_generator() : output(nullptr), output_len(0), hide_outer_array_tokens(false)
  {
  }
  CUDF_HOST_DEVICE json_generator(char* _output, bool _hide_outer_array_tokens)
    : output(_output), output_len(0), hide_outer_array_tokens(_hide_outer_array_tokens)
  {
  }
  CUDF_HOST_DEVICE json_generator(bool _hide_outer_array_tokens)
    : output(nullptr), output_len(0), hide_outer_array_tokens(_hide_outer_array_tokens)
  {
  }

  // create a nested child generator based on this parent generator
  // child generator is a view
  CUDF_HOST_DEVICE json_generator new_child_generator(bool hide_outer_array_tokens)
  {
    if (nullptr == output) {
      return json_generator(hide_outer_array_tokens);
    } else {
      return json_generator(output + output_len, hide_outer_array_tokens);
    }
  }

  CUDF_HOST_DEVICE void write_start_array()
  {
    if (!hide_outer_array_tokens) {
      if (output) { *(output + output_len) = '['; }
      output_len++;
      is_first_item[array_depth] = true;
      array_depth++;
    } else {
      // hide the outer start array token
      // Note: do not inc output_len
      is_first_item[array_depth] = true;
      array_depth++;
    }
  }

  CUDF_HOST_DEVICE void write_end_array()
  {
    if (!hide_outer_array_tokens) {
      if (output) { *(output + output_len) = ']'; }
      output_len++;
      array_depth--;
    } else {
      // hide the outer end array token
      array_depth--;
    }
  }

  // return true if it's in a array context and it's not writing the first item.
  CUDF_HOST_DEVICE bool need_comma()
  {
    return (array_depth > 0 && !is_first_item[array_depth - 1]);
  }

  /**
   * write comma accroding to current generator state
   */
  CUDF_HOST_DEVICE void try_write_comma()
  {
    if (need_comma()) {
      // in array context and writes first item
      if (output) { *(output + output_len) = ','; }
      output_len++;
    }
  }

  /**
   * copy current structure when parsing. If current token is start
   * object/array, then copy to corresponding matched end object/array. return
   * false if JSON format is invalid return true if JSON format is valid
   */
  CUDF_HOST_DEVICE bool copy_current_structure(json_parser<>& parser)
  {
    // first try add comma
    try_write_comma();

    if (array_depth > 0) { is_first_item[array_depth - 1] = false; }

    if (nullptr != output) {
      auto copy_to       = output + output_len;
      auto [b, copy_len] = parser.copy_current_structure(copy_to);
      output_len += copy_len;
      return b;
    } else {
      char* copy_to      = nullptr;
      auto [b, copy_len] = parser.copy_current_structure(copy_to);
      output_len += copy_len;
      return b;
    }
  }

  /**
   * Get current text from JSON parser and then write the text
   * Note: Because JSON strings contains '\' to do escape,
   * JSON parser should do unescape to remove '\' and JSON parser
   * then can not return a pointer and length pair (char *, len),
   * For number token, JSON parser can return a pair (char *, len)
   */
  CUDF_HOST_DEVICE void write_raw(json_parser<>& parser)
  {
    if (array_depth > 0) { is_first_item[array_depth - 1] = false; }

    if (nullptr != output) {
      auto copied = parser.write_unescaped_text(output + output_len);
      output_len += copied;
    } else {
      auto len = parser.compute_unescaped_len();
      output_len += len;
    }
  }

  /**
   * write child raw value
   * e.g.:
   *
   * write_array_tokens = false
   * need_comma = true
   * [1,2,3]1,2,3
   *        ^
   *        |
   *    child pointer
   * ==>>
   * [1,2,3],1,2,3
   *
   *
   * write_array_tokens = true
   * need_comma = true
   *   [12,3,4
   *     ^
   *     |
   * child pointer
   * ==>>
   *   [1,[2,3,4]
   *
   * @param child_block_begin
   * @param child_block_len
   */
  CUDF_HOST_DEVICE void write_child_raw_value(char* child_block_begin,
                                              size_t child_block_len,
                                              bool write_outer_array_tokens)
  {
    bool insert_comma = need_comma();

    is_first_item[array_depth - 1] = false;

    if (nullptr != output) {
      if (write_outer_array_tokens) {
        if (insert_comma) {
          *(child_block_begin + child_block_len + 2) = ']';
          move_forward(child_block_begin, child_block_len, 2);
          *(child_block_begin + 1) = '[';
          *(child_block_begin)     = ',';
        } else {
          *(child_block_begin + child_block_len + 1) = ']';
          move_forward(child_block_begin, child_block_len, 1);
          *(child_block_begin) = '[';
        }
      } else {
        if (insert_comma) {
          move_forward(child_block_begin, child_block_len, 1);
          *(child_block_begin) = ',';
        } else {
          // do not need comma && do not need write outer array tokens
          // do nothing, because child generator buff is directly after the
          // parent generator
        }
      }
    }

    // update length
    if (insert_comma) { output_len++; }
    if (write_outer_array_tokens) { output_len += 2; }
    output_len += child_block_len;
  }

  CUDF_HOST_DEVICE void move_forward(char* begin, size_t len, int forward)
  {
    char* pos = begin + len + forward - 1;
    char* e   = begin + forward - 1;
    // should add outer array tokens
    // First move chars [2, end) a byte forward
    while (pos > e) {
      *pos = *(pos - 1);
      pos--;
    }
  }

  CUDF_HOST_DEVICE void reset() { output_len = 0; }

  CUDF_HOST_DEVICE inline size_t get_output_len() const { return output_len; }
  CUDF_HOST_DEVICE inline char* get_output_start_position() const { return output; }
  CUDF_HOST_DEVICE inline char* get_current_output_position() const { return output + output_len; }

  /**
   * generator may contain trash output, e.g.: generator writes some output,
   * then JSON format is invalid, the previous output becomes trash.
   */
  CUDF_HOST_DEVICE inline void set_output_len_zero() { output_len = 0; }

 private:
  char* output;
  size_t output_len;
  bool hide_outer_array_tokens;

  bool is_first_item[max_json_nesting_depth];
  int array_depth = 0;
};

/**
 * path evaluator which can run on both CPU and GPU
 */
struct path_evaluator {
  static CUDF_HOST_DEVICE inline bool path_is_empty(size_t path_size) { return path_size == 0; }

  static CUDF_HOST_DEVICE inline bool path_match_element(path_instruction const* path_ptr,
                                                         size_t path_size,
                                                         path_instruction_type path_type0)
  {
    if (path_size < 1) { return false; }
    return path_ptr[0].type == path_type0;
  }

  static CUDF_HOST_DEVICE inline bool path_match_elements(path_instruction const* path_ptr,
                                                          size_t path_size,
                                                          path_instruction_type path_type0,
                                                          path_instruction_type path_type1)
  {
    if (path_size < 2) { return false; }
    return path_ptr[0].type == path_type0 && path_ptr[1].type == path_type1;
  }

  static CUDF_HOST_DEVICE inline bool path_match_elements(path_instruction const* path_ptr,
                                                          size_t path_size,
                                                          path_instruction_type path_type0,
                                                          path_instruction_type path_type1,
                                                          path_instruction_type path_type2,
                                                          path_instruction_type path_type3)
  {
    if (path_size < 4) { return false; }
    return path_ptr[0].type == path_type0 && path_ptr[1].type == path_type1 &&
           path_ptr[2].type == path_type2 && path_ptr[3].type == path_type3;
  }

  static CUDF_HOST_DEVICE inline thrust::tuple<bool, int> path_match_subscript_index(
    path_instruction const* path_ptr, size_t path_size)
  {
    auto match = path_match_elements(
      path_ptr, path_size, path_instruction_type::subscript, path_instruction_type::index);
    if (match) {
      return thrust::make_tuple(true, path_ptr[1].index);
    } else {
      return thrust::make_tuple(false, 0);
    }
  }

  static CUDF_HOST_DEVICE inline thrust::tuple<bool, cudf::string_view> path_match_named(
    path_instruction const* path_ptr, size_t path_size)
  {
    auto match = path_match_element(path_ptr, path_size, path_instruction_type::named);
    if (match) {
      return thrust::make_tuple(true, path_ptr[0].name);
    } else {
      return thrust::make_tuple(false, cudf::string_view());
    }
  }

  static CUDF_HOST_DEVICE inline thrust::tuple<bool, int>
  path_match_subscript_index_subscript_wildcard(path_instruction const* path_ptr, size_t path_size)
  {
    auto match = path_match_elements(path_ptr,
                                     path_size,
                                     path_instruction_type::subscript,
                                     path_instruction_type::index,
                                     path_instruction_type::subscript,
                                     path_instruction_type::wildcard);
    if (match) {
      return thrust::make_tuple(true, path_ptr[1].index);
    } else {
      return thrust::make_tuple(false, 0);
    }
  }

  static CUDF_HOST_DEVICE bool evaluate_path(json_parser<>& p,
                                             json_generator<>& g,
                                             write_style style,
                                             path_instruction const* path_ptr,
                                             int path_size)
  {
    auto token = p.get_current_token();

    // case (VALUE_STRING, Nil) if style == RawStyle
    if (json_token::VALUE_STRING == token && path_is_empty(path_size) &&
        style == write_style::raw_style) {
      // there is no array wildcard or slice parent, emit this string without
      // quotes write current string in parser to generator
      g.write_raw(p);
      return true;
    }
    // case (START_ARRAY, Nil) if style == FlattenStyle
    else if (json_token::START_ARRAY == token && path_is_empty(path_size) &&
             style == write_style::flatten_style) {
      // flatten this array into the parent
      bool dirty = false;
      while (json_token::END_ARRAY != p.next_token()) {
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return false; }

        dirty |= path_evaluator::evaluate_path(p, g, style, nullptr, 0);
      }
      return dirty;
    }
    // case (_, Nil)
    else if (path_is_empty(path_size)) {
      // general case: just copy the child tree verbatim
      return g.copy_current_structure(p);
    }
    // case (START_OBJECT, Key :: xs)
    else if (json_token::START_OBJECT == token &&
             path_match_element(path_ptr, path_size, path_instruction_type::key)) {
      bool dirty = false;
      while (json_token::END_OBJECT != p.next_token()) {
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return false; }

        if (dirty) {
          // once a match has been found we can skip other fields
          if (!p.try_skip_children()) {
            // JSON validation check
            return false;
          }
        } else {
          dirty = path_evaluator::evaluate_path(p, g, style, path_ptr + 1, path_size - 1);
        }
      }
      return dirty;
    }
    // case (START_ARRAY, Subscript :: Wildcard :: Subscript :: Wildcard :: xs)
    else if (json_token::START_ARRAY == token &&
             path_match_elements(path_ptr,
                                 path_size,
                                 path_instruction_type::subscript,
                                 path_instruction_type::wildcard,
                                 path_instruction_type::subscript,
                                 path_instruction_type::wildcard)) {
      // special handling for the non-structure preserving double wildcard
      // behavior in Hive
      bool dirty = false;
      g.write_start_array();
      while (p.next_token() != json_token::END_ARRAY) {
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return false; }

        dirty |= path_evaluator::evaluate_path(
          p, g, write_style::flatten_style, path_ptr + 4, path_size - 4);
      }
      g.write_end_array();
      return dirty;
    }
    // case (START_ARRAY, Subscript :: Wildcard :: xs) if style != QuotedStyle
    else if (json_token::START_ARRAY == token &&
             path_match_elements(path_ptr,
                                 path_size,
                                 path_instruction_type::subscript,
                                 path_instruction_type::wildcard) &&
             style != write_style::quoted_style) {
      // retain Flatten, otherwise use Quoted... cannot use Raw within an array
      write_style next_style = write_style::raw_style;
      switch (style) {
        case write_style::raw_style: next_style = write_style::quoted_style; break;
        case write_style::flatten_style: next_style = write_style::flatten_style; break;
        case write_style::quoted_style: next_style = write_style::quoted_style;  // never happen
      }

      // temporarily buffer child matches, the emitted json will need to be
      // modified slightly if there is only a single element written

      int dirty = 0;
      // create a child generator with hide outer array tokens mode.
      auto child_g = g.new_child_generator(/*hide_outer_array_tokens*/ true);

      // Note: child generator does not actually write the outer start array
      // token into buffer it only updates internal nested state
      child_g.write_start_array();

      while (p.next_token() != json_token::END_ARRAY) {
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return false; }

        // track the number of array elements and only emit an outer array if
        // we've written more than one element, this matches Hive's behavior
        dirty +=
          (path_evaluator::evaluate_path(p, child_g, next_style, path_ptr + 2, path_size - 2) ? 1
                                                                                              : 0);
      }

      // Note: child generator does not actually write the outer end array token
      // into buffer it only updates internal nested state
      child_g.write_end_array();

      char* child_g_start = child_g.get_output_start_position();
      size_t child_g_len  = child_g.get_output_len();  // len already excluded outer [ ]

      if (dirty > 1) {
        // add outer array tokens
        g.write_child_raw_value(child_g_start, child_g_len, true);
      } else if (dirty == 1) {
        // remove outer array tokens
        g.write_child_raw_value(child_g_start, child_g_len, false);
      }  // else do not write anything

      return dirty > 0;
    }
    // case (START_ARRAY, Subscript :: Wildcard :: xs)
    else if (json_token::START_ARRAY == token &&
             path_match_elements(path_ptr,
                                 path_size,
                                 path_instruction_type::subscript,
                                 path_instruction_type::wildcard)) {
      bool dirty = false;
      g.write_start_array();
      while (p.next_token() != json_token::END_ARRAY) {
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return false; }

        // wildcards can have multiple matches, continually update the dirty
        // count
        dirty |= path_evaluator::evaluate_path(
          p, g, write_style::quoted_style, path_ptr + 2, path_size - 2);
      }
      g.write_end_array();

      return dirty;
    }
    // case (START_ARRAY, Subscript :: Index(idx) :: (xs@Subscript :: Wildcard
    // ::
    // _))
    else if (json_token::START_ARRAY == token &&
             thrust::get<0>(path_match_subscript_index_subscript_wildcard(path_ptr, path_size))) {
      int idx = thrust::get<1>(path_match_subscript_index_subscript_wildcard(path_ptr, path_size));
      p.next_token();
      // JSON validation check
      if (json_token::ERROR == p.get_current_token()) { return false; }

      int i = idx;
      while (i >= 0) {
        if (p.get_current_token() == json_token::END_ARRAY) {
          // terminate, nothing has been written
          return false;
        }
        if (0 == i) {
          bool dirty = path_evaluator::evaluate_path(
            p, g, write_style::quoted_style, path_ptr + 2, path_size - 2);
          while (p.next_token() != json_token::END_ARRAY) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }

            // advance the token stream to the end of the array
            if (!p.try_skip_children()) { return false; }
          }
          return dirty;
        } else {
          // i > 0
          if (!p.try_skip_children()) { return false; }

          p.next_token();
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return false; }
        }
        --i;
      }
      // path parser guarantees idx >= 0
      // will never reach to here
      return false;
    }
    // case (START_ARRAY, Subscript :: Index(idx) :: xs)
    else if (json_token::START_ARRAY == token &&
             thrust::get<0>(path_match_subscript_index(path_ptr, path_size))) {
      int idx = thrust::get<1>(path_match_subscript_index(path_ptr, path_size));
      p.next_token();
      // JSON validation check
      if (json_token::ERROR == p.get_current_token()) { return false; }

      int i = idx;
      while (i >= 0) {
        if (p.get_current_token() == json_token::END_ARRAY) {
          // terminate, nothing has been written
          return false;
        }
        if (0 == i) {
          bool dirty = path_evaluator::evaluate_path(p, g, style, path_ptr + 2, path_size - 2);
          while (p.next_token() != json_token::END_ARRAY) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }

            // advance the token stream to the end of the array
            if (!p.try_skip_children()) { return false; }
          }
          return dirty;
        } else {
          // i > 0
          if (!p.try_skip_children()) { return false; }

          p.next_token();
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return false; }
        }
        --i;
      }
      // path parser guarantees idx >= 0
      // will never reach to here
      return false;
    }
    // case (FIELD_NAME, Named(name) :: xs) if p.getCurrentName == name
    else if (json_token::FIELD_NAME == token &&
             thrust::get<0>(path_match_named(path_ptr, path_size)) &&
             p.match_current_field_name(thrust::get<1>(path_match_named(path_ptr, path_size)))) {
      if (p.next_token() != json_token::VALUE_NULL) {
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return false; }

        return path_evaluator::evaluate_path(p, g, style, path_ptr + 1, path_size - 1);
      } else {
        return false;
      }
    }
    // case (FIELD_NAME, Wildcard :: xs)
    else if (json_token::FIELD_NAME == token &&
             path_match_element(path_ptr, path_size, path_instruction_type::wildcard)) {
      p.next_token();
      // JSON validation check
      if (json_token::ERROR == p.get_current_token()) { return false; }

      return path_evaluator::evaluate_path(p, g, style, path_ptr + 1, path_size - 1);
      // case _ =>
    } else {
      if (!p.try_skip_children()) { return false; }
      return false;
    }
  }
};

}  // namespace detail

/**
 * Extracts json object from a json string based on json path specified, and
 * returns json string of the extracted json object. It will return null if the
 * input json string is invalid.
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  std::vector<std::tuple<path_instruction_type, std::string, int64_t>> const& instructions,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
