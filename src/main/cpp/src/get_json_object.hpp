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
  CUDF_HOST_DEVICE CUDF_HOST_DEVICE json_generator<>& operator=(const json_generator<>& other)
  {
    this->output = other.output;
    this->output_len = other.output_len;
    this->array_depth = other.array_depth;
    this->hide_outer_array_tokens = other.hide_outer_array_tokens;
    for (size_t i = 0; i < max_json_nesting_depth; i++)
    {
      this->is_first_item[i] = other.is_first_item[i];
    }

    return *this;
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

  CUDF_HOST_DEVICE inline void set_output_len(size_t len) { output_len = len; }

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


  /**
   *
   * The following commented function is recursive version,
   * The next function below is the rewritten version,
   * Keep version here is for review purpuse, because rewritten version(iterative)
   * is not human friendly.
   * 
  */
  // static CUDF_HOST_DEVICE bool evaluate_path(json_parser<>& p,
  //                                            json_generator<>& g,
  //                                            write_style style,
  //                                            path_instruction const* path_ptr,
  //                                            int path_size)
  // {
  //   auto token = p.get_current_token();

  //   // case (VALUE_STRING, Nil) if style == RawStyle
  //   // case path 1
  //   if (json_token::VALUE_STRING == token && path_is_empty(path_size) &&
  //       style == write_style::raw_style) {
  //     // there is no array wildcard or slice parent, emit this string without
  //     // quotes write current string in parser to generator
  //     g.write_raw(p);
  //     return true;
  //   }
  //   // case (START_ARRAY, Nil) if style == FlattenStyle
  //   // case path 2
  //   else if (json_token::START_ARRAY == token && path_is_empty(path_size) &&
  //            style == write_style::flatten_style) {
  //     // flatten this array into the parent
  //     bool dirty = false;
  //     while (json_token::END_ARRAY != p.next_token()) {
  //       // JSON validation check
  //       if (json_token::ERROR == p.get_current_token()) { return false; }

  //       dirty |= path_evaluator::evaluate_path(p, g, style, nullptr, 0);
  //     }
  //     return dirty;
  //   }
  //   // case (_, Nil)
  //   // case path 3
  //   else if (path_is_empty(path_size)) {
  //     // general case: just copy the child tree verbatim
  //     return g.copy_current_structure(p);
  //   }
  //   // case (START_OBJECT, Key :: xs)
  //   // case path 4
  //   else if (json_token::START_OBJECT == token &&
  //            path_match_element(path_ptr, path_size, path_instruction_type::key)) {
  //     bool dirty = false;
  //     while (json_token::END_OBJECT != p.next_token()) {
  //       // JSON validation check
  //       if (json_token::ERROR == p.get_current_token()) { return false; }

  //       if (dirty) {
  //         // once a match has been found we can skip other fields
  //         if (!p.try_skip_children()) {
  //           // JSON validation check
  //           return false;
  //         }
  //       } else {
  //         dirty = path_evaluator::evaluate_path(p, g, style, path_ptr + 1, path_size - 1);
  //       }
  //     }
  //     return dirty;
  //   }
  //   // case (START_ARRAY, Subscript :: Wildcard :: Subscript :: Wildcard :: xs)
  //   // case path 5
  //   else if (json_token::START_ARRAY == token &&
  //            path_match_elements(path_ptr,
  //                                path_size,
  //                                path_instruction_type::subscript,
  //                                path_instruction_type::wildcard,
  //                                path_instruction_type::subscript,
  //                                path_instruction_type::wildcard)) {
  //     // special handling for the non-structure preserving double wildcard
  //     // behavior in Hive
  //     bool dirty = false;
  //     g.write_start_array();
  //     while (p.next_token() != json_token::END_ARRAY) {
  //       // JSON validation check
  //       if (json_token::ERROR == p.get_current_token()) { return false; }

  //       dirty |= path_evaluator::evaluate_path(
  //         p, g, write_style::flatten_style, path_ptr + 4, path_size - 4);
  //     }
  //     g.write_end_array();
  //     return dirty;
  //   }
  //   // case (START_ARRAY, Subscript :: Wildcard :: xs) if style != QuotedStyle
  //   // case path 6
  //   else if (json_token::START_ARRAY == token &&
  //            path_match_elements(path_ptr,
  //                                path_size,
  //                                path_instruction_type::subscript,
  //                                path_instruction_type::wildcard) &&
  //            style != write_style::quoted_style) {
  //     // retain Flatten, otherwise use Quoted... cannot use Raw within an array
  //     write_style next_style = write_style::raw_style;
  //     switch (style) {
  //       case write_style::raw_style: next_style = write_style::quoted_style; break;
  //       case write_style::flatten_style: next_style = write_style::flatten_style; break;
  //       case write_style::quoted_style: next_style = write_style::quoted_style;  // never happen
  //     }

  //     // temporarily buffer child matches, the emitted json will need to be
  //     // modified slightly if there is only a single element written

  //     int dirty = 0;
  //     // create a child generator with hide outer array tokens mode.
  //     auto child_g = g.new_child_generator(/*hide_outer_array_tokens*/ true);

  //     // Note: child generator does not actually write the outer start array
  //     // token into buffer it only updates internal nested state
  //     child_g.write_start_array();

  //     while (p.next_token() != json_token::END_ARRAY) {
  //       // JSON validation check
  //       if (json_token::ERROR == p.get_current_token()) { return false; }

  //       // track the number of array elements and only emit an outer array if
  //       // we've written more than one element, this matches Hive's behavior
  //       dirty +=
  //         (path_evaluator::evaluate_path(p, child_g, next_style, path_ptr + 2, path_size - 2) ? 1
  //                                                                                             : 0);
  //     }

  //     // Note: child generator does not actually write the outer end array token
  //     // into buffer it only updates internal nested state
  //     child_g.write_end_array();

  //     char* child_g_start = child_g.get_output_start_position();
  //     size_t child_g_len  = child_g.get_output_len();  // len already excluded outer [ ]

  //     if (dirty > 1) {
  //       // add outer array tokens
  //       g.write_child_raw_value(child_g_start, child_g_len, true);
  //     } else if (dirty == 1) {
  //       // remove outer array tokens
  //       g.write_child_raw_value(child_g_start, child_g_len, false);
  //     }  // else do not write anything

  //     return dirty > 0;
  //   }
  //   // case (START_ARRAY, Subscript :: Wildcard :: xs)
  //   // case path 7
  //   else if (json_token::START_ARRAY == token &&
  //            path_match_elements(path_ptr,
  //                                path_size,
  //                                path_instruction_type::subscript,
  //                                path_instruction_type::wildcard)) {
  //     bool dirty = false;
  //     g.write_start_array();
  //     while (p.next_token() != json_token::END_ARRAY) {
  //       // JSON validation check
  //       if (json_token::ERROR == p.get_current_token()) { return false; }

  //       // wildcards can have multiple matches, continually update the dirty
  //       // count
  //       dirty |= path_evaluator::evaluate_path(
  //         p, g, write_style::quoted_style, path_ptr + 2, path_size - 2);
  //     }
  //     g.write_end_array();

  //     return dirty;
  //   }
  //   /* case (START_ARRAY, Subscript :: Index(idx) :: (xs@Subscript :: Wildcard :: _)) */
  //   // case path 8
  //   else if (json_token::START_ARRAY == token &&
  //            thrust::get<0>(path_match_subscript_index_subscript_wildcard(path_ptr, path_size))) {
  //     int idx = thrust::get<1>(path_match_subscript_index_subscript_wildcard(path_ptr, path_size));
  //     p.next_token();
  //     // JSON validation check
  //     if (json_token::ERROR == p.get_current_token()) { return false; }

  //     int i = idx;
  //     while (i >= 0) {
  //       if (p.get_current_token() == json_token::END_ARRAY) {
  //         // terminate, nothing has been written
  //         return false;
  //       }
  //       if (0 == i) {
  //         bool dirty = path_evaluator::evaluate_path(
  //           p, g, write_style::quoted_style, path_ptr + 2, path_size - 2);
  //         while (p.next_token() != json_token::END_ARRAY) {
  //           // JSON validation check
  //           if (json_token::ERROR == p.get_current_token()) { return false; }

  //           // advance the token stream to the end of the array
  //           if (!p.try_skip_children()) { return false; }
  //         }
  //         return dirty;
  //       } else {
  //         // i > 0
  //         if (!p.try_skip_children()) { return false; }

  //         p.next_token();
  //         // JSON validation check
  //         if (json_token::ERROR == p.get_current_token()) { return false; }
  //       }
  //       --i;
  //     }
  //     // path parser guarantees idx >= 0
  //     // will never reach to here
  //     return false;
  //   }
  //   // case (START_ARRAY, Subscript :: Index(idx) :: xs)
  //   // case path 9
  //   else if (json_token::START_ARRAY == token &&
  //            thrust::get<0>(path_match_subscript_index(path_ptr, path_size))) {
  //     int idx = thrust::get<1>(path_match_subscript_index(path_ptr, path_size));
  //     p.next_token();
  //     // JSON validation check
  //     if (json_token::ERROR == p.get_current_token()) { return false; }

  //     int i = idx;
  //     while (i >= 0) {
  //       if (p.get_current_token() == json_token::END_ARRAY) {
  //         // terminate, nothing has been written
  //         return false;
  //       }
  //       if (0 == i) {
  //         bool dirty = path_evaluator::evaluate_path(p, g, style, path_ptr + 2, path_size - 2);
  //         while (p.next_token() != json_token::END_ARRAY) {
  //           // JSON validation check
  //           if (json_token::ERROR == p.get_current_token()) { return false; }

  //           // advance the token stream to the end of the array
  //           if (!p.try_skip_children()) { return false; }
  //         }
  //         return dirty;
  //       } else {
  //         // i > 0
  //         if (!p.try_skip_children()) { return false; }

  //         p.next_token();
  //         // JSON validation check
  //         if (json_token::ERROR == p.get_current_token()) { return false; }
  //       }
  //       --i;
  //     }
  //     // path parser guarantees idx >= 0
  //     // will never reach to here
  //     return false;
  //   }
  //   // case (FIELD_NAME, Named(name) :: xs) if p.getCurrentName == name
  //   // case path 10
  //   else if (json_token::FIELD_NAME == token &&
  //            thrust::get<0>(path_match_named(path_ptr, path_size)) &&
  //            p.match_current_field_name(thrust::get<1>(path_match_named(path_ptr, path_size)))) {
  //     if (p.next_token() != json_token::VALUE_NULL) {
  //       // JSON validation check
  //       if (json_token::ERROR == p.get_current_token()) { return false; }

  //       return path_evaluator::evaluate_path(p, g, style, path_ptr + 1, path_size - 1);
  //     } else {
  //       return false;
  //     }
  //   }
  //   // case (FIELD_NAME, Wildcard :: xs)
  //   // case path 11
  //   else if (json_token::FIELD_NAME == token &&
  //            path_match_element(path_ptr, path_size, path_instruction_type::wildcard)) {
  //     p.next_token();
  //     // JSON validation check
  //     if (json_token::ERROR == p.get_current_token()) { return false; }

  //     return path_evaluator::evaluate_path(p, g, style, path_ptr + 1, path_size - 1);
  //   }
  //   // case _ =>
  //   // case path 12
  //   else {
  //     if (!p.try_skip_children()) { return false; }
  //     return false;
  //   }
  // }

  /**
   * 
   * This function is rewritten from above commented recursive function.
   * this function is equivalent to the above commented recursive function.
  */
  static CUDF_HOST_DEVICE bool evaluate_path(json_parser<>& p,
                                             json_generator<>& root_g,
                                             write_style root_style,
                                             path_instruction const* root_path_ptr,
                                             int root_path_size)
  {

    // manually maintained context stack in lieu of calling evaluate_path recursively.
    struct context {
      // current token
      json_token token;

      // which case path that this task is from
      int case_path;
      
      // used to save current generator
      json_generator<> g;

      write_style style;
      path_instruction const* path_ptr;
      int path_size;

      // is this context task is done
      bool task_is_done = false;

      // whether written output
      // if dirty > 0, indicates success
      int dirty         = 0;

      // for some case paths
      bool is_first_enter = true;

      // used to save child JSON generator for case path 8
      json_generator<> child_g;

      CUDF_HOST_DEVICE context()
        : token(json_token::INIT),
          case_path(-1),
          g(json_generator<>()),
          style(write_style::raw_style),
          path_ptr(nullptr),
          path_size(0)
      {
      }

      CUDF_HOST_DEVICE context(json_token _token,
                               int _case_path,
                               json_generator<> _g,
                               write_style _style,
                               path_instruction const* _path_ptr,
                               int _path_size)
        : token(_token),
          case_path(_case_path),
          g(_g),
          style(_style),
          path_ptr(_path_ptr),
          path_size(_path_size)
      {
      }

      CUDF_HOST_DEVICE context& operator=(const context& other)
      {
        token = other.token;
        case_path     = other.case_path;
        g     = other.g;
        style     = other.style;
        path_ptr  = other.path_ptr;
        path_size = other.path_size;
        task_is_done = other.task_is_done;
        dirty = other.dirty;
        is_first_enter = other.is_first_enter;
        child_g = other.child_g;

        return *this;
      }
    };

    // path max depth limitation
    constexpr int max_path_depth = 32;

    // stack
    context stack[max_path_depth];
    int stack_pos     = 0;

    // push context function
    auto push_context = [&stack, &stack_pos](json_token _token,
                                             int _case_path,
                                             json_generator<> _g,
                                             write_style _style,
                                             path_instruction const* _path_ptr,
                                             int _path_size) {
      if (stack_pos == max_path_depth - 1) { return false; }
      stack[stack_pos++] = context(_token, _case_path, _g, _style, _path_ptr, _path_size);
      return true;
    };

    // push context function
    auto push_ctx = [&stack, &stack_pos](context ctx) {
      if (stack_pos == max_path_depth - 1) { return false; }
      stack[stack_pos++] = ctx;
      return true;
    };

    // pop context function
    auto pop_context = [&stack, &stack_pos](context& c) {
      if (stack_pos > 0) {
        c = stack[--stack_pos];
        return true;
      }
      return false;
    };
    
    // put the first context task
    push_context(p.get_current_token(), -1, root_g, root_style, root_path_ptr, root_path_size);

    // current context task
    context ctx;

    // parent context task
    context p_ctx;

    while (pop_context(ctx)) {
      if (!ctx.task_is_done) {
        // task is not done.

        // case (VALUE_STRING, Nil) if style == RawStyle
        // case path 1
        if (json_token::VALUE_STRING == ctx.token && path_is_empty(ctx.path_size) &&
            ctx.style == write_style::raw_style) {
          // there is no array wildcard or slice parent, emit this string without
          // quotes write current string in parser to generator
          ctx.g.write_raw(p);
          ctx.dirty = 1;
          ctx.task_is_done = true;
          push_ctx(ctx);
        }
        // case (START_ARRAY, Nil) if style == FlattenStyle
        // case path 2
        else if (json_token::START_ARRAY == ctx.token && path_is_empty(ctx.path_size) &&
                ctx.style == write_style::flatten_style) {
          // flatten this array into the parent
          if (json_token::END_ARRAY != p.next_token()) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }
            // push back task
            push_ctx(ctx);
            // add child task
            push_context(p.get_current_token(), 2, ctx.g, ctx.style, nullptr, 0);
          } else {
            // END_ARRAY
            ctx.task_is_done = true;
            push_ctx(ctx);
          }
        }
        // case (_, Nil)
        // case path 3
        else if (path_is_empty(ctx.path_size)) {
          // general case: just copy the child tree verbatim
          if (!(ctx.g.copy_current_structure(p))) {
            // JSON validation check
            return false;
          }
          ctx.dirty = 1;
          ctx.task_is_done = true;
          push_ctx(ctx);
        }
        // case (START_OBJECT, Key :: xs)
        // case path 4
        else if (json_token::START_OBJECT == ctx.token &&
                path_match_element(ctx.path_ptr, ctx.path_size, path_instruction_type::key)) {
          if (json_token::END_OBJECT != p.next_token()) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }

            if (ctx.dirty > 0) {
              // once a match has been found we can skip other fields
              if (!p.try_skip_children()) {
                // JSON validation check
                return false;
              }
              push_ctx(ctx);
            } else {
              // need to try more children
              push_ctx(ctx);
              push_context(p.get_current_token(), 4, ctx.g, ctx.style, ctx.path_ptr + 1, ctx.path_size - 1);
            }
          }
          else
          {
            ctx.task_is_done = true;
            push_ctx(ctx);
          }
        }
        // case (START_ARRAY, Subscript :: Wildcard :: Subscript :: Wildcard :: xs)
        // case path 5
        else if (json_token::START_ARRAY == ctx.token &&
                path_match_elements(ctx.path_ptr,
                                    ctx.path_size,
                                    path_instruction_type::subscript,
                                    path_instruction_type::wildcard,
                                    path_instruction_type::subscript,
                                    path_instruction_type::wildcard)) {
          // special handling for the non-structure preserving double wildcard
          // behavior in Hive
          if (ctx.is_first_enter) {
            ctx.is_first_enter = false;
            ctx.g.write_start_array();
          }

          if (p.next_token() != json_token::END_ARRAY) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }
            push_ctx(ctx);
            push_context(p.get_current_token(), 5, ctx.g, write_style::flatten_style, ctx.path_ptr + 4, ctx.path_size - 4);
          } else {
            ctx.g.write_end_array();
            ctx.task_is_done = true;
            push_ctx(ctx);
          }
        }
        // case (START_ARRAY, Subscript :: Wildcard :: xs) if style != QuotedStyle
        // case path 6
        else if (json_token::START_ARRAY == ctx.token &&
                path_match_elements(ctx.path_ptr,
                                    ctx.path_size,
                                    path_instruction_type::subscript,
                                    path_instruction_type::wildcard) &&
                ctx.style != write_style::quoted_style) {
          // retain Flatten, otherwise use Quoted... cannot use Raw within an array
          write_style next_style = write_style::raw_style;
          switch (ctx.style) {
            case write_style::raw_style: next_style = write_style::quoted_style; break;
            case write_style::flatten_style: next_style = write_style::flatten_style; break;
            case write_style::quoted_style: next_style = write_style::quoted_style;  // never happen
          }

          // temporarily buffer child matches, the emitted json will need to be
          // modified slightly if there is only a single element written

          json_generator<> child_g;
          if (ctx.is_first_enter) {
            ctx.is_first_enter = false;
            // create a child generator with hide outer array tokens mode.
            child_g = ctx.g.new_child_generator(/*hide_outer_array_tokens*/ true);
            // Note: child generator does not actually write the outer start array
            // token into buffer it only updates internal nested state
            child_g.write_start_array();

            // copy to stack
            ctx.child_g = child_g;
          } else {
            child_g = ctx.child_g;
          }

          if (p.next_token() != json_token::END_ARRAY) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }

            push_ctx(ctx);
            // track the number of array elements and only emit an outer array if
            // we've written more than one element, this matches Hive's behavior
            push_context(p.get_current_token(), 6, child_g, next_style, ctx.path_ptr + 2, ctx.path_size - 2);
          } else {
            // Note: child generator does not actually write the outer end array token
            // into buffer it only updates internal nested state
            child_g.write_end_array();

            char* child_g_start = child_g.get_output_start_position();
            size_t child_g_len  = child_g.get_output_len();  // len already excluded outer [ ]

            if (ctx.dirty > 1) {
              // add outer array tokens
              ctx.g.write_child_raw_value(child_g_start, child_g_len, true);
              ctx.task_is_done = true;
              push_ctx(ctx);
            } else if (ctx.dirty == 1) {
              // remove outer array tokens
              ctx.g.write_child_raw_value(child_g_start, child_g_len, false);
              ctx.task_is_done = true;
              push_ctx(ctx);
            }  // else do not write anything
          }
        }
        // case (START_ARRAY, Subscript :: Wildcard :: xs)
        // case path 7
        else if (json_token::START_ARRAY == ctx.token &&
                path_match_elements(ctx.path_ptr,
                                    ctx.path_size,
                                    path_instruction_type::subscript,
                                    path_instruction_type::wildcard)) {
          if (ctx.is_first_enter) {
            ctx.is_first_enter = false;
            ctx.g.write_start_array();
          }

          if (p.next_token() != json_token::END_ARRAY) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }

            // wildcards can have multiple matches, continually update the dirty
            // count
            push_ctx(ctx);
            push_context(p.get_current_token(), 7, ctx.g, write_style::quoted_style, ctx.path_ptr + 2, ctx.path_size - 2);
          } else {
            ctx.g.write_end_array();
            ctx.task_is_done = true;
            push_ctx(ctx);
          }
        }
        /* case (START_ARRAY, Subscript :: Index(idx) :: (xs@Subscript :: Wildcard :: _)) */
        // case path 8
        else if (json_token::START_ARRAY == ctx.token &&
                thrust::get<0>(path_match_subscript_index_subscript_wildcard(ctx.path_ptr, ctx.path_size))) {
          int idx = thrust::get<1>(path_match_subscript_index_subscript_wildcard(ctx.path_ptr, ctx.path_size));

          p.next_token();
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return false; }
          ctx.is_first_enter = false;

          int i = idx;
          while (i > 0) {
            if (p.get_current_token() == json_token::END_ARRAY) {
              // terminate, nothing has been written
              return false;
            }

            if (!p.try_skip_children()) { return false; }

            p.next_token();
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }
          
            --i;
          }

          // i == 0
          push_ctx(ctx);
          push_context(p.get_current_token(), 8, ctx.g, write_style::quoted_style, ctx.path_ptr + 2, ctx.path_size - 2);
        }
        // case (START_ARRAY, Subscript :: Index(idx) :: xs)
        // case path 9
        else if (json_token::START_ARRAY == ctx.token &&
                thrust::get<0>(path_match_subscript_index(ctx.path_ptr, ctx.path_size))) {
          int idx = thrust::get<1>(path_match_subscript_index_subscript_wildcard(ctx.path_ptr, ctx.path_size));

          p.next_token();
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return false; }
          ctx.is_first_enter = false;

          int i = idx;
          while (i > 0) {
            if (p.get_current_token() == json_token::END_ARRAY) {
              // terminate, nothing has been written
              return false;
            }

            if (!p.try_skip_children()) { return false; }

            p.next_token();
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }
          
            --i;
          }

          // i == 0
          push_ctx(ctx);
          push_context(p.get_current_token(), 9, ctx.g, ctx.style, ctx.path_ptr + 2, ctx.path_size - 2);
        }
        // case (FIELD_NAME, Named(name) :: xs) if p.getCurrentName == name
        // case path 10
        else if (json_token::FIELD_NAME == ctx.token &&
                thrust::get<0>(path_match_named(ctx.path_ptr, ctx.path_size)) &&
                p.match_current_field_name(thrust::get<1>(path_match_named(ctx.path_ptr, ctx.path_size)))) {
          if (p.next_token() != json_token::VALUE_NULL) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }
            push_ctx(ctx);
            push_context(p.get_current_token(), 10, ctx.g, ctx.style, ctx.path_ptr + 1, ctx.path_size - 1);
          } else {
            return false;
          }
        }
        // case (FIELD_NAME, Wildcard :: xs)
        // case path 11
        else if (json_token::FIELD_NAME == ctx.token &&
                path_match_element(ctx.path_ptr, ctx.path_size, path_instruction_type::wildcard)) {
          p.next_token();
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return false; }
          push_ctx(ctx);
          push_context(p.get_current_token(), 11, ctx.g, ctx.style, ctx.path_ptr + 1, ctx.path_size - 1);
        }
        // case _ =>
        // case path 12
        else {
          if (!p.try_skip_children()) { return false; }
          // default case path, return false for this task
          ctx.dirty = 0;
          ctx.task_is_done = true;
          push_ctx(ctx);
        } 
      }
      else
      {
        // current context is done.

        // pop parent task
        // update parent task info according to current task result
        if (pop_context(p_ctx)) {
          // case (VALUE_STRING, Nil) if style == RawStyle
          // case path 1
          if (1 == ctx.case_path) {
            // never happen
          }
          // case (START_ARRAY, Nil) if style == FlattenStyle
          // case path 2
          else if (2 == ctx.case_path) {
            // collect result from child task
            p_ctx.dirty += ctx.dirty;
            // copy generator states to parent task;
            p_ctx.g = ctx.g;
            push_ctx(p_ctx);
          }
          // case (_, Nil)
          // case path 3
          else if (3 == ctx.case_path) {
            // never happen
          }
          // case (START_OBJECT, Key :: xs)
          // case path 4
          else if (4 == ctx.case_path) {
            if (p_ctx.dirty < 1 && ctx.dirty > 0)
            {
              p_ctx.dirty = ctx.dirty;
            }
            // copy generator states to parent task;
            p_ctx.g = ctx.g;
            push_ctx(p_ctx);
          }
          // case (START_ARRAY, Subscript :: Wildcard :: Subscript :: Wildcard :: xs)
          // case path 5
          else if (5 == ctx.case_path) {
            // collect result from child task
            p_ctx.dirty += ctx.dirty;
            // copy generator states to parent task;
            p_ctx.g = ctx.g;
            push_ctx(p_ctx);
          }
          // case (START_ARRAY, Subscript :: Wildcard :: xs) if style != QuotedStyle
          // case path 6
          else if (6 == ctx.case_path) {
            // collect result from child task
            p_ctx.dirty += ctx.dirty;
            // update child generator for parent task
            p_ctx.child_g = ctx.g;
            push_ctx(p_ctx);
          }
          // case (START_ARRAY, Subscript :: Wildcard :: xs)
          // case path 7
          else if (7 == ctx.case_path) {
            // collect result from child task
            p_ctx.dirty += ctx.dirty;
            // copy generator states to parent task;
            p_ctx.g = ctx.g;
            push_ctx(p_ctx);
          }
          /* case (START_ARRAY, Subscript :: Index(idx) :: (xs@Subscript :: Wildcard :: _)) */
          // case path 8
          else if (8 == ctx.case_path || 9 == ctx.case_path) {
            // collect result from child task
            p_ctx.dirty += ctx.dirty;

            // post logic:
            if (p.next_token() != json_token::END_ARRAY) {
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return false; }
              // advance the token stream to the end of the array
              if (!p.try_skip_children()) { return false; }
            }
            // task is done
            p_ctx.task_is_done = true;
            // copy generator states to parent task;
            p_ctx.g = ctx.g;
            push_ctx(p_ctx);
          }
          // case (FIELD_NAME, Named(name) :: xs) if p.getCurrentName == name
          // case path 10
          else if (10 == ctx.case_path) {
            // collect result from child task
            p_ctx.dirty += ctx.dirty;
            // task is done
            p_ctx.task_is_done = true;
            // copy generator states to parent task;
            p_ctx.g = ctx.g;
            push_ctx(p_ctx);
          }
          // case (FIELD_NAME, Wildcard :: xs)
          // case path 11
          else if (11 == ctx.case_path) {
            // collect result from child task
            p_ctx.dirty += ctx.dirty;
            // task is done
            p_ctx.task_is_done = true;
            // copy generator states to parent task;
            p_ctx.g = ctx.g;
            push_ctx(p_ctx);
          }
          // case _ =>
          // case path 12
          else {
            // never happen
          }
        }
        else
        {
          // has no parent task, stack is empty, will exit
        }
      }
    }

    // copy output len
    root_g.set_output_len(ctx.g.get_output_len());
    return ctx.dirty > 0;
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
