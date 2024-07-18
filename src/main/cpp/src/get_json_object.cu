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
#include "json_parser.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace spark_rapids_jni {

namespace detail {

// path max depth limitation
// There is a same constant in JSONUtil.java, keep them consistent when changing
// Note: Spark-Rapids should guarantee the path depth is less or equal to this limit,
// or GPU reports cudaErrorIllegalAddress
constexpr int max_path_depth = 16;

/**
 * @brief JSON style to write.
 */
enum class write_style : int8_t { RAW, QUOTED, FLATTEN };

/**

 * @brief JSON generator used to write out JSON content.
 *
 * Because of get_json_object only outputs JSON object as a whole item,
 * it's no need to store internal state for JSON object when outputing,
 * only need to store internal state for JSON array.
 */
class json_generator {
 public:
  __device__ json_generator(int _offset = 0) : offset(_offset), output_len(0) {}

  // create a nested child generator based on this parent generator,
  // child generator is a view, parent and child share the same byte array
  __device__ json_generator new_child_generator() const
  {
    return json_generator(offset + output_len);
  }

  // write [
  // add an extra comma if needed,
  // e.g.: when JSON content is: [[1,2,3]
  // writing a new [ should result: [[1,2,3],[
  __device__ void write_start_array(char* out_begin)
  {
    try_write_comma(out_begin);

    out_begin[offset + output_len] = '[';
    output_len++;
    array_depth++;
    // new array is empty
    is_curr_array_empty = true;
  }

  // write ]
  __device__ void write_end_array(char* out_begin)
  {
    out_begin[offset + output_len] = ']';
    output_len++;
    // point to parent array
    array_depth--;
    // set parent array as non-empty because already had a closed child item.
    is_curr_array_empty = false;
  }

  // write first start array without output, only update internal state
  __device__ void write_first_start_array_without_output()
  {
    // hide the outer start array token
    // Note: do not inc output_len
    array_depth++;
    // new array is empty
    is_curr_array_empty = true;
  }

  // return true if it's in a array context and it's not writing the first item.
  __device__ inline bool need_comma() const { return (array_depth > 0 && !is_curr_array_empty); }

  /**
   * write comma accroding to current generator state
   */
  __device__ void try_write_comma(char* out_begin)
  {
    if (need_comma()) {
      // in array context and writes first item
      out_begin[offset + output_len] = ',';
      output_len++;
    }
  }

  /**
   * copy current structure when parsing. If current token is start
   * object/array, then copy to corresponding matched end object/array. return
   * false if JSON format is invalid return true if JSON format is valid
   */
  __device__ bool copy_current_structure(json_parser& parser, char* out_begin)
  {
    // first try add comma
    try_write_comma(out_begin);

    if (array_depth > 0) { is_curr_array_empty = false; }

    auto [b, copy_len] = parser.copy_current_structure(out_begin + offset + output_len);
    output_len += copy_len;
    return b;
  }

  /**
   * Get current text from JSON parser and then write the text
   * Note: Because JSON strings contains '\' to do escape,
   * JSON parser should do unescape to remove '\' and JSON parser
   * then can not return a pointer and length pair (char *, len),
   * For number token, JSON parser can return a pair (char *, len)
   */
  __device__ void write_raw(json_parser& parser, char* out_begin)
  {
    if (array_depth > 0) { is_curr_array_empty = false; }

    auto copied = parser.write_unescaped_text(out_begin + offset + output_len);
    output_len += copied;
  }

  /**
   * write child raw value
   * e.g.:
   *
   * write_outer_array_tokens = false
   * need_comma = true
   * [1,2,3]1,2,3
   *        ^
   *        |
   *    child pointer
   * ==>>
   * [1,2,3],1,2,3
   *
   *
   * write_outer_array_tokens = true
   * need_comma = true
   *   [12,3,4
   *     ^
   *     |
   * child pointer
   * ==>>
   *   [1,[2,3,4]
   *
   * For more information about param write_outer_array_tokens, refer to
   * `write_first_start_array_without_output`
   * @param child_block_begin
   * @param child_block_len
   * @param write_outer_array_tokens whether write outer array tokens for child
   * block
   */
  __device__ void write_child_raw_value(char* child_block_begin,
                                        int child_block_len,
                                        bool write_outer_array_tokens)
  {
    bool insert_comma = need_comma();

    if (array_depth > 0) { is_curr_array_empty = false; }

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

    // update length
    if (insert_comma) { output_len++; }
    if (write_outer_array_tokens) { output_len += 2; }
    output_len += child_block_len;
  }

  // move memory block forward by specified bytes
  // e.g.:  memory is: 1 2 0 0, begin is 1, len is 2, after moving,
  // memory is: 1 2 1 2.
  // e.g.:  memory is: 1 2 0 0, begin is 1, len is 1, after moving,
  // memory is: 1 1 2 0.
  // Note: should move from end to begin to avoid overwrite buffer
  static __device__ void move_forward(char* begin, size_t len, int forward)
  {
    // TODO copy by 8 bytes
    char* pos = begin + len + forward - 1;
    char* e   = begin + forward - 1;
    while (pos > e) {
      *pos = *(pos - forward);
      pos--;
    }
  }

  __device__ inline int get_offset() const { return offset; }
  __device__ inline int get_output_len() const { return output_len; }

  /**
   * generator may contain trash output, e.g.: generator writes some output,
   * then JSON format is invalid, the previous output becomes trash.
   */
  __device__ inline void set_output_len_zero() { output_len = 0; }

  __device__ inline void set_output_len(size_t len) { output_len = len; }

 private:
  int offset;  // offset from the global output buffer
  int output_len;

  int array_depth = 0;

  // whether already worte a item in current array
  // used to decide whether add a comma before writing out a new item.
  bool is_curr_array_empty;
};

/**
 * path evaluator which can run on both CPU and GPU
 */
__device__ inline bool path_is_empty(size_t path_size) { return path_size == 0; }

__device__ inline bool path_match_element(cudf::device_span<path_instruction const> path,
                                          path_instruction_type path_type0)
{
  if (path.size() < 1) { return false; }
  return path.data()[0].type == path_type0;
}

__device__ inline bool path_match_elements(cudf::device_span<path_instruction const> path,
                                           path_instruction_type path_type0,
                                           path_instruction_type path_type1)
{
  if (path.size() < 2) { return false; }
  return path.data()[0].type == path_type0 && path.data()[1].type == path_type1;
}

__device__ inline thrust::tuple<bool, int> path_match_index(
  cudf::device_span<path_instruction const> path)
{
  auto match = path_match_element(path, path_instruction_type::INDEX);
  if (match) {
    return thrust::make_tuple(true, path.data()[0].index);
  } else {
    return thrust::make_tuple(false, 0);
  }
}

__device__ inline thrust::tuple<bool, cudf::string_view> path_match_named(
  cudf::device_span<path_instruction const> path)
{
  auto match = path_match_element(path, path_instruction_type::NAMED);
  if (match) {
    return thrust::make_tuple(true, path.data()[0].name);
  } else {
    return thrust::make_tuple(false, cudf::string_view());
  }
}

__device__ inline thrust::tuple<bool, int> path_match_index_wildcard(
  cudf::device_span<path_instruction const> path)
{
  auto match =
    path_match_elements(path, path_instruction_type::INDEX, path_instruction_type::WILDCARD);
  if (match) {
    return thrust::make_tuple(true, path.data()[0].index);
  } else {
    return thrust::make_tuple(false, 0);
  }
}

/**
 * @brief The cases that mirro Apache Spark case path in `jsonExpressions.scala#evaluatePath()`.
 */
enum class evaluation_case_path : int8_t {
  INVALID                                           = -1,
  START_ARRAY___EMPTY_PATH___FLATTEN_STYLE          = 2,
  START_OBJECT___MATCHED_NAME_PATH                  = 4,
  START_ARRAY___MATCHED_DOUBLE_WILDCARD             = 5,
  START_ARRAY___MATCHED_WILDCARD___STYLE_NOT_QUOTED = 6,
  START_ARRAY___MATCHED_WILDCARD                    = 7,
  START_ARRAY___MATCHED_INDEX_AND_WILDCARD          = 8,
  START_ARRAY___MATCHED_INDEX                       = 9
};

/**
 * @brief The struct to store states during processing JSON through different nested levels.
 */
struct context {
  // used to save current generator
  json_generator g;

  // used to save child JSON generator for case path 6
  json_generator child_g;

  cudf::device_span<path_instruction const> path;

  // whether written output
  // if dirty > 0, indicates success
  int dirty;

  // which case path that this task is from
  evaluation_case_path case_path;

  // current token
  json_token token;

  write_style style;

  // for some case paths
  bool is_first_enter;

  // is this context task is done
  bool task_is_done;
};

/**
 * @brief Parse a single json string using the provided command buffer.
 *
 * @param input The incoming json string
 * @param path_commands The command buffer to be applied to the string
 * @param out_buf Buffer user to store the string resulted from the query
 * @return A pair containing the result code and the output size
 */
__device__ thrust::pair<bool, cudf::size_type> evaluate_path(
  char_range input, cudf::device_span<path_instruction const> path_commands, char* out_buf)
{
  json_parser p{input};
  p.next_token();
  if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

  // define stack; plus 1 indicates root context task needs an extra memory
  context stack[max_path_depth + 1];
  int stack_size = 0;

  // push context function
  auto push_context = [&p, &stack, &stack_size](evaluation_case_path _case_path,
                                                json_generator _g,
                                                write_style _style,
                                                cudf::device_span<path_instruction const> _path) {
    // no need to check stack is full
    // because Spark-Rapids already checked maximum length of `path_instruction`
    auto& ctx          = stack[stack_size++];
    ctx.g              = std::move(_g);
    ctx.path           = std::move(_path);
    ctx.dirty          = 0;
    ctx.case_path      = _case_path;
    ctx.token          = p.get_current_token();
    ctx.style          = _style;
    ctx.is_first_enter = true;
    ctx.task_is_done   = false;
  };

  // put the first context task
  push_context(evaluation_case_path::INVALID, json_generator{}, write_style::RAW, path_commands);

  while (stack_size > 0) {
    auto& ctx = stack[stack_size - 1];
    if (!ctx.task_is_done) {
      // case (VALUE_STRING, Nil) if style == RawStyle
      // case path 1
      if (json_token::VALUE_STRING == ctx.token && path_is_empty(ctx.path.size()) &&
          ctx.style == write_style::RAW) {
        // there is no array wildcard or slice parent, emit this string without
        // quotes write current string in parser to generator
        ctx.g.write_raw(p, out_buf);
        ctx.dirty        = 1;
        ctx.task_is_done = true;
      }
      // case (START_ARRAY, Nil) if style == FlattenStyle
      // case path 2
      else if (json_token::START_ARRAY == ctx.token && path_is_empty(ctx.path.size()) &&
               ctx.style == write_style::FLATTEN) {
        // flatten this array into the parent
        if (json_token::END_ARRAY != p.next_token()) {
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }
          // push back task
          // add child task
          push_context(evaluation_case_path::START_ARRAY___EMPTY_PATH___FLATTEN_STYLE,
                       ctx.g,
                       ctx.style,
                       {nullptr, 0});
        } else {
          // END_ARRAY
          ctx.task_is_done = true;
        }
      }
      // case (_, Nil)
      // case path 3
      else if (path_is_empty(ctx.path.size())) {
        // general case: just copy the child tree verbatim
        if (!(ctx.g.copy_current_structure(p, out_buf))) {
          // JSON validation check
          return {false, 0};
        }
        ctx.dirty        = 1;
        ctx.task_is_done = true;
      }
      // case (START_OBJECT, Named :: xs)
      // case path 4
      else if (json_token::START_OBJECT == ctx.token &&
               thrust::get<0>(path_match_named(ctx.path))) {
        if (!ctx.is_first_enter) {
          // 2st enter
          // skip the following children after the expect
          if (ctx.dirty > 0) {
            while (json_token::END_OBJECT != p.next_token()) {
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

              // skip FIELD_NAME token
              p.next_token();
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

              // skip value of FIELD_NAME
              if (!p.try_skip_children()) {
                // JSON validation check
                return {false, 0};
              }
            }
          }
          // Mark task is done regardless whether the expected child was found.
          ctx.task_is_done = true;
        } else {
          // below is 1st enter
          ctx.is_first_enter = false;
          // match first mached children with expected name
          bool found_expected_child = false;
          while (json_token::END_OBJECT != p.next_token()) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

            // need to try more children
            auto match_named = path_match_named(ctx.path);
            auto named       = thrust::get<1>(match_named);
            // current token is FIELD_NAME
            if (p.match_current_field_name(named)) {
              // skip FIELD_NAME token
              p.next_token();
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

              // meets null token, it's not expected, return false
              if (json_token::VALUE_NULL == p.get_current_token()) { return {false, 0}; }
              // push sub task; sub task will update the result of path 4
              push_context(evaluation_case_path::START_OBJECT___MATCHED_NAME_PATH,
                           ctx.g,
                           ctx.style,
                           {ctx.path.data() + 1, ctx.path.size() - 1});
              found_expected_child = true;
              break;
            } else {
              // skip FIELD_NAME token
              p.next_token();
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

              // current child is not expected, skip current child
              if (!p.try_skip_children()) {
                // JSON validation check
                return {false, 0};
              }
            }
          }
          if (!found_expected_child) {
            // did not find any expected sub child
            ctx.task_is_done = true;
            ctx.dirty        = false;
          }
        }
      }
      // case (START_ARRAY, Wildcard :: Wildcard :: xs)
      // case path 5
      else if (json_token::START_ARRAY == ctx.token &&
               path_match_elements(
                 ctx.path, path_instruction_type::WILDCARD, path_instruction_type::WILDCARD)) {
        // special handling for the non-structure preserving double wildcard
        // behavior in Hive
        if (ctx.is_first_enter) {
          ctx.is_first_enter = false;
          ctx.g.write_start_array(out_buf);
        }

        if (p.next_token() != json_token::END_ARRAY) {
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }
          push_context(evaluation_case_path::START_ARRAY___MATCHED_DOUBLE_WILDCARD,
                       ctx.g,
                       write_style::FLATTEN,
                       {ctx.path.data() + 2, ctx.path.size() - 2});
        } else {
          ctx.g.write_end_array(out_buf);
          ctx.task_is_done = true;
        }
      }
      // case (START_ARRAY, Wildcard :: xs) if style != QuotedStyle
      // case path 6
      else if (json_token::START_ARRAY == ctx.token &&
               path_match_element(ctx.path, path_instruction_type::WILDCARD) &&
               ctx.style != write_style::QUOTED) {
        // retain Flatten, otherwise use Quoted... cannot use Raw within an array
        write_style next_style = write_style::RAW;
        switch (ctx.style) {
          case write_style::RAW: next_style = write_style::QUOTED; break;
          case write_style::FLATTEN: next_style = write_style::FLATTEN; break;
          case write_style::QUOTED: next_style = write_style::QUOTED;  // never happen
        }

        // temporarily buffer child matches, the emitted json will need to be
        // modified slightly if there is only a single element written

        json_generator child_g;
        if (ctx.is_first_enter) {
          ctx.is_first_enter = false;
          // create a child generator with hide outer array tokens mode.
          child_g = ctx.g.new_child_generator();
          // write first [ without output, without update len, only update internal state
          child_g.write_first_start_array_without_output();
        } else {
          child_g = ctx.child_g;
        }

        if (p.next_token() != json_token::END_ARRAY) {
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }
          // track the number of array elements and only emit an outer array if
          // we've written more than one element, this matches Hive's behavior
          push_context(evaluation_case_path::START_ARRAY___MATCHED_WILDCARD___STYLE_NOT_QUOTED,
                       child_g,
                       next_style,
                       {ctx.path.data() + 1, ctx.path.size() - 1});
        } else {
          char* child_g_start = out_buf + child_g.get_offset();
          int child_g_len     = child_g.get_output_len();
          if (ctx.dirty > 1) {
            // add outer array tokens
            ctx.g.write_child_raw_value(
              child_g_start, child_g_len, /* write_outer_array_tokens */ true);
          } else if (ctx.dirty == 1) {
            // remove outer array tokens
            ctx.g.write_child_raw_value(
              child_g_start, child_g_len, /* write_outer_array_tokens */ false);
          }  // else do not write anything

          // Done anyway, since we already reached the end array.
          ctx.task_is_done = true;
        }
      }
      // case (START_ARRAY, Wildcard :: xs)
      // case path 7
      else if (json_token::START_ARRAY == ctx.token &&
               path_match_element(ctx.path, path_instruction_type::WILDCARD)) {
        if (ctx.is_first_enter) {
          ctx.is_first_enter = false;
          ctx.g.write_start_array(out_buf);
        }
        if (p.next_token() != json_token::END_ARRAY) {
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

          // wildcards can have multiple matches, continually update the dirty
          // count
          push_context(evaluation_case_path::START_ARRAY___MATCHED_WILDCARD,
                       ctx.g,
                       write_style::QUOTED,
                       {ctx.path.data() + 1, ctx.path.size() - 1});
        } else {
          ctx.g.write_end_array(out_buf);
          ctx.task_is_done = true;
        }
      }
      /* case (START_ARRAY, Index(idx) :: (xs@Wildcard :: _)) */
      // case path 8
      else if (json_token::START_ARRAY == ctx.token &&
               thrust::get<0>(path_match_index_wildcard(ctx.path))) {
        int idx = thrust::get<1>(path_match_index_wildcard(ctx.path));

        p.next_token();
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }
        ctx.is_first_enter = false;

        int i = idx;
        while (i > 0) {
          if (p.get_current_token() == json_token::END_ARRAY) {
            // terminate, nothing has been written
            return {false, 0};
          }

          if (!p.try_skip_children()) { return {false, 0}; }

          p.next_token();
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

          --i;
        }

        // i == 0
        push_context(evaluation_case_path::START_ARRAY___MATCHED_INDEX_AND_WILDCARD,
                     ctx.g,
                     write_style::QUOTED,
                     {ctx.path.data() + 1, ctx.path.size() - 1});
      }
      // case (START_ARRAY, Index(idx) :: xs)
      // case path 9
      else if (json_token::START_ARRAY == ctx.token && thrust::get<0>(path_match_index(ctx.path))) {
        int idx = thrust::get<1>(path_match_index(ctx.path));

        p.next_token();
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

        int i = idx;
        while (i > 0) {
          if (p.get_current_token() == json_token::END_ARRAY) {
            // terminate, nothing has been written
            return {false, 0};
          }

          if (!p.try_skip_children()) { return {false, 0}; }

          p.next_token();
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

          --i;
        }

        // i == 0
        push_context(evaluation_case_path::START_ARRAY___MATCHED_INDEX,
                     ctx.g,
                     ctx.style,
                     {ctx.path.data() + 1, ctx.path.size() - 1});
      }
      // case _ =>
      // case path 12
      else {
        if (!p.try_skip_children()) { return {false, 0}; }
        // default case path, return false for this task
        ctx.dirty        = 0;
        ctx.task_is_done = true;
      }
    }       // if (!ctx.task_is_done)
    else {  // current context is done.
      // pop current top context
      stack_size--;

      // has no parent task, stack is empty, will exit
      if (stack_size == 0) { break; }

      // peek parent context task
      // update parent task info according to current task result
      auto& p_ctx = stack[stack_size - 1];

      switch (ctx.case_path) {
          // path 2: case (START_ARRAY, Nil) if style == FlattenStyle
          // path 5: case (START_ARRAY, Wildcard :: Wildcard :: xs)
          // path 7: case (START_ARRAY, Wildcard :: xs)
        case evaluation_case_path::START_ARRAY___EMPTY_PATH___FLATTEN_STYLE:
        case evaluation_case_path::START_ARRAY___MATCHED_DOUBLE_WILDCARD:
        case evaluation_case_path::START_ARRAY___MATCHED_WILDCARD: {
          // collect result from child task
          p_ctx.dirty += ctx.dirty;
          // copy generator states to parent task;
          p_ctx.g = ctx.g;

          break;
        }

          // case (START_OBJECT, Named :: xs)
          // case path 4
        case evaluation_case_path::START_OBJECT___MATCHED_NAME_PATH: {
          p_ctx.dirty = ctx.dirty;
          // copy generator states to parent task;
          p_ctx.g = ctx.g;

          break;
        }

          // case (START_ARRAY, Wildcard :: xs) if style != QuotedStyle
          // case path 6
        case evaluation_case_path::START_ARRAY___MATCHED_WILDCARD___STYLE_NOT_QUOTED: {
          // collect result from child task
          p_ctx.dirty += ctx.dirty;
          // update child generator for parent task
          p_ctx.child_g = ctx.g;

          break;
        }

          /* case (START_ARRAY, Index(idx) :: (xs@Wildcard :: _)) */
          // case path 8
          // case (START_ARRAY, Index(idx) :: xs)
          // case path 9
        case evaluation_case_path::START_ARRAY___MATCHED_INDEX_AND_WILDCARD:
        case evaluation_case_path::START_ARRAY___MATCHED_INDEX: {
          // collect result from child task
          p_ctx.dirty += ctx.dirty;

          // post logic:
          while (p.next_token() != json_token::END_ARRAY) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }
            // advance the token stream to the end of the array
            if (!p.try_skip_children()) { return {false, 0}; }
          }
          // task is done
          p_ctx.task_is_done = true;
          // copy generator states to parent task;
          p_ctx.g = ctx.g;

          break;
        }

        default:;  // Never happens!
      }            // end switch (ctx.case_path)
    }              // ctx.task_is_done
  }                // while (stack_size > 0)

  auto const success = stack[0].dirty > 0;

  // generator may contain trash output, e.g.: generator writes some output,
  // then JSON format is invalid, the previous output becomes trash.
  // We need to return output size as zero.
  return {success, success ? stack[0].g.get_output_len() : 0};
}

/**
 * @brief Struct storing data such as path instructions, output buffer etc, corresponding to a
 * single JSON path.
 */
struct json_path_processing_data {
  cudf::device_span<path_instruction const> path_commands;
  cudf::detail::input_offsetalator offsets;
  thrust::pair<char const*, cudf::size_type>* out_stringviews;
  char* out_buf;
  int8_t* has_out_of_bound;
};

/**
 * @brief Extract JSON object, from one input row, on one path.
 *
 * @param input The entire input strings column
 * @param path_data Array containing all path data
 * @param row_path_idx The index to identify row index and path index
 */
__device__ void process_row_path(cudf::column_device_view input,
                                 cudf::device_span<json_path_processing_data> path_data,
                                 int64_t row_path_idx)
{
  auto const row_idx  = row_path_idx / path_data.size();
  auto const path_idx = row_path_idx % path_data.size();
  auto const& path    = path_data[path_idx];

  char* const dst          = path.out_buf + path.offsets[row_idx];
  bool is_valid            = false;
  cudf::size_type out_size = 0;

  auto const str = input.element<cudf::string_view>(row_idx);
  if (str.size_bytes() > 0) {
    thrust::tie(is_valid, out_size) = evaluate_path(char_range{str}, path.path_commands, dst);

    auto const max_size = path.offsets[row_idx + 1] - path.offsets[row_idx];
    if (out_size > max_size) { *(path.has_out_of_bound) = 1; }
  }

  // Write out `nullptr` in the output string_view to indicate that the output is a null.
  // The situation `out_stringviews == nullptr` should only happen if the kernel is launched a
  // second time due to out-of-bound write in the first launch.
  if (path.out_stringviews) {
    path.out_stringviews[row_idx] = {is_valid ? dst : nullptr, out_size};
  }
}

/**
 * @brief Kernel for running the JSONPath query, using all threads for processing.
 *
 * This kernel writes out the output strings and their lengths at the same time. If any output
 * length exceed buffer size limit, a boolean flag will be turned on to inform to the caller.
 * In such situation, another (larger) output buffer will be generated and the kernel is launched
 * again. Otherwise, launching this kernel only once is sufficient to produce the desired output.
 *
 * @param input The input JSON strings stored in a strings column
 * @param path_data Array containing all path data
 */
template <int block_size, int min_block_per_sm>
__launch_bounds__(block_size, min_block_per_sm) CUDF_KERNEL
  void get_json_object_kernel_thread_parallel(
    cudf::column_device_view input, cudf::device_span<json_path_processing_data> path_data)
{
  auto const max_tid = static_cast<int64_t>(input.size()) * path_data.size();
  auto const stride  = cudf::detail::grid_1d::grid_stride();

  for (auto tid = cudf::detail::grid_1d::global_thread_id(); tid < max_tid; tid += stride) {
    process_row_path(input, path_data, tid);
  }
}

/**
 * @brief Kernel for running the JSONPath query, using one warp to process a row.
 *
 * The behavior of this kernel should be identical to that of the version using all threads for
 * processing.
 *
 * @param input The input JSON strings stored in a strings column
 * @param path_data Array containing all path data
 */
template <int block_size, int min_block_per_sm>
__launch_bounds__(block_size, min_block_per_sm) CUDF_KERNEL
  void get_json_object_kernel_warp_parallel(cudf::column_device_view input,
                                            cudf::device_span<json_path_processing_data> path_data)
{
  auto const max_tid =
    static_cast<int64_t>(input.size()) * path_data.size() * cudf::detail::warp_size;
  auto const stride  = cudf::detail::grid_1d::grid_stride();
  auto const lane_id = threadIdx.x % cudf::detail::warp_size;

  for (auto tid = cudf::detail::grid_1d::global_thread_id(); tid < max_tid; tid += stride) {
    if (lane_id == 0) {
      auto const warp_idx = tid / cudf::detail::warp_size;
      process_row_path(input, path_data, warp_idx);
    }
    __syncwarp();
  }
}

/**
 * @brief Launch the main kernel.
 *
 * Either a thread-parallel or warp-parallel kernel is launched, depending on the value of
 * `exec_thread_parallel`.
 */
void launch_kernel(bool exec_thread_parallel,
                   cudf::column_device_view const& input,
                   cudf::device_span<json_path_processing_data> path_data,
                   rmm::cuda_stream_view stream)
{
  // We explicitly set the minBlocksPerMultiprocessor parameter in the launch bounds to avoid
  // spilling from the kernel itself. By default NVCC uses a heuristic to find a balance between
  // the maximum number of registers used by a kernel and the parallelism of the kernel.
  // If lots of registers are used the parallelism may suffer. But in our case
  // NVCC gets this wrong and we want to avoid spilling all the time or else
  // the performance is really bad. This essentially tells NVCC to prefer using lots
  // of registers over spilling.
  if (exec_thread_parallel) {
    constexpr int block_size       = 256;
    constexpr int min_block_per_sm = 1;
    auto const num_blocks =
      cudf::util::div_rounding_up_safe(static_cast<std::size_t>(input.size()) * path_data.size(),
                                       static_cast<std::size_t>(block_size));

    get_json_object_kernel_thread_parallel<block_size, min_block_per_sm>
      <<<num_blocks, block_size, 0, stream.value()>>>(input, path_data);
  } else {
    // The optimal values for block_size and min_block_per_sm were found through testing,
    // which are 128-8 or 256-4.
    constexpr int block_size       = 128;
    constexpr int min_block_per_sm = 8;
    auto const num_blocks          = cudf::util::div_rounding_up_safe(
      static_cast<std::size_t>(input.size()) * path_data.size() * cudf::detail::warp_size,
      static_cast<std::size_t>(block_size));

    get_json_object_kernel_warp_parallel<block_size, min_block_per_sm>
      <<<num_blocks, block_size, 0, stream.value()>>>(input, path_data);
  }
}

/**
 * @brief Construct the device vector containing necessary data for the input JSON paths.
 *
 * All JSON paths are processed at once, without stream synchronization, to minimize overhead.
 *
 * A tuple of values are returned, however, only the first element is needed for further kernel
 * launch. The remaining are unused but need to be kept alive as they contains data for later
 * asynchronous host-device memcpy.
 */
std::tuple<std::vector<rmm::device_uvector<path_instruction>>,
           std::unique_ptr<std::vector<std::vector<path_instruction>>>,
           cudf::string_scalar,
           std::string>
construct_path_commands(
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>> const&
    json_paths,
  rmm::cuda_stream_view stream)
{
  // Concatenate all names from path instructions.
  auto h_inst_names = [&] {
    std::size_t length{0};
    for (auto const& instructions : json_paths) {
      for (auto const& [type, name, index] : instructions) {
        if (type == path_instruction_type::NAMED) { length += name.length(); }
      }
    }
    std::string all_names;
    all_names.reserve(length);
    for (auto const& instructions : json_paths) {
      for (auto const& [type, name, index] : instructions) {
        if (type == path_instruction_type::NAMED) { all_names += name; }
      }
    }
    return all_names;
  }();
  auto d_inst_names = cudf::string_scalar(h_inst_names, true, stream);

  std::size_t name_pos{0};
  auto h_path_commands = std::make_unique<std::vector<std::vector<path_instruction>>>();
  h_path_commands->reserve(json_paths.size());

  for (auto const& instructions : json_paths) {
    h_path_commands->emplace_back();
    auto& path_commands = h_path_commands->back();
    path_commands.reserve(instructions.size());

    for (auto const& [type, name, index] : instructions) {
      path_commands.emplace_back(path_instruction{type});

      if (type == path_instruction_type::INDEX) {
        path_commands.back().index = index;
      } else if (type == path_instruction_type::NAMED) {
        path_commands.back().name = cudf::string_view(d_inst_names.data() + name_pos, name.size());
        name_pos += name.size();
      } else if (type != path_instruction_type::WILDCARD) {
        CUDF_FAIL("Invalid path instruction type");
      }
    }
  }

  auto d_path_commands = std::vector<rmm::device_uvector<path_instruction>>{};
  d_path_commands.reserve(h_path_commands->size());
  for (auto const& path_commands : *h_path_commands) {
    d_path_commands.emplace_back(cudf::detail::make_device_uvector_async(
      path_commands, stream, rmm::mr::get_current_device_resource()));
  }

  return {std::move(d_path_commands),
          std::move(h_path_commands),
          std::move(d_inst_names),
          std::move(h_inst_names)};
}

std::pair<std::unique_ptr<std::vector<rmm::device_uvector<path_instruction>>>,
          std::unique_ptr<cudf::string_scalar>>
generate_device_json_paths(
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>> const&
    json_paths,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Concatenate all names from path instructions.
  auto h_inst_names = [&] {
    std::size_t length{0};
    for (auto const& instructions : json_paths) {
      for (auto const& [type, name, index] : instructions) {
        if (type == path_instruction_type::NAMED) { length += name.length(); }
      }
    }
    std::string all_names;
    all_names.reserve(length);
    for (auto const& instructions : json_paths) {
      for (auto const& [type, name, index] : instructions) {
        if (type == path_instruction_type::NAMED) { all_names += name; }
      }
    }
    return all_names;
  }();
  auto d_inst_names = std::make_unique<cudf::string_scalar>(h_inst_names, true, stream);

  std::size_t name_pos{0};
  auto h_path_commands = std::make_unique<std::vector<std::vector<path_instruction>>>();
  h_path_commands->reserve(json_paths.size());

  for (auto const& instructions : json_paths) {
    h_path_commands->emplace_back();
    auto& path_commands = h_path_commands->back();
    path_commands.reserve(instructions.size());

    for (auto const& [type, name, index] : instructions) {
      path_commands.emplace_back(path_instruction{type});

      if (type == path_instruction_type::INDEX) {
        path_commands.back().index = index;
      } else if (type == path_instruction_type::NAMED) {
        path_commands.back().name = cudf::string_view(d_inst_names->data() + name_pos, name.size());
        name_pos += name.size();
      } else if (type != path_instruction_type::WILDCARD) {
        CUDF_FAIL("Invalid path instruction type");
      }
    }
  }

  auto d_path_commands = std::make_unique<std::vector<rmm::device_uvector<path_instruction>>>();
  d_path_commands->reserve(h_path_commands->size());
  for (auto const& path_commands : *h_path_commands) {
    d_path_commands->emplace_back(cudf::detail::make_device_uvector_async(
      path_commands, stream, rmm::mr::get_current_device_resource()));
  }
  stream.synchronize();

  return {std::move(d_path_commands), std::move(d_inst_names)};
}

std::vector<std::unique_ptr<cudf::column>> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<rmm::device_uvector<path_instruction>> const& d_json_paths,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_outputs = d_json_paths.size();
  std::vector<std::unique_ptr<cudf::column>> output;

  // Input is empty or all nulls - just return all null columns.
  if (input.is_empty() || input.size() == input.null_count()) {
    for (std::size_t idx = 0; idx < num_outputs; ++idx) {
      output.emplace_back(std::make_unique<cudf::column>(input.parent(), stream, mr));
    }
    return output;
  }

  auto const d_input_ptr = cudf::column_device_view::create(input.parent(), stream);
  auto const in_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  auto const [max_row_size, sum_row_size] =
    thrust::transform_reduce(rmm::exec_policy(stream),
                             thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(input.size()),
                             cuda::proclaim_return_type<thrust::pair<int64_t, int64_t>>(
                               [in_offsets] __device__(auto const idx) {
                                 auto const size = in_offsets[idx + 1] - in_offsets[idx];
                                 return thrust::pair<int64_t, int64_t>{size, size};
                               }),
                             thrust::pair<int64_t, int64_t>{0, 0},
                             cuda::proclaim_return_type<thrust::pair<int64_t, int64_t>>(
                               [] __device__(auto const& lhs, auto const& rhs) {
                                 return thrust::pair<int64_t, int64_t>{
                                   std::max(lhs.first, rhs.first), lhs.second + rhs.second};
                               }));

  // We will use scratch buffers to store the output strings without knowing their sizes.
  // Since we do not know their sizes, we need to allocate the buffer a bit larger than the input
  // size so that we will not write output strings into an out-of-bound position.
  // Checking out-of-bound needs to be performed in the main kernel to make sure we will not have
  // data corruption.
  auto const scratch_size = [&, max_row_size = max_row_size] {
    // Pad the scratch buffer by an additional size that is a multiple of max row size.
    auto constexpr padding_rows = 10;
    return input.chars_size(stream) + max_row_size * padding_rows;
  }();

  rmm::device_uvector<int8_t> d_has_out_of_bound(num_outputs, stream);
  std::vector<rmm::device_uvector<char>> scratch_buffers;
  std::vector<rmm::device_uvector<thrust::pair<char const*, cudf::size_type>>> out_stringviews;
  std::vector<json_path_processing_data> h_path_data;
  scratch_buffers.reserve(d_json_paths.size());
  out_stringviews.reserve(d_json_paths.size());
  h_path_data.reserve(d_json_paths.size());

  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    // auto const& instructions = json_paths[idx];
    // if (instructions.size() > max_path_depth) { CUDF_FAIL("JSONPath query exceeds maximum
    // depth"); }

    scratch_buffers.emplace_back(rmm::device_uvector<char>(scratch_size, stream));
    out_stringviews.emplace_back(rmm::device_uvector<thrust::pair<char const*, cudf::size_type>>{
      static_cast<std::size_t>(input.size()), stream});

    h_path_data.emplace_back(json_path_processing_data{d_json_paths[idx],
                                                       in_offsets,
                                                       out_stringviews.back().data(),
                                                       scratch_buffers.back().data(),
                                                       d_has_out_of_bound.data() + idx});
  }
  auto d_path_data = cudf::detail::make_device_uvector_async(
    h_path_data, stream, rmm::mr::get_current_device_resource());
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), d_has_out_of_bound.begin(), d_has_out_of_bound.end(), 0);

  // Threshold to decide on using thread parallel or warp parallel algorithms.
  constexpr int64_t AVG_CHAR_BYTES_THRESHOLD = 256;
  auto const exec_thread_parallel =
    (sum_row_size / (input.size() - input.null_count())) < AVG_CHAR_BYTES_THRESHOLD;
  launch_kernel(exec_thread_parallel, *d_input_ptr, d_path_data, stream);

  // Do not use parallel check since we do not have many elements.
  auto h_has_out_of_bound = cudf::detail::make_host_vector_sync(d_has_out_of_bound, stream);
  auto has_no_oob         = std::none_of(
    h_has_out_of_bound.begin(), h_has_out_of_bound.end(), [](auto const val) { return val != 0; });

  // If we didn't see any out-of-bound write, everything is good so far.
  // Just gather the output strings and return.
  if (has_no_oob) {
    for (auto const& out_sview : out_stringviews) {
      output.emplace_back(cudf::make_strings_column(out_sview, stream, mr));
    }
    return output;
  }
  // From here, we had out-of-bound write. Although this is very rare, it may still happen.

  std::vector<std::pair<rmm::device_buffer, cudf::size_type>> out_null_masks_and_null_counts;
  std::vector<std::pair<std::unique_ptr<cudf::column>, int64_t>> out_offsets_and_sizes;
  std::vector<rmm::device_uvector<char>> out_char_buffers;
  std::vector<std::size_t> oob_indices;

  // Check validity from the stored char pointers.
  auto const validator = [] __device__(thrust::pair<char const*, cudf::size_type> const item) {
    return item.first != nullptr;
  };

  // Rebuild the data only for paths that had out of bound write.
  h_path_data.clear();
  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    auto const& out_sview = out_stringviews[idx];

    if (h_has_out_of_bound[idx]) {
      oob_indices.emplace_back(idx);
      output.emplace_back(nullptr);  // just placeholder.

      out_null_masks_and_null_counts.emplace_back(
        cudf::detail::valid_if(out_sview.begin(), out_sview.end(), validator, stream, mr));

      // The string sizes computed in the previous kernel call will be used to allocate a new char
      // buffer to store the output.
      auto const size_it = cudf::detail::make_counting_transform_iterator(
        0,
        cuda::proclaim_return_type<cudf::size_type>(
          [string_pairs = out_sview.data()] __device__(auto const idx) {
            return string_pairs[idx].second;
          }));
      out_offsets_and_sizes.emplace_back(cudf::strings::detail::make_offsets_child_column(
        size_it, size_it + input.size(), stream, mr));
      out_char_buffers.emplace_back(
        rmm::device_uvector<char>(out_offsets_and_sizes.back().second, stream, mr));

      h_path_data.emplace_back(
        json_path_processing_data{d_json_paths[idx],
                                  cudf::detail::offsetalator_factory::make_input_iterator(
                                    out_offsets_and_sizes.back().first->view()),
                                  nullptr /*out_stringviews*/,
                                  out_char_buffers.back().data(),
                                  d_has_out_of_bound.data() + idx});
    } else {
      output.emplace_back(cudf::make_strings_column(out_sview, stream, mr));
    }
  }
  // These buffers are no longer needed.
  scratch_buffers.clear();
  out_stringviews.clear();

  // Push data to the GPU and launch the kernel again.
  d_path_data = cudf::detail::make_device_uvector_async(
    h_path_data, stream, rmm::mr::get_current_device_resource());
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), d_has_out_of_bound.begin(), d_has_out_of_bound.end(), 0);
  launch_kernel(exec_thread_parallel, *d_input_ptr, d_path_data, stream);

  // Check out of bound again to make sure everything looks right.
  h_has_out_of_bound = cudf::detail::make_host_vector_sync(d_has_out_of_bound, stream);
  has_no_oob         = std::none_of(
    h_has_out_of_bound.begin(), h_has_out_of_bound.end(), [](auto const val) { return val != 0; });

  // The last kernel call should not encounter any out-of-bound write.
  // If OOB is still detected, there must be something wrong happened.
  CUDF_EXPECTS(has_no_oob, "Unexpected out-of-bound write in get_json_object kernel.");

  for (auto const idx : oob_indices) {
    output[idx] = cudf::make_strings_column(input.size(),
                                            std::move(out_offsets_and_sizes[idx].first),
                                            out_char_buffers[idx].release(),
                                            out_null_masks_and_null_counts[idx].second,
                                            std::move(out_null_masks_and_null_counts[idx].first));
  }
  return output;
}

}  // namespace detail

std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& input,
  rmm::device_uvector<path_instruction> const& json_path,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return std::move(detail::get_json_object(input, {instructions}, stream, mr).front());
}

std::vector<std::unique_ptr<cudf::column>> get_json_object_multiple_paths(
  cudf::strings_column_view const& input,
  std::vector<rmm::device_uvector<path_instruction>> const& json_paths,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return detail::get_json_object(input, json_paths, stream, mr);
}

std::pair<std::unique_ptr<std::vector<rmm::device_uvector<path_instruction>>>,
          std::unique_ptr<cudf::string_scalar>>
generate_device_json_paths(
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>> const&
    json_paths,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return detail::generate_device_json_paths(json_paths, stream, mr);
}

}  // namespace spark_rapids_jni
