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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/pair.h>
#include <thrust/tuple.h>

namespace spark_rapids_jni {

namespace detail {

/**
 * write JSON style
 */
enum class write_style { RAW, QUOTED, FLATTEN };

/**
 * path instruction
 */
struct path_instruction {
  __device__ inline path_instruction(path_instruction_type _type) : type(_type) {}

  path_instruction_type type;

  // used when type is named type
  cudf::string_view name;

  // used when type is index
  int index{-1};
};

/**
 * JSON generator is used to write out JSON content.
 * Because of get_json_object only outputs JSON object as a whole item,
 * it's no need to store internal state for JSON object when outputing,
 * only need to store internal state for JSON array.
 */
class json_generator {
 public:
  __device__ json_generator(char* _output) : output(_output), output_len(0) {}
  __device__ json_generator() : output(nullptr), output_len(0) {}

  // create a nested child generator based on this parent generator,
  // child generator is a view, parent and child share the same byte array
  __device__ json_generator new_child_generator()
  {
    if (nullptr == output) {
      return json_generator();
    } else {
      return json_generator(output + output_len);
    }
  }

  // write [
  // add an extra comma if needed,
  // e.g.: when JSON content is: [[1,2,3]
  // writing a new [ should result: [[1,2,3],[
  __device__ void write_start_array()
  {
    try_write_comma();

    if (output) { *(output + output_len) = '['; }

    output_len++;
    array_depth++;
    // new array is empty
    is_curr_array_empty = true;
  }

  // write ]
  __device__ void write_end_array()
  {
    if (output) { *(output + output_len) = ']'; }
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
  __device__ inline bool need_comma() { return (array_depth > 0 && !is_curr_array_empty); }

  /**
   * write comma accroding to current generator state
   */
  __device__ void try_write_comma()
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
  __device__ bool copy_current_structure(json_parser& parser)
  {
    // first try add comma
    try_write_comma();

    if (array_depth > 0) { is_curr_array_empty = false; }

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
  __device__ void write_raw(json_parser& parser)
  {
    if (array_depth > 0) { is_curr_array_empty = false; }

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
                                        size_t child_block_len,
                                        bool write_outer_array_tokens)
  {
    bool insert_comma = need_comma();

    if (array_depth > 0) { is_curr_array_empty = false; }

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

  // move memory block forward by specified bytes
  // e.g.:  memory is: 1 2 0 0, begin is 1, len is 2, after moving,
  // memory is: 1 2 1 2.
  // e.g.:  memory is: 1 2 0 0, begin is 1, len is 1, after moving,
  // memory is: 1 1 2 0.
  // Note: should move from end to begin to avoid overwrite buffer
  __device__ void move_forward(char* begin, size_t len, int forward)
  {
    // TODO copy by 8 bytes
    char* pos = begin + len + forward - 1;
    char* e   = begin + forward - 1;
    while (pos > e) {
      *pos = *(pos - forward);
      pos--;
    }
  }

  __device__ void reset() { output_len = 0; }

  __device__ inline size_t get_output_len() const { return output_len; }
  __device__ inline char* get_output_start_position() const { return output; }
  __device__ inline char* get_current_output_position() const { return output + output_len; }

  /**
   * generator may contain trash output, e.g.: generator writes some output,
   * then JSON format is invalid, the previous output becomes trash.
   */
  __device__ inline void set_output_len_zero() { output_len = 0; }

  __device__ inline void set_output_len(size_t len) { output_len = len; }

 private:
  char* output;
  size_t output_len;

  // whether already worte a item in current array
  // used to decide whether add a comma before writing out a new item.
  bool is_curr_array_empty;
  int array_depth = 0;
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
 *
 * This function is rewritten from above commented recursive function.
 * this function is equivalent to the above commented recursive function.
 */
__device__ bool evaluate_path(json_parser& p,
                              json_generator& root_g,
                              write_style root_style,
                              cudf::device_span<path_instruction const> root_path)
{
  // manually maintained context stack in lieu of calling evaluate_path recursively.
  struct context {
    // current token
    json_token token;

    // which case path that this task is from
    int case_path;

    // used to save current generator
    json_generator g;

    write_style style;

    cudf::device_span<path_instruction const> path;

    // is this context task is done
    bool task_is_done;

    // whether written output
    // if dirty > 0, indicates success
    int dirty;

    // for some case paths
    bool is_first_enter;

    // used to save child JSON generator for case path 8
    json_generator child_g;
  };

  // path max depth limitation
  // There is a same constant in JSONUtil.java, keep them consistent when changing
  // Note: Spark-Rapids should guarantee the path depth is less or equal to this limit,
  // or GPU reports cudaErrorIllegalAddress
  constexpr int max_path_depth = 8;

  // define stack; plus 1 indicates root context task needs an extra memory
  context stack[max_path_depth + 1];
  int stack_pos = 0;

  // push context function
  auto push_context = [&stack, &stack_pos](json_token _token,
                                           int _case_path,
                                           json_generator _g,
                                           write_style _style,
                                           cudf::device_span<path_instruction const> _path) {
    // no need to check stack is full
    // because Spark-Rapids already checked maximum length of `path_instruction`
    auto& ctx          = stack[stack_pos];
    ctx.token          = _token;
    ctx.case_path      = _case_path;
    ctx.g              = _g;
    ctx.style          = _style;
    ctx.path           = _path;
    ctx.task_is_done   = false;
    ctx.dirty          = 0;
    ctx.is_first_enter = true;

    stack_pos++;
  };

  // put the first context task
  push_context(p.get_current_token(), -1, root_g, root_style, root_path);

  while (stack_pos > 0) {
    auto& ctx = stack[stack_pos - 1];
    if (!ctx.task_is_done) {
      // task is not done.

      // case (VALUE_STRING, Nil) if style == RawStyle
      // case path 1
      if (json_token::VALUE_STRING == ctx.token && path_is_empty(ctx.path.size()) &&
          ctx.style == write_style::RAW) {
        // there is no array wildcard or slice parent, emit this string without
        // quotes write current string in parser to generator
        ctx.g.write_raw(p);
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
          if (json_token::ERROR == p.get_current_token()) { return false; }
          // push back task
          // add child task
          push_context(p.get_current_token(), 2, ctx.g, ctx.style, {nullptr, 0});
        } else {
          // END_ARRAY
          ctx.task_is_done = true;
        }
      }
      // case (_, Nil)
      // case path 3
      else if (path_is_empty(ctx.path.size())) {
        // general case: just copy the child tree verbatim
        if (!(ctx.g.copy_current_structure(p))) {
          // JSON validation check
          return false;
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
              if (json_token::ERROR == p.get_current_token()) { return false; }

              // skip FIELD_NAME token
              p.next_token();
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return false; }

              // skip value of FIELD_NAME
              if (!p.try_skip_children()) {
                // JSON validation check
                return false;
              }
            }
            ctx.task_is_done = true;
          } else {
            return false;
          }
        } else {
          // below is 1st enter
          ctx.is_first_enter = false;
          // match first mached children with expected name
          bool found_expected_child = false;
          while (json_token::END_OBJECT != p.next_token()) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }

            // need to try more children
            auto match_named = path_match_named(ctx.path);
            auto named       = thrust::get<1>(match_named);
            // current token is FIELD_NAME
            if (p.match_current_field_name(named)) {
              // skip FIELD_NAME token
              p.next_token();
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return false; }

              // meets null token, it's not expected, return false
              if (json_token::VALUE_NULL == p.get_current_token()) { return false; }
              // push sub task; sub task will update the result of path 4
              push_context(p.get_current_token(),
                           4,
                           ctx.g,
                           ctx.style,
                           {ctx.path.data() + 1, ctx.path.size() - 1});
              found_expected_child = true;
              break;
            } else {
              // skip FIELD_NAME token
              p.next_token();
              // JSON validation check
              if (json_token::ERROR == p.get_current_token()) { return false; }

              // current child is not expected, skip current child
              if (!p.try_skip_children()) {
                // JSON validation check
                return false;
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
          ctx.g.write_start_array();
        }

        if (p.next_token() != json_token::END_ARRAY) {
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return false; }
          push_context(p.get_current_token(),
                       5,
                       ctx.g,
                       write_style::FLATTEN,
                       {ctx.path.data() + 2, ctx.path.size() - 2});
        } else {
          ctx.g.write_end_array();
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
          if (json_token::ERROR == p.get_current_token()) { return false; }
          // track the number of array elements and only emit an outer array if
          // we've written more than one element, this matches Hive's behavior
          push_context(p.get_current_token(),
                       6,
                       child_g,
                       next_style,
                       {ctx.path.data() + 1, ctx.path.size() - 1});
        } else {
          char* child_g_start = child_g.get_output_start_position();
          size_t child_g_len  = child_g.get_output_len();
          if (ctx.dirty > 1) {
            // add outer array tokens
            ctx.g.write_child_raw_value(
              child_g_start, child_g_len, /* write_outer_array_tokens */ true);
            ctx.task_is_done = true;
          } else if (ctx.dirty == 1) {
            // remove outer array tokens
            ctx.g.write_child_raw_value(
              child_g_start, child_g_len, /* write_outer_array_tokens */ false);
            ctx.task_is_done = true;
          }  // else do not write anything
        }
      }
      // case (START_ARRAY, Wildcard :: xs)
      // case path 7
      else if (json_token::START_ARRAY == ctx.token &&
               path_match_element(ctx.path, path_instruction_type::WILDCARD)) {
        if (ctx.is_first_enter) {
          ctx.is_first_enter = false;
          ctx.g.write_start_array();
        }
        if (p.next_token() != json_token::END_ARRAY) {
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return false; }

          // wildcards can have multiple matches, continually update the dirty
          // count
          push_context(p.get_current_token(),
                       7,
                       ctx.g,
                       write_style::QUOTED,
                       {ctx.path.data() + 1, ctx.path.size() - 1});
        } else {
          ctx.g.write_end_array();
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
        push_context(p.get_current_token(),
                     8,
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
        if (json_token::ERROR == p.get_current_token()) { return false; }

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
        push_context(
          p.get_current_token(), 9, ctx.g, ctx.style, {ctx.path.data() + 1, ctx.path.size() - 1});
      }
      // case _ =>
      // case path 12
      else {
        if (!p.try_skip_children()) { return false; }
        // default case path, return false for this task
        ctx.dirty        = 0;
        ctx.task_is_done = true;
      }
    } else {
      // current context is done.

      // pop current top context
      stack_pos--;

      // pop parent task
      // update parent task info according to current task result
      if (stack_pos > 0) {
        // peek parent context task
        auto& p_ctx = stack[stack_pos - 1];

        // case (VALUE_STRING, Nil) if style == RawStyle
        // case path 1
        if (1 == ctx.case_path) {
          // never happen
        }
        // path 2: case (START_ARRAY, Nil) if style == FlattenStyle
        // path 5: case (START_ARRAY, Wildcard :: Wildcard :: xs)
        // path 7: case (START_ARRAY, Wildcard :: xs)
        else if (2 == ctx.case_path || 5 == ctx.case_path || 7 == ctx.case_path) {
          // collect result from child task
          p_ctx.dirty += ctx.dirty;
          // copy generator states to parent task;
          p_ctx.g = ctx.g;
        }
        // case (START_OBJECT, Named :: xs)
        // case path 4
        else if (4 == ctx.case_path) {
          p_ctx.dirty = ctx.dirty;
          // copy generator states to parent task;
          p_ctx.g = ctx.g;
        }
        // case (START_ARRAY, Wildcard :: xs) if style != QuotedStyle
        // case path 6
        else if (6 == ctx.case_path) {
          // collect result from child task
          p_ctx.dirty += ctx.dirty;
          // update child generator for parent task
          p_ctx.child_g = ctx.g;
        }
        /* case (START_ARRAY, Index(idx) :: (xs@Wildcard :: _)) */
        // case path 8
        // case (START_ARRAY, Index(idx) :: xs)
        // case path 9
        else if (8 == ctx.case_path || 9 == ctx.case_path) {
          // collect result from child task
          p_ctx.dirty += ctx.dirty;

          // post logic:
          while (p.next_token() != json_token::END_ARRAY) {
            // JSON validation check
            if (json_token::ERROR == p.get_current_token()) { return false; }
            // advance the token stream to the end of the array
            if (!p.try_skip_children()) { return false; }
          }
          // task is done
          p_ctx.task_is_done = true;
          // copy generator states to parent task;
          p_ctx.g = ctx.g;
        }
        // case path 3: case (_, Nil)
        // case path 12: case _ =>
        // others
        else {
          // never happen
        }
      } else {
        // has no parent task, stack is empty, will exit
      }
    }
  }

  // copy output len
  root_g.set_output_len(stack[0].g.get_output_len());
  return stack[0].dirty > 0;
}

rmm::device_uvector<path_instruction> construct_path_commands(
  std::vector<std::tuple<path_instruction_type, std::string, int64_t>> const& instructions,
  cudf::string_scalar const& all_names_scalar,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  int name_pos = 0;

  // construct the path commands
  std::vector<path_instruction> path_commands;
  for (auto const& inst : instructions) {
    auto const& [type, name, index] = inst;
    switch (type) {
      case path_instruction_type::SUBSCRIPT:
      case path_instruction_type::KEY:
        // skip SUBSCRIPT and KEY to save stack size in `evaluate_path`
        break;
      case path_instruction_type::WILDCARD:
        path_commands.emplace_back(path_instruction{path_instruction_type::WILDCARD});
        break;
      case path_instruction_type::INDEX:
        path_commands.emplace_back(path_instruction{path_instruction_type::INDEX});
        path_commands.back().index = index;
        break;
      case path_instruction_type::NAMED:
        path_commands.emplace_back(path_instruction{path_instruction_type::NAMED});
        path_commands.back().name =
          cudf::string_view(all_names_scalar.data() + name_pos, name.size());
        name_pos += name.size();
        break;
      default: CUDF_FAIL("Invalid path instruction type");
    }
  }
  // convert to uvector
  return cudf::detail::make_device_uvector_sync(path_commands, stream, mr);
}

/**
 * @brief Parse a single json string using the provided command buffer
 *
 *
 * @param input The incoming json string
 * @param input_len Size of the incoming json string
 * @param path_commands_ptr The command buffer to be applied to the string.
 * @param path_commands_size The command buffer size.
 * @param out_buf Buffer user to store the results of the query
 *                (nullptr in the size computation step)
 * @param out_buf_size Size of the output buffer
 * @returns A pair containing the result code and the output buffer.
 */
__device__ thrust::pair<bool, size_t> get_json_object_single(
  char const* input,
  cudf::size_type input_len,
  cudf::device_span<path_instruction const> path_commands,
  char* out_buf,
  size_t out_buf_size)
{
  json_parser j_parser(input, input_len);
  j_parser.next_token();
  // JSON validation check
  if (json_token::ERROR == j_parser.get_current_token()) { return {false, 0}; }

  // First pass: preprocess sizes.
  // Second pass: writes output.
  // The generator automatically determines which pass based on `out_buf`.
  // If `out_buf_size` is zero, pass in `nullptr` to avoid generator writing trash output.
  json_generator generator((out_buf_size == 0) ? nullptr : out_buf);

  bool const success = evaluate_path(
    j_parser, generator, write_style::RAW, {path_commands.data(), path_commands.size()});

  if (!success) {
    // generator may contain trash output, e.g.: generator writes some output,
    // then JSON format is invalid, the previous output becomes trash.
    // set output as zero to tell second step
    generator.set_output_len_zero();
  }

  return {success, generator.get_output_len()};
}

/**
 * @brief Kernel for running the JSONPath query.
 *
 * This kernel operates in a 2-pass way. On the first pass it computes the
 * output sizes. On the second pass, it fills in the provided output buffers
 * (chars and validity).
 *
 * @param col Device view of the incoming string
 * @param commands JSONPath command buffer
 * @param output_offsets Buffer used to store the string offsets for the results
 *        of the query
 * @param out_buf Buffer used to store the results of the query
 * @param out_validity Output validity buffer
 * @param out_valid_count Output count of # of valid bits
 * @param options Options controlling behavior
 */
template <int block_size>
__launch_bounds__(block_size) CUDF_KERNEL
  void get_json_object_kernel(cudf::column_device_view col,
                              cudf::device_span<path_instruction const> path_commands,
                              cudf::size_type* d_sizes,
                              cudf::detail::input_offsetalator output_offsets,
                              char* out_buf,
                              cudf::bitmask_type* out_validity,
                              cudf::size_type* out_valid_count)
{
  auto tid          = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();

  cudf::size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffff'ffffu, tid < col.size());
  while (tid < col.size()) {
    bool is_valid               = false;
    cudf::string_view const str = col.element<cudf::string_view>(tid);
    if (str.size_bytes() > 0) {
      char* dst = out_buf != nullptr ? out_buf + output_offsets[tid] : nullptr;
      size_t const dst_size =
        out_buf != nullptr ? output_offsets[tid + 1] - output_offsets[tid] : 0;

      // process one single row
      auto [result, output_size] = get_json_object_single(
        str.data(), str.size_bytes(), {path_commands.data(), path_commands.size()}, dst, dst_size);
      if (result) { is_valid = true; }

      // filled in only during the precompute step. during the compute step, the
      // offsets are fed back in so we do -not- want to write them out
      if (out_buf == nullptr) { d_sizes[tid] = static_cast<cudf::size_type>(output_size); }
    } else {
      // valid JSON length is always greater than 0
      // if `str` size len is zero, output len is 0 and `is_valid` is false
      if (out_buf == nullptr) { d_sizes[tid] = 0; }
    }

    // validity filled in only during the output step
    if (out_validity != nullptr) {
      uint32_t mask = __ballot_sync(active_threads, is_valid);
      // 0th lane of the warp writes the validity
      if (!(tid % cudf::detail::warp_size)) {
        out_validity[cudf::word_index(tid)] = mask;
        warp_valid_count += __popc(mask);
      }
    }

    tid += stride;
    active_threads = __ballot_sync(active_threads, tid < col.size());
  }

  // sum the valid counts across the whole block
  if (out_valid_count != nullptr) {
    cudf::size_type block_valid_count =
      cudf::detail::single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
    if (threadIdx.x == 0) { atomicAdd(out_valid_count, block_valid_count); }
  }
}

std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<std::tuple<path_instruction_type, std::string, int64_t>> const& instructions,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);

  // get a string buffer to store all the names and convert to device
  std::string all_names;
  for (auto const& inst : instructions) {
    all_names += std::get<1>(inst);
  }
  cudf::string_scalar all_names_scalar(all_names, true, stream);
  // parse the json_path into a command buffer
  auto path_commands = construct_path_commands(
    instructions, all_names_scalar, stream, rmm::mr::get_current_device_resource());

  // compute output sizes
  auto sizes = rmm::device_uvector<cudf::size_type>(
    input.size(), stream, rmm::mr::get_current_device_resource());
  auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(input.offsets());

  constexpr int block_size = 512;
  cudf::detail::grid_1d const grid{input.size(), block_size};
  auto d_input_ptr = cudf::column_device_view::create(input.parent(), stream);
  // preprocess sizes (returned in the offsets buffer)
  get_json_object_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *d_input_ptr, path_commands, sizes.data(), d_offsets, nullptr, nullptr, nullptr);

  // convert sizes to offsets
  auto [offsets, output_size] =
    cudf::strings::detail::make_offsets_child_column(sizes.begin(), sizes.end(), stream, mr);
  d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // allocate output string column
  rmm::device_uvector<char> chars(output_size, stream, mr);

  // potential optimization : if we know that all outputs are valid, we could
  // skip creating the validity mask altogether
  rmm::device_buffer validity =
    cudf::detail::create_null_mask(input.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  // compute results
  rmm::device_scalar<cudf::size_type> d_valid_count{0, stream};

  get_json_object_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *d_input_ptr,
      path_commands,
      sizes.data(),
      d_offsets,
      chars.data(),
      static_cast<cudf::bitmask_type*>(validity.data()),
      d_valid_count.data());

  return make_strings_column(input.size(),
                             std::move(offsets),
                             chars.release(),
                             input.size() - d_valid_count.value(stream),
                             std::move(validity));
}

}  // namespace detail

std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<std::tuple<path_instruction_type, std::string, int64_t>> const& instructions,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return detail::get_json_object(input, instructions, stream, mr);
}

}  // namespace spark_rapids_jni
