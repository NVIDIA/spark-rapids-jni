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

#include "from_json.hpp"
#include "get_json_object.hpp"
#include "json_parser.cuh"

// #include <cudf_test/debug_utilities.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/json.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <numeric>
#include <unordered_set>

namespace spark_rapids_jni {

namespace detail {

namespace test {

/**
 * @brief JSON style to write.
 */
enum class write_style : int8_t { RAW, QUOTED, FLATTEN };

/**
 * @brief Instruction along a JSON path.
 */
struct path_instruction {
  __device__ inline path_instruction(path_instruction_type _type) : type(_type) {}

  // used when type is named type
  cudf::string_view name;

  // used when type is index
  int index{-1};

  path_instruction_type type;
};

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
  __device__ void write_start_array(char* out_begin, char element_delimiter)
  {
    try_write_comma(out_begin, element_delimiter);

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
  __device__ void try_write_comma(char* out_begin, char element_delimiter)
  {
    if (need_comma()) {
      // in array context and writes first item
      out_begin[offset + output_len] = element_delimiter;
      output_len++;
    }
  }

  /**
   * copy current structure when parsing. If current token is start
   * object/array, then copy to corresponding matched end object/array. return
   * false if JSON format is invalid return true if JSON format is valid
   */
  __device__ bool copy_current_structure(json_parser& parser,
                                         char* out_begin,
                                         char element_delimiter)
  {
    // first try add comma
    try_write_comma(out_begin, element_delimiter);

    if (array_depth > 0) { is_curr_array_empty = false; }

    // printf("parser line %d\n", __LINE__);

    auto [b, copy_len] = parser.copy_current_structure(out_begin + offset + output_len);
    output_len += copy_len;
    return b;
  }

  static __device__ cudf::size_type write_quote(char* out, bool keep_quotes)
  {
    if (!keep_quotes) { return 0; }
    *out = '"';
    return 1;
  }

  /**
   * Get current text from JSON parser and then write the text
   * Note: Because JSON strings contains '\' to do escape,
   * JSON parser should do unescape to remove '\' and JSON parser
   * then can not return a pointer and length pair (char *, len),
   * For number token, JSON parser can return a pair (char *, len)
   */
  __device__ void write_raw(json_parser& parser, char* out_begin, bool keep_quotes)
  {
    if (array_depth > 0) { is_curr_array_empty = false; }

    output_len += write_quote(out_begin + offset + output_len, keep_quotes);
    output_len += parser.write_unescaped_text(out_begin + offset + output_len);
    output_len += write_quote(out_begin + offset + output_len, keep_quotes);
  }

  __device__ void write_null_placeholder(char* out_begin, char null)
  {
    out_begin[offset + output_len] = null;
    output_len += 1;
    is_curr_array_empty = false;
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

  bool is_in_array;

  // for some case paths
  bool is_first_enter;

  // is this context task is done
  bool task_is_done;
};

/**
 * @brief Parse a single json string using the provided command buffer.
 *
 * @param p The JSON parser for input string
 * @param path_commands The command buffer to be applied to the string
 * TODO: update
 * @param out_buf Buffer user to store the string resulted from the query
 * @param max_path_depth_exceeded A marker to record if the maximum path depth has been reached
 *        during parsing the input string
 * @return A pair containing the result code and the output size
 */
__device__ thrust::pair<bool, cudf::size_type> evaluate_path(
  json_parser& p,
  cudf::device_span<path_instruction const> path_commands,
  cudf::type_id path_type_id,
  bool keep_quotes,
  char element_delimiter,
  char null_placeholder,
  char* out_buf,
  int8_t* max_path_depth_exceeded)
{
  p.next_token();
  if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

  // Define stack; plus 1 indicates root context task needs an extra memory.
  context stack[MAX_JSON_PATH_DEPTH + 1];
  int stack_size = 0;

  auto const push_context = [&](evaluation_case_path _case_path,
                                json_generator _g,
                                write_style _style,
                                cudf::device_span<path_instruction const> _path,
                                bool is_in_array) {
    if (stack_size > MAX_JSON_PATH_DEPTH) {
      *max_path_depth_exceeded = 1;
      // Because no more context is pushed, the evaluation output should be wrong.
      // But that is not important, since we will throw exception after the kernel finishes.
      return;
    }
    auto& ctx          = stack[stack_size++];
    ctx.g              = std::move(_g);
    ctx.path           = std::move(_path);
    ctx.dirty          = 0;
    ctx.case_path      = _case_path;
    ctx.token          = p.get_current_token();
    ctx.style          = _style;
    ctx.is_in_array    = is_in_array;
    ctx.is_first_enter = true;
    ctx.task_is_done   = false;
  };

  push_context(
    evaluation_case_path::INVALID, json_generator{}, write_style::RAW, path_commands, false);

  while (stack_size > 0) {
    auto& ctx = stack[stack_size - 1];
    if (!ctx.task_is_done) {
      // case (VALUE_STRING, Nil) if style == RawStyle
      // case path 1
      if (json_token::VALUE_STRING == ctx.token && path_is_empty(ctx.path.size())) {
        // there is no array wildcard or slice parent, emit this string without
        // quotes write current string in parser to generator
        ctx.g.try_write_comma(out_buf, element_delimiter);
        ctx.g.write_raw(p, out_buf, keep_quotes);
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
                       {nullptr, 0},
                       true);
        } else {
          // END_ARRAY
          ctx.task_is_done = true;
        }
      }
      // case (_, Nil)
      // case path 3
      else if (path_is_empty(ctx.path.size())) {
        // printf("path is empty, path type = %d, token = %d\n",
        //        (int)path_type_id,
        //        (int)p.get_current_token());

        // If this is a struct column, we only need to check to see if there exists a struct.
        if (path_type_id == cudf::type_id::STRUCT || path_type_id == cudf::type_id::LIST) {
          if (path_type_id == cudf::type_id::STRUCT &&
              p.get_current_token() != json_token::START_OBJECT) {
            return {false, 0};
          }
          if (path_type_id == cudf::type_id::LIST &&
              p.get_current_token() != json_token::START_ARRAY) {
            return {false, 0};
          }

          if (path_type_id == cudf::type_id::STRUCT) {
            // Or copy current structure?
            if (!p.try_skip_children()) { return {false, 0}; }
          } else if (!(ctx.g.copy_current_structure(p, nullptr, ','))) {
            // not copy only if there is struct?
            return {false, 0};
          }

          // Just write anything into the output, to mark the output as a non-null row.
          // Such output will be discarded anyway.
          ctx.g.write_start_array(out_buf, element_delimiter);
        } else if (!(ctx.g.copy_current_structure(p, out_buf, element_delimiter))) {
          return {false, 0};
        }
        ctx.dirty        = 1;
        ctx.task_is_done = true;
      }
      // case (START_OBJECT, Named :: xs)
      // case path 4
      else if (json_token::START_OBJECT == ctx.token &&
               thrust::get<0>(path_match_named(ctx.path))) {
        // printf("start object\n");

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
                           {ctx.path.data() + 1, ctx.path.size() - 1},
                           ctx.is_in_array);
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
            if (ctx.is_in_array) {
              ctx.g.try_write_comma(out_buf, element_delimiter);
              ctx.g.write_null_placeholder(out_buf, null_placeholder);
              ctx.dirty = 1;
            } else {
              ctx.dirty = false;
            }
            ctx.task_is_done = true;
          }
        }
      }
      // case (START_ARRAY, Wildcard :: xs)
      // case path 7
      else if (json_token::START_ARRAY == ctx.token &&
               path_match_element(ctx.path, path_instruction_type::WILDCARD)) {
        // printf("array *\n");

        if (ctx.is_first_enter) {
          ctx.is_first_enter = false;
          ctx.g.write_first_start_array_without_output();
        }
        if (p.next_token() != json_token::END_ARRAY) {
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return {false, 0}; }

          // wildcards can have multiple matches, continually update the dirty
          // count
          push_context(evaluation_case_path::START_ARRAY___MATCHED_WILDCARD,
                       ctx.g,
                       write_style::QUOTED,
                       {ctx.path.data() + 1, ctx.path.size() - 1},
                       true);
        } else {
          // ctx.g.write_end_array(out_buf);
          ctx.task_is_done = true;
        }
      }
      // case _ =>
      // case path 12
      else {
        // printf("get obj line %d\n", __LINE__);

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
  bool keep_quotes;
  cudf::type_id type_id;
};

/**
 * @brief Kernel for running the JSONPath query, in which one input row is processed by entire
 * warp (or multiple warps) of threads.
 *
 * The number of warps processing each row is computed as `ceil(num_paths / warp_size)`.
 *
 * We explicitly set a value for `min_block_per_sm` parameter in the launch bounds to avoid
 * spilling from the kernel itself. By default NVCC uses a heuristic to find a balance between
 * the maximum number of registers used by a kernel and the parallelism of the kernel.
 * If lots of registers are used the parallelism may suffer. But in our case NVCC gets this wrong
 * and we want to avoid spilling all the time or else the performance is really bad. This
 * essentially tells NVCC to prefer using lots of registers over spilling.
 *
 * TODO update
 * @param input The input JSON strings stored in a strings column
 * @param path_data Array containing all path data
 * @param num_threads_per_row Number of threads processing each input row
 * @param max_path_depth_exceeded A marker to record if the maximum path depth has been reached
 *        during parsing the input string
 */
template <int block_size, int min_block_per_sm>
__launch_bounds__(block_size, min_block_per_sm) CUDF_KERNEL
  void get_json_object_kernel(cudf::column_device_view input,
                              cudf::device_span<json_path_processing_data> path_data,
                              char element_delimiter,
                              char null_placeholder,
                              bool allow_leading_zero_numbers,
                              bool allow_non_numeric_numbers,
                              bool allow_unquoted_control_chars,
                              std::size_t num_threads_per_row,
                              int8_t* max_path_depth_exceeded)
{
  auto const tidx    = cudf::detail::grid_1d::global_thread_id();
  auto const row_idx = tidx / num_threads_per_row;
  if (row_idx >= input.size()) { return; }

  auto const path_idx = tidx % num_threads_per_row;
  if (path_idx >= path_data.size()) { return; }

  auto const& path         = path_data[path_idx];
  char* const dst          = path.out_buf + path.offsets[row_idx];
  bool is_valid            = false;
  cudf::size_type out_size = 0;

  auto const str = input.element<cudf::string_view>(row_idx);
  if (str.size_bytes() > 0) {
    json_parser p{char_range{str}};
    p.set_allow_leading_zero_numbers(allow_leading_zero_numbers);
    p.set_allow_non_numeric_numbers(allow_non_numeric_numbers);
    p.set_allow_unquoted_control_chars(allow_unquoted_control_chars);
    thrust::tie(is_valid, out_size) = evaluate_path(p,
                                                    path.path_commands,
                                                    path.type_id,
                                                    path.keep_quotes,
                                                    element_delimiter,
                                                    null_placeholder,
                                                    dst,
                                                    max_path_depth_exceeded);

    // We did not terminate the `evaluate_path` function early to reduce complexity of the code.
    // Instead, if max depth was encountered, we've just continued the evaluation until here
    // then discard the output entirely.
    if (p.max_nesting_depth_exceeded()) {
      *max_path_depth_exceeded = 1;
      return;
    }

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
 * @brief A utility class to launch the main kernel.
 */
struct kernel_launcher {
  static void exec(cudf::column_device_view const& input,
                   cudf::device_span<json_path_processing_data> path_data,
                   char element_delimiter,
                   char null_placeholder,
                   bool allow_leading_zero_numbers,
                   bool allow_non_numeric_numbers,
                   bool allow_unquoted_control_chars,
                   int8_t* max_path_depth_exceeded,
                   rmm::cuda_stream_view stream)
  {
    // The optimal values for block_size and min_block_per_sm were found through testing,
    // which are either 128-8 or 256-4. The pair 128-8 seems a bit better.
    static constexpr int block_size       = 128;
    static constexpr int min_block_per_sm = 8;

    // The number of threads for processing one input row is at least one warp.
    auto const num_threads_per_row =
      cudf::util::div_rounding_up_safe(path_data.size(),
                                       static_cast<std::size_t>(cudf::detail::warp_size)) *
      cudf::detail::warp_size;
    auto const num_blocks = cudf::util::div_rounding_up_safe(num_threads_per_row * input.size(),
                                                             static_cast<std::size_t>(block_size));
    get_json_object_kernel<block_size, min_block_per_sm>
      <<<num_blocks, block_size, 0, stream.value()>>>(input,
                                                      path_data,
                                                      element_delimiter,
                                                      null_placeholder,
                                                      allow_leading_zero_numbers,
                                                      allow_non_numeric_numbers,
                                                      allow_unquoted_control_chars,
                                                      num_threads_per_row,
                                                      max_path_depth_exceeded);
  }
};

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
  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>> const&
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

int64_t calc_scratch_size(cudf::strings_column_view const& input,
                          cudf::detail::input_offsetalator const& in_offsets,
                          rmm::cuda_stream_view stream)
{
  auto const max_row_size = thrust::transform_reduce(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(input.size()),
    cuda::proclaim_return_type<int64_t>(
      [in_offsets] __device__(auto const idx) { return in_offsets[idx + 1] - in_offsets[idx]; }),
    int64_t{0},
    thrust::maximum{});

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
  return scratch_size;
}

/**
 * @brief Error handling using error markers gathered after kernel launch.
 *
 * If the input JSON has nesting depth exceeds the maximum allowed value, an exception will be
 * thrown as it is unacceptable. Otherwise, out of bound write is checked and returned.
 *
 * @param error_check The array of markers to check for error
 * @return A boolean value indicating if there is any out of bound write
 */
bool check_error(cudf::detail::host_vector<int8_t> const& error_check)
{
  // The last value is to mark if nesting depth has exceeded.
  CUDF_EXPECTS(error_check.back() == 0,
               "The processed input has nesting depth exceeds depth limit.");

  // Do not use parallel check since we do not have many elements.
  // The last element is not related, but its value is already `0` thus just check until
  // the end of the array for simplicity.
  return std::none_of(
    error_check.cbegin(), error_check.cend(), [](auto const val) { return val != 0; });
}

std::vector<std::unique_ptr<cudf::column>> get_json_object_batch(
  cudf::column_device_view const& input,
  cudf::detail::input_offsetalator const& in_offsets,
  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>> const&
    json_paths,
  std::vector<cudf::type_id> const& type_ids,
  std::vector<std::size_t> const& output_ids,
  std::unordered_set<std::size_t> const& keep_quotes,
  char element_delimiter,
  char null_placeholder,
  int64_t scratch_size,
  bool allow_leading_zero_numbers,
  bool allow_non_numeric_numbers,
  bool allow_unquoted_control_chars,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const [d_json_paths, h_json_paths, d_inst_names, h_inst_names] =
    construct_path_commands(json_paths, stream);

  auto const num_outputs = json_paths.size();
  std::vector<std::unique_ptr<cudf::column>> output;

  // The error check array contains markers denoting if there is any out-of-bound write occurs
  // (first `num_outputs` elements), or if the nesting depth exceeded its limits (the last element).
  rmm::device_uvector<int8_t> d_error_check(num_outputs + 1, stream);
  auto const d_max_path_depth_exceeded = d_error_check.data() + num_outputs;

  std::vector<rmm::device_uvector<char>> scratch_buffers;
  std::vector<rmm::device_uvector<thrust::pair<char const*, cudf::size_type>>> out_stringviews;
  std::vector<json_path_processing_data> h_path_data;
  scratch_buffers.reserve(json_paths.size());
  out_stringviews.reserve(json_paths.size());
  h_path_data.reserve(json_paths.size());

  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    auto const& path = json_paths[idx];
    if (path.size() > MAX_JSON_PATH_DEPTH) {
      CUDF_FAIL("JSON Path has depth exceeds the maximum allowed value.");
    }

    scratch_buffers.emplace_back(rmm::device_uvector<char>(scratch_size, stream));
    out_stringviews.emplace_back(rmm::device_uvector<thrust::pair<char const*, cudf::size_type>>{
      static_cast<std::size_t>(input.size()), stream});

    // printf("idx: %d, output_ids[idx]: %d\n", (int)idx, (int)output_ids[idx]);
    // printf("keep_quotes.find(output_ids[idx]) != keep_quotes.end(): %d\n",
    //        (int)(keep_quotes.find(output_ids[idx]) != keep_quotes.end()));
    // fflush(stdout);

    h_path_data.emplace_back(
      json_path_processing_data{d_json_paths[idx],
                                in_offsets,
                                out_stringviews.back().data(),
                                scratch_buffers.back().data(),
                                d_error_check.data() + idx,
                                keep_quotes.find(output_ids[idx]) != keep_quotes.end(),
                                type_ids[idx]});
  }
  auto d_path_data = cudf::detail::make_device_uvector_async(
    h_path_data, stream, rmm::mr::get_current_device_resource());
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), d_error_check.begin(), d_error_check.end(), 0);

  kernel_launcher::exec(input,
                        d_path_data,
                        element_delimiter,
                        null_placeholder,
                        allow_leading_zero_numbers,
                        allow_non_numeric_numbers,
                        allow_unquoted_control_chars,
                        d_max_path_depth_exceeded,
                        stream);
  auto h_error_check = cudf::detail::make_host_vector_sync(d_error_check, stream);
  auto has_no_oob    = check_error(h_error_check);

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

    if (h_error_check[idx]) {
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
                                  d_error_check.data() + idx,
                                  keep_quotes.find(output_ids[idx]) != keep_quotes.end(),
                                  type_ids[idx]});
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
    rmm::exec_policy(stream), d_error_check.begin(), d_error_check.end(), 0);
  kernel_launcher::exec(input,
                        d_path_data,
                        element_delimiter,
                        null_placeholder,
                        allow_leading_zero_numbers,
                        allow_non_numeric_numbers,
                        allow_unquoted_control_chars,
                        d_max_path_depth_exceeded,
                        stream);
  h_error_check = cudf::detail::make_host_vector_sync(d_error_check, stream);
  has_no_oob    = check_error(h_error_check);

  // The last kernel call should not encounter any out-of-bound write.
  // If OOB is still detected, there must be something wrong happened.
  CUDF_EXPECTS(has_no_oob, "Unexpected out-of-bound write in get_json_object kernel.");

  for (std::size_t idx = 0; idx < oob_indices.size(); ++idx) {
    auto const out_idx = oob_indices[idx];
    output[out_idx] =
      cudf::make_strings_column(input.size(),
                                std::move(out_offsets_and_sizes[idx].first),
                                out_char_buffers[idx].release(),
                                out_null_masks_and_null_counts[idx].second,
                                std::move(out_null_masks_and_null_counts[idx].first));
  }
  return output;
}

// TODO: update docs for keep_quotes
std::vector<std::unique_ptr<cudf::column>> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const&
    json_paths,
  std::vector<cudf::type_id> const& type_ids,
  std::unordered_set<std::size_t> const& keep_quotes,
  char element_delimiter,
  char null_placeholder,
  int64_t memory_budget_bytes,
  int32_t parallel_override,
  bool allow_leading_zero_numbers,
  bool allow_non_numeric_numbers,
  bool allow_unquoted_control_chars,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_outputs = json_paths.size();

  // Input is empty or all nulls - just return all null columns.
  if (input.is_empty() || input.size() == input.null_count()) {
    std::vector<std::unique_ptr<cudf::column>> output;
    for (std::size_t idx = 0; idx < num_outputs; ++idx) {
      output.emplace_back(std::make_unique<cudf::column>(input.parent(), stream, mr));
    }
    return output;
  }

  std::vector<std::size_t> sorted_indices(json_paths.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);  // Fill with 0, 1, 2, ...

  // Sort indices based on the corresponding paths.
  std::sort(sorted_indices.begin(), sorted_indices.end(), [&json_paths](size_t i, size_t j) {
    return json_paths[i] < json_paths[j];
  });

  auto const in_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  auto const scratch_size = calc_scratch_size(input, in_offsets, stream);
  if (memory_budget_bytes <= 0 && parallel_override <= 0) {
    parallel_override = static_cast<int>(sorted_indices.size());
  }
  auto const d_input_ptr = cudf::column_device_view::create(input.parent(), stream);
  std::vector<std::unique_ptr<cudf::column>> output(num_outputs);

  // TODO: reserve
  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>> batch;
  std::vector<cudf::type_id> batch_type_ids;
  std::vector<std::size_t> output_ids;

  std::size_t starting_path = 0;
  while (starting_path < num_outputs) {
    std::size_t at = starting_path;
    batch.resize(0);
    batch_type_ids.resize(0);
    output_ids.resize(0);
    if (parallel_override > 0) {
      int count = 0;
      while (at < num_outputs && count < parallel_override) {
        auto output_location = sorted_indices[at];
        batch.emplace_back(json_paths[output_location]);
        batch_type_ids.push_back(type_ids[output_location]);
        output_ids.push_back(output_location);
        at++;
        count++;
      }
    } else {
      long budget = 0;
      while (at < num_outputs && budget < memory_budget_bytes) {
        auto output_location = sorted_indices[at];
        batch.emplace_back(json_paths[output_location]);
        batch_type_ids.push_back(type_ids[output_location]);
        output_ids.push_back(output_location);
        at++;
        budget += scratch_size;
      }
    }
    auto tmp = get_json_object_batch(*d_input_ptr,
                                     in_offsets,
                                     batch,
                                     batch_type_ids,
                                     output_ids,
                                     keep_quotes,
                                     element_delimiter,
                                     null_placeholder,
                                     scratch_size,
                                     allow_leading_zero_numbers,
                                     allow_non_numeric_numbers,
                                     allow_unquoted_control_chars,
                                     stream,
                                     mr);
    for (std::size_t i = 0; i < tmp.size(); i++) {
      std::size_t out_i = output_ids[i];
      output[out_i]     = std::move(tmp[i]);
    }
    starting_path = at;
  }
  return output;
}

}  // namespace test

void travel_path(
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>>& paths,
  std::vector<std::tuple<path_instruction_type, std::string, int32_t>>& current_path,
  std::vector<cudf::type_id>& type_ids,
  std::unordered_set<std::size_t>& keep_quotes,
  bool& has_list_type,
  bool parent_is_list,
  std::string const& name,
  json_schema_element const& column_schema)
{
  bool popped{false};
  current_path.emplace_back(path_instruction_type::NAMED, name, -1);
  if (column_schema.child_types.size() == 0) {  // leaf of the schema
    if (cudf::is_fixed_width(column_schema.type)) {
      // TODO: comment
      keep_quotes.insert(paths.size());
    }
    // printf("column_schema type: %d\n", static_cast<int>(column_schema.type.id()));
    paths.push_back(current_path);  // this will copy
    type_ids.push_back(column_schema.type.id());
  } else {
    if (column_schema.type.id() == cudf::type_id::STRUCT) {
      type_ids.push_back(column_schema.type.id());

      // STRUCT directly under array does not have name field.
      if (parent_is_list) {
        popped = true;
        current_path.pop_back();  // remove the last NAMED instruction.
      }
      paths.push_back(current_path);  // this will copy
                                      // printf("column_schema type: STRUCT\n");
      for (auto const& [child_name, child_schema] : column_schema.child_types) {
        travel_path(paths,
                    current_path,
                    type_ids,
                    keep_quotes,
                    has_list_type,
                    false /*parent_is_list*/,
                    child_name,
                    child_schema);
      }
    } else if (column_schema.type.id() == cudf::type_id::LIST) {
      // printf("column_schema type: LIST\n");

      CUDF_EXPECTS(column_schema.child_types.size() == 1, "TODO");
      has_list_type = true;

      bool has_struct_child{false};
      for (auto const& [child_name, child_schema] : column_schema.child_types) {
        if (child_schema.type.id() == cudf::type_id::STRUCT) {
          has_struct_child = true;
          break;
        }
      }

      // TODO: is this needed, if there is no struct child?
      if (has_struct_child) {
        paths.push_back(current_path);  // this will copy
        type_ids.push_back(column_schema.type.id());
      }

      current_path.emplace_back(path_instruction_type::WILDCARD, "*", -1);

      // Only add a path name if this column is not under a list type.
      if (has_struct_child) {
        for (auto const& [child_name, child_schema] : column_schema.child_types) {
          travel_path(paths,
                      current_path,
                      type_ids,
                      keep_quotes,
                      has_list_type,
                      true /*parent_is_list*/,
                      child_name,
                      child_schema);
        }
      } else {
        auto const child_type = column_schema.child_types.front().second.type;
        if (cudf::is_fixed_width(child_type)) { keep_quotes.insert(paths.size()); }
        paths.push_back(current_path);  // this will copy
        type_ids.push_back(child_type.id());
      }

      current_path.pop_back();  // remove WILDCARD

    } else {
      // TODO
      CUDF_FAIL("Unsupported type");
    }
  }
  // if (column_schema.type.id() != cudf::type_id::STRUCT || !has_list_type) {
  if (!popped) { current_path.pop_back(); }
}

std::tuple<std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>>,
           std::vector<cudf::type_id>,
           std::unordered_set<std::size_t>,
           bool>
flatten_schema_to_paths(std::vector<std::pair<std::string, json_schema_element>> const& schema)
{
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> paths;
  std::vector<cudf::type_id> type_ids;
  std::unordered_set<std::size_t> keep_quotes;
  bool has_list_type{false};

  std::vector<std::tuple<path_instruction_type, std::string, int32_t>> current_path;
  std::for_each(schema.begin(), schema.end(), [&](auto const& kv) {
    travel_path(paths,
                current_path,
                type_ids,
                keep_quotes,
                has_list_type,
                false /*parent_is_list*/,
                kv.first,
                kv.second);
  });

  return {std::move(paths), std::move(type_ids), std::move(keep_quotes), has_list_type};
}

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> extract_lists(
  std::unique_ptr<cudf::column>& input,
  json_schema_element const& column_schema,
  char element_delimiter,
  char null_placeholder,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (column_schema.type.id() == cudf::type_id::STRUCT) {
    std::unique_ptr<cudf::column> offsets{nullptr};
    std::vector<std::unique_ptr<cudf::column>> new_children;
    cudf::size_type num_child_rows{-1};
    auto children = std::move(input->release().children);
    for (std::size_t child_idx = 0; child_idx < children.size(); ++child_idx) {
      auto& child = children[child_idx];
      auto [new_child_offsets, new_child] =
        extract_lists(child,
                      column_schema.child_types[child_idx].second,
                      element_delimiter,
                      null_placeholder,
                      stream,
                      mr);
      if (num_child_rows < 0) { num_child_rows = new_child->size(); }
      if (num_child_rows != new_child->size()) {
        // printf("num_child_rows != new_child->size(): %d != %d\n",
        // (int)num_child_rows,
        // (int)new_child->size());
      }
      CUDF_EXPECTS(num_child_rows == new_child->size(), "num_child_rows != new_child->size()");

      if (!offsets) { offsets = std::move(new_child_offsets); }
      new_children.emplace_back(std::move(new_child));
    }

    // return cudf::make_structs_column(
    //             num_child_rows, std::move(children), null_count, std::move(*null_mask), stream,
    //             mr);
    // TODO: fix null mask
    return {std::move(offsets),
            cudf::make_structs_column(num_child_rows, std::move(new_children), 0, {}, stream, mr)};
  }

  // printf("before split:\n");
  // cudf::test::print(input->view());

  auto tmp           = cudf::strings::split_record(cudf::strings_column_view{input->view()},
                                         cudf::string_scalar{std::string{element_delimiter}},
                                         -1,
                                         stream,
                                         mr);
  auto split_content = tmp->release();

  if (input->size() == input->null_count()) {
    return {std::move(split_content.children[cudf::lists_column_view::offsets_column_index]),
            std::move(split_content.children[cudf::lists_column_view::child_column_index])};
  }

  auto const child_cv = split_content.children[cudf::lists_column_view::child_column_index]->view();
  auto const child_strview = cudf::strings_column_view{child_cv};

  // printf("child_cv:\n");
  // cudf::test::print(child_cv);

  // Convert a row index into an invalid value (-1) if that row contains a null placeholder.
  // Don't care about nulls in the child column, as they will be gathered to the output.
  auto const gather_it = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [null_placeholder,
       offsets = child_strview.offsets().begin<cudf::size_type>(),
       chars   = child_strview.chars_begin(stream)] __device__(cudf::size_type idx) {
        if (offsets[idx + 1] - offsets[idx] == 1) {
          return chars[offsets[idx]] == null_placeholder ? -1 : idx;
        }
        return idx;
      }));

  // TODO: report issue when the input is strings column has null == size
  auto out_child = std::move(cudf::detail::gather(cudf::table_view{{child_cv}},
                                                  gather_it,
                                                  gather_it + child_cv.size(),
                                                  cudf::out_of_bounds_policy::NULLIFY,
                                                  stream,
                                                  mr)
                               ->release()
                               .front());
  // printf("out_child:\n");
  // cudf::test::print(out_child->view());

  if (out_child->null_count() == 0) { out_child->set_null_mask(rmm::device_buffer{}, 0); }

  // auto split_content =
  //   cudf::strings::split_record(cudf::strings_column_view{input->view()},
  //                               cudf::string_scalar{std::string{element_delimiter}},
  //                               -1,
  //                               stream,
  //                               mr)
  //     ->release();
  // printf("after split:\n");
  // cudf::test::print(tmp->view());

  return {std::move(split_content.children[cudf::lists_column_view::offsets_column_index]),
          std::move(out_child)};
}

void assemble_column(std::size_t& column_order,
                     std::vector<std::unique_ptr<cudf::column>>& output,
                     std::vector<std::unique_ptr<cudf::column>>& read_columns,
                     std::string const& name,
                     json_schema_element const& column_schema,
                     char element_delimiter,
                     char null_placeholder,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  if (column_schema.child_types.size() == 0) {  // leaf of the schema
    output.emplace_back(std::move(read_columns[column_order]));
    ++column_order;
  } else {
    if (column_schema.type.id() == cudf::type_id::STRUCT) {
      auto const null_count = read_columns[column_order]->null_count();
      auto const null_mask  = std::move(read_columns[column_order]->release().null_mask);
      ++column_order;

      std::vector<std::unique_ptr<cudf::column>> children;
      for (auto const& [child_name, child_schema] : column_schema.child_types) {
        assemble_column(column_order,
                        children,
                        read_columns,
                        child_name,
                        child_schema,
                        element_delimiter,
                        null_placeholder,
                        stream,
                        mr);
      }

      // TODO: generate null mask from input.
      auto const num_rows = children.front()->size();
      output.emplace_back(cudf::make_structs_column(
        num_rows, std::move(children), null_count, std::move(*null_mask), stream, mr));
    } else if (column_schema.type.id() == cudf::type_id::LIST) {
      // TODO: split LIST into child column
      // For now, just output as a strings column.

      bool has_struct_child{false};
      for (auto const& [child_name, child_schema] : column_schema.child_types) {
        if (child_schema.type.id() == cudf::type_id::STRUCT) {
          has_struct_child = true;
          break;
        }
      }

      auto const num_rows   = read_columns[column_order]->size();
      auto const null_count = read_columns[column_order]->null_count();
      std::unique_ptr<rmm::device_buffer> null_mask{nullptr};

      // printf("num rows: %d\n", num_rows);
      // If there is struct child, ..... TODO
      if (has_struct_child) {
        null_mask = std::move(read_columns[column_order]->release().null_mask);
        ++column_order;
      }

      std::vector<std::unique_ptr<cudf::column>> children;
      for (auto const& [child_name, child_schema] : column_schema.child_types) {
        assemble_column(column_order,
                        children,
                        read_columns,
                        child_name,
                        child_schema,
                        element_delimiter,
                        null_placeholder,
                        stream,
                        mr);
      }

      // printf("line %d\n", __LINE__);
      // cudf::test::print(children.front()->view());

      auto [offsets, child] = extract_lists(children.front(),
                                            column_schema.child_types.front().second,
                                            element_delimiter,
                                            null_placeholder,
                                            stream,
                                            mr);

      // printf("line %d\n", __LINE__);
      // cudf::test::print(child->view());
      // printf("line %d\n", __LINE__);
      // cudf::test::print(offsets->view());

      // TODO: fix null mask
      if (!has_struct_child) { null_mask = std::move(children.front()->release().null_mask); }

      output.emplace_back(cudf::make_lists_column(num_rows,
                                                  std::move(offsets),
                                                  std::move(child),
                                                  null_count,
                                                  std::move(*null_mask),
                                                  stream,
                                                  mr));

      // printf("line %d\n", __LINE__);
      // cudf::test::print(output.back()->view());
    } else {
      CUDF_FAIL("Unsupported type");
    }
  }
}

std::pair<char, char> find_delimiter(cudf::strings_column_view const& input,
                                     rmm::cuda_stream_view stream)
{
  auto constexpr num_levels  = 256;
  auto constexpr lower_level = std::numeric_limits<char>::min();
  auto constexpr upper_level = std::numeric_limits<char>::max();
  auto const num_chars       = input.chars_size(stream);  // stream sync

  // TODO: return when num_chars==0

  rmm::device_uvector<uint32_t> d_histogram(num_levels, stream);
  thrust::fill(rmm::exec_policy(stream), d_histogram.begin(), d_histogram.end(), 0);

  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(nullptr,
                                      temp_storage_bytes,
                                      input.chars_begin(stream),
                                      d_histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_chars,
                                      stream.value());
  rmm::device_buffer d_temp(temp_storage_bytes, stream);
  cub::DeviceHistogram::HistogramEven(d_temp.data(),
                                      temp_storage_bytes,
                                      input.chars_begin(stream),
                                      d_histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_chars,
                                      stream.value());

  auto const zero_level = d_histogram.begin() - lower_level;
  auto first_zero_count_pos =
    thrust::find(rmm::exec_policy(stream), zero_level, d_histogram.end(), 0);
  if (first_zero_count_pos == d_histogram.end()) {
    // Try again...
    first_zero_count_pos =
      thrust::find(rmm::exec_policy(stream), d_histogram.begin(), d_histogram.end(), 0);
    if (first_zero_count_pos == d_histogram.end()) {
      // TODO: change message
      throw std::logic_error(
        "can't find a character suitable as delimiter for combining json strings to json lines "
        "with "
        "custom delimiter");
    }
  }

  auto second_zero_count_pos =
    thrust::find(rmm::exec_policy(stream), first_zero_count_pos + 1, d_histogram.end(), 0);
  if (second_zero_count_pos == d_histogram.end()) {
    // TODO: change message
    throw std::logic_error(
      "can't find a character suitable as delimiter for combining json strings to json lines "
      "with "
      "custom delimiter");
  }

  return {static_cast<char>(first_zero_count_pos - zero_level),
          static_cast<char>(second_zero_count_pos - zero_level)};
}

std::vector<std::unique_ptr<cudf::column>> assemble_output(
  std::vector<std::pair<std::string, json_schema_element>> const& schema,
  std::vector<std::unique_ptr<cudf::column>>& read_columns,
  char element_delimiter,
  char null_placeholder,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> output;
  output.reserve(read_columns.size());

  std::size_t column_order{0};
  std::for_each(schema.begin(), schema.end(), [&](auto const& kv) {
    assemble_column(column_order,
                    output,
                    read_columns,
                    kv.first,
                    kv.second,
                    element_delimiter,
                    null_placeholder,
                    stream,
                    mr);
  });

  return output;
}

std::vector<std::unique_ptr<cudf::column>> from_json_to_structs(
  cudf::strings_column_view const& input,
  std::vector<std::pair<std::string, json_schema_element>> const& schema,
  bool allow_leading_zero_numbers,
  bool allow_non_numeric_numbers,
  bool allow_unquoted_control_chars,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // printf("line %d\n", __LINE__);
  // fflush(stdout);
  auto const [json_paths, type_ids, keep_quotes, has_list_type] = flatten_schema_to_paths(schema);

  // printf("line %d\n", __LINE__);
  // fflush(stdout);

#if 0
  int count{0};
  for (auto const& path : json_paths) {
    printf("\n\npath (%d/%d): \n", count++, (int)json_paths.size());
    for (auto node : path) {
      printf(".%s", std::get<1>(node).c_str());
    }
    printf("\n");
  }

  printf("keep quotes: \n");
  for (auto const i : keep_quotes) {
    printf("%d, ", (int)i);
  }
  printf("\n\n\n");
  fflush(stdout);

  auto ptr  = input.chars_begin(stream);
  auto size = input.chars_size(stream);
  std::vector<char> h_v(size);
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(h_v.data(), ptr, sizeof(char) * size, cudaMemcpyDefault, stream.value()));
  stream.synchronize();

  printf("input (size = %d): ", (int)size);
  for (auto c : h_v) {
    printf("%c", c);
  }
  printf("\n");

#endif

  // array<struct<a: struct<b: int>>>
  // [{'a': {'b': 1, 'c' : 2}, 'x': []}, {}]

  // This should only run when there is LIST column.
  char delimiter{','}, null_placeholder{'\0'};
  if (has_list_type) { std::tie(delimiter, null_placeholder) = find_delimiter(input, stream); }
  // printf("delimiter: %c (code: %d)\n", delimiter, (int)delimiter);
  // printf("null_placeholder: %c (code: %d)\n", null_placeholder, (int)null_placeholder);

  auto tmp = test::get_json_object(input,
                                   json_paths,
                                   type_ids,
                                   keep_quotes,
                                   delimiter,
                                   null_placeholder,
                                   1024 * 1024 * 1024 * 4L,
                                   -1,
                                   allow_leading_zero_numbers,
                                   allow_non_numeric_numbers,
                                   allow_unquoted_control_chars,
                                   stream,
                                   mr);
  // printf("line %d\n", __LINE__);
  // fflush(stdout);

  if constexpr (0) {
    for (std::size_t i = 0; i < tmp.size(); ++i) {
      auto out  = cudf::strings_column_view{tmp[i]->view()};
      auto ptr  = out.chars_begin(stream);
      auto size = out.chars_size(stream);
      std::vector<char> h_v(size);
      CUDF_CUDA_TRY(
        cudaMemcpyAsync(h_v.data(), ptr, sizeof(char) * size, cudaMemcpyDefault, stream.value()));
      stream.synchronize();

      printf("out %d / %d (size = %d): ", (int)i, (int)tmp.size(), (int)size);
      for (auto c : h_v) {
        printf("%c", c);
      }
      printf("\n");

      // cudf::test::print(tmp[i]->view());
    }
  }

  return assemble_output(schema, tmp, delimiter, null_placeholder, stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> is_null_or_empty(cudf::strings_column_view const& input,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  auto const d_input_ptr = cudf::column_device_view::create(input.parent(), stream);
  rmm::device_uvector<bool> output(input.size(), stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(input.size()),
                    output.begin(),
                    [input = *d_input_ptr] __device__(cudf::size_type idx) -> bool {
                      if (input.is_null(idx)) { return true; }

                      auto const d_str = input.element<cudf::string_view>(idx);
                      int i            = 0;
                      for (; i < d_str.size_bytes(); ++i) {
                        if (d_str[i] != ' ') { break; }
                      }
                      auto const empty = i == d_str.size_bytes();
                      return empty;
                    });

  return std::make_unique<cudf::column>(std::move(output), rmm::device_buffer{}, 0);
}

std::vector<std::unique_ptr<cudf::column>> from_json_to_structs(
  cudf::strings_column_view const& input,
  std::vector<std::pair<std::string, json_schema_element>> const& schema,
  bool allow_leading_zero_numbers,
  bool allow_non_numeric_numbers,
  bool allow_unquoted_control_chars,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_json_to_structs(input,
                                      schema,
                                      allow_leading_zero_numbers,
                                      allow_non_numeric_numbers,
                                      allow_unquoted_control_chars,
                                      stream,
                                      mr);
}

}  // namespace spark_rapids_jni
