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

#include <memory>

namespace spark_rapids_jni {

namespace detail {

/**
 * path instruction type
 */
enum class path_instruction_type { SUBSCRIPT, WILDCARD, KEY, INDEX, NAMED };

/**
 * path instruction
 */
struct path_instruction {
  CUDF_HOST_DEVICE inline path_instruction(path_instruction_type _type) : type(_type) {}

  path_instruction_type type;

  // used when type is named type
  cudf::string_view name{"", 0};

  // used when type is index
  long index{-1};
};

class JsonPathParser {
 public:
  std::vector<path_instruction> instructions;

  thrust::optional<path_instruction> current_instruction = thrust::nullopt;

  size_t current_start = 0;
  size_t current_len   = 0;

  bool is_digit(char c) { return c >= '0' && c <= '9'; }

  thrust::optional<std::vector<path_instruction>> parse(const std::string& str)
  {
    instructions.clear();
    current_start = 0;
    current_len   = 0;
    size_t pos    = 0;
    if (str.size() == 2) {
      // the while loop below will not run if str.size() == 2
      // and there is no valid path
      return thrust::nullopt;
    }
    if (str[pos++] != '$') { return thrust::nullopt; }
    bool in_brackets = false;
    bool in_quotes   = false;
    if (str[pos] == '[') { in_brackets = true; }
    if (str[pos] == '\'') {
      // .'name' is invalid
      return thrust::nullopt;
    }
    while (pos < str.size() - 1) {
      char from = str[pos];
      char to   = str[pos + 1];
      // std::cout << "from: " << from << " to: " << to << " " << in_brackets << " " << in_quotes <<
      // " " << current_start << ' ' << current_len << std::endl;
      if (from == '.' && !in_quotes) {
        if (to == '*') {
          // child wildcard
          current_instruction = path_instruction{path_instruction_type::WILDCARD};
          instructions.push_back(*current_instruction);
          current_instruction = thrust::nullopt;
        } else if (to == '.' || to == '[') {
          // . and [ => empty key
          return thrust::nullopt;
        } else {
          // named
          current_instruction = path_instruction{path_instruction_type::KEY};
          instructions.push_back(*current_instruction);
          current_instruction = thrust::nullopt;
          current_instruction = path_instruction{path_instruction_type::NAMED};
          if (to != ' ') {  // if a space follows the dot, strip it directly
            current_start = pos + 1;
            current_len   = 1;
          }
        }
      } else if (from == '[' && !in_quotes) {
        if (to == '\'') {
          // named: ['name']
          in_quotes           = true;
          current_instruction = path_instruction{path_instruction_type::NAMED};
        } else {
          // subscript: [*] or [123]
          in_brackets = true;
          instructions.push_back(path_instruction{path_instruction_type::SUBSCRIPT});
          if (to == '*') {
            // subscript wildcard
            current_instruction = path_instruction{path_instruction_type::WILDCARD};
          } else if (is_digit(to)) {
            // subscript index
            // TODO long overflow
            current_instruction        = path_instruction{path_instruction_type::INDEX};
            current_instruction->index = (to - '0');
          } else {
            return thrust::nullopt;
          }
        }
      } else if (from == ']' && !in_quotes) {
        if (to == '.') {
          // do nothing
        } else if (to == '[') {
          in_brackets = true;
        } else if (current_instruction->type == path_instruction_type::NAMED) {
          // `.name` case, chars except . and [ are allowed
          current_len++;
        } else {
          return thrust::nullopt;
        }
      } else if (from == '*') {
        if (to == ']') {
          // subscript wildcard
          if (!in_quotes) {
            in_brackets = false;
            instructions.push_back(*current_instruction);
            current_instruction = thrust::nullopt;
          } else {
            // in quotes
            current_len++;
          }
        } else if (to == '.') {
          // do nothing
        } else if (to == '\'' && in_quotes) {
          // child wildcard OR named, don't know yet
          // will be pushed when meet ']'
          in_quotes = false;
        } else if (in_quotes ||
                   current_instruction->type == path_instruction_type::NAMED) {  // normal char
          // named, change to NAMED
          if (current_instruction->type != path_instruction_type::NAMED) {
            // must in quote, so it is safe to add a key
            current_instruction = path_instruction{path_instruction_type::KEY};
            instructions.push_back(*current_instruction);
            current_instruction->type = path_instruction_type::NAMED;
            current_start             = pos;
            current_len               = 1;
          }
          current_len++;
        } else {
          return thrust::nullopt;
        }
      } else if (from == '\'') {
        if (to == '*' && in_brackets) {
          // maybe ['*'], will update current_instruction later if it is ['*abc']
          current_instruction = path_instruction{path_instruction_type::WILDCARD};
        } else if (to == ']' && !in_quotes) {
          // named
          if (!in_brackets) {
            // .']abc => Named(']abc)
            current_len++;
          }
          in_brackets = false;
          if (current_instruction->type == path_instruction_type::NAMED) {
            // end of ['a'] => Named(a)
            current_instruction->name = cudf::string_view(str.c_str() + current_start, current_len);
            current_len               = 0;
          }
          // could be wildcard
          instructions.push_back(*current_instruction);
          current_instruction = thrust::nullopt;
        } else if (to == '?' || to == '\'') {
          // ? => invalid
          // ' => empty key
          return thrust::nullopt;
        } else {  // normal char
          if (in_brackets) {
            current_instruction = path_instruction{path_instruction_type::KEY};
            instructions.push_back(*current_instruction);
          }
          current_instruction = thrust::nullopt;
          // begin of ['a']
          current_instruction = path_instruction{path_instruction_type::NAMED};
          if (current_len > 0 || to != ' ') {
            // strip leading space
            if (current_len == 0) {
              current_start = pos + 1;
              current_len   = 1;
            } else {
              current_len++;
            }
          }
        }
      } else if (is_digit(from) && in_brackets && !in_quotes) {
        if (is_digit(to)) {
          // compatibility issue: crashed in Spark but overflow quietly in kernel
          current_instruction->index = current_instruction->index * 10 + (to - '0');
        } else if (to == ']') {
          in_brackets = false;
          instructions.push_back(*current_instruction);
          current_instruction = thrust::nullopt;
        } else {
          return thrust::nullopt;
        }
      } else {  // normal char
        if (!in_quotes && (to == '.' || to == '[')) {
          if (current_instruction->type == path_instruction_type::NAMED) {
            // end of a named
            if (current_len > 0) {
              current_instruction->name =
                cudf::string_view(str.c_str() + current_start, current_len);
              current_len = 0;
              instructions.push_back(*current_instruction);
              current_instruction = thrust::nullopt;
            } else {
              // empty key
              return thrust::nullopt;
            }
            if (to == '[') { in_brackets = true; }
          } else {
            return thrust::nullopt;
          }
        } else if (in_quotes && to == '\'') {
          in_quotes = false;
        } else if (in_quotes && to == '?') {
          return thrust::nullopt;
        } else {  // normal char
          if (current_len > 0 || to != ' ') {
            if (current_len == 0) {
              current_start = pos + 1;
              current_len   = 1;
            } else {
              current_len++;
            }
          }
        }
      }
      pos++;
    }
    if (str[pos] == '.') {
      // empty key
      return thrust::nullopt;
    }
    if (current_instruction) {
      if (current_instruction->type == path_instruction_type::NAMED) {
        if (current_len > 0) {
          current_instruction->name = cudf::string_view(str.c_str() + current_start, current_len);
          // std::cout << "name: " << current_instruction->name << std::endl;
          current_len = 0;
          instructions.push_back(*current_instruction);
          current_instruction = thrust::nullopt;
        } else {
          // empty key
          return thrust::nullopt;
        }
      } else {
        return thrust::nullopt;
      }
    }

    return instructions;
  }
};

}  // namespace detail

std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  cudf::string_scalar const& json_path,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
}  // namespace spark_rapids_jni
