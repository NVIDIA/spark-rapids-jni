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

#pragma once

#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/resource_ref.hpp>

#include <memory>
#include <vector>

namespace spark_rapids_jni {

/**
 * @brief Type of instruction in a JSON path.
 */
enum class path_instruction_type : int8_t { WILDCARD, INDEX, NAMED };

/**
 * @brief Instruction along a JSON path.
 */
struct path_instruction {
  path_instruction(path_instruction_type _type) : type(_type) {}

  // used when type is named type
  cudf::string_view name;

  // used when type is index
  int index{-1};

  path_instruction_type type;
};

/**
 * @brief The class to store data of a JSON path on the GPU.
 */
class json_path_device_storage {
  json_path_device_storage(rmm::device_uvector<path_instruction>&& _instructions,
                           cudf::string_scalar&& _instruction_names)
    : instructions{std::move(_instructions)}, instruction_names{std::move(_instruction_names)}
  {
  }

 public:
  rmm::device_uvector<path_instruction> const instructions;

 private:
  // This scalar is unused but needs to be kept alive as its data is access through the
  // `instructions` array.
  cudf::string_scalar const instruction_names;
};

/**
 * @brief Construct the vector containing device data for the input JSON paths.
 *
 * All JSON paths are processed at once, to minimize overhead.
 */
std::vector<std::unique_ptr<json_path_device_storage>> generate_device_json_paths(
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int64_t>>> const&
    json_paths,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Extract JSON object from a JSON string based on the specified JSON path.
 *
 * If the input JSON string is invalid, or it does not contain the object at the given path, a null
 * will be returned.
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& input,
  cudf::device_span<path_instruction const> json_path,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Extract multiple JSON objects from a JSON string based on the specified JSON paths.
 *
 * This function processes all the JSON paths in parallel, which may be faster than calling
 * to `get_json_object` on the individual JSON paths. However, it may consume much more GPU
 * memory, proportional to the number of JSON paths.
 */
std::vector<std::unique_ptr<cudf::column>> get_json_object_multiple_paths(
  cudf::strings_column_view const& input,
  std::vector<cudf::device_span<path_instruction const>> const& json_paths,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
