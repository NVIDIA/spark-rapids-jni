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

#include "cudf_jni_apis.hpp"
#include "get_json_object.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <vector>

using path_instruction_type = spark_rapids_jni::path_instruction_type;

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_JSONUtils_getJsonObject(JNIEnv* env,
                                                         jclass,
                                                         jlong input_column,
                                                         jbyteArray j_type_nums,
                                                         jobjectArray j_names,
                                                         jintArray j_indexes)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, j_type_nums, "j_type_nums is null", 0);
  JNI_NULL_CHECK(env, j_names, "j_names is null", 0);
  JNI_NULL_CHECK(env, j_indexes, "j_indexes is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const n_column_view      = reinterpret_cast<cudf::column_view const*>(input_column);
    auto const n_strings_col_view = cudf::strings_column_view{*n_column_view};

    std::vector<std::tuple<path_instruction_type, std::string, int32_t>> instructions;

    auto const type_nums = cudf::jni::native_jbyteArray(env, j_type_nums).to_vector();
    auto const names     = cudf::jni::native_jstringArray(env, j_names);
    auto const indexes   = cudf::jni::native_jintArray(env, j_indexes).to_vector();
    int size             = type_nums.size();
    if (names.size() != size || indexes.size() != static_cast<uint64_t>(size) ||
        type_nums.size() != static_cast<uint64_t>(size)) {
      JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS, "wrong number of entries passed in", 0);
    }

    for (int i = 0; i < size; i++) {
      path_instruction_type instruction_type = static_cast<path_instruction_type>(type_nums[i]);
      const char* name_str                   = names[i].get();
      jlong index                            = indexes[i];
      instructions.emplace_back(instruction_type, name_str, index);
    }

    return cudf::jni::release_as_jlong(
      spark_rapids_jni::get_json_object(n_strings_col_view, instructions));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_JSONUtils_getJsonObjectMultiplePaths(JNIEnv* env,
                                                                      jclass,
                                                                      jlong j_input,
                                                                      jbyteArray j_type_nums,
                                                                      jobjectArray j_names,
                                                                      jintArray j_indexes,
                                                                      jintArray j_path_offsets)
{
  JNI_NULL_CHECK(env, j_input, "j_input column is null", 0);
  JNI_NULL_CHECK(env, j_type_nums, "j_type_nums is null", 0);
  JNI_NULL_CHECK(env, j_names, "j_names is null", 0);
  JNI_NULL_CHECK(env, j_indexes, "j_indexes is null", 0);
  JNI_NULL_CHECK(env, j_path_offsets, "j_path_offsets is null", 0);

  using path_type = std::vector<std::tuple<path_instruction_type, std::string, int32_t>>;

  try {
    cudf::jni::auto_set_device(env);

    auto const path_offsets = cudf::jni::native_jintArray(env, j_path_offsets).to_vector();
    CUDF_EXPECTS(path_offsets.size() > 1, "Invalid path offsets.");
    auto const type_nums = cudf::jni::native_jbyteArray(env, j_type_nums).to_vector();
    auto const names     = cudf::jni::native_jstringArray(env, j_names);
    auto const indexes   = cudf::jni::native_jintArray(env, j_indexes).to_vector();
    auto const num_paths = path_offsets.size() - 1;
    std::vector<path_type> paths(num_paths);
    auto const num_entries = path_offsets[num_paths];

    if (num_entries < 0 || names.size() != num_entries ||
        indexes.size() != static_cast<uint64_t>(num_entries) ||
        type_nums.size() != static_cast<uint64_t>(num_entries)) {
      JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS, "wrong number of entries passed in", 0);
    }

    for (std::size_t i = 0; i < num_paths; ++i) {
      auto const path_size = path_offsets[i + 1] - path_offsets[i];
      auto path            = path_type{};
      path.reserve(path_size);
      for (int j = path_offsets[i]; j < path_offsets[i + 1]; ++j) {
        path_instruction_type instruction_type = static_cast<path_instruction_type>(type_nums[j]);
        const char* name_str                   = names[j].get();
        jlong index                            = indexes[j];
        path.emplace_back(instruction_type, name_str, index);
      }

      paths[i] = std::move(path);
    }

    auto const input_cv = reinterpret_cast<cudf::column_view const*>(j_input);
    auto output =
      spark_rapids_jni::get_json_object_multiple_paths(cudf::strings_column_view{*input_cv}, paths);

    auto out_handles = cudf::jni::native_jlongArray(env, output.size());
    std::transform(output.begin(), output.end(), out_handles.begin(), [](auto& col) {
      return cudf::jni::release_as_jlong(col);
    });
    return out_handles.get_jArray();
  }
  CATCH_STD(env, 0);
}
}
