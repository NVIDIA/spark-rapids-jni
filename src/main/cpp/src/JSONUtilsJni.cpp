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
#include "from_json.hpp"
#include "get_json_object.hpp"

#include <cudf/io/json.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <map>
#include <vector>

using path_instruction_type = spark_rapids_jni::path_instruction_type;

namespace spark_rapids_jni {
json_schema_element read_schema_element(int& index,
                                        cudf::jni::native_jstringArray const& names,
                                        cudf::jni::native_jintArray const& children,
                                        cudf::jni::native_jintArray const& types,
                                        cudf::jni::native_jintArray const& scales)
{
  printf("JNI line %d\n", __LINE__);
  fflush(stdout);

  auto d_type = cudf::data_type{static_cast<cudf::type_id>(types[index]), scales[index]};
  if (d_type.id() == cudf::type_id::STRUCT || d_type.id() == cudf::type_id::LIST) {
    printf("JNI line %d\n", __LINE__);
    fflush(stdout);

    std::vector<std::pair<std::string, json_schema_element>> child_elems;
    int num_children = children[index];
    // go to the next entry, so recursion can parse it.
    index++;
    for (int i = 0; i < num_children; i++) {
      printf("JNI line %d\n", __LINE__);
      fflush(stdout);

      auto const name = std::string{names.get(index).get()};
      child_elems.emplace_back(name, read_schema_element(index, names, children, types, scales));
    }
    return json_schema_element{d_type, std::move(child_elems)};
  } else {
    printf("JNI line %d\n", __LINE__);

    printf("children size: %d, idx = %d\n", children.size(), index);

    fflush(stdout);

    if (children[index] != 0) {
      throw std::invalid_argument("found children for a type that should have none");
    }
    // go to the next entry before returning...
    index++;
    printf("JNI line %d\n", __LINE__);
    fflush(stdout);
    return json_schema_element{d_type, {}};
  }
}
}  // namespace spark_rapids_jni

extern "C" {

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_getMaxJSONPathDepth(JNIEnv* env,
                                                                                      jclass)
{
  try {
    cudf::jni::auto_set_device(env);
    return spark_rapids_jni::MAX_JSON_PATH_DEPTH;
  }
  CATCH_STD(env, 0);
}

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
    if (names.size() != size || indexes.size() != static_cast<std::size_t>(size) ||
        type_nums.size() != static_cast<std::size_t>(size)) {
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
                                                                      jintArray j_path_offsets,
                                                                      jlong memory_budget_bytes,
                                                                      jint parallel_override)
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
        indexes.size() != static_cast<std::size_t>(num_entries) ||
        type_nums.size() != static_cast<std::size_t>(num_entries)) {
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
    auto output         = spark_rapids_jni::get_json_object_multiple_paths(
      cudf::strings_column_view{*input_cv}, paths, memory_budget_bytes, parallel_override);

    auto out_handles = cudf::jni::native_jlongArray(env, output.size());
    std::transform(output.begin(), output.end(), out_handles.begin(), [](auto& col) {
      return cudf::jni::release_as_jlong(col);
    });
    return out_handles.get_jArray();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_extractRawMapFromJsonString(
  JNIEnv* env, jclass, jlong j_input)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input_cv = reinterpret_cast<cudf::column_view const*>(j_input);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::from_json_to_raw_map(cudf::strings_column_view{*input_cv}).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_JSONUtils_fromJsonToStructs(JNIEnv* env,
                                                             jclass,
                                                             jlong j_input,
                                                             jobjectArray j_col_names,
                                                             jintArray j_num_children,
                                                             jintArray j_types,
                                                             jintArray j_scales,
                                                             jboolean allow_leading_zero_numbers,
                                                             jboolean allow_non_numeric_numbers)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);
  JNI_NULL_CHECK(env, j_col_names, "j_col_names is null", 0);
  JNI_NULL_CHECK(env, j_num_children, "j_num_children is null", 0);
  JNI_NULL_CHECK(env, j_types, "j_types is null", 0);
  JNI_NULL_CHECK(env, j_scales, "j_scales is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstringArray n_col_names(env, j_col_names);
    cudf::jni::native_jintArray n_types(env, j_types);
    cudf::jni::native_jintArray n_scales(env, j_scales);
    cudf::jni::native_jintArray n_children(env, j_num_children);

    if (n_types.size() != n_scales.size()) {
      JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS, "types and scales must match size", 0);
    }
    if (n_col_names.size() != n_types.size()) {
      JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS, "types and column names must match size", 0);
    }
    if (n_children.size() != n_types.size()) {
      JNI_THROW_NEW(env, cudf::jni::ILLEGAL_ARG_CLASS, "types and num children must match size", 0);
    }

    printf("JNI line %d, size = %d\n", __LINE__, (int)n_types.size());
    fflush(stdout);

    std::vector<std::pair<std::string, spark_rapids_jni::json_schema_element>> schema;
    int idx = 0;
    while (idx < n_types.size()) {
      printf("JNI line %d\n", __LINE__);
      fflush(stdout);

      auto const name = std::string{n_col_names.get(idx).get()};
      schema.emplace_back(
        name,
        spark_rapids_jni::read_schema_element(idx, n_col_names, n_children, n_types, n_scales));

      // auto const name = n_col_names.get(at).get();
      printf("JNI line %d\n", __LINE__);
      fflush(stdout);

      // auto child = cudf::jni::read_schema_element(at, n_children, n_col_names, n_types,
      // n_scales); printf("JNI line %d\n", __LINE__); fflush(stdout);

      // schema.emplace(name, std::move(child));
    }
    printf("JNI line %d\n", __LINE__);
    fflush(stdout);

    auto const input_cv = reinterpret_cast<cudf::column_view const*>(j_input);
    auto output = spark_rapids_jni::from_json_to_structs(cudf::strings_column_view{*input_cv},
                                                         schema,
                                                         allow_leading_zero_numbers,
                                                         allow_non_numeric_numbers);

    printf("JNI line %d\n", __LINE__);
    fflush(stdout);

    auto out_handles = cudf::jni::native_jlongArray(env, output.size());
    std::transform(output.begin(), output.end(), out_handles.begin(), [](auto& col) {
      return cudf::jni::release_as_jlong(col);
    });
    return out_handles.get_jArray();
  }
  CATCH_STD(env, 0);
}
}
