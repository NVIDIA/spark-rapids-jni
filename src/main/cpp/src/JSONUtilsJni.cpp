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
#include "json_utils.hpp"

#include <cudf/strings/strings_column_view.hpp>

#include <vector>

using path_instruction_type = spark_rapids_jni::path_instruction_type;

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

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_concatenateJsonStrings(
  JNIEnv* env, jclass, jlong j_input)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input_cv = reinterpret_cast<cudf::column_view const*>(j_input);
    auto [is_valid, joined_strings, delimiter] =
      spark_rapids_jni::concat_json(cudf::strings_column_view{*input_cv});

    // The output array contains 5 elements:
    // [0]: address of the cudf::column object `is_valid` in host memory
    // [1]: address of data buffer of the concatenated strings in device memory
    // [2]: data length
    // [3]: address of the rmm::device_buffer object (of the concatenated strings) in host memory
    // [4]: delimiter char
    auto out_handles = cudf::jni::native_jlongArray(env, 5);
    out_handles[0]   = reinterpret_cast<jlong>(is_valid.release());
    out_handles[1]   = reinterpret_cast<jlong>(joined_strings->data());
    out_handles[2]   = static_cast<jlong>(joined_strings->size());
    out_handles[3]   = reinterpret_cast<jlong>(joined_strings.release());
    out_handles[4]   = static_cast<jlong>(delimiter);
    return out_handles.get_jArray();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_makeStructs(
  JNIEnv* env, jclass, jlongArray j_children, jlong j_is_null)
{
  JNI_NULL_CHECK(env, j_children, "j_children is null", 0);
  JNI_NULL_CHECK(env, j_is_null, "j_is_null is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const children =
      cudf::jni::native_jpointerArray<cudf::column_view>{env, j_children}.get_dereferenced();
    auto const is_null = *reinterpret_cast<cudf::column_view const*>(j_is_null);
    return cudf::jni::ptr_as_jlong(spark_rapids_jni::make_structs(children, is_null).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_JSONUtils_castStringsToBooleans(JNIEnv* env, jclass, jlong j_input)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input = *reinterpret_cast<cudf::column_view const*>(j_input);
    return cudf::jni::ptr_as_jlong(spark_rapids_jni::cast_strings_to_booleans(input).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_castStringsToDecimals(
  JNIEnv* env, jclass, jlong j_input, jint precision, jint scale, jboolean is_us_locale)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input = *reinterpret_cast<cudf::column_view const*>(j_input);

    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::cast_strings_to_decimals(input, precision, scale, is_us_locale).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_removeQuotes(
  JNIEnv* env, jclass, jlong j_input, jboolean nullify_if_not_quoted)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input = *reinterpret_cast<cudf::column_view const*>(j_input);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::remove_quotes(input, nullify_if_not_quoted).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_JSONUtils_removeQuotesForFloats(JNIEnv* env, jclass, jlong j_input)
{
  JNI_NULL_CHECK(env, j_input, "j_input is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input = *reinterpret_cast<cudf::column_view const*>(j_input);
    return cudf::jni::ptr_as_jlong(spark_rapids_jni::remove_quotes_for_floats(input).release());
  }
  CATCH_STD(env, 0);
}

}  // extern "C"
