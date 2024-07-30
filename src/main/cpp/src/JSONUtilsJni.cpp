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

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_generateDeviceJSONPaths(
  JNIEnv* env, jclass, jobjectArray j_paths, jintArray j_dpath_offsets)
{
  JNI_NULL_CHECK(env, j_paths, "j_dpath_ptrs is null", 0);
  JNI_NULL_CHECK(env, j_dpath_offsets, "j_dpath_offsets is null", 0);

  using path_type = std::vector<std::tuple<path_instruction_type, std::string, int64_t>>;

  try {
    cudf::jni::auto_set_device(env);

    auto const path_offsets = cudf::jni::native_jintArray(env, j_dpath_offsets).to_vector();
    CUDF_EXPECTS(path_offsets.size() > 1, "Invalid path offsets.");
    auto const num_paths = path_offsets.size() - 1;
    std::vector<path_type> paths(num_paths);

    for (std::size_t i = 0; i < num_paths; ++i) {
      auto const path_size = path_offsets[i + 1] - path_offsets[i];
      auto& path           = paths[i];
      path.reserve(path_size);

      for (int j = path_offsets[i]; j < path_offsets[i + 1]; ++j) {
        jobject instruction = env->GetObjectArrayElement(j_paths, j);
        JNI_NULL_CHECK(env, instruction, "path_instruction is null", 0);
        jclass instruction_class = env->GetObjectClass(instruction);
        JNI_NULL_CHECK(env, instruction_class, "instruction_class is null", 0);

        jfieldID field_id = env->GetFieldID(instruction_class, "type", "I");
        JNI_NULL_CHECK(env, field_id, "field_id is null", 0);
        jint type                              = env->GetIntField(instruction, field_id);
        path_instruction_type instruction_type = static_cast<path_instruction_type>(type);

        field_id = env->GetFieldID(instruction_class, "name", "Ljava/lang/String;");
        JNI_NULL_CHECK(env, field_id, "field_id is null", 0);
        jstring name = (jstring)env->GetObjectField(instruction, field_id);
        JNI_NULL_CHECK(env, name, "name is null", 0);
        const char* name_str = env->GetStringUTFChars(name, JNI_FALSE);

        field_id = env->GetFieldID(instruction_class, "index", "J");
        JNI_NULL_CHECK(env, field_id, "field_id is null", 0);
        jlong index = env->GetLongField(instruction, field_id);

        path.emplace_back(instruction_type, name_str, index);
        env->ReleaseStringUTFChars(name, name_str);
      }
    }

    auto output      = spark_rapids_jni::generate_device_json_paths(paths);
    auto out_handles = cudf::jni::native_jlongArray(env, output.size());
    std::transform(output.begin(), output.end(), out_handles.begin(), [](auto& d_path) {
      return cudf::jni::release_as_jlong(d_path);
    });
    return out_handles.get_jArray();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_getJsonObject(JNIEnv* env,
                                                                                 jclass,
                                                                                 jlong j_input,
                                                                                 jlong j_dpath)
{
  JNI_NULL_CHECK(env, j_input, "input column is null", 0);
  JNI_NULL_CHECK(env, j_dpath, "j_dpath is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input_ptr = reinterpret_cast<cudf::column_view const*>(j_input);
    auto const d_path_ptr =
      reinterpret_cast<spark_rapids_jni::json_path_device_storage const*>(j_dpath);
    return cudf::jni::release_as_jlong(spark_rapids_jni::get_json_object(
      cudf::strings_column_view{*input_ptr}, d_path_ptr->instructions));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_getJsonObjectMultiplePaths(
  JNIEnv* env, jclass, jlong j_input, jlongArray j_dpath_ptrs)
{
  JNI_NULL_CHECK(env, j_input, "j_input column is null", 0);
  JNI_NULL_CHECK(env, j_dpath_ptrs, "j_dpath_ptrs is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input_ptr = reinterpret_cast<cudf::column_view const*>(j_input);
    auto const path_ptrs =
      cudf::jni::native_jpointerArray<spark_rapids_jni::json_path_device_storage>{env,
                                                                                  j_dpath_ptrs};
    std::vector<cudf::device_span<spark_rapids_jni::path_instruction const>> paths;
    paths.reserve(path_ptrs.size());
    for (auto ptr : path_ptrs) {
      paths.emplace_back(ptr->instructions.data(), ptr->instructions.size());
    }
    auto output = spark_rapids_jni::get_json_object_multiple_paths(
      cudf::strings_column_view{*input_ptr}, paths);
    auto out_handles = cudf::jni::native_jlongArray(env, output.size());
    std::transform(output.begin(), output.end(), out_handles.begin(), [](auto& col) {
      return cudf::jni::release_as_jlong(col);
    });
    return out_handles.get_jArray();
  }
  CATCH_STD(env, 0);
}
}
