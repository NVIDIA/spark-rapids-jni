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
JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_getJsonObject(
  JNIEnv* env, jclass, jlong input_column, jint size, jobjectArray path_instructions)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, path_instructions, "path_instructions is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const n_column_view      = reinterpret_cast<cudf::column_view const*>(input_column);
    auto const n_strings_col_view = cudf::strings_column_view{*n_column_view};

    std::vector<std::tuple<path_instruction_type, std::string, int64_t>> instructions;
    for (int i = 0; i < size; i++) {
      jobject instruction = env->GetObjectArrayElement(path_instructions, i);
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

      instructions.emplace_back(instruction_type, name_str, index);

      env->ReleaseStringUTFChars(name, name_str);
    }

    return cudf::jni::release_as_jlong(
      spark_rapids_jni::get_json_object(n_strings_col_view, instructions));
  }
  CATCH_STD(env, 0);
}
}
