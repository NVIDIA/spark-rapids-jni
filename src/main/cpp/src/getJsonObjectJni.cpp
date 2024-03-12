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

extern "C" {
JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GetJsonObject_getJsonObject(
  JNIEnv* env, jclass, jlong input_column, jlong json_path)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, json_path, "json path is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const*>(input_column);
    auto const n_strings_col_view(*n_column_view);
    auto const* n_scalar_path = reinterpret_cast<cudf::string_scalar*>(j_scalar_handle);
    auto const result = spark_rapids_jni::get_json_object(n_strings_col_view, *n_scalar_path);
    return release_as_jlong(result);
  }
  CATCH_STD(env, 0);
}
}
