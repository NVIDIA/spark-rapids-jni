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

extern "C" {
JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_JSONUtils_getJsonObject(
  JNIEnv* env, jclass, jlong input_column, jlong instructions_table)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, instructions_table, "path_ins_types is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view* n_column_view = reinterpret_cast<cudf::column_view*>(input_column);
    cudf::strings_column_view n_strings_col_view(*n_column_view);

    auto const instructions = reinterpret_cast<cudf::table_view const*>(instructions_table);

    return cudf::jni::release_as_jlong(
      spark_rapids_jni::get_json_object(n_strings_col_view, *instructions));
  }
  CATCH_STD(env, 0);
}
}
