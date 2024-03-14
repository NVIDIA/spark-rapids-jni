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
JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_JSONUtils_getJsonObject(JNIEnv* env,
                                                         jclass,
                                                         jlong input_column,
                                                         jintArray path_ins_types,
                                                         jobjectArray path_ins_names,
                                                         jlongArray path_ins_indexes)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, path_ins_types, "path_ins_types is null", 0);
  JNI_NULL_CHECK(env, path_ins_names, "path_ins_names is null", 0);
  JNI_NULL_CHECK(env, path_ins_indexes, "path_ins_indexes is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const n_column_view = reinterpret_cast<cudf::column_view const*>(input_column);
    auto const n_strings_col_view = cudf::strings_column_view{*n_column_view};

    cudf::jni::native_jintArray path_ins_types_n(env, path_ins_types);
    cudf::jni::native_jstringArray path_ins_names_n(env, path_ins_names);
    cudf::jni::native_jlongArray path_ins_indexes_n(env, path_ins_indexes);

    auto const path_ins_types_v = std::vector<int32_t>(path_ins_types_n.begin(), path_ins_types_n.end());

    auto const path_ins_indexes_v = std::vector<int64_t>(path_ins_indexes_n.begin(), path_ins_indexes_n.end());

    return cudf::jni::release_as_jlong(spark_rapids_jni::get_json_object(
      n_strings_col_view, path_ins_types_v, path_ins_names_n.as_cpp_vector(), path_ins_indexes_v));
  }
  CATCH_STD(env, 0);
}
}
