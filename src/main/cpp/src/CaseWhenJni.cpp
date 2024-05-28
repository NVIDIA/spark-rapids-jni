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

#include "case_when.hpp"
#include "cudf_jni_apis.hpp"

#include <vector>

extern "C" {
JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CaseWhen_selectFirstTrueIndex(
  JNIEnv* env, jclass, jlongArray bool_cols)
{
  JNI_NULL_CHECK(env, bool_cols, "array of column handles is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jpointerArray<cudf::column_view> n_cudf_bool_columns(env, bool_cols);
    auto bool_column_views = n_cudf_bool_columns.get_dereferenced();
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::select_first_true_index(cudf::table_view(bool_column_views)));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CaseWhen_selectFromIndex(JNIEnv* env,
                                                                                  jclass,
                                                                                  jlong scalar_cols,
                                                                                  jlong index_col)
{
  JNI_NULL_CHECK(env, scalar_cols, "Column handles is null", 0);
  JNI_NULL_CHECK(env, index_col, "Column handles is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const scalar_column_view      = reinterpret_cast<cudf::column_view const*>(scalar_cols);
    auto const scalar_strings_col_view = cudf::strings_column_view{*scalar_column_view};
    auto const index_column_view       = reinterpret_cast<cudf::column_view const*>(index_col);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::select_from_index(scalar_strings_col_view, *index_column_view));
  }
  CATCH_STD(env, 0);
}
}
