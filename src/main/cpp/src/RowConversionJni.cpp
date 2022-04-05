/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "row_conversion.hpp"

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"

extern "C" {

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_RowConversion_convertToRows(JNIEnv *env, jclass, jlong input_table)
{
  JNI_NULL_CHECK(env, input_table, "input table is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::table_view *n_input_table = reinterpret_cast<cudf::table_view *>(input_table);
    std::vector<std::unique_ptr<cudf::column>> cols = spark_rapids_jni::convert_to_rows(*n_input_table);
    int num_columns = cols.size();
    cudf::jni::native_jlongArray outcol_handles(env, num_columns);
    for (int i = 0; i < num_columns; i++) {
      outcol_handles[i] = reinterpret_cast<jlong>(cols[i].release());
    }
    return outcol_handles.get_jArray();
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_RowConversion_convertFromRows(JNIEnv *env, jclass,
                                                               jlong input_column,
                                                               jintArray types,
                                                               jintArray scale)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, types, "types is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    cudf::column_view *input = reinterpret_cast<cudf::column_view *>(input_column);
    cudf::lists_column_view list_input(*input);
    cudf::jni::native_jintArray n_types(env, types);
    cudf::jni::native_jintArray n_scale(env, scale);
    std::vector<cudf::data_type> types_vec;
    for (int i = 0; i < n_types.size(); i++) {
      types_vec.emplace_back(cudf::jni::make_data_type(n_types[i], n_scale[i]));
    }
    std::unique_ptr<cudf::table> result = spark_rapids_jni::convert_from_rows(list_input, types_vec);
    return cudf::jni::convert_table_for_return(env, result);
  }
  CATCH_STD(env, 0);
}

}
