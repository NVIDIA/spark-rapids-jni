/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include "list_slice.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GpuListSliceUtils_listSliceIntInt(
  JNIEnv* env, jclass, jlong input_column, jint start, jint length)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  try {
    cudf::jni::auto_set_device(env);

    // The following constructor expects that the type of the input_column is LIST.
    // If the type is not LIST, an exception will be thrown.
    cudf::lists_column_view lcv{*reinterpret_cast<cudf::column_view const*>(input_column)};
    return cudf::jni::release_as_jlong(spark_rapids_jni::list_slice(lcv, start, length));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GpuListSliceUtils_listSliceIntCol(
  JNIEnv* env, jclass, jlong input_column, jint start, jlong length)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, length, "length column is null", 0);
  try {
    cudf::jni::auto_set_device(env);

    // The following constructor expects that the type of the input_column is LIST.
    // If the type is not LIST, an exception will be thrown.
    cudf::lists_column_view lcv{*reinterpret_cast<cudf::column_view const*>(input_column)};
    auto const& length_cv = *reinterpret_cast<cudf::column_view const*>(length);
    return cudf::jni::release_as_jlong(spark_rapids_jni::list_slice(lcv, start, length_cv));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GpuListSliceUtils_listSliceColInt(
  JNIEnv* env, jclass, jlong input_column, jlong start, jint length)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, start, "start column is null", 0);
  try {
    cudf::jni::auto_set_device(env);

    // The following constructor expects that the type of the input_column is LIST.
    // If the type is not LIST, an exception will be thrown.
    cudf::lists_column_view lcv{*reinterpret_cast<cudf::column_view const*>(input_column)};
    auto const& start_cv = *reinterpret_cast<cudf::column_view const*>(start);
    return cudf::jni::release_as_jlong(spark_rapids_jni::list_slice(lcv, start_cv, length));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GpuListSliceUtils_listSliceColCol(
  JNIEnv* env, jclass, jlong input_column, jlong start, jlong length)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, start, "start column is null", 0);
  JNI_NULL_CHECK(env, length, "length column is null", 0);
  try {
    cudf::jni::auto_set_device(env);

    // The following constructor expects that the type of the input_column is LIST.
    // If the type is not LIST, an exception will be thrown.
    cudf::lists_column_view lcv{*reinterpret_cast<cudf::column_view const*>(input_column)};
    auto const& start_cv  = *reinterpret_cast<cudf::column_view const*>(start);
    auto const& length_cv = *reinterpret_cast<cudf::column_view const*>(length);
    return cudf::jni::release_as_jlong(spark_rapids_jni::list_slice(lcv, start_cv, length_cv));
  }
  CATCH_STD(env, 0);
}

}  // extern "C"
