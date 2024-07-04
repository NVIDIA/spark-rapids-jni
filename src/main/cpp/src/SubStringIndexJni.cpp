/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "substring_index.hpp"
#include "cudf_jni_apis.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_SubstringIndex_substringIndexWithScalar(
  JNIEnv* env, jclass, jlong strings_handle, jlong delimiter_handle, jint count) {
  JNI_NULL_CHECK(env, strings_handle, "strings column handle is null", 0);
  JNI_NULL_CHECK(env, delimiter_handle, "delimiter scalar handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    cudf::strings_column_view strings_view{*reinterpret_cast<cudf::column_view const*>(strings_handle)};
    //auto strings_view = reinterpret_cast<cudf::strings_column_view*>(strings_handle);
    auto delimiter_scalar = reinterpret_cast<cudf::string_scalar*>(delimiter_handle);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::substring_index(strings_view, *delimiter_scalar, count));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_SubstringIndex_substringIndexWithColumn(
  JNIEnv* env, jclass, jlong strings_handle, jlong delimiter_handle, jint count) {
  JNI_NULL_CHECK(env, strings_handle, "strings column handle is null", 0);
  JNI_NULL_CHECK(env, delimiter_handle, "delimiter column handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);

    cudf::strings_column_view strings_view{*reinterpret_cast<cudf::column_view const*>(strings_handle)};
    cudf::strings_column_view delimiter_view{*reinterpret_cast<cudf::column_view const*>(delimiter_handle)};
    //auto strings_view = reinterpret_cast<cudf::strings_column_view*>(strings_handle);
    //auto delimiter_view = reinterpret_cast<cudf::strings_column_view*>(delimiter_handle);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::substring_index(strings_view, delimiter_view, count));
  }
  CATCH_STD(env, 0);
}

} // extern "C"
