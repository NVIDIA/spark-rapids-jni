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

#include "cudf_jni_apis.hpp"
#include "substring_index.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GpuSubstringIndexUtils_substringIndex(
  JNIEnv* env, jclass, jlong strings_handle, jstring delimiter_object, jint count)
{
  JNI_NULL_CHECK(env, strings_handle, "strings column handle is null", 0);
  JNI_NULL_CHECK(env, delimiter_object, "delimiter scalar handle is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const input          = reinterpret_cast<cudf::column_view const*>(strings_handle);
    auto const strings_column = cudf::strings_column_view{*input};
    auto const delimiter_jstr = cudf::jni::native_jstring(env, delimiter_object);
    auto const delimiter      = std::string(delimiter_jstr.get(), delimiter_jstr.size_bytes());
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::substring_index(strings_column, cudf::string_scalar{delimiter}, count));
  }
  CATCH_STD(env, 0);
}
}  // extern "C"
