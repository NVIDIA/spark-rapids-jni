/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include "datetime_utils.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_DateTimeUtils_rebaseGregorianToJulian(
  JNIEnv* env, jclass, jlong input)
{
  JNI_NULL_CHECK(env, input, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input_cv = reinterpret_cast<cudf::column_view const*>(input);
    auto output         = spark_rapids_jni::rebase_gregorian_to_julian(*input_cv);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_DateTimeUtils_rebaseJulianToGregorian(
  JNIEnv* env, jclass, jlong input)
{
  JNI_NULL_CHECK(env, input, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input_cv = reinterpret_cast<cudf::column_view const*>(input);
    auto output         = spark_rapids_jni::rebase_julian_to_gregorian(*input_cv);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_DateTimeUtils_truncate(JNIEnv* env,
                                                                                jclass,
                                                                                jlong input,
                                                                                jstring component)
{
  JNI_NULL_CHECK(env, input, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input_cv       = reinterpret_cast<cudf::column_view const*>(input);
    auto const component_jstr = cudf::jni::native_jstring(env, component);
    auto const component_str  = std::string(component_jstr.get(), component_jstr.size_bytes());
    auto output               = spark_rapids_jni::truncate(*input_cv, component_str);
    return reinterpret_cast<jlong>(output.release());
  }
  CATCH_STD(env, 0);
}

}  // extern "C"
