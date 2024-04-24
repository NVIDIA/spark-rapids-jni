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
#include "dtype_utils.hpp"
#include "jni_utils.hpp"
#include "string_digits_pattern.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_StringDigitsPattern_stringDigitsPattern(
  JNIEnv* env, jclass, jlong column_view, jlong target, jint d)
{
  JNI_NULL_CHECK(env, column_view, "input column is null", 0);
  JNI_NULL_CHECK(env, target, "target is null", 0);

  try {
    cudf::jni::auto_set_device(env);

    // printf("StringDigitsPatternJni.cpp:
    // Java_com_nvidia_spark_rapids_jni_StringDigitsPattern_stringDigitsPattern\n");

    cudf::column_view* cv = reinterpret_cast<cudf::column_view*>(column_view);
    cudf::strings_column_view scv(*cv);
    cudf::string_scalar* ss_scalar = reinterpret_cast<cudf::string_scalar*>(target);
    return cudf::jni::release_as_jlong(spark_rapids_jni::string_digits_pattern(scv, *ss_scalar, d));
  }
  CATCH_STD(env, 0);
}
}