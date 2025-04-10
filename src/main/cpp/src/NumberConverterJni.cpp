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
#include "number_converter.hpp"

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_NumberConverter_convert(JNIEnv* env,
                                                         jclass,
                                                         jlong input,
                                                         jboolean is_input_cv,
                                                         jlong from_base,
                                                         jboolean is_from_cv,
                                                         jlong to_base,
                                                         jboolean is_to_cv)
{
  JNI_NULL_CHECK(env, input, "input column/scalar handle is null", 0);
  if (is_from_cv) { JNI_NULL_CHECK(env, from_base, "from_base column handle is null", 0); }
  if (is_to_cv) { JNI_NULL_CHECK(env, to_base, "to_base column handle is null", 0); }

  try {
    cudf::jni::auto_set_device(env);

    spark_rapids_jni::convert_number_t input_variant = [&] {
      if (is_input_cv) {
        spark_rapids_jni::convert_number_t t = *reinterpret_cast<cudf::column_view*>(input);
        return t;
      } else {
        spark_rapids_jni::convert_number_t t = *reinterpret_cast<cudf::string_scalar*>(input);
        return t;
      }
    }();
    spark_rapids_jni::convert_number_t from_base_variant = [&] {
      if (is_from_cv) {
        spark_rapids_jni::convert_number_t t = *reinterpret_cast<cudf::column_view*>(from_base);
        return t;
      } else {
        spark_rapids_jni::convert_number_t t = static_cast<int>(from_base);
        return t;
      }
    }();
    spark_rapids_jni::convert_number_t to_base_variant = [&] {
      if (is_to_cv) {
        spark_rapids_jni::convert_number_t t = *reinterpret_cast<cudf::column_view*>(to_base);
        return t;
      } else {
        spark_rapids_jni::convert_number_t t = static_cast<int>(to_base);
        return t;
      }
    }();

    return cudf::jni::release_as_jlong(
      spark_rapids_jni::convert(input_variant, from_base_variant, to_base_variant));
    return 0L;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jboolean JNICALL
Java_com_nvidia_spark_rapids_jni_NumberConverter_isConvertOverflow(JNIEnv* env,
                                                                   jclass,
                                                                   jlong input,
                                                                   jboolean is_input_cv,
                                                                   jlong from_base,
                                                                   jboolean is_from_cv,
                                                                   jlong to_base,
                                                                   jboolean is_to_cv)
{
  JNI_NULL_CHECK(env, input, "input column/scalar handle is null", 0);
  if (is_from_cv) { JNI_NULL_CHECK(env, from_base, "from_base column handle is null", 0); }
  if (is_to_cv) { JNI_NULL_CHECK(env, to_base, "to_base column handle is null", 0); }

  try {
    cudf::jni::auto_set_device(env);

    spark_rapids_jni::convert_number_t input_variant = [&] {
      if (is_input_cv) {
        spark_rapids_jni::convert_number_t t = *reinterpret_cast<cudf::column_view*>(input);
        return t;
      } else {
        spark_rapids_jni::convert_number_t t = *reinterpret_cast<cudf::string_scalar*>(input);
        return t;
      }
    }();
    spark_rapids_jni::convert_number_t from_base_variant = [&] {
      if (is_from_cv) {
        spark_rapids_jni::convert_number_t t = *reinterpret_cast<cudf::column_view*>(from_base);
        return t;
      } else {
        spark_rapids_jni::convert_number_t t = static_cast<int>(from_base);
        return t;
      }
    }();
    spark_rapids_jni::convert_number_t to_base_variant = [&] {
      if (is_to_cv) {
        spark_rapids_jni::convert_number_t t = *reinterpret_cast<cudf::column_view*>(to_base);
        return t;
      } else {
        spark_rapids_jni::convert_number_t t = static_cast<int>(to_base);
        return t;
      }
    }();

    return spark_rapids_jni::is_convert_overflow(input_variant, from_base_variant, to_base_variant);
    return 0L;
  }
  CATCH_STD(env, 0);
}
}
