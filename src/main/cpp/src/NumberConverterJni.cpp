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

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_NumberConverter_convertCvCvCv(
  JNIEnv* env, jclass, jlong input, jlong from_base, jlong to_base)
{
  JNI_NULL_CHECK(env, input, "input column handle is null", 0);
  JNI_NULL_CHECK(env, from_base, "from_base column handle is null", 0);
  JNI_NULL_CHECK(env, to_base, "to_base column handle is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto input_view{*reinterpret_cast<cudf::column_view const*>(input)};
    auto from_base_view{*reinterpret_cast<cudf::column_view const*>(from_base)};
    auto to_base_view{*reinterpret_cast<cudf::column_view const*>(to_base)};

    return cudf::jni::release_as_jlong(
      spark_rapids_jni::convert_cv_cv_cv(input_view, from_base_view, to_base_view));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_NumberConverter_convertCvCvS(
  JNIEnv* env, jclass, jlong input, jlong from_base, int to_base)
{
  JNI_NULL_CHECK(env, input, "input column handle is null", 0);
  JNI_NULL_CHECK(env, from_base, "from_base column handle is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto input_view{*reinterpret_cast<cudf::column_view const*>(input)};
    auto from_base_view{*reinterpret_cast<cudf::column_view const*>(from_base)};

    return cudf::jni::release_as_jlong(
      spark_rapids_jni::convert_cv_cv_s(input_view, from_base_view, to_base));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_NumberConverter_convertCvSCv(
  JNIEnv* env, jclass, jlong input, int from_base, jlong to_base)
{
  JNI_NULL_CHECK(env, input, "input column handle is null", 0);
  JNI_NULL_CHECK(env, to_base, "to_base column handle is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto input_view{*reinterpret_cast<cudf::column_view const*>(input)};
    auto to_base_view{*reinterpret_cast<cudf::column_view const*>(to_base)};

    return cudf::jni::release_as_jlong(
      spark_rapids_jni::convert_cv_s_cv(input_view, from_base, to_base_view));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_NumberConverter_convertCvSS(
  JNIEnv* env, jclass, jlong input, int from_base, int to_base)
{
  JNI_NULL_CHECK(env, input, "input column handle is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto input_view{*reinterpret_cast<cudf::column_view const*>(input)};

    return cudf::jni::release_as_jlong(
      spark_rapids_jni::convert_cv_s_s(input_view, from_base, to_base));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_NumberConverter_isConvertOverflowCvCvCv(
  JNIEnv* env, jclass, jlong input, jlong from_base, jlong to_base)
{
  JNI_NULL_CHECK(env, input, "input column handle is null", 0);
  JNI_NULL_CHECK(env, from_base, "from_base column handle is null", 0);
  JNI_NULL_CHECK(env, to_base, "to_base column handle is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto input_view{*reinterpret_cast<cudf::column_view const*>(input)};
    auto from_base_view{*reinterpret_cast<cudf::column_view const*>(from_base)};
    auto to_base_view{*reinterpret_cast<cudf::column_view const*>(to_base)};
    return spark_rapids_jni::is_convert_overflow_cv_cv_cv(input_view, from_base_view, to_base_view);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_NumberConverter_isConvertOverflowCvCvS(
  JNIEnv* env, jclass, jlong input, jlong from_base, int to_base)
{
  JNI_NULL_CHECK(env, input, "input column handle is null", 0);
  JNI_NULL_CHECK(env, from_base, "from_base column handle is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto input_view{*reinterpret_cast<cudf::column_view const*>(input)};
    auto from_base_view{*reinterpret_cast<cudf::column_view const*>(from_base)};
    return spark_rapids_jni::is_convert_overflow_cv_cv_s(input_view, from_base_view, to_base);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_NumberConverter_isConvertOverflowCvSCv(
  JNIEnv* env, jclass, jlong input, int from_base, jlong to_base)
{
  JNI_NULL_CHECK(env, input, "input column handle is null", 0);
  JNI_NULL_CHECK(env, to_base, "to_base column handle is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto input_view{*reinterpret_cast<cudf::column_view const*>(input)};
    auto to_base_view{*reinterpret_cast<cudf::column_view const*>(to_base)};
    return spark_rapids_jni::is_convert_overflow_cv_s_cv(input_view, from_base, to_base_view);
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_NumberConverter_isConvertOverflowCvSS(
  JNIEnv* env, jclass, jlong input, int from_base, int to_base)
{
  JNI_NULL_CHECK(env, input, "input column handle is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto input_view{*reinterpret_cast<cudf::column_view const*>(input)};
    return spark_rapids_jni::is_convert_overflow_cv_s_s(input_view, from_base, to_base);
  }
  CATCH_STD(env, 0);
}
}
