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

#include "cast_string.hpp"

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"
#include "jni_utils.hpp"

constexpr char const* JNI_CAST_ERROR_CLASS = "com/nvidia/spark/rapids/jni/CastException";

#define CATCH_CAST_EXCEPTION(env, ret_val)                                                \
  catch (const spark_rapids_jni::cast_error& e)                                           \
  {                                                                                       \
    if (env->ExceptionOccurred()) { return ret_val; }                                     \
    jclass ex_class = env->FindClass(JNI_CAST_ERROR_CLASS);                               \
    if (ex_class != NULL) {                                                               \
      jmethodID ctor_id = env->GetMethodID(ex_class, "<init>", "(Ljava/lang/String;I)V"); \
      if (ctor_id != NULL) {                                                              \
        std::string n_msg = e.get_string_with_error();                                    \
        jstring j_msg     = env->NewStringUTF(n_msg.c_str());                             \
        if (j_msg != NULL) {                                                              \
          jint e_row         = static_cast<jint>(e.get_row_number());                     \
          jobject cuda_error = env->NewObject(ex_class, ctor_id, j_msg, e_row);           \
          if (cuda_error != NULL) { env->Throw((jthrowable)cuda_error); }                 \
        }                                                                                 \
      }                                                                                   \
    }                                                                                     \
    return ret_val;                                                                       \
  }                                                                                       \
  CATCH_STD(env, 0);

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CastStrings_toInteger(
  JNIEnv* env, jclass, jlong input_column, jboolean ansi_enabled, jint j_dtype)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);

    cudf::strings_column_view scv{*reinterpret_cast<cudf::column_view const*>(input_column)};
    return cudf::jni::release_as_jlong(spark_rapids_jni::string_to_integer(
      cudf::jni::make_data_type(j_dtype, 0), scv, ansi_enabled, cudf::default_stream_value));
  }
  CATCH_CAST_EXCEPTION(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CastStrings_toDecimal(
  JNIEnv* env, jclass, jlong input_column, jboolean ansi_enabled, jint precision, jint scale)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);

    cudf::strings_column_view scv{*reinterpret_cast<cudf::column_view const*>(input_column)};
    return cudf::jni::release_as_jlong(spark_rapids_jni::string_to_decimal(
      precision, scale, scv, ansi_enabled, cudf::default_stream_value));
  }
  CATCH_CAST_EXCEPTION(env, 0);
}
}