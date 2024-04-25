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
#include "dtype_utils.hpp"
#include "parse_uri.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParseURI_parseProtocol(JNIEnv* env,
                                                                                jclass,
                                                                                jlong input_column)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const*>(input_column);
    return cudf::jni::ptr_as_jlong(spark_rapids_jni::parse_uri_to_protocol(*input).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParseURI_parseHost(JNIEnv* env,
                                                                            jclass,
                                                                            jlong input_column)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const*>(input_column);
    return cudf::jni::ptr_as_jlong(spark_rapids_jni::parse_uri_to_host(*input).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParseURI_parseQuery(JNIEnv* env,
                                                                             jclass,
                                                                             jlong input_column)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const*>(input_column);
    return cudf::jni::ptr_as_jlong(spark_rapids_jni::parse_uri_to_query(*input).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParseURI_parseQueryWithLiteral(
  JNIEnv* env, jclass, jlong input_column, jstring query)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, query, "query is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const*>(input_column);
    cudf::jni::native_jstring native_query(env, query);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::parse_uri_to_query(*input, native_query.get()).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParseURI_parseQueryWithColumn(
  JNIEnv* env, jclass, jlong input_column, jlong query_column)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_NULL_CHECK(env, query_column, "query column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const*>(input_column);
    auto const query = reinterpret_cast<cudf::column_view const*>(query_column);
    return cudf::jni::ptr_as_jlong(spark_rapids_jni::parse_uri_to_query(*input, *query).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ParseURI_parsePath(JNIEnv* env,
                                                                            jclass,
                                                                            jlong input_column)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto const input = reinterpret_cast<cudf::column_view const*>(input_column);
    return cudf::jni::ptr_as_jlong(spark_rapids_jni::parse_uri_to_path(*input).release());
  }
  CATCH_STD(env, 0);
}
}
