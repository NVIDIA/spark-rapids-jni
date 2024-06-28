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
#include "hash.hpp"
#include "jni_utils.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Hash_murmurHash32(
  JNIEnv* env, jclass, jint seed, jlongArray column_handles)
{
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto column_views =
      cudf::jni::native_jpointerArray<cudf::column_view>{env, column_handles}.get_dereferenced();
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::murmur_hash3_32(cudf::table_view{column_views}, seed));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Hash_xxhash64(JNIEnv* env,
                                                                       jclass,
                                                                       jlong seed,
                                                                       jlongArray column_handles)
{
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto column_views =
      cudf::jni::native_jpointerArray<cudf::column_view>{env, column_handles}.get_dereferenced();
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::xxhash64(cudf::table_view{column_views}, seed));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Hash_hiveHash(JNIEnv* env,
                                                                       jclass,
                                                                       jlongArray column_handles)
{
  JNI_NULL_CHECK(env, column_handles, "array of column handles is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto column_views =
      cudf::jni::native_jpointerArray<cudf::column_view>{env, column_handles}.get_dereferenced();
    return cudf::jni::release_as_jlong(spark_rapids_jni::hive_hash(cudf::table_view{column_views}));
  }
  CATCH_STD(env, 0);
}
}
