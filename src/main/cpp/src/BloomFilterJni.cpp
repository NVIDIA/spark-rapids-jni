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

#include "bloom_filter.hpp"
#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"
#include "jni_utils.hpp"
#include "utilities.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_creategpu(
  JNIEnv* env, jclass, jint numHashes, jlong bloomFilterBits)
{
  try {
    cudf::jni::auto_set_device(env);

    int bloom_filter_longs = static_cast<int>((bloomFilterBits + 63) / 64);
    auto bloom_filter      = spark_rapids_jni::bloom_filter_create(numHashes, bloom_filter_longs);
    return reinterpret_cast<jlong>(bloom_filter.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_put(JNIEnv* env,
                                                                        jclass,
                                                                        jlong bloomFilter,
                                                                        jlong cv)
{
  try {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_column = *reinterpret_cast<cudf::column_view const*>(cv);
    spark_rapids_jni::bloom_filter_put(*(reinterpret_cast<cudf::list_scalar*>(bloomFilter)),
                                       input_column);
    return 0;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_merge(JNIEnv* env,
                                                                           jclass,
                                                                           jlong bloomFilters)
{
  try {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_bloom_filter =
      *reinterpret_cast<cudf::column_view const*>(bloomFilters);
    auto bloom_filter = spark_rapids_jni::bloom_filter_merge(input_bloom_filter);
    return reinterpret_cast<jlong>(bloom_filter.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_probe(JNIEnv* env,
                                                                           jclass,
                                                                           jlong bloomFilter,
                                                                           jlong cv)
{
  try {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_column = *reinterpret_cast<cudf::column_view const*>(cv);
    return cudf::jni::release_as_jlong(spark_rapids_jni::bloom_filter_probe(
      input_column, *(reinterpret_cast<cudf::list_scalar*>(bloomFilter))));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_probebuffer(
  JNIEnv* env, jclass, jlong bloomFilter, jlong bloomFilterSize, jlong cv)
{
  try {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_column = *reinterpret_cast<cudf::column_view const*>(cv);
    auto buf                              = reinterpret_cast<uint8_t const*>(bloomFilter);
    return cudf::jni::release_as_jlong(spark_rapids_jni::bloom_filter_probe(
      input_column, cudf::device_span<uint8_t const>{buf, static_cast<size_t>(bloomFilterSize)}));
  }
  CATCH_STD(env, 0);
}
}
