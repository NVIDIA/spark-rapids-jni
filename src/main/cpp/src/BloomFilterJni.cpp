/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

#include <limits>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_creategpu(
  JNIEnv* env, jclass, jint version, jint numHashes, jlong bloomFilterBits, jint seed)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // TODO (future): There is an impedance mismatch between the C++ and Java APIs.
    // This seems to have been introduced in https://github.com/NVIDIA/spark-rapids-jni/pull/1303.
    // The Java API accepts a long for the bloom filter bit count, but the C++ API accepts an int.
    // This means that the Java API can represent a bloom filter bit count that is too large to
    // be represented as an int in the C++ API.
    // We should fix this by changing the C++ API to accept a long for the bloom filter bit count.
    // We will address this in a future PR.  For now, we add error checking to avoid overflow.
    JNI_ARG_CHECK(env,
                  bloomFilterBits >= 0 && bloomFilterBits <= std::numeric_limits<int>::max() - 63,
                  "bloom filter bit count overflows int when converted to longs",
                  0);
    auto const bloom_filter_longs_long = (bloomFilterBits + 63) / 64;
    auto const bloom_filter_longs      = static_cast<int>(bloom_filter_longs_long);
    auto bloom_filter =
      spark_rapids_jni::bloom_filter_create(version, numHashes, bloom_filter_longs, seed);
    return reinterpret_cast<jlong>(bloom_filter.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_put(JNIEnv* env,
                                                                        jclass,
                                                                        jlong bloomFilter,
                                                                        jlong cv)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_column = *reinterpret_cast<cudf::column_view const*>(cv);
    spark_rapids_jni::bloom_filter_put(*(reinterpret_cast<cudf::list_scalar*>(bloomFilter)),
                                       input_column);
    return 0;
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_merge(JNIEnv* env,
                                                                           jclass,
                                                                           jlong bloomFilters)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_bloom_filter =
      *reinterpret_cast<cudf::column_view const*>(bloomFilters);
    auto bloom_filter = spark_rapids_jni::bloom_filter_merge(input_bloom_filter);
    return reinterpret_cast<jlong>(bloom_filter.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_probe(JNIEnv* env,
                                                                           jclass,
                                                                           jlong bloomFilter,
                                                                           jlong cv)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_column = *reinterpret_cast<cudf::column_view const*>(cv);
    return cudf::jni::release_as_jlong(spark_rapids_jni::bloom_filter_probe(
      input_column, *(reinterpret_cast<cudf::list_scalar*>(bloomFilter))));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_probebuffer(
  JNIEnv* env, jclass, jlong bloomFilter, jlong bloomFilterSize, jlong cv)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    cudf::column_view const& input_column = *reinterpret_cast<cudf::column_view const*>(cv);
    auto buf                              = reinterpret_cast<uint8_t const*>(bloomFilter);
    return cudf::jni::release_as_jlong(spark_rapids_jni::bloom_filter_probe(
      input_column, cudf::device_span<uint8_t const>{buf, static_cast<size_t>(bloomFilterSize)}));
  }
  JNI_CATCH(env, 0);
}
}
