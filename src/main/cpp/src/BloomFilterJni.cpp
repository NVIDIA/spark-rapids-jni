/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "bloom_filter.hpp"
#include "utilities.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_build(
  JNIEnv* env, jclass, jlong bloom_filter, jlong bloom_filter_bytes, jint bloom_filter_bits, jlong cv, jint num_hashes)
{
  try {
    cudf::jni::auto_set_device(env);

    cudf::column_view input_column{*reinterpret_cast<cudf::column_view const*>(cv)};
    spark_rapids_jni::bloom_filter_build({reinterpret_cast<cudf::bitmask_type*>(bloom_filter), static_cast<std::size_t>(bloom_filter_bytes / 4)},
                                         bloom_filter_bits,
                                         input_column,
                                         num_hashes);
    return 0;
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_probe(
  JNIEnv* env, jclass, jlong cv, jlong bloom_filter, jlong bloom_filter_bytes, jint bloom_filter_bits, jint num_hashes)
{
  try {
    cudf::jni::auto_set_device(env);

    cudf::column_view input_column{*reinterpret_cast<cudf::column_view const*>(cv)};
    return cudf::jni::release_as_jlong(spark_rapids_jni::bloom_filter_probe(input_column,
                                                                            {reinterpret_cast<cudf::bitmask_type const*>(bloom_filter), static_cast<std::size_t>(bloom_filter_bytes / 4)},
                                                                            bloom_filter_bits,
                                                                            num_hashes));
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_BloomFilter_merge(
  JNIEnv* env, jclass, jlongArray bloom_filters, jlong bloom_filter_bytes)
{
  try {
    cudf::jni::auto_set_device(env);
        
    cudf::jni::native_jpointerArray<cudf::bitmask_type> jbuffers{env, bloom_filters};
    std::vector<cudf::device_span<cudf::bitmask_type const>> cbloom_filters(jbuffers.size());
    std::transform(jbuffers.begin(), jbuffers.end(), cbloom_filters.begin(), [bloom_filter_bytes](cudf::bitmask_type const* buf){
      return cudf::device_span<cudf::bitmask_type const>{buf, static_cast<std::size_t>(bloom_filter_bytes / 4)};
    });    
  
    auto merged = spark_rapids_jni::bitmask_bitwise_or(cbloom_filters);

    cudf::jni::native_jlongArray result(env, 2);
    result[0] = cudf::jni::ptr_as_jlong(merged->data());
    result[1] = cudf::jni::release_as_jlong(merged);
    return result.get_jArray();
  }
  CATCH_STD(env, 0);
}

}