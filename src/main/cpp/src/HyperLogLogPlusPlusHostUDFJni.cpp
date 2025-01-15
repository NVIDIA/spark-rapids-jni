/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include "hyper_log_log_plus_plus.hpp"
#include "hyper_log_log_plus_plus_host_udf.hpp"

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_HyperLogLogPlusPlusHostUDF_createHLLPPHostUDF(JNIEnv* env,
                                                                               jclass,
                                                                               jint agg_type,
                                                                               int precision)
{
  try {
    cudf::jni::auto_set_device(env);
    auto udf_ptr = [&] {
      // The value of agg_type must be sync with
      // `HyperLogLogPlusPlusHostUDF.java#AggregationType`.
      switch (agg_type) {
        case 0: return spark_rapids_jni::create_hllpp_reduction_host_udf(precision);
        case 1: return spark_rapids_jni::create_hllpp_reduction_merge_host_udf(precision);
        case 2: return spark_rapids_jni::create_hllpp_groupby_host_udf(precision);
        case 3: return spark_rapids_jni::create_hllpp_groupby_merge_host_udf(precision);
        default: CUDF_FAIL("Invalid aggregation type.");
      }
    }();
    CUDF_EXPECTS(udf_ptr != nullptr, "Invalid HyperLogLogPlusPlus(HLLPP) UDF instance.");

    return reinterpret_cast<jlong>(udf_ptr.release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_HyperLogLogPlusPlusHostUDF_estimateDistinctValueFromSketches(
  JNIEnv* env, jclass, jlong sketches, jint precision)
{
  JNI_NULL_CHECK(env, sketches, "Sketch column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const sketch_view = reinterpret_cast<cudf::column_view const*>(sketches);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::estimate_from_hll_sketches(*sketch_view, precision).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_HyperLogLogPlusPlusHostUDF_close(
  JNIEnv* env, jclass class_object, jlong ptr)
try {
  cudf::jni::auto_set_device(env);
  auto to_del = reinterpret_cast<cudf::host_udf_base*>(ptr);
  delete to_del;
}
CATCH_STD(env, );
}

}  // extern "C"
