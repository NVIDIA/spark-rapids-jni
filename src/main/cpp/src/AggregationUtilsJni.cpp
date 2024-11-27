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

#include "aggregation_utils.hpp"
#include "cudf_jni_apis.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_AggregationUtils_createNativeTestHostUDF(
  JNIEnv* env, jclass, jint agg_type)
{
  try {
    cudf::jni::auto_set_device(env);
    auto udf_ptr = [&] {
      // The value of agg_type must be in sync with `AggregationUtils.java#AggregationType`.
      switch (agg_type) {
        case 0: return spark_rapids_jni::create_test_reduction_host_udf();
        case 1: return spark_rapids_jni::create_test_segmented_reduction_host_udf();
        default: return spark_rapids_jni::create_test_groupby_host_udf();
      }
    }();
    CUDF_EXPECTS(udf_ptr != nullptr, "Invalid host udf instance.");

    return reinterpret_cast<jlong>(udf_ptr.release());
  }
  CATCH_STD(env, 0);
}

}  // extern "C"
