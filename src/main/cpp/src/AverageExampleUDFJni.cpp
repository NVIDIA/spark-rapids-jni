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

#include "average_example.hpp"
#include "average_example_host_udf.hpp"
#include "cudf_jni_apis.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_AverageExampleUDF_createAverageHostUDF(
  JNIEnv* env, jclass, jint agg_type)
{
  JNI_TRY
  {
    auto udf_ptr = [&] {
      // The value of agg_type must be sync with
      // `AverageExampleUDF.java#AggregationType`.
      switch (agg_type) {
        case 0: return spark_rapids_jni::create_average_example_reduction_host_udf();
        case 1: return spark_rapids_jni::create_average_example_reduction_merge_host_udf();
        case 2: return spark_rapids_jni::create_average_example_groupby_host_udf();
        case 3: return spark_rapids_jni::create_average_example_groupby_merge_host_udf();
        default: CUDF_FAIL("Invalid aggregation type.");
      }
    }();
    CUDF_EXPECTS(udf_ptr != nullptr, "Invalid AverageExample UDF instance.");

    return reinterpret_cast<jlong>(udf_ptr);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_AverageExampleUDF_computeAvg(JNIEnv* env,
                                                                                      jclass,
                                                                                      jlong input)
{
  JNI_NULL_CHECK(env, input, "input column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input_view = reinterpret_cast<cudf::column_view const*>(input);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::compute_average_example(*input_view).release());
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
