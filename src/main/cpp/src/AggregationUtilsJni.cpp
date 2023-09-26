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

#include "aggregation_utils.hpp"
#include "cudf_jni_apis.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_AggregationUtils_percentileFromHistogram(
    JNIEnv *env, jclass, jlong input_handle, jlong percentage_handle) {
  JNI_NULL_CHECK(env, input_handle, "input_handle is null", 0);
  JNI_NULL_CHECK(env, percentage_handle, "percentage_handle is null", 0);

  try {
    cudf::jni::auto_set_device(env);

    auto const value = reinterpret_cast<cudf::column_view const *>(input_handle);
    auto const percentage = reinterpret_cast<cudf::column_view const *>(percentage_handle);

    return cudf::jni::ptr_as_jlong(
        spark_rapids_jni::percentile_from_histogram(*value, *percentage).release());
  }
  CATCH_STD(env, 0);
}

} // extern "C"
