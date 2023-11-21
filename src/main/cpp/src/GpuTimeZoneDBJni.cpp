/* Copyright (c) 2023, NVIDIA CORPORATION.
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
#include "timezones.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GpuTimeZoneDB_convertTimestampColumnToUTC(
  JNIEnv* env, jclass, jlong input_handle, jlong transitions_handle, jint tz_index)
{
  JNI_NULL_CHECK(env, input_handle, "column is null", 0);
  JNI_NULL_CHECK(env, transitions_handle, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const input       = reinterpret_cast<cudf::column_view const*>(input_handle);
    auto const transitions = reinterpret_cast<cudf::table_view const*>(transitions_handle);
    auto const index       = static_cast<cudf::size_type>(tz_index);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::convert_timestamp_to_utc(*input, *transitions, index).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_GpuTimeZoneDB_convertUTCTimestampColumnToTimeZone(
  JNIEnv* env, jclass, jlong input_handle, jlong transitions_handle, jint tz_index)
{
  JNI_NULL_CHECK(env, input_handle, "column is null", 0);
  JNI_NULL_CHECK(env, transitions_handle, "column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const input       = reinterpret_cast<cudf::column_view const*>(input_handle);
    auto const transitions = reinterpret_cast<cudf::table_view const*>(transitions_handle);
    auto const index       = static_cast<cudf::size_type>(tz_index);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::convert_utc_timestamp_to_timezone(*input, *transitions, index).release());
  }
  CATCH_STD(env, 0);
}
}
