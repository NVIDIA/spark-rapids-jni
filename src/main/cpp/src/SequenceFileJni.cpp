/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include "sequence_file.hpp"

#include <vector>

extern "C" {

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_SequenceFile_parseSequenceFileNative(JNIEnv* env,
                                                                      jclass,
                                                                      jlong j_data_address,
                                                                      jlong j_data_size,
                                                                      jbyteArray j_sync_marker,
                                                                      jboolean j_wants_key,
                                                                      jboolean j_wants_value)
{
  JNI_NULL_CHECK(env, j_sync_marker, "sync_marker is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Convert sync marker from Java byte array to std::vector
    auto const sync_marker_raw = cudf::jni::native_jbyteArray(env, j_sync_marker);
    if (sync_marker_raw.size() != spark_rapids_jni::SYNC_MARKER_SIZE) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "sync_marker must be exactly 16 bytes",
                    nullptr);
    }
    std::vector<uint8_t> sync_marker(sync_marker_raw.begin(), sync_marker_raw.end());

    auto const data_ptr    = reinterpret_cast<uint8_t const*>(j_data_address);
    auto const data_size   = static_cast<size_t>(j_data_size);
    bool const wants_key   = j_wants_key;
    bool const wants_value = j_wants_value;

    // Parse the SequenceFile
    auto result = spark_rapids_jni::parse_sequence_file(
      data_ptr, data_size, sync_marker, wants_key, wants_value);

    // Build the result array
    std::vector<jlong> column_handles;
    if (wants_key && result.key_column) {
      column_handles.push_back(cudf::jni::release_as_jlong(result.key_column));
    }
    if (wants_value && result.value_column) {
      column_handles.push_back(cudf::jni::release_as_jlong(result.value_column));
    }

    // Create and return Java long array
    auto out_handles = cudf::jni::native_jlongArray(env, column_handles.size());
    for (size_t i = 0; i < column_handles.size(); ++i) {
      out_handles[i] = column_handles[i];
    }
    return out_handles.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_SequenceFile_countRecordsNative(
  JNIEnv* env, jclass, jlong j_data_address, jlong j_data_size, jbyteArray j_sync_marker)
{
  JNI_NULL_CHECK(env, j_sync_marker, "sync_marker is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Convert sync marker from Java byte array to std::vector
    auto const sync_marker_raw = cudf::jni::native_jbyteArray(env, j_sync_marker);
    if (sync_marker_raw.size() != spark_rapids_jni::SYNC_MARKER_SIZE) {
      JNI_THROW_NEW(
        env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "sync_marker must be exactly 16 bytes", 0);
    }
    std::vector<uint8_t> sync_marker(sync_marker_raw.begin(), sync_marker_raw.end());

    auto const data_ptr  = reinterpret_cast<uint8_t const*>(j_data_address);
    auto const data_size = static_cast<size_t>(j_data_size);

    return static_cast<jlong>(spark_rapids_jni::count_records(data_ptr, data_size, sync_marker));
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
