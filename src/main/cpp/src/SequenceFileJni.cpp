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

/**
 * Parse multiple SequenceFiles from a combined buffer.
 *
 * @param j_data_address Address of the combined device buffer
 * @param j_file_offsets Array of file offsets in the combined buffer
 * @param j_file_sizes Array of file sizes
 * @param j_sync_markers 2D array of sync markers (one 16-byte array per file)
 * @param j_wants_key Whether to extract keys
 * @param j_wants_value Whether to extract values
 * @return Object array: [keyColumn (long), valueColumn (long), fileRowCounts (int[]), totalRows (int)]
 */
JNIEXPORT jobject JNICALL
Java_com_nvidia_spark_rapids_jni_SequenceFile_parseMultipleFilesNative(JNIEnv* env,
                                                                       jclass,
                                                                       jlong j_data_address,
                                                                       jlongArray j_file_offsets,
                                                                       jlongArray j_file_sizes,
                                                                       jobjectArray j_sync_markers,
                                                                       jboolean j_wants_key,
                                                                       jboolean j_wants_value)
{
  JNI_NULL_CHECK(env, j_file_offsets, "file_offsets is null", nullptr);
  JNI_NULL_CHECK(env, j_file_sizes, "file_sizes is null", nullptr);
  JNI_NULL_CHECK(env, j_sync_markers, "sync_markers is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Get array lengths
    jsize const num_files = env->GetArrayLength(j_file_offsets);
    if (env->GetArrayLength(j_file_sizes) != num_files ||
        env->GetArrayLength(j_sync_markers) != num_files) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "file_offsets, file_sizes, and sync_markers must have the same length",
                    nullptr);
    }

    // Convert file offsets and sizes
    auto const file_offsets = cudf::jni::native_jlongArray(env, j_file_offsets);
    auto const file_sizes   = cudf::jni::native_jlongArray(env, j_file_sizes);

    // Build file descriptors
    std::vector<spark_rapids_jni::file_descriptor> file_descs(num_files);
    for (jsize i = 0; i < num_files; ++i) {
      file_descs[i].data_offset = static_cast<int64_t>(file_offsets[i]);
      file_descs[i].data_size   = static_cast<int64_t>(file_sizes[i]);

      // Get sync marker for this file
      auto j_sync = static_cast<jbyteArray>(env->GetObjectArrayElement(j_sync_markers, i));
      if (j_sync == nullptr) {
        JNI_THROW_NEW(
          env, cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS, "sync_marker is null for file", nullptr);
      }
      auto const sync_marker_raw = cudf::jni::native_jbyteArray(env, j_sync);
      if (sync_marker_raw.size() != spark_rapids_jni::SYNC_MARKER_SIZE) {
        JNI_THROW_NEW(env,
                      cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                      "sync_marker must be exactly 16 bytes",
                      nullptr);
      }
      std::copy(sync_marker_raw.begin(), sync_marker_raw.end(), file_descs[i].sync_marker);
    }

    auto const data_ptr    = reinterpret_cast<uint8_t const*>(j_data_address);
    bool const wants_key   = j_wants_key;
    bool const wants_value = j_wants_value;

    // Parse multiple files
    auto result =
      spark_rapids_jni::parse_multiple_sequence_files(data_ptr, file_descs, wants_key, wants_value);

    // Get the MultiFileParseResult class
    jclass result_class = env->FindClass("com/nvidia/spark/rapids/jni/SequenceFile$MultiFileParseResult");
    if (result_class == nullptr) {
      JNI_THROW_NEW(env, cudf::jni::RUNTIME_EXCEPTION_CLASS, "Cannot find MultiFileParseResult class", nullptr);
    }

    // Get constructor: (long keyColumnHandle, long valueColumnHandle, int[] fileRowCounts, int totalRows)
    jmethodID constructor = env->GetMethodID(result_class, "<init>", "(JJ[II)V");
    if (constructor == nullptr) {
      JNI_THROW_NEW(env, cudf::jni::RUNTIME_EXCEPTION_CLASS, "Cannot find MultiFileParseResult constructor", nullptr);
    }

    // Convert file row counts to Java int array
    jintArray j_file_row_counts = env->NewIntArray(num_files);
    if (j_file_row_counts == nullptr) {
      JNI_THROW_NEW(env, cudf::jni::RUNTIME_EXCEPTION_CLASS, "Failed to allocate int array", nullptr);
    }
    env->SetIntArrayRegion(j_file_row_counts, 0, num_files, result.file_row_counts.data());

    // Get column handles (0 if null)
    jlong key_handle   = result.key_column ? cudf::jni::release_as_jlong(result.key_column) : 0;
    jlong value_handle = result.value_column ? cudf::jni::release_as_jlong(result.value_column) : 0;

    // Create result object
    jobject result_obj = env->NewObject(result_class,
                                        constructor,
                                        key_handle,
                                        value_handle,
                                        j_file_row_counts,
                                        static_cast<jint>(result.total_rows));

    return result_obj;
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"
