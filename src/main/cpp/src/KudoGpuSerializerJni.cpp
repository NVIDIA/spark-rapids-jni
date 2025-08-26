/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include "shuffle_split.hpp"

extern "C" {

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_kudo_KudoGpuSerializer_splitAndSerializeToDevice(
  JNIEnv* env, jclass, jlong j_table_view, jintArray j_splits)
{
  JNI_NULL_CHECK(env, j_table_view, "table is null", NULL);
  JNI_NULL_CHECK(env, j_splits, "splits is null", NULL);
  try {
    cudf::jni::auto_set_device(env);

    auto table = reinterpret_cast<cudf::table_view const*>(j_table_view);
    const cudf::jni::native_jintArray n_splits(env, j_splits);
    std::vector<cudf::size_type> splits = n_splits.to_vector<int>();

    auto [split_result, split_meta] = spark_rapids_jni::shuffle_split(
      *table, splits, cudf::get_default_stream(), cudf::get_current_device_resource());

    // The following code is ugly. We need to return two device buffers to java, but
    // there is no good way to do this. For this we return three values for each buffer.
    // These values are {data_address, data_size, rmm::device_buffer*}
    // These values are then returned to java and we call into `DeviceMemoryBuffer.fromRmm`
    // That creates a new DeviceMemoryBuffer that takes ownership of the rmm::device_buffer*
    // and will free it when the DeviceMemoryBuffer is closed.
    // To make this work it looks like we pull data out of the rmm::device_buffer and
    // then either leak it or release the memory held by it, but that is not technically
    // the case.
    cudf::jni::native_jlongArray result(env, 6);
    result[0] = reinterpret_cast<jlong>(split_result.partitions->data());
    result[1] = static_cast<jlong>(split_result.partitions->size());
    result[2] = reinterpret_cast<jlong>(split_result.partitions.release());

    // split_result.offsets is an rmm::device_uvector<size_t> so we have to
    // pull out the rmm::device_buffer * from inside it to return the data in a way that
    // java can handle it.
    auto offsets = std::make_unique<rmm::device_buffer>(std::move(split_result.offsets.release()));
    result[3]    = reinterpret_cast<jlong>(offsets->data());
    result[4]    = static_cast<jlong>(offsets->size());
    result[5]    = reinterpret_cast<jlong>(offsets.release());

    return result.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_kudo_KudoGpuSerializer_assembleFromDeviceRawNative(
  JNIEnv* env,
  jclass,
  jlong part_addr,
  jlong part_len,
  jlong offset_addr,
  jlong offset_len,
  jintArray flat_num_children,
  jintArray flat_type_ids,
  jintArray flat_scale)
{
  JNI_NULL_CHECK(env, part_addr, "part_addr is null", NULL);
  JNI_NULL_CHECK(env, offset_addr, "offset_addr is null", NULL);
  JNI_NULL_CHECK(env, flat_num_children, "num_children is null", NULL);
  JNI_NULL_CHECK(env, flat_type_ids, "type_ids is null", NULL);
  JNI_NULL_CHECK(env, flat_scale, "scale is null", NULL);

  try {
    cudf::jni::auto_set_device(env);

    cudf::device_span<uint8_t const> partitions(reinterpret_cast<uint8_t*>(part_addr), part_len);
    cudf::device_span<size_t const> offsets(reinterpret_cast<size_t*>(offset_addr),
                                            offset_len / sizeof(size_t));

    cudf::jni::native_jintArray nnc(env, flat_num_children);
    cudf::jni::native_jintArray nti(env, flat_type_ids);
    cudf::jni::native_jintArray ns(env, flat_scale);

    spark_rapids_jni::shuffle_split_metadata meta;
    meta.col_info.reserve(nnc.size());

    for (int i = 0; i < nnc.size(); ++i) {
      auto tid          = static_cast<cudf::type_id>(nti[i]);
      auto scale        = ns[i];
      auto num_children = static_cast<cudf::size_type>(nnc[i]);
      cudf::size_type param =
        spark_rapids_jni::is_fixed_point(cudf::data_type{tid, scale}) ? scale : num_children;
      meta.col_info.emplace_back(tid, param);
    }

    // Get single allocation buffers from shuffle_assemble
    auto single_alloc = shuffle_assemble(meta,
                                        partitions,
                                        offsets,
                                        cudf::get_default_stream(),
                                        cudf::get_current_device_resource());

    // Return native handles to cudf::column_view objects
    auto num_columns = single_alloc.column_views.size();
    auto buffer_size = single_alloc.single_buffer.size();

    // Create array for: buffer handle + buffer size + column view handles
    cudf::jni::native_jlongArray result(env, num_columns + 2);

    // First element is the single buffer handle
    result[0] = cudf::jni::release_as_jlong(std::make_unique<rmm::device_buffer>(std::move(single_alloc.single_buffer)));
    // Second element is the buffer size
    result[1] = static_cast<jlong>(buffer_size);

    // Remaining elements are column view handles
    for (size_t i = 0; i < num_columns; i++) {
      result[i + 2] = cudf::jni::ptr_as_jlong(single_alloc.column_views[i].release());
    }

    return result.get_jArray();
  }
  CATCH_STD(env, NULL);
}
}
