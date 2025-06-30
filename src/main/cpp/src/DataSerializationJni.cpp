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

#include "cookie_serializer.hpp"
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

    return cudf::jni::convert_table_for_return(
      env,
      shuffle_assemble(meta,
                       partitions,
                       offsets,
                       cudf::get_default_stream(),
                       cudf::get_current_device_resource()));
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CookieSerializer_serialize(
  JNIEnv* env, jclass, jlongArray j_addrs_sizes)
{
  JNI_NULL_CHECK(env, j_addrs_sizes, "Array containing buffers' address/size is null", 0);
  try {
    auto const addrs_sizes = cudf::jni::native_jlongArray{env, j_addrs_sizes};
    if (addrs_sizes.size() % 2 != 0) {
      throw std::logic_error("Length of addrs_sizes is not a multiple of 2.");
    }
    std::size_t const num_buffers = addrs_sizes.size() / 2;
    std::vector<cudf::host_span<uint8_t const>> buffers(num_buffers);
    for (int i = 0; i < addrs_sizes.size(); i += 2) {
      buffers[i] = cudf::host_span<uint8_t const>{reinterpret_cast<uint8_t const*>(addrs_sizes[i]),
                                                  static_cast<std::size_t>(addrs_sizes[i + 1])};
    }
    return reinterpret_cast<jlong>(spark_rapids_jni::serialize_cookie(buffers).release());
  }
  CATCH_STD(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_CookieSerializer_deserialize(
  JNIEnv* env, jclass, jlong j_addr, jlong j_size)
{
  JNI_NULL_CHECK(env, j_addr, "Input buffer address is null", NULL);
  CUDF_EXPECTS(j_size > 0, "Input buffer size is non-positive", std::invalid_argument);

  try {
    auto deserialized = spark_rapids_jni::deserialize_cookie(cudf::host_span<uint8_t const>{
      reinterpret_cast<uint8_t const*>(j_addr), static_cast<std::size_t>(j_size)});
    cudf::jni::native_jlongArray result(env, deserialized.size());
    for (std::size_t i = 0; i < deserialized.size(); ++i) {
      result[i] = reinterpret_cast<jlong>(deserialized[i].release());
    }
    return result.get_jArray();
  }
  CATCH_STD(env, NULL);
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_CookieSerializer_NativeBuffer_closeStdVector(
  JNIEnv* env, jclass, jlong j_std_vector_handle)
{
  JNI_NULL_CHECK(env, j_std_vector_handle, "std_vector_handle is null", );
  try {
    auto const ptr = reinterpret_cast<std::unique_ptr<std::vector<uint8_t>>*>(j_std_vector_handle);
    ptr->reset();  // reset, not release, as we are deleting the underlying vector
    delete ptr;    // must also delete the handler
  }
  CATCH_STD(env, );
}
}