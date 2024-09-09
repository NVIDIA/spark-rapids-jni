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

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

namespace spark_rapids_jni {

constexpr int64_t DEFAULT_XXHASH64_SEED = 42;

/**
 * @brief Computes the murmur32 hash value of each row in the input set of columns.
 *
 * @param input The table of columns to hash
 * @param seed Optional seed value to use for the hash function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input.
 */
std::unique_ptr<cudf::column> murmur_hash3_32(
  cudf::table_view const& input,
  uint32_t seed                     = 0,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Computes the xxhash64 hash value of each row in the input set of columns.
 *
 * @param input The table of columns to hash
 * @param seed Optional seed value to use for the hash function
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input.
 */
std::unique_ptr<cudf::column> xxhash64(
  cudf::table_view const& input,
  int64_t seed                      = DEFAULT_XXHASH64_SEED,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Computes the Hive hash value of each row in the input set of columns.
 *
 * @param input The table of columns to hash
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns A column where each row is the hash of a column from the input.
 */
std::unique_ptr<cudf::column> hive_hash(
  cudf::table_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
