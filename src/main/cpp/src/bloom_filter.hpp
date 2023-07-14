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

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace spark_rapids_jni {

/**
 * @brief Create an empty bloom filter of the specified size in bits.
 *
 * @param bloom_filter_bits Size of the bloom filter in bits.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned bloom filter's memory.
 * @returns An allocated bloom filter initialized to empty.
 *
 */
std::unique_ptr<rmm::device_buffer> bloom_filter_create(
  cudf::size_type bloom_filter_bits,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Builds a bloom filter by hashing input int64_t values using xxhash64.
 *
 * @param[in,out] bloom_filter The bloom filter to be constructed. The function expects that the
 * buffer has already been initialized to 0.
 * @param bloom_filter_bits Size of the bloom filter in bits.
 * @param input Input column of int64_t values to be inserted into the bloom filter.
 * @param num_hashes Number of hashes to apply.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 */
void bloom_filter_build(cudf::device_span<cudf::bitmask_type> bloom_filter,
                        cudf::size_type bloom_filter_bits,
                        cudf::column_view const& input,
                        cudf::size_type num_hashes,
                        rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Probe a bloom filter with an input column of int64_t values.
 *
 * @param input The column of int64_t values to probe with.
 * @param bloom_filter The bloom filter to be probed.
 * @param bloom_filter_bits The size in bits of the bloom filter.
 * @param num_hashes The number of hashes to apply.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned boolean column's memory.
 *
 * @returns A column of booleans where a true value indicates a value may be present in the bloom
 * filter, and a false indicates the value is not present.
 */
std::unique_ptr<cudf::column> bloom_filter_probe(
  cudf::column_view const& input,
  cudf::device_span<cudf::bitmask_type const> bloom_filter,
  cudf::size_type bloom_filter_bits,
  cudf::size_type num_hashes,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Create a device span from an rmm::device_buffer
 *
 * @param bloom_filter The bloom filter buffer to be converted.
 *
 * @returns A device span representing a view into the bloom filter
 *
 */
cudf::device_span<cudf::bitmask_type> bloom_filter_to_span(rmm::device_buffer& bloom_filter);

}  // namespace spark_rapids_jni
