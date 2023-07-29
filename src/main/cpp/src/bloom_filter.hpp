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
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace spark_rapids_jni {

// included only for testing purposes
struct bloom_filter_header {
  int version;
  int num_hashes;
  int num_longs;
};
constexpr int bloom_filter_header_size = sizeof(bloom_filter_header);

/**
 * @brief Create an empty bloom filter of the specified size in (64 bit) longs with using
 * the specified number of hashes to be used when operating on the filter.
 *
 * @param num_hashes The number of hashes to use.
 * @param bloom_filter_longs Size of the bloom filter in bits.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned bloom filter's memory.
 * @returns An list_scalar wrapping a packed Spark bloom_filter.
 *
 */
std::unique_ptr<cudf::list_scalar> bloom_filter_create(
  int num_hashes,
  int bloom_filter_longs,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Inserts input values into a bloom filter.
 *
 * Can be called multiple times on the same bloom_filter buffer.
 *
 * @param[in,out] bloom_filter The bloom filter to be added to.
 * @param input Input column of int64_t values to be inserted into the bloom filter.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 */
void bloom_filter_put(cudf::list_scalar& bloom_filter,
                      cudf::column_view const& input,
                      rmm::cuda_stream_view stream = cudf::get_default_stream());

/**
 * @brief Probe a bloom filter with an input column of int64_t values.
 *
 * @param input The column of int64_t values to probe with.
 * @param bloom_filter The bloom filter to be probed.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned boolean column's memory.
 *
 * @returns A column of booleans where a true value indicates a value may be present in the bloom
 * filter, and a false indicates the value is not present.
 */
std::unique_ptr<cudf::column> bloom_filter_probe(
  cudf::column_view const& input,
  cudf::list_scalar& bloom_filter,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Merge multiple bloom filters into a single output.
 *
 * The incoming bloom filters are expected to be in the form of a list column, with
 * each row corresponding to an invidual bloom filter.  Each bloom filter must have the
 * same number of hashes and size.
 *
 * @param bloom_filters The bloom filters to be probed.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned boolean column's memory.
 *
 * @returns The new bloom filter.
 */
std::unique_ptr<cudf::list_scalar> bloom_filter_merge(
  cudf::column_view const& bloom_filters,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
