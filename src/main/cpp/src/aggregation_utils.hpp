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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

//
#include <rmm/cuda_stream_view.hpp>

namespace spark_rapids_jni {

/**
 * @brief
 */
std::unique_ptr<cudf::column> create_histograms_if_valid(
    cudf::column_view const &values, cudf::column_view const &frequencies,
    cudf::size_type output_size, rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute percentiles from the given histograms and percentage values.
 *
 * The input histograms must be given in the form of `List<Struct<ElementType, LongType>>`.
 *
 * @param input The lists of input histograms
 * @param percentages The input percentage values
 * @param output_as_list Specify whether the output percentiles will be wrapped into a list
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A lists column, each list stores the percentile value(s) of the corresponding row in the
 * input column
 */
std::unique_ptr<cudf::column> percentile_from_histogram(
    cudf::column_view const &input, std::vector<double> const &percentage, bool output_as_list,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

} // namespace spark_rapids_jni
