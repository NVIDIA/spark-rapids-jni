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

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace spark_rapids_jni {

/**
 * @brief Bitwise-or an array of equally-sized bitmask buffers into a single output buffer
 *
 * @param input The array of input bitmask buffers.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned bloom filter's memory.
 *
 */
rmm::device_uvector<cudf::bitmask_type> bitmask_bitwise_or(std::vector<rmm::device_uvector<cudf::bitmask_type>const*> const& input,  
                                                           rmm::cuda_stream_view stream = cudf::get_default_stream(),
                                                           rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

} // namespace spark_rapids_jni
