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

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace spark_rapids_jni {

rmm::device_uvector<cudf::bitmask_type> bitmask_bitwise_or(
  std::vector<rmm::device_uvector<cudf::bitmask_type> const*> const& input,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.size() > 0, "Empty input");
  auto const mask_size = (*input.begin())->size();
  CUDF_EXPECTS(
    std::all_of(
      input.begin(), input.end(), [mask_size](auto mask) { return mask->size() == mask_size; }),
    "Encountered size mismatch in inputs");
  if (mask_size == 0) { return rmm::device_uvector<cudf::bitmask_type>(0, stream, mr); }

  // move the pointers to the gpu
  std::vector<cudf::bitmask_type const*> h_input(input.size());
  std::transform(
    input.begin(), input.end(), h_input.begin(), [](auto mask) { return mask->data(); });
  rmm::device_uvector<cudf::bitmask_type const*> d_input(
    h_input.size(), stream, rmm::mr::get_current_device_resource());
  cudaMemcpyAsync(d_input.data(),
                  h_input.data(),
                  sizeof(cudf::bitmask_type const*) * h_input.size(),
                  cudaMemcpyHostToDevice);

  rmm::device_uvector<cudf::bitmask_type> out(mask_size, stream, mr);
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + mask_size,
    out.begin(),
    [buffers = d_input.data(), num_buffers = input.size()] __device__(cudf::size_type word_index) {
      cudf::bitmask_type out = buffers[0][word_index];
      for (auto idx = 1; idx < num_buffers; idx++) {
        out |= buffers[idx][word_index];
      }
      return out;
    });

  return out;
}

}  // namespace spark_rapids_jni