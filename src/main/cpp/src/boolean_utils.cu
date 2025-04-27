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

#include "boolean_utils.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>

namespace spark_rapids_jni {

int64_t false_count(cudf::column_view const& input, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::UINT8 ||
                 input.type().id() == cudf::type_id::INT8 ||
                 input.type().id() == cudf::type_id::BOOL8,
               "Input column must be of type of UINT8, INT8, or BOOL8");
  auto const dcv = cudf::column_device_view::create(input);
  return thrust::count_if(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(input.size()),
    [d_col = *dcv] __device__(cudf::size_type idx) { return d_col.element<uint8_t>(idx) == 0; });
}

}  // namespace spark_rapids_jni
