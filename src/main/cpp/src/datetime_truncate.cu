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

#include "datetime_utils.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>

namespace spark_rapids_jni {

namespace detail {

std::unique_ptr<cudf::column> truncate(cudf::column_view const& input,
                                       cudf::datetime::datetime_component component,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
}

}  // namespace detail

std::unique_ptr<cudf::column> truncate(cudf::column_view const& input,
                                       cudf::datetime::datetime_component component,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::truncate(input, component, stream, mr);
}

}  // namespace spark_rapids_jni
