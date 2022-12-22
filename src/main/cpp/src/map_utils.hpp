/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <memory>

#include <cudf/column/column_view.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace spark_rapids_jni {

std::unique_ptr<cudf::column>
from_json(cudf::column_view const &input, bool throw_if_keys_duplicate,
          rmm::cuda_stream_view stream = cudf::get_default_stream(),
          rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource());

} // namespace spark_rapids_jni
