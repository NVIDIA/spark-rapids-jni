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
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace spark_rapids_jni {

/**
 * @brief Multiply two columns with optional ANSI SQL mode.
 * Only supports integer types.
 * This function performs element-wise multiplication of two columns.
 * If `is_ansi_mode` is true, it checks for overflow and sets the validity accordingly.
 *
 * @param left_input The left input column view.
 * @param right_input The right input column view.
 * @param is_ansi_mode If true, enables ANSI SQL mode for overflow checking.
 * @param stream CUDA stream to use for the operation.
 * @param mr Memory resource to use for allocations.
 * @return A new column containing the results of the multiplication.
 * @throws spark_rapids_jni::error_at_row exception if it's ansi mode and overflow occurs.
 */
std::unique_ptr<cudf::column> multiply(
  cudf::column_view const& left_input,
  cudf::column_view const& right_input,
  bool is_ansi_mode,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
