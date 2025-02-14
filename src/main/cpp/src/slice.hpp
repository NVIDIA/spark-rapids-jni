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

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>


namespace spark_rapids_jni {

std::unique_ptr<cudf::column> slice(
  cudf::lists_column_view const& input,
  cudf::size_type const start,
  cudf::size_type const length,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> slice(
  cudf::lists_column_view const& input,
  cudf::size_type const start,
  cudf::column_view const& length,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> slice(
  cudf::lists_column_view const& input,
  cudf::column_view const& start,
  cudf::size_type const length,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> slice(
  cudf::lists_column_view const& input,
  cudf::column_view const& start,
  cudf::column_view const& length,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
