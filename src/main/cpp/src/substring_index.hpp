/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

namespace spark_rapids_jni{

std::unique_ptr<cudf::column> substring_index(
    cudf::strings_column_view const& strings,
    cudf::string_scalar const& delimiter,
    cudf::size_type count,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> substring_index(
    cudf::strings_column_view const& strings,
    cudf::strings_column_view const& delimiter,
    cudf::size_type count,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());
}   // namespace spark_rapids_jni
