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

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace spark_rapids_jni {

/**
 * select the first column index with true value.
 * e.g.:
 * column 0 in table: true,  false, false
 * column 1 in table: false, true,  false
 * column 2 in table: false, false, true
 * 
 * return column: 0, 1, 2
*/
std::unique_ptr<cudf::column> select_first_true_index(
  cudf::table_view const& when_bool_columns,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * Select strings int scalar column according to index column
 * scalar column: s0, s1, s2
 * index  column: 0,  1,  2,  2,  1,  0,  3
 * output column: s0, s1, s2, s2, s1, s0, null
 * 
*/
std::unique_ptr<cudf::column> select_from_index(
  cudf::strings_column_view const& then_and_else_scalar_column,
  cudf::column_view const& select_index_column,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
