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
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace spark_rapids_jni {

/**
 * @brief Converts a LIST(STRUCT(KEY, VALUE)) column to a map column following Spark semantics.
 *
 * Spark semantics for map_from_entries:
 *  - If a row's array contains any null struct entry (the whole struct is null), the output row
 *    is null — even if another entry in that same row has a null key.
 *  - If a row's array contains no null struct entry but does contain a null key inside a valid
 *    struct, behavior depends on throw_on_null_key:
 *      - true  → throws a logic_error ("Cannot use null as map key.")
 *      - false → returns the row as-is (caller is responsible for deduplication policy)
 *
 * This function only handles null-struct masking and null-key validation.
 * Duplicate-key deduplication is left to the caller.
 *
 * @param input           Input LIST(STRUCT(KEY, VALUE)) column.
 * @param throw_on_null_key  When true, throw if any valid-struct entry has a null key.
 * @param stream          CUDA stream used for device memory operations and kernel launches.
 * @param mr              Device memory resource used to allocate the returned column's memory.
 * @return A new column equal to @p input except that rows containing null struct entries are
 *         replaced with a null outer row.
 * @throws cudf::logic_error if the input is not a LIST(STRUCT(KEY,...)) column.
 * @throws cudf::logic_error if @p throw_on_null_key is true and any row (with no null struct
 *         entries) contains a null key inside a valid struct.
 */
std::unique_ptr<cudf::column> map_from_entries(
  cudf::column_view const& input,
  bool throw_on_null_key,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
