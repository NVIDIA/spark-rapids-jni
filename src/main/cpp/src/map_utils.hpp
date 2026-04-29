/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
 * Fast-path contract: when every row would be returned unchanged (no null struct entries, and
 * either no null key in a valid struct, or @p throw_on_null_key is false), this function
 * returns @c nullptr.  A Spark MAP<K,V> and a cuDF LIST<STRUCT<K,V>> share the same physical
 * layout (offsets, struct child, null mask), so the caller can reinterpret the input as the
 * result with a single @c incRefCount — no device allocation, no kernel past Phase 1, no copy.
 * An empty input also returns @c nullptr under the same contract.
 *
 * Sliced input is not supported: this function throws if @c input.offset() != 0 or the struct
 * child has a non-zero offset.  Callers must materialize a slice (e.g. via @c cudf::copy)
 * before invoking.
 *
 * Duplicate-key deduplication is left to the caller.
 *
 * @param input              Input LIST(STRUCT(KEY, VALUE)) column.  Must not be sliced.
 * @param throw_on_null_key  When true, throw if any valid-struct entry has a null key.
 * @param stream             CUDA stream used for device memory operations and kernel launches.
 * @param mr                 Device memory resource used to allocate the returned column's memory.
 * @return A new column equal to @p input except that rows containing null struct entries are
 *         replaced with null outer rows; or @c nullptr if no row needs masking (fast path).
 * @throws cudf::logic_error if the input is not a LIST(STRUCT(KEY,...)) column.
 * @throws cudf::logic_error if @p input or its struct child has a non-zero offset.
 * @throws cudf::logic_error if @p throw_on_null_key is true and any row (with no null struct
 *         entries) contains a null key inside a valid struct.
 */
[[nodiscard]] std::unique_ptr<cudf::column> map_from_entries(
  cudf::column_view const& input,
  bool throw_on_null_key,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
