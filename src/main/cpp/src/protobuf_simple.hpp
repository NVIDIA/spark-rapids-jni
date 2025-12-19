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
#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace spark_rapids_jni {

/**
 * Decode protobuf messages (one message per row) from a LIST<INT8/UINT8> column into a STRUCT column.
 *
 * This is intentionally limited to "simple types" (top-level scalar fields).
 *
 * Supported output child types:
 * - BOOL8, INT32, INT64, FLOAT32, FLOAT64, STRING
 *
 * @param binary_input LIST<INT8/UINT8> column, each row is one protobuf message
 * @param field_numbers protobuf field numbers (one per output child)
 * @param out_types output cudf data types (one per output child)
 * @return STRUCT column with the given children types, with nullability propagated from input rows
 */
std::unique_ptr<cudf::column> decode_protobuf_simple_to_struct(
  cudf::column_view const& binary_input,
  std::vector<int> const& field_numbers,
  std::vector<cudf::data_type> const& out_types);

}  // namespace spark_rapids_jni



