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
 * Decode protobuf messages (one message per row) from a LIST<INT8/UINT8> column into a STRUCT
 * column.
 *
 * This is intentionally limited to "simple types" (top-level scalar fields).
 *
 * Supported output child types (cudf dtypes) and corresponding protobuf field types:
 * - BOOL8   : protobuf `bool` (varint wire type)
 * - INT32   : protobuf `int32`, `sint32` (with zigzag), `fixed32`/`sfixed32` (with fixed encoding)
 * - UINT32  : protobuf `uint32`, `fixed32` (with fixed encoding)
 * - INT64   : protobuf `int64`, `sint64` (with zigzag), `fixed64`/`sfixed64` (with fixed encoding)
 * - UINT64  : protobuf `uint64`, `fixed64` (with fixed encoding)
 * - FLOAT32 : protobuf `float`  (fixed32 wire type)
 * - FLOAT64 : protobuf `double` (fixed64 wire type)
 * - STRING  : protobuf `string` (length-delimited wire type, UTF-8 text)
 * - LIST    : protobuf `bytes`  (length-delimited wire type, raw bytes as LIST<INT8>)
 *
 * Integer handling:
 * - For standard varint-encoded fields (`int32`, `int64`, `uint32`, `uint64`), use encoding=0.
 * - For zigzag-encoded signed fields (`sint32`, `sint64`), use encoding=2.
 * - For fixed-width fields (`fixed32`, `fixed64`, `sfixed32`, `sfixed64`), use encoding=1.
 *
 * Nested messages, repeated fields, map fields, and oneof fields are out of scope for this API.
 *
 * @param binary_input LIST<INT8/UINT8> column, each row is one protobuf message
 * @param field_numbers protobuf field numbers (one per output child)
 * @param out_types output cudf data types (one per output child)
 * @param encodings encoding type for each field (0=default, 1=fixed, 2=zigzag)
 * @param fail_on_errors whether to throw on malformed messages
 * @return STRUCT column with the given children types, with nullability propagated from input rows
 */
std::unique_ptr<cudf::column> decode_protobuf_simple_to_struct(
  cudf::column_view const& binary_input,
  std::vector<int> const& field_numbers,
  std::vector<cudf::data_type> const& out_types,
  std::vector<int> const& encodings,
  bool fail_on_errors);

}  // namespace spark_rapids_jni
