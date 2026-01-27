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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <memory>
#include <vector>

namespace spark_rapids_jni {

// Encoding constants
constexpr int ENC_DEFAULT = 0;
constexpr int ENC_FIXED   = 1;
constexpr int ENC_ZIGZAG  = 2;

/**
 * Decode protobuf messages (one message per row) from a LIST<INT8/UINT8> column into a STRUCT
 * column.
 *
 * This uses a two-pass approach for efficiency:
 * - Pass 1: Scan all messages once, recording (offset, length) for each requested field
 * - Pass 2: Extract data in parallel using the recorded locations
 *
 * This is significantly faster than the per-field approach when decoding multiple fields,
 * as each message is only parsed once regardless of the number of fields.
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
 * @param total_num_fields Total number of fields in the output struct (including null columns)
 * @param decoded_field_indices Indices into the output struct for fields that should be decoded.
 *                              Fields not in this list will be null columns in the output.
 * @param field_numbers Protobuf field numbers for decoded fields (parallel to
 * decoded_field_indices)
 * @param all_types Output cudf data types for ALL fields in the struct (size = total_num_fields)
 * @param encodings Encoding type for each decoded field (0=default, 1=fixed, 2=zigzag)
 *                  (parallel to decoded_field_indices)
 * @param is_required Whether each decoded field is required (parallel to decoded_field_indices).
 *                    If a required field is missing and fail_on_errors is true, an exception is
 * thrown.
 * @param has_default_value Whether each decoded field has a default value (parallel to
 * decoded_field_indices)
 * @param default_ints Default values for int/long/enum fields (parallel to decoded_field_indices)
 * @param default_floats Default values for float/double fields (parallel to decoded_field_indices)
 * @param default_bools Default values for bool fields (parallel to decoded_field_indices)
 * @param default_strings Default values for string/bytes fields (parallel to decoded_field_indices)
 * @param enum_valid_values Valid enum values for each field (parallel to decoded_field_indices).
 *                          Empty vector means not an enum field. Non-empty vector contains the
 *                          valid enum values. Unknown enum values will be set to null.
 * @param fail_on_errors Whether to throw on malformed messages or missing required fields.
 *        Note: error checking is performed after all kernels complete (not between kernel launches)
 *        to avoid synchronization overhead.
 * @return STRUCT column with total_num_fields children. Decoded fields contain the parsed data,
 *         other fields contain all nulls. The STRUCT itself is always non-null.
 */
std::unique_ptr<cudf::column> decode_protobuf_to_struct(
  cudf::column_view const& binary_input,
  int total_num_fields,
  std::vector<int> const& decoded_field_indices,
  std::vector<int> const& field_numbers,
  std::vector<cudf::data_type> const& all_types,
  std::vector<int> const& encodings,
  std::vector<bool> const& is_required,
  std::vector<bool> const& has_default_value,
  std::vector<int64_t> const& default_ints,
  std::vector<double> const& default_floats,
  std::vector<bool> const& default_bools,
  std::vector<std::vector<uint8_t>> const& default_strings,
  std::vector<std::vector<int32_t>> const& enum_valid_values,
  bool fail_on_errors);

}  // namespace spark_rapids_jni
