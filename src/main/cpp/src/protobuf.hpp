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

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace spark_rapids_jni {

// Encoding constants
constexpr int ENC_DEFAULT      = 0;
constexpr int ENC_FIXED        = 1;
constexpr int ENC_ZIGZAG       = 2;
constexpr int ENC_ENUM_STRING  = 3;
constexpr int MAX_FIELD_NUMBER = (1 << 29) - 1;

// Wire type constants
constexpr int WT_VARINT = 0;
constexpr int WT_64BIT  = 1;
constexpr int WT_LEN    = 2;
constexpr int WT_32BIT  = 5;

// Maximum nesting depth for nested messages
constexpr int MAX_NESTING_DEPTH = 10;

/**
 * Descriptor for a field in a nested protobuf schema.
 * Used to represent flattened schema with parent-child relationships.
 */
struct nested_field_descriptor {
  int field_number;           // Protobuf field number
  int parent_idx;             // Index of parent field in schema (-1 for top-level)
  int depth;                  // Nesting depth (0 for top-level)
  int wire_type;              // Expected wire type
  cudf::type_id output_type;  // Output cudf type
  int encoding;               // Encoding type (ENC_DEFAULT, ENC_FIXED, ENC_ZIGZAG)
  bool is_repeated;           // Whether this field is repeated (array)
  bool is_required;           // Whether this field is required (proto2)
  bool has_default_value;     // Whether this field has a default value
};

/**
 * Context and schema information for decoding protobuf messages.
 */
struct ProtobufDecodeContext {
  std::vector<nested_field_descriptor> schema;
  std::vector<cudf::data_type> schema_output_types;
  std::vector<int64_t> default_ints;
  std::vector<double> default_floats;
  std::vector<bool> default_bools;
  std::vector<std::vector<uint8_t>> default_strings;
  std::vector<std::vector<int32_t>> enum_valid_values;
  std::vector<std::vector<std::vector<uint8_t>>> enum_names;
  bool fail_on_errors;
};

struct ProtobufFieldMetaView {
  nested_field_descriptor const& schema;
  cudf::data_type const& output_type;
  int64_t default_int;
  double default_float;
  bool default_bool;
  std::vector<uint8_t> const& default_string;
  std::vector<int32_t> const& enum_valid_values;
  std::vector<std::vector<uint8_t>> const& enum_names;
};

inline bool is_encoding_compatible(nested_field_descriptor const& field,
                                   cudf::data_type const& type)
{
  switch (field.encoding) {
    case ENC_DEFAULT:
      switch (type.id()) {
        case cudf::type_id::BOOL8:
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64: return field.wire_type == WT_VARINT;
        case cudf::type_id::FLOAT32: return field.wire_type == WT_32BIT;
        case cudf::type_id::FLOAT64: return field.wire_type == WT_64BIT;
        case cudf::type_id::STRING:
        case cudf::type_id::LIST:
        case cudf::type_id::STRUCT: return field.wire_type == WT_LEN;
        default: return false;
      }
    case ENC_FIXED:
      switch (type.id()) {
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::FLOAT32: return field.wire_type == WT_32BIT;
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64:
        case cudf::type_id::FLOAT64: return field.wire_type == WT_64BIT;
        default: return false;
      }
    case ENC_ZIGZAG:
      return field.wire_type == WT_VARINT &&
             (type.id() == cudf::type_id::INT32 || type.id() == cudf::type_id::INT64);
    case ENC_ENUM_STRING: return field.wire_type == WT_VARINT && type.id() == cudf::type_id::STRING;
    default: return false;
  }
}

inline void validate_decode_context(ProtobufDecodeContext const& context)
{
  auto const num_fields = context.schema.size();
  auto const fail_size  = [&](char const* name, size_t actual) {
    throw std::invalid_argument(std::string("protobuf decode context: ") + name +
                                " size mismatch with schema (" + std::to_string(actual) + " vs " +
                                std::to_string(num_fields) + ")");
  };

  if (context.schema_output_types.size() != num_fields)
    fail_size("schema_output_types", context.schema_output_types.size());
  if (context.default_ints.size() != num_fields)
    fail_size("default_ints", context.default_ints.size());
  if (context.default_floats.size() != num_fields)
    fail_size("default_floats", context.default_floats.size());
  if (context.default_bools.size() != num_fields)
    fail_size("default_bools", context.default_bools.size());
  if (context.default_strings.size() != num_fields)
    fail_size("default_strings", context.default_strings.size());
  if (context.enum_valid_values.size() != num_fields)
    fail_size("enum_valid_values", context.enum_valid_values.size());
  if (context.enum_names.size() != num_fields) fail_size("enum_names", context.enum_names.size());

  std::set<std::pair<int, int>> seen_field_numbers;
  for (size_t i = 0; i < num_fields; ++i) {
    auto const& field = context.schema[i];
    auto const& type  = context.schema_output_types[i];
    if (type.id() != field.output_type) {
      throw std::invalid_argument(
        "protobuf decode context: schema_output_types id mismatch at field " + std::to_string(i));
    }
    if (field.field_number <= 0 || field.field_number > MAX_FIELD_NUMBER) {
      throw std::invalid_argument("protobuf decode context: invalid field number at field " +
                                  std::to_string(i));
    }
    if (field.depth < 0 || field.depth >= MAX_NESTING_DEPTH) {
      throw std::invalid_argument(
        "protobuf decode context: field depth exceeds supported limit at field " +
        std::to_string(i));
    }
    if (field.parent_idx < -1 || field.parent_idx >= static_cast<int>(i)) {
      throw std::invalid_argument("protobuf decode context: invalid parent index at field " +
                                  std::to_string(i));
    }
    if (!seen_field_numbers.emplace(field.parent_idx, field.field_number).second) {
      throw std::invalid_argument(
        "protobuf decode context: duplicate field number under same parent at field " +
        std::to_string(i));
    }
    if (field.parent_idx == -1) {
      if (field.depth != 0) {
        throw std::invalid_argument(
          "protobuf decode context: top-level field must have depth 0 at field " +
          std::to_string(i));
      }
    } else {
      auto const& parent = context.schema[field.parent_idx];
      if (field.depth != parent.depth + 1) {
        throw std::invalid_argument("protobuf decode context: child depth mismatch at field " +
                                    std::to_string(i));
      }
      if (context.schema_output_types[field.parent_idx].id() != cudf::type_id::STRUCT) {
        throw std::invalid_argument("protobuf decode context: parent must be STRUCT at field " +
                                    std::to_string(i));
      }
    }
    if (!(field.wire_type == WT_VARINT || field.wire_type == WT_64BIT ||
          field.wire_type == WT_LEN || field.wire_type == WT_32BIT)) {
      throw std::invalid_argument("protobuf decode context: invalid wire type at field " +
                                  std::to_string(i));
    }
    if (field.encoding < ENC_DEFAULT || field.encoding > ENC_ENUM_STRING) {
      throw std::invalid_argument("protobuf decode context: invalid encoding at field " +
                                  std::to_string(i));
    }
    if (field.is_repeated && field.has_default_value) {
      throw std::invalid_argument(
        "protobuf decode context: repeated field cannot carry default value at field " +
        std::to_string(i));
    }
    if (field.has_default_value &&
        (type.id() == cudf::type_id::STRUCT || type.id() == cudf::type_id::LIST)) {
      throw std::invalid_argument(
        "protobuf decode context: STRUCT/LIST field cannot carry default value at field " +
        std::to_string(i));
    }
    if (!is_encoding_compatible(field, type)) {
      throw std::invalid_argument(
        "protobuf decode context: incompatible wire type/encoding/output type at field " +
        std::to_string(i));
    }
    if (field.encoding == ENC_ENUM_STRING) {
      if (context.enum_valid_values[i].empty() || context.enum_names[i].empty()) {
        throw std::invalid_argument(
          "protobuf decode context: enum-as-string field requires non-empty metadata at field " +
          std::to_string(i));
      }
      if (context.enum_valid_values[i].size() != context.enum_names[i].size()) {
        throw std::invalid_argument(
          "protobuf decode context: enum-as-string metadata mismatch at field " +
          std::to_string(i));
      }
    }
  }
}

inline ProtobufFieldMetaView make_field_meta_view(ProtobufDecodeContext const& context,
                                                  int schema_idx)
{
  auto const idx = static_cast<size_t>(schema_idx);
  return ProtobufFieldMetaView{context.schema.at(idx),
                               context.schema_output_types.at(idx),
                               context.default_ints.at(idx),
                               context.default_floats.at(idx),
                               context.default_bools.at(idx),
                               context.default_strings.at(idx),
                               context.enum_valid_values.at(idx),
                               context.enum_names.at(idx)};
}

/**
 * Decode protobuf messages (one message per row) from a LIST<INT8/UINT8> column into a STRUCT
 * column, with support for nested messages and repeated fields.
 *
 * This uses a multi-pass approach:
 * - Pass 1: Scan all messages, count nested elements and repeated field occurrences
 * - Pass 2: Prefix sum to compute output offsets for arrays and nested structs
 * - Pass 3: Extract data using pre-computed offsets
 * - Pass 4: Build nested column structure
 *
 * The schema is represented as a flattened array of field descriptors with parent-child
 * relationships. Top-level fields have parent_idx == -1 and depth == 0. For pure scalar
 * schemas, all fields are top-level with is_repeated == false.
 *
 * Supported output child types (cudf dtypes) and corresponding protobuf field types:
 * - BOOL8   : protobuf `bool` (varint wire type)
 * - INT32   : protobuf `int32`, `sint32` (with zigzag), `fixed32`/`sfixed32` (with fixed encoding)
 * - INT64   : protobuf `int64`, `sint64` (with zigzag), `fixed64`/`sfixed64` (with fixed encoding)
 * - FLOAT32 : protobuf `float`  (fixed32 wire type)
 * - FLOAT64 : protobuf `double` (fixed64 wire type)
 * - STRING  : protobuf `string` (length-delimited wire type, UTF-8 text)
 * - LIST    : protobuf `bytes`  (length-delimited wire type, raw bytes as LIST<UINT8>)
 * - STRUCT  : protobuf nested `message`
 *
 * @param binary_input LIST<INT8/UINT8> column, each row is one protobuf message
 * @param context Decoding context containing schema and default values
 * @return STRUCT column with nested structure
 */
std::unique_ptr<cudf::column> decode_protobuf_to_struct(cudf::column_view const& binary_input,
                                                        ProtobufDecodeContext const& context,
                                                        rmm::cuda_stream_view stream);

}  // namespace spark_rapids_jni
