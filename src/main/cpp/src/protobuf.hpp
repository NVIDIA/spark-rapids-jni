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
#include <stdexcept>
#include <string>
#include <vector>

namespace spark_rapids_jni {

// Encoding constants
constexpr int ENC_DEFAULT     = 0;
constexpr int ENC_FIXED       = 1;
constexpr int ENC_ZIGZAG      = 2;
constexpr int ENC_ENUM_STRING = 3;

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

  for (size_t i = 0; i < num_fields; ++i) {
    auto const& field = context.schema[i];
    if (field.encoding == ENC_ENUM_STRING &&
        context.enum_valid_values[i].size() != context.enum_names[i].size()) {
      throw std::invalid_argument(
        "protobuf decode context: enum-as-string metadata mismatch at field " + std::to_string(i));
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
 * - LIST    : protobuf `bytes`  (length-delimited wire type, raw bytes as LIST<INT8>)
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
