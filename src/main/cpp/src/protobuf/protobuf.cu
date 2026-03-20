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

#include "protobuf/protobuf_host_helpers.hpp"

#include <set>
#include <string>

namespace spark_rapids_jni::protobuf {

using namespace detail;

namespace {

std::unique_ptr<cudf::column> make_null_column_with_schema(
  std::vector<nested_field_descriptor> const& schema,
  int schema_idx,
  int num_fields,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const& field = schema[schema_idx];
  auto const dtype  = cudf::data_type{schema[schema_idx].output_type};

  if (field.is_repeated) {
    std::unique_ptr<cudf::column> empty_child;
    if (dtype.id() == cudf::type_id::STRUCT) {
      empty_child =
        make_empty_struct_column_with_schema(schema, schema_idx, num_fields, stream, mr);
    } else {
      empty_child = make_empty_column_safe(dtype, stream, mr);
    }
    return make_null_list_column_with_child(std::move(empty_child), num_rows, stream, mr);
  }

  if (dtype.id() == cudf::type_id::STRUCT) {
    auto child_indices = find_child_field_indices(schema, num_fields, schema_idx);
    std::vector<std::unique_ptr<cudf::column>> children;
    for (auto const child_idx : child_indices) {
      children.push_back(
        make_null_column_with_schema(schema, child_idx, num_fields, num_rows, stream, mr));
    }
    auto null_mask = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    return cudf::make_structs_column(
      num_rows, std::move(children), num_rows, std::move(null_mask), stream, mr);
  }

  return make_null_column(dtype, num_rows, stream, mr);
}

}  // namespace

bool is_encoding_compatible(nested_field_descriptor const& field, cudf::data_type const& type)
{
  switch (field.encoding) {
    case proto_encoding::DEFAULT:
      switch (type.id()) {
        case cudf::type_id::BOOL8:
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64: return field.wire_type == proto_wire_type::VARINT;
        case cudf::type_id::FLOAT32: return field.wire_type == proto_wire_type::I32BIT;
        case cudf::type_id::FLOAT64: return field.wire_type == proto_wire_type::I64BIT;
        case cudf::type_id::STRING:
        case cudf::type_id::LIST:
        case cudf::type_id::STRUCT: return field.wire_type == proto_wire_type::LEN;
        default: return false;
      }
    case proto_encoding::FIXED:
      switch (type.id()) {
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::FLOAT32: return field.wire_type == proto_wire_type::I32BIT;
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64:
        case cudf::type_id::FLOAT64: return field.wire_type == proto_wire_type::I64BIT;
        default: return false;
      }
    case proto_encoding::ZIGZAG:
      return field.wire_type == proto_wire_type::VARINT &&
             (type.id() == cudf::type_id::INT32 || type.id() == cudf::type_id::INT64);
    case proto_encoding::ENUM_STRING:
      return field.wire_type == proto_wire_type::VARINT && type.id() == cudf::type_id::STRING;
    default: return false;
  }
}

void validate_decode_context(ProtobufDecodeContext const& context)
{
  auto const num_fields = context.schema.size();
  CUDF_EXPECTS(context.default_ints.size() == num_fields,
               "protobuf decode context: default_ints size mismatch with schema (" +
                 std::to_string(context.default_ints.size()) + " vs " + std::to_string(num_fields) +
                 ")",
               std::invalid_argument);
  CUDF_EXPECTS(context.default_floats.size() == num_fields,
               "protobuf decode context: default_floats size mismatch with schema (" +
                 std::to_string(context.default_floats.size()) + " vs " +
                 std::to_string(num_fields) + ")",
               std::invalid_argument);
  CUDF_EXPECTS(context.default_bools.size() == num_fields,
               "protobuf decode context: default_bools size mismatch with schema (" +
                 std::to_string(context.default_bools.size()) + " vs " +
                 std::to_string(num_fields) + ")",
               std::invalid_argument);
  CUDF_EXPECTS(context.default_strings.size() == num_fields,
               "protobuf decode context: default_strings size mismatch with schema (" +
                 std::to_string(context.default_strings.size()) + " vs " +
                 std::to_string(num_fields) + ")",
               std::invalid_argument);
  CUDF_EXPECTS(context.enum_valid_values.size() == num_fields,
               "protobuf decode context: enum_valid_values size mismatch with schema (" +
                 std::to_string(context.enum_valid_values.size()) + " vs " +
                 std::to_string(num_fields) + ")",
               std::invalid_argument);
  CUDF_EXPECTS(context.enum_names.size() == num_fields,
               "protobuf decode context: enum_names size mismatch with schema (" +
                 std::to_string(context.enum_names.size()) + " vs " + std::to_string(num_fields) +
                 ")",
               std::invalid_argument);

  std::set<std::pair<int, int>> seen_field_numbers;
  for (size_t i = 0; i < num_fields; ++i) {
    auto const& field = context.schema[i];
    auto const type   = cudf::data_type{field.output_type};
    CUDF_EXPECTS(field.field_number > 0 && field.field_number <= MAX_FIELD_NUMBER,
                 "protobuf decode context: invalid field number at field " + std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(
      field.depth >= 0 && field.depth < MAX_NESTING_DEPTH,
      "protobuf decode context: field depth exceeds supported limit at field " + std::to_string(i),
      std::invalid_argument);
    CUDF_EXPECTS(field.parent_idx >= -1 && field.parent_idx < static_cast<int>(i),
                 "protobuf decode context: invalid parent index at field " + std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(seen_field_numbers.emplace(field.parent_idx, field.field_number).second,
                 "protobuf decode context: duplicate field number under same parent at field " +
                   std::to_string(i),
                 std::invalid_argument);

    if (field.parent_idx == -1) {
      CUDF_EXPECTS(
        field.depth == 0,
        "protobuf decode context: top-level field must have depth 0 at field " + std::to_string(i),
        std::invalid_argument);
    } else {
      auto const& parent = context.schema[field.parent_idx];
      CUDF_EXPECTS(field.depth == parent.depth + 1,
                   "protobuf decode context: child depth mismatch at field " + std::to_string(i),
                   std::invalid_argument);
      CUDF_EXPECTS(
        cudf::data_type{context.schema[field.parent_idx].output_type}.id() == cudf::type_id::STRUCT,
        "protobuf decode context: parent must be STRUCT at field " + std::to_string(i),
        std::invalid_argument);
    }

    CUDF_EXPECTS(
      field.wire_type == proto_wire_type::VARINT || field.wire_type == proto_wire_type::I64BIT ||
        field.wire_type == proto_wire_type::LEN || field.wire_type == proto_wire_type::I32BIT,
      "protobuf decode context: invalid wire type at field " + std::to_string(i),
      std::invalid_argument);
    CUDF_EXPECTS(
      field.encoding >= proto_encoding::DEFAULT && field.encoding <= proto_encoding::ENUM_STRING,
      "protobuf decode context: invalid encoding at field " + std::to_string(i),
      std::invalid_argument);
    CUDF_EXPECTS(!(field.is_repeated && field.is_required),
                 "protobuf decode context: field cannot be both repeated and required at field " +
                   std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(!(field.is_repeated && field.has_default_value),
                 "protobuf decode context: repeated field cannot carry default value at field " +
                   std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(!(field.has_default_value &&
                   (type.id() == cudf::type_id::STRUCT || type.id() == cudf::type_id::LIST)),
                 "protobuf decode context: STRUCT/LIST field cannot carry default value at field " +
                   std::to_string(i),
                 std::invalid_argument);
    CUDF_EXPECTS(is_encoding_compatible(field, type),
                 "protobuf decode context: incompatible wire type/encoding/output type at field " +
                   std::to_string(i),
                 std::invalid_argument);

    if (field.encoding == proto_encoding::ENUM_STRING) {
      CUDF_EXPECTS(
        !(context.enum_valid_values[i].empty() || context.enum_names[i].empty()),
        "protobuf decode context: enum-as-string field requires non-empty metadata at field " +
          std::to_string(i),
        std::invalid_argument);
      CUDF_EXPECTS(
        context.enum_valid_values[i].size() == context.enum_names[i].size(),
        "protobuf decode context: enum-as-string metadata mismatch at field " + std::to_string(i),
        std::invalid_argument);
      auto const& ev = context.enum_valid_values[i];
      for (size_t j = 1; j < ev.size(); ++j) {
        CUDF_EXPECTS(
          ev[j] > ev[j - 1],
          "protobuf decode context: enum_valid_values must be strictly sorted at field " +
            std::to_string(i),
          std::invalid_argument);
      }
    }
  }
}

ProtobufFieldMetaView make_field_meta_view(ProtobufDecodeContext const& context, int schema_idx)
{
  auto const idx = static_cast<size_t>(schema_idx);
  return ProtobufFieldMetaView{context.schema.at(idx),
                               cudf::data_type{context.schema.at(idx).output_type},
                               context.default_ints.at(idx),
                               context.default_floats.at(idx),
                               context.default_bools.at(idx),
                               context.default_strings.at(idx),
                               context.enum_valid_values.at(idx),
                               context.enum_names.at(idx)};
}

std::unique_ptr<cudf::column> decode_protobuf_to_struct(cudf::column_view const& binary_input,
                                                        ProtobufDecodeContext const& context,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  validate_decode_context(context);
  auto const& schema = context.schema;
  CUDF_EXPECTS(binary_input.type().id() == cudf::type_id::LIST,
               "binary_input must be a LIST<INT8/UINT8> column");
  cudf::lists_column_view const in_list(binary_input);
  auto const child_type = in_list.child().type().id();
  CUDF_EXPECTS(child_type == cudf::type_id::INT8 || child_type == cudf::type_id::UINT8,
               "binary_input must be a LIST<INT8/UINT8> column");

  auto const num_rows   = binary_input.size();
  auto const num_fields = static_cast<int>(schema.size());

  if (num_fields == 0) {
    auto const input_null_count = binary_input.null_count();
    if (input_null_count > 0) {
      auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
      return cudf::make_structs_column(
        num_rows, {}, input_null_count, std::move(null_mask), stream, mr);
    }
    return cudf::make_structs_column(num_rows, {}, 0, rmm::device_buffer{}, stream, mr);
  }

  if (num_rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_children;
    for (int i = 0; i < num_fields; i++) {
      if (schema[i].parent_idx == -1) {
        auto field_type = cudf::data_type{schema[i].output_type};
        if (schema[i].is_repeated && field_type.id() == cudf::type_id::STRUCT) {
          rmm::device_uvector<int32_t> offsets(1, stream, mr);
          CUDF_CUDA_TRY(cudaMemsetAsync(offsets.data(), 0, sizeof(int32_t), stream.value()));
          auto offsets_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::INT32}, 1, offsets.release(), rmm::device_buffer{}, 0);
          auto empty_struct =
            make_empty_struct_column_with_schema(schema, i, num_fields, stream, mr);
          empty_children.push_back(cudf::make_lists_column(
            0, std::move(offsets_col), std::move(empty_struct), 0, rmm::device_buffer{}));
        } else if (schema[i].is_repeated) {
          auto empty_child = make_empty_column_safe(field_type, stream, mr);
          empty_children.push_back(make_empty_list_column(std::move(empty_child), stream, mr));
        } else if (field_type.id() == cudf::type_id::STRUCT) {
          empty_children.push_back(
            make_empty_struct_column_with_schema(schema, i, num_fields, stream, mr));
        } else {
          empty_children.push_back(make_empty_column_safe(field_type, stream, mr));
        }
      }
    }
    return cudf::make_structs_column(
      0, std::move(empty_children), 0, rmm::device_buffer{}, stream, mr);
  }

  std::vector<std::unique_ptr<cudf::column>> column_map(num_fields);

  std::vector<std::unique_ptr<cudf::column>> top_level_children;
  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == -1) {
      if (column_map[i]) {
        top_level_children.push_back(std::move(column_map[i]));
      } else {
        top_level_children.push_back(
          make_null_column_with_schema(schema, i, num_fields, num_rows, stream, mr));
      }
    }
  }

  auto const input_null_count = binary_input.null_count();
  if (input_null_count > 0) {
    auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
    return cudf::make_structs_column(
      num_rows, std::move(top_level_children), input_null_count, std::move(null_mask), stream, mr);
  }

  return cudf::make_structs_column(
    num_rows, std::move(top_level_children), 0, rmm::device_buffer{}, stream, mr);
}

}  // namespace spark_rapids_jni::protobuf
