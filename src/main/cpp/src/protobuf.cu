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

#include "protobuf_common.cuh"

#include <limits>

using namespace spark_rapids_jni::protobuf_detail;

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> decode_protobuf_to_struct(cudf::column_view const& binary_input,
                                                        ProtobufDecodeContext const& context,
                                                        rmm::cuda_stream_view stream)
{
  validate_decode_context(context);
  auto const& schema              = context.schema;
  auto const& schema_output_types = context.schema_output_types;
  CUDF_EXPECTS(binary_input.type().id() == cudf::type_id::LIST,
               "binary_input must be a LIST<INT8/UINT8> column");
  cudf::lists_column_view const in_list(binary_input);
  auto const child_type = in_list.child().type().id();
  CUDF_EXPECTS(child_type == cudf::type_id::INT8 || child_type == cudf::type_id::UINT8,
               "binary_input must be a LIST<INT8/UINT8> column");

  auto mr         = cudf::get_current_device_resource_ref();
  auto num_rows   = binary_input.size();
  auto num_fields = static_cast<int>(schema.size());

  if (num_fields == 0) {
    return cudf::make_structs_column(
      num_rows, std::vector<std::unique_ptr<cudf::column>>{}, 0, rmm::device_buffer{}, stream, mr);
  }

  if (num_rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_children;
    for (int i = 0; i < num_fields; i++) {
      if (schema[i].parent_idx == -1) {
        auto field_type = schema_output_types[i];
        if (schema[i].is_repeated && field_type.id() == cudf::type_id::STRUCT) {
          rmm::device_uvector<int32_t> offsets(1, stream, mr);
          CUDF_CUDA_TRY(cudaMemsetAsync(offsets.data(), 0, sizeof(int32_t), stream.value()));
          auto offsets_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::INT32}, 1, offsets.release(), rmm::device_buffer{}, 0);
          auto empty_struct = make_empty_struct_column_with_schema(
            schema, schema_output_types, i, num_fields, stream, mr);
          empty_children.push_back(cudf::make_lists_column(
            0, std::move(offsets_col), std::move(empty_struct), 0, rmm::device_buffer{}));
        } else if (schema[i].is_repeated) {
          auto empty_child = make_empty_column_safe(field_type, stream, mr);
          empty_children.push_back(make_empty_list_column(std::move(empty_child), stream, mr));
        } else if (field_type.id() == cudf::type_id::STRUCT) {
          empty_children.push_back(make_empty_struct_column_with_schema(
            schema, schema_output_types, i, num_fields, stream, mr));
        } else {
          empty_children.push_back(make_empty_column_safe(field_type, stream, mr));
        }
      }
    }
    return cudf::make_structs_column(
      0, std::move(empty_children), 0, rmm::device_buffer{}, stream, mr);
  }

  // Classify top-level fields by kind.
  std::vector<int> repeated_field_indices;
  std::vector<int> nested_field_indices;
  std::vector<int> scalar_field_indices;

  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == -1) {
      if (schema[i].is_repeated) {
        repeated_field_indices.push_back(i);
      } else if (schema[i].output_type == cudf::type_id::STRUCT) {
        nested_field_indices.push_back(i);
      } else {
        scalar_field_indices.push_back(i);
      }
    }
  }

  // Column map: populated by decode passes (added in follow-up PRs).
  // Any entry left as nullptr will become a null column in the assembly below.
  std::vector<std::unique_ptr<cudf::column>> column_map(num_fields);

  // --- Follow-up PRs insert scalar / repeated / nested decode passes here. ---

  // Assemble top_level_children in schema order.
  std::vector<std::unique_ptr<cudf::column>> top_level_children;
  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == -1) {
      if (column_map[i]) {
        top_level_children.push_back(std::move(column_map[i]));
      } else {
        if (schema[i].is_repeated) {
          auto const element_type = schema_output_types[i];
          std::unique_ptr<cudf::column> empty_child;
          if (element_type.id() == cudf::type_id::STRUCT) {
            empty_child = make_empty_struct_column_with_schema(
              schema, schema_output_types, i, num_fields, stream, mr);
          } else {
            empty_child = make_empty_column_safe(element_type, stream, mr);
          }
          top_level_children.push_back(
            make_null_list_column_with_child(std::move(empty_child), num_rows, stream, mr));
        } else {
          top_level_children.push_back(
            make_null_column(schema_output_types[i], num_rows, stream, mr));
        }
      }
    }
  }

  return cudf::make_structs_column(
    num_rows, std::move(top_level_children), 0, rmm::device_buffer{}, stream, mr);
}

}  // namespace spark_rapids_jni
