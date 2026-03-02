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

using namespace spark_rapids_jni::protobuf_detail;

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> decode_protobuf_to_struct(cudf::column_view const& binary_input,
                                                        ProtobufDecodeContext const& context,
                                                        rmm::cuda_stream_view stream)
{
  auto const& schema              = context.schema;
  auto const& schema_output_types = context.schema_output_types;
  auto const& default_ints        = context.default_ints;
  auto const& default_floats      = context.default_floats;
  auto const& default_bools       = context.default_bools;
  auto const& default_strings     = context.default_strings;
  auto const& enum_valid_values   = context.enum_valid_values;
  auto const& enum_names          = context.enum_names;
  bool fail_on_errors             = context.fail_on_errors;
  CUDF_EXPECTS(binary_input.type().id() == cudf::type_id::LIST,
               "binary_input must be a LIST<INT8/UINT8> column");
  cudf::lists_column_view const in_list(binary_input);
  auto const child_type = in_list.child().type().id();
  CUDF_EXPECTS(child_type == cudf::type_id::INT8 || child_type == cudf::type_id::UINT8,
               "binary_input must be a LIST<INT8/UINT8> column");

  auto mr         = cudf::get_current_device_resource_ref();
  auto num_rows   = binary_input.size();
  auto num_fields = static_cast<int>(schema.size());

  if (num_rows == 0 || num_fields == 0) {
    // Build empty struct based on top-level fields with proper nested structure
    std::vector<std::unique_ptr<cudf::column>> empty_children;
    for (int i = 0; i < num_fields; i++) {
      if (schema[i].parent_idx == -1) {
        auto field_type = schema_output_types[i];
        if (schema[i].is_repeated && field_type.id() == cudf::type_id::STRUCT) {
          // Repeated message field - build empty LIST with proper struct element
          rmm::device_uvector<int32_t> offsets(1, stream, mr);
          int32_t zero = 0;
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            offsets.data(), &zero, sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));
          auto offsets_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::INT32}, 1, offsets.release(), rmm::device_buffer{}, 0);
          auto empty_struct = make_empty_struct_column_with_schema(
            schema, schema_output_types, i, num_fields, stream, mr);
          empty_children.push_back(cudf::make_lists_column(0,
                                                           std::move(offsets_col),
                                                           std::move(empty_struct),
                                                           0,
                                                           rmm::device_buffer{},
                                                           stream,
                                                           mr));
        } else if (field_type.id() == cudf::type_id::STRUCT && !schema[i].is_repeated) {
          // Non-repeated nested message field
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

  // Copy schema to device
  std::vector<device_nested_field_descriptor> h_device_schema(num_fields);
  for (int i = 0; i < num_fields; i++) {
    h_device_schema[i] = device_nested_field_descriptor{schema[i]};
  }

  rmm::device_uvector<device_nested_field_descriptor> d_schema(num_fields, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_schema.data(),
                                h_device_schema.data(),
                                num_fields * sizeof(device_nested_field_descriptor),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  auto d_in = cudf::column_device_view::create(binary_input, stream);

  // Identify repeated and nested fields at depth 0
  std::vector<int> repeated_field_indices;
  std::vector<int> nested_field_indices;
  std::vector<int> scalar_field_indices;

  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == -1) {  // Top-level fields only
      if (schema[i].is_repeated) {
        repeated_field_indices.push_back(i);
      } else if (schema[i].output_type == cudf::type_id::STRUCT) {
        nested_field_indices.push_back(i);
      } else {
        scalar_field_indices.push_back(i);
      }
    }
  }

  int num_repeated = static_cast<int>(repeated_field_indices.size());
  int num_nested   = static_cast<int>(nested_field_indices.size());
  int num_scalar   = static_cast<int>(scalar_field_indices.size());

  // Error flag
  rmm::device_uvector<int> d_error(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 0, sizeof(int), stream.value()));
  auto check_error_and_throw = [&]() {
    if (!fail_on_errors) return;
    int h_error = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &h_error, d_error.data(), sizeof(int), cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
    CUDF_EXPECTS(h_error == 0,
                 "Malformed protobuf message, unsupported wire type, or missing required field");
  };

  // Enum validation support (PERMISSIVE mode)
  bool has_enum_fields = std::any_of(
    enum_valid_values.begin(), enum_valid_values.end(), [](auto const& v) { return !v.empty(); });
  rmm::device_uvector<bool> d_row_has_invalid_enum(has_enum_fields ? num_rows : 0, stream, mr);
  if (has_enum_fields) {
    CUDF_CUDA_TRY(
      cudaMemsetAsync(d_row_has_invalid_enum.data(), 0, num_rows * sizeof(bool), stream.value()));
  }

  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((num_rows + threads - 1) / threads);

  // Allocate for counting repeated fields
  rmm::device_uvector<repeated_field_info> d_repeated_info(
    num_repeated > 0 ? static_cast<size_t>(num_rows) * num_repeated : 1, stream, mr);
  rmm::device_uvector<field_location> d_nested_locations(
    num_nested > 0 ? static_cast<size_t>(num_rows) * num_nested : 1, stream, mr);

  rmm::device_uvector<int> d_repeated_indices(num_repeated > 0 ? num_repeated : 1, stream, mr);
  rmm::device_uvector<int> d_nested_indices(num_nested > 0 ? num_nested : 1, stream, mr);

  if (num_repeated > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_repeated_indices.data(),
                                  repeated_field_indices.data(),
                                  num_repeated * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
  }
  if (num_nested > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_nested_indices.data(),
                                  nested_field_indices.data(),
                                  num_nested * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));
  }

  // Count repeated fields at depth 0
  if (num_repeated > 0 || num_nested > 0) {
    count_repeated_fields_kernel<<<blocks, threads, 0, stream.value()>>>(*d_in,
                                                                         d_schema.data(),
                                                                         num_fields,
                                                                         0,  // depth_level
                                                                         d_repeated_info.data(),
                                                                         num_repeated,
                                                                         d_repeated_indices.data(),
                                                                         d_nested_locations.data(),
                                                                         num_nested,
                                                                         d_nested_indices.data(),
                                                                         d_error.data());
    check_error_and_throw();
  }

  // For scalar fields at depth 0, use the existing scan_all_fields_kernel
  // Use a map to store columns by schema index, then assemble in order at the end
  std::map<int, std::unique_ptr<cudf::column>> column_map;

  // Process scalar fields using existing infrastructure
  if (num_scalar > 0) {
    std::vector<field_descriptor> h_field_descs(num_scalar);
    for (int i = 0; i < num_scalar; i++) {
      int schema_idx                      = scalar_field_indices[i];
      h_field_descs[i].field_number       = schema[schema_idx].field_number;
      h_field_descs[i].expected_wire_type = schema[schema_idx].wire_type;
    }

    rmm::device_uvector<field_descriptor> d_field_descs(num_scalar, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_field_descs.data(),
                                  h_field_descs.data(),
                                  num_scalar * sizeof(field_descriptor),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));

    rmm::device_uvector<field_location> d_locations(
      static_cast<size_t>(num_rows) * num_scalar, stream, mr);

    auto h_field_lookup = build_field_lookup_table(h_field_descs.data(), num_scalar);
    rmm::device_uvector<int> d_field_lookup(h_field_lookup.size(), stream, mr);
    if (!h_field_lookup.empty()) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_field_lookup.data(),
                                    h_field_lookup.data(),
                                    h_field_lookup.size() * sizeof(int),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
    }

    scan_all_fields_kernel<<<blocks, threads, 0, stream.value()>>>(
      *d_in,
      d_field_descs.data(),
      num_scalar,
      h_field_lookup.empty() ? nullptr : d_field_lookup.data(),
      static_cast<int>(h_field_lookup.size()),
      d_locations.data(),
      d_error.data());
    check_error_and_throw();

    // Check required fields (after scan pass)
    {
      bool has_required = false;
      for (int i = 0; i < num_scalar; i++) {
        int si = scalar_field_indices[i];
        if (schema[si].is_required) {
          has_required = true;
          break;
        }
      }
      if (has_required) {
        rmm::device_uvector<uint8_t> d_is_required(num_scalar, stream, mr);
        std::vector<uint8_t> h_is_required(num_scalar);
        for (int i = 0; i < num_scalar; i++) {
          h_is_required[i] = schema[scalar_field_indices[i]].is_required ? 1 : 0;
        }
        CUDF_CUDA_TRY(cudaMemcpyAsync(d_is_required.data(),
                                      h_is_required.data(),
                                      num_scalar * sizeof(uint8_t),
                                      cudaMemcpyHostToDevice,
                                      stream.value()));
        check_required_fields_kernel<<<blocks, threads, 0, stream.value()>>>(
          d_locations.data(), d_is_required.data(), num_scalar, num_rows, d_error.data());
        check_error_and_throw();
      }
    }

    // Extract scalar values (reusing existing extraction logic)
    cudf::lists_column_view const in_list_view(binary_input);
    auto const* message_data =
      reinterpret_cast<uint8_t const*>(in_list_view.child().data<int8_t>());
    auto const* list_offsets = in_list_view.offsets().data<cudf::size_type>();

    cudf::size_type base_offset = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &base_offset, list_offsets, sizeof(cudf::size_type), cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();

    for (int i = 0; i < num_scalar; i++) {
      int schema_idx = scalar_field_indices[i];
      auto const dt  = schema_output_types[schema_idx];
      auto const enc = schema[schema_idx].encoding;
      bool has_def   = schema[schema_idx].has_default_value;

      switch (dt.id()) {
        case cudf::type_id::BOOL8:
        case cudf::type_id::INT32:
        case cudf::type_id::UINT32:
        case cudf::type_id::INT64:
        case cudf::type_id::UINT64:
        case cudf::type_id::FLOAT32:
        case cudf::type_id::FLOAT64: {
          TopLevelLocationProvider loc_provider{
            list_offsets, base_offset, d_locations.data(), i, num_scalar};
          column_map[schema_idx] = extract_typed_column(dt,
                                                        enc,
                                                        message_data,
                                                        loc_provider,
                                                        num_rows,
                                                        blocks,
                                                        threads,
                                                        has_def,
                                                        has_def ? default_ints[schema_idx] : 0,
                                                        has_def ? default_floats[schema_idx] : 0.0,
                                                        has_def ? default_bools[schema_idx] : false,
                                                        default_strings[schema_idx],
                                                        schema_idx,
                                                        enum_valid_values,
                                                        enum_names,
                                                        d_row_has_invalid_enum,
                                                        d_error,
                                                        stream,
                                                        mr);
          break;
        }
        case cudf::type_id::STRING: {
          if (enc == spark_rapids_jni::ENC_ENUM_STRING) {
            // ENUM-as-string path:
            // 1. Decode enum numeric value as INT32 varint.
            // 2. Validate against enum_valid_values.
            // 3. Convert INT32 -> UTF-8 enum name bytes.
            rmm::device_uvector<int32_t> out(num_rows, stream, mr);
            rmm::device_uvector<bool> valid((num_rows > 0 ? num_rows : 1), stream, mr);
            int64_t def_int = has_def ? default_ints[schema_idx] : 0;
            TopLevelLocationProvider loc_provider{
              list_offsets, base_offset, d_locations.data(), i, num_scalar};
            extract_varint_kernel<int32_t, false, TopLevelLocationProvider>
              <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                       loc_provider,
                                                       num_rows,
                                                       out.data(),
                                                       valid.data(),
                                                       d_error.data(),
                                                       has_def,
                                                       def_int);

            if (schema_idx < static_cast<int>(enum_valid_values.size()) &&
                schema_idx < static_cast<int>(enum_names.size())) {
              auto const& valid_enums     = enum_valid_values[schema_idx];
              auto const& enum_name_bytes = enum_names[schema_idx];
              if (!valid_enums.empty() && valid_enums.size() == enum_name_bytes.size()) {
                column_map[schema_idx] = build_enum_string_column(out,
                                                                  valid,
                                                                  valid_enums,
                                                                  enum_name_bytes,
                                                                  d_row_has_invalid_enum,
                                                                  num_rows,
                                                                  stream,
                                                                  mr);
              } else {
                // Missing enum metadata for enum-as-string field; mark as decode error.
                CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 1, sizeof(int), stream.value()));
                column_map[schema_idx] = make_null_column(dt, num_rows, stream, mr);
              }
            } else {
              CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 1, sizeof(int), stream.value()));
              column_map[schema_idx] = make_null_column(dt, num_rows, stream, mr);
            }
          } else {
            // Regular protobuf STRING (length-delimited)
            bool has_def_str    = has_def;
            auto const& def_str = default_strings[schema_idx];
            TopLevelLocationProvider len_provider{nullptr, 0, d_locations.data(), i, num_scalar};
            TopLevelLocationProvider copy_provider{
              list_offsets, base_offset, d_locations.data(), i, num_scalar};
            auto valid_fn = [locs = d_locations.data(), i, num_scalar, has_def_str] __device__(
                              cudf::size_type row) {
              return locs[row * num_scalar + i].offset >= 0 || has_def_str;
            };
            column_map[schema_idx] = extract_and_build_string_or_bytes_column(false,
                                                                              message_data,
                                                                              num_rows,
                                                                              len_provider,
                                                                              copy_provider,
                                                                              valid_fn,
                                                                              has_def_str,
                                                                              def_str,
                                                                              d_error,
                                                                              stream,
                                                                              mr);
          }
          break;
        }
        case cudf::type_id::LIST: {
          // bytes (BinaryType) represented as LIST<UINT8>
          bool has_def_bytes    = has_def;
          auto const& def_bytes = default_strings[schema_idx];
          TopLevelLocationProvider len_provider{nullptr, 0, d_locations.data(), i, num_scalar};
          TopLevelLocationProvider copy_provider{
            list_offsets, base_offset, d_locations.data(), i, num_scalar};
          auto valid_fn = [locs = d_locations.data(), i, num_scalar, has_def_bytes] __device__(
                            cudf::size_type row) {
            return locs[row * num_scalar + i].offset >= 0 || has_def_bytes;
          };
          column_map[schema_idx] = extract_and_build_string_or_bytes_column(true,
                                                                            message_data,
                                                                            num_rows,
                                                                            len_provider,
                                                                            copy_provider,
                                                                            valid_fn,
                                                                            has_def_bytes,
                                                                            def_bytes,
                                                                            d_error,
                                                                            stream,
                                                                            mr);
          break;
        }
        default:
          // For LIST (bytes) and other unsupported types, create placeholder columns
          column_map[schema_idx] = make_null_column(dt, num_rows, stream, mr);
          break;
      }
    }
  }

  // Process repeated fields
  if (num_repeated > 0) {
    cudf::lists_column_view const in_list_view(binary_input);
    auto const* list_offsets = in_list_view.offsets().data<cudf::size_type>();
    auto const* message_data =
      reinterpret_cast<uint8_t const*>(in_list_view.child().data<int8_t>());
    cudf::size_type base_offset = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &base_offset, list_offsets, sizeof(cudf::size_type), cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();

    for (int ri = 0; ri < num_repeated; ri++) {
      int schema_idx    = repeated_field_indices[ri];
      auto element_type = schema_output_types[schema_idx];

      // Get per-row counts for this repeated field entirely on GPU (performance fix!)
      rmm::device_uvector<int32_t> d_field_counts(num_rows, stream, mr);
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(num_rows),
                        d_field_counts.data(),
                        extract_strided_count{d_repeated_info.data(), ri, num_repeated});

      int64_t total_count = thrust::reduce(
        rmm::exec_policy(stream), d_field_counts.begin(), d_field_counts.end(), int64_t{0});

      if (total_count > 0) {
        // Build offsets for occurrence scanning on GPU (performance fix!)
        rmm::device_uvector<int32_t> d_occ_offsets(num_rows + 1, stream, mr);
        thrust::exclusive_scan(rmm::exec_policy(stream),
                               d_field_counts.begin(),
                               d_field_counts.end(),
                               d_occ_offsets.data(),
                               0);
        // Set last element
        CUDF_CUDA_TRY(cudaMemcpyAsync(d_occ_offsets.data() + num_rows,
                                      &total_count,
                                      sizeof(int32_t),
                                      cudaMemcpyHostToDevice,
                                      stream.value()));

        // Scan for all occurrences
        rmm::device_uvector<repeated_occurrence> d_occurrences(total_count, stream, mr);
        scan_repeated_field_occurrences_kernel<<<blocks, threads, 0, stream.value()>>>(
          *d_in,
          d_schema.data(),
          schema_idx,
          0,
          d_occ_offsets.data(),
          d_occurrences.data(),
          d_error.data());

        // Build the appropriate column type based on element type
        // For now, support scalar repeated fields
        auto child_type_id = static_cast<cudf::type_id>(h_device_schema[schema_idx].output_type_id);

        // The output_type in schema is the LIST type, but we need element type
        // For repeated int32, output_type should indicate the element is INT32
        switch (child_type_id) {
          case cudf::type_id::INT32:
            column_map[schema_idx] =
              build_repeated_scalar_column<int32_t>(binary_input,
                                                    message_data,
                                                    list_offsets,
                                                    base_offset,
                                                    h_device_schema[schema_idx],
                                                    d_field_counts,
                                                    d_occurrences,
                                                    total_count,
                                                    num_rows,
                                                    stream,
                                                    mr);
            break;
          case cudf::type_id::INT64:
            column_map[schema_idx] =
              build_repeated_scalar_column<int64_t>(binary_input,
                                                    message_data,
                                                    list_offsets,
                                                    base_offset,
                                                    h_device_schema[schema_idx],
                                                    d_field_counts,
                                                    d_occurrences,
                                                    total_count,
                                                    num_rows,
                                                    stream,
                                                    mr);
            break;
          case cudf::type_id::UINT32:
            column_map[schema_idx] =
              build_repeated_scalar_column<uint32_t>(binary_input,
                                                     message_data,
                                                     list_offsets,
                                                     base_offset,
                                                     h_device_schema[schema_idx],
                                                     d_field_counts,
                                                     d_occurrences,
                                                     total_count,
                                                     num_rows,
                                                     stream,
                                                     mr);
            break;
          case cudf::type_id::UINT64:
            column_map[schema_idx] =
              build_repeated_scalar_column<uint64_t>(binary_input,
                                                     message_data,
                                                     list_offsets,
                                                     base_offset,
                                                     h_device_schema[schema_idx],
                                                     d_field_counts,
                                                     d_occurrences,
                                                     total_count,
                                                     num_rows,
                                                     stream,
                                                     mr);
            break;
          case cudf::type_id::FLOAT32:
            column_map[schema_idx] =
              build_repeated_scalar_column<float>(binary_input,
                                                  message_data,
                                                  list_offsets,
                                                  base_offset,
                                                  h_device_schema[schema_idx],
                                                  d_field_counts,
                                                  d_occurrences,
                                                  total_count,
                                                  num_rows,
                                                  stream,
                                                  mr);
            break;
          case cudf::type_id::FLOAT64:
            column_map[schema_idx] =
              build_repeated_scalar_column<double>(binary_input,
                                                   message_data,
                                                   list_offsets,
                                                   base_offset,
                                                   h_device_schema[schema_idx],
                                                   d_field_counts,
                                                   d_occurrences,
                                                   total_count,
                                                   num_rows,
                                                   stream,
                                                   mr);
            break;
          case cudf::type_id::BOOL8:
            column_map[schema_idx] =
              build_repeated_scalar_column<uint8_t>(binary_input,
                                                    message_data,
                                                    list_offsets,
                                                    base_offset,
                                                    h_device_schema[schema_idx],
                                                    d_field_counts,
                                                    d_occurrences,
                                                    total_count,
                                                    num_rows,
                                                    stream,
                                                    mr);
            break;
          case cudf::type_id::STRING: {
            auto enc = schema[schema_idx].encoding;
            if (enc == spark_rapids_jni::ENC_ENUM_STRING &&
                schema_idx < static_cast<int>(enum_valid_values.size()) &&
                schema_idx < static_cast<int>(enum_names.size()) &&
                !enum_valid_values[schema_idx].empty() &&
                enum_valid_values[schema_idx].size() == enum_names[schema_idx].size()) {
              // Repeated enum-as-string: extract varints, then convert to strings.
              auto const& valid_enums = enum_valid_values[schema_idx];
              auto const& name_bytes  = enum_names[schema_idx];

              // 1. Extract enum integer values from occurrences
              rmm::device_uvector<int32_t> enum_ints(total_count, stream, mr);
              auto const rep_blocks =
                static_cast<int>((total_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
              RepeatedLocationProvider rep_loc{list_offsets, base_offset, d_occurrences.data()};
              extract_varint_kernel<int32_t, false>
                <<<rep_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
                  message_data, rep_loc, total_count, enum_ints.data(), nullptr, d_error.data());

              // 2. Build device-side enum lookup tables
              rmm::device_uvector<int32_t> d_valid_enums(valid_enums.size(), stream, mr);
              CUDF_CUDA_TRY(cudaMemcpyAsync(d_valid_enums.data(),
                                            valid_enums.data(),
                                            valid_enums.size() * sizeof(int32_t),
                                            cudaMemcpyHostToDevice,
                                            stream.value()));

              std::vector<int32_t> h_name_offsets(valid_enums.size() + 1, 0);
              int32_t total_name_chars = 0;
              for (size_t k = 0; k < name_bytes.size(); ++k) {
                total_name_chars += static_cast<int32_t>(name_bytes[k].size());
                h_name_offsets[k + 1] = total_name_chars;
              }
              std::vector<uint8_t> h_name_chars(total_name_chars);
              int32_t cursor = 0;
              for (auto const& nm : name_bytes) {
                if (!nm.empty()) {
                  std::copy(nm.data(), nm.data() + nm.size(), h_name_chars.data() + cursor);
                  cursor += static_cast<int32_t>(nm.size());
                }
              }
              rmm::device_uvector<int32_t> d_name_offsets(h_name_offsets.size(), stream, mr);
              CUDF_CUDA_TRY(cudaMemcpyAsync(d_name_offsets.data(),
                                            h_name_offsets.data(),
                                            h_name_offsets.size() * sizeof(int32_t),
                                            cudaMemcpyHostToDevice,
                                            stream.value()));
              rmm::device_uvector<uint8_t> d_name_chars(total_name_chars, stream, mr);
              if (total_name_chars > 0) {
                CUDF_CUDA_TRY(cudaMemcpyAsync(d_name_chars.data(),
                                              h_name_chars.data(),
                                              total_name_chars * sizeof(uint8_t),
                                              cudaMemcpyHostToDevice,
                                              stream.value()));
              }

              // 3. Validate enum values (sets row_has_invalid_enum for PERMISSIVE mode).
              //    We also need per-element validity for string building.
              rmm::device_uvector<bool> elem_valid(total_count, stream, mr);
              thrust::fill(rmm::exec_policy(stream), elem_valid.data(), elem_valid.end(), true);
              // validate_enum_values_kernel works on per-row basis; here we need per-element.
              // Binary-search each element inline via the lengths kernel below.

              // 4. Compute per-element string lengths
              rmm::device_uvector<int32_t> elem_lengths(total_count, stream, mr);
              compute_enum_string_lengths_kernel<<<rep_blocks,
                                                   THREADS_PER_BLOCK,
                                                   0,
                                                   stream.value()>>>(
                enum_ints.data(),
                elem_valid.data(),
                d_valid_enums.data(),
                d_name_offsets.data(),
                static_cast<int>(valid_enums.size()),
                elem_lengths.data(),
                total_count);

              // 5. Build string offsets
              auto [str_offs_col, total_chars] = cudf::strings::detail::make_offsets_child_column(
                elem_lengths.begin(), elem_lengths.end(), stream, mr);

              // 6. Copy string chars
              rmm::device_uvector<char> chars(total_chars, stream, mr);
              if (total_chars > 0) {
                copy_enum_string_chars_kernel<<<rep_blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
                  enum_ints.data(),
                  elem_valid.data(),
                  d_valid_enums.data(),
                  d_name_offsets.data(),
                  d_name_chars.data(),
                  static_cast<int>(valid_enums.size()),
                  str_offs_col->view().data<int32_t>(),
                  chars.data(),
                  total_count);
              }

              // 7. Assemble LIST<STRING> column
              auto child_col = cudf::make_strings_column(
                total_count, std::move(str_offs_col), chars.release(), 0, rmm::device_buffer{});

              // Build list offsets from per-row counts
              rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
              thrust::exclusive_scan(rmm::exec_policy(stream),
                                     d_field_counts.begin(),
                                     d_field_counts.end(),
                                     list_offs.begin(),
                                     0);
              CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data() + num_rows,
                                            &total_count,
                                            sizeof(int32_t),
                                            cudaMemcpyHostToDevice,
                                            stream.value()));

              auto list_offs_col =
                std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                               num_rows + 1,
                                               list_offs.release(),
                                               rmm::device_buffer{},
                                               0);

              auto input_null_count = binary_input.null_count();
              if (input_null_count > 0) {
                auto null_mask         = cudf::copy_bitmask(binary_input, stream, mr);
                column_map[schema_idx] = cudf::make_lists_column(num_rows,
                                                                 std::move(list_offs_col),
                                                                 std::move(child_col),
                                                                 input_null_count,
                                                                 std::move(null_mask),
                                                                 stream,
                                                                 mr);
              } else {
                column_map[schema_idx] = cudf::make_lists_column(num_rows,
                                                                 std::move(list_offs_col),
                                                                 std::move(child_col),
                                                                 0,
                                                                 rmm::device_buffer{},
                                                                 stream,
                                                                 mr);
              }
            } else {
              column_map[schema_idx] = build_repeated_string_column(binary_input,
                                                                    message_data,
                                                                    list_offsets,
                                                                    base_offset,
                                                                    h_device_schema[schema_idx],
                                                                    d_field_counts,
                                                                    d_occurrences,
                                                                    total_count,
                                                                    num_rows,
                                                                    false,
                                                                    stream,
                                                                    mr);
            }
            break;
          }
          case cudf::type_id::LIST:  // bytes as LIST<INT8>
            column_map[schema_idx] = build_repeated_string_column(binary_input,
                                                                  message_data,
                                                                  list_offsets,
                                                                  base_offset,
                                                                  h_device_schema[schema_idx],
                                                                  d_field_counts,
                                                                  d_occurrences,
                                                                  total_count,
                                                                  num_rows,
                                                                  true,
                                                                  stream,
                                                                  mr);
            break;
          case cudf::type_id::STRUCT: {
            // Repeated message field - ArrayType(StructType)
            auto child_field_indices = find_child_field_indices(schema, num_fields, schema_idx);
            if (child_field_indices.empty()) {
              auto empty_struct_child = make_empty_struct_column_with_schema(
                schema, schema_output_types, schema_idx, num_fields, stream, mr);
              column_map[schema_idx] = make_null_list_column_with_child(
                std::move(empty_struct_child), num_rows, stream, mr);
            } else {
              column_map[schema_idx] = build_repeated_struct_column(binary_input,
                                                                    message_data,
                                                                    list_offsets,
                                                                    base_offset,
                                                                    h_device_schema[schema_idx],
                                                                    d_field_counts,
                                                                    d_occurrences,
                                                                    total_count,
                                                                    num_rows,
                                                                    h_device_schema,
                                                                    child_field_indices,
                                                                    schema_output_types,
                                                                    default_ints,
                                                                    default_floats,
                                                                    default_bools,
                                                                    default_strings,
                                                                    schema,
                                                                    enum_valid_values,
                                                                    enum_names,
                                                                    d_row_has_invalid_enum,
                                                                    d_error,
                                                                    stream,
                                                                    mr);
            }
            break;
          }
          default:
            // Unsupported element type - create null column
            column_map[schema_idx] = make_null_list_column_with_child(
              make_empty_column_safe(element_type, stream, mr), num_rows, stream, mr);
            break;
        }
      } else {
        // All rows have count=0 - create list of empty elements
        rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
        thrust::fill(rmm::exec_policy(stream), offsets.begin(), offsets.end(), 0);
        auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                          num_rows + 1,
                                                          offsets.release(),
                                                          rmm::device_buffer{},
                                                          0);

        // Build appropriate empty child column
        std::unique_ptr<cudf::column> child_col;
        auto child_type_id = static_cast<cudf::type_id>(h_device_schema[schema_idx].output_type_id);
        if (child_type_id == cudf::type_id::STRUCT) {
          // Use helper to build empty struct with proper nested structure
          child_col = make_empty_struct_column_with_schema(
            schema, schema_output_types, schema_idx, num_fields, stream, mr);
        } else {
          child_col = make_empty_column_safe(schema_output_types[schema_idx], stream, mr);
        }

        auto const input_null_count = binary_input.null_count();
        if (input_null_count > 0) {
          auto null_mask         = cudf::copy_bitmask(binary_input, stream, mr);
          column_map[schema_idx] = cudf::make_lists_column(num_rows,
                                                           std::move(offsets_col),
                                                           std::move(child_col),
                                                           input_null_count,
                                                           std::move(null_mask),
                                                           stream,
                                                           mr);
        } else {
          column_map[schema_idx] = cudf::make_lists_column(num_rows,
                                                           std::move(offsets_col),
                                                           std::move(child_col),
                                                           0,
                                                           rmm::device_buffer{},
                                                           stream,
                                                           mr);
        }
      }
    }
  }

  // Process nested struct fields (Phase 2)
  if (num_nested > 0) {
    cudf::lists_column_view const in_list_view(binary_input);
    auto const* message_data =
      reinterpret_cast<uint8_t const*>(in_list_view.child().data<int8_t>());
    auto const* list_offsets = in_list_view.offsets().data<cudf::size_type>();

    cudf::size_type base_offset = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &base_offset, list_offsets, sizeof(cudf::size_type), cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();

    for (int ni = 0; ni < num_nested; ni++) {
      int parent_schema_idx = nested_field_indices[ni];

      // Find child fields of this nested message
      auto child_field_indices = find_child_field_indices(schema, num_fields, parent_schema_idx);

      if (child_field_indices.empty()) {
        // No child fields - create empty struct
        column_map[parent_schema_idx] =
          make_null_column(schema_output_types[parent_schema_idx], num_rows, stream, mr);
        continue;
      }

      // Extract parent locations for this nested field directly on GPU
      rmm::device_uvector<field_location> d_parent_locs(num_rows, stream, mr);
      extract_strided_locations_kernel<<<blocks, threads, 0, stream.value()>>>(
        d_nested_locations.data(), ni, num_nested, d_parent_locs.data(), num_rows);

      column_map[parent_schema_idx] = build_nested_struct_column(message_data,
                                                                 list_offsets,
                                                                 base_offset,
                                                                 d_parent_locs,
                                                                 child_field_indices,
                                                                 schema,
                                                                 num_fields,
                                                                 schema_output_types,
                                                                 default_ints,
                                                                 default_floats,
                                                                 default_bools,
                                                                 default_strings,
                                                                 enum_valid_values,
                                                                 enum_names,
                                                                 d_row_has_invalid_enum,
                                                                 d_error,
                                                                 num_rows,
                                                                 stream,
                                                                 mr,
                                                                 0);
    }
  }

  // Assemble top_level_children in schema order (not processing order)
  std::vector<std::unique_ptr<cudf::column>> top_level_children;
  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == -1) {  // Top-level field
      auto it = column_map.find(i);
      if (it != column_map.end()) {
        top_level_children.push_back(std::move(it->second));
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

  CUDF_CUDA_TRY(cudaPeekAtLastError());
  int h_error = 0;
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(&h_error, d_error.data(), sizeof(int), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  if (fail_on_errors) {
    CUDF_EXPECTS(h_error == 0,
                 "Malformed protobuf message, unsupported wire type, or missing required field");
  }

  // Build final struct with PERMISSIVE mode null mask for invalid enums
  cudf::size_type struct_null_count = 0;
  rmm::device_buffer struct_mask{0, stream, mr};

  if (has_enum_fields) {
    auto [mask, null_count] = cudf::detail::valid_if(
      thrust::make_counting_iterator<cudf::size_type>(0),
      thrust::make_counting_iterator<cudf::size_type>(num_rows),
      [row_invalid = d_row_has_invalid_enum.data()] __device__(cudf::size_type row) {
        return !row_invalid[row];
      },
      stream,
      mr);
    struct_mask       = std::move(mask);
    struct_null_count = null_count;
  }

  // cuDF struct child views do not inherit parent nulls. Push PERMISSIVE invalid-enum nulls
  // down into every top-level child so extracted fields respect "null struct => null field".
  if (has_enum_fields && struct_null_count > 0) {
    auto const* struct_mask_ptr = static_cast<cudf::bitmask_type const*>(struct_mask.data());
    for (auto& child : top_level_children) {
      auto child_view = child->mutable_view();
      if (child_view.nullable()) {
        auto const child_mask_words =
          cudf::num_bitmask_words(static_cast<size_t>(child_view.size() + child_view.offset()));
        std::array<cudf::bitmask_type const*, 2> masks{child_view.null_mask(), struct_mask_ptr};
        std::array<cudf::size_type, 2> begin_bits{child_view.offset(), 0};
        auto const valid_count = cudf::detail::inplace_bitmask_and(
          cudf::device_span<cudf::bitmask_type>(child_view.null_mask(), child_mask_words),
          cudf::host_span<cudf::bitmask_type const* const>(masks.data(), masks.size()),
          cudf::host_span<cudf::size_type const>(begin_bits.data(), begin_bits.size()),
          child_view.size(),
          stream);
        child->set_null_count(child_view.size() - valid_count);
      } else {
        auto child_mask = cudf::detail::copy_bitmask(struct_mask_ptr, 0, num_rows, stream, mr);
        child->set_null_mask(std::move(child_mask), struct_null_count);
      }
    }
  }

  return cudf::make_structs_column(
    num_rows, std::move(top_level_children), struct_null_count, std::move(struct_mask), stream, mr);
}

}  // namespace spark_rapids_jni
