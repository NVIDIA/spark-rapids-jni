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
          empty_children.push_back(cudf::make_lists_column(
            0, std::move(offsets_col), std::move(empty_struct), 0, rmm::device_buffer{}));
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

  // Extract shared input data pointers (used by scalar, repeated, and nested sections)
  cudf::lists_column_view const in_list_view(binary_input);
  auto const* message_data = reinterpret_cast<uint8_t const*>(in_list_view.child().data<int8_t>());
  auto const message_data_size = static_cast<cudf::size_type>(in_list_view.child().size());
  auto const* list_offsets     = in_list_view.offsets().data<cudf::size_type>();

  cudf::size_type base_offset = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &base_offset, list_offsets, sizeof(cudf::size_type), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();

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
  auto error_message = [](int code) -> char const* {
    switch (code) {
      case ERR_BOUNDS: return "Protobuf decode error: message data out of bounds";
      case ERR_VARINT: return "Protobuf decode error: invalid or truncated varint";
      case ERR_FIELD_NUMBER: return "Protobuf decode error: invalid field number";
      case ERR_WIRE_TYPE: return "Protobuf decode error: unexpected wire type";
      case ERR_OVERFLOW: return "Protobuf decode error: length-delimited field overflows message";
      case ERR_FIELD_SIZE: return "Protobuf decode error: invalid field size";
      case ERR_SKIP: return "Protobuf decode error: unable to skip unknown field";
      case ERR_FIXED_LEN:
        return "Protobuf decode error: invalid fixed-width or packed field length";
      case ERR_REQUIRED: return "Protobuf decode error: missing required field";
      case ERR_SCHEMA_TOO_LARGE:
        return "Protobuf decode error: schema exceeds maximum supported repeated fields per kernel "
               "(128)";
      case ERR_MISSING_ENUM_META:
        return "Protobuf decode error: missing or mismatched enum metadata for enum-as-string "
               "field";
      default: return "Protobuf decode error: unknown error";
    }
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
  auto const blocks  = static_cast<int>((num_rows + threads - 1u) / threads);

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

  // Count repeated fields at depth 0 (with O(1) field_number lookup tables)
  rmm::device_uvector<int> d_fn_to_rep(0, stream, mr);
  rmm::device_uvector<int> d_fn_to_nested(0, stream, mr);

  if (num_repeated > 0 || num_nested > 0) {
    auto h_fn_to_rep = protobuf_detail::build_index_lookup_table(
      schema.data(), repeated_field_indices.data(), num_repeated);
    auto h_fn_to_nested = protobuf_detail::build_index_lookup_table(
      schema.data(), nested_field_indices.data(), num_nested);

    if (!h_fn_to_rep.empty()) {
      d_fn_to_rep = rmm::device_uvector<int>(h_fn_to_rep.size(), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_fn_to_rep.data(),
                                    h_fn_to_rep.data(),
                                    h_fn_to_rep.size() * sizeof(int),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
    }
    if (!h_fn_to_nested.empty()) {
      d_fn_to_nested = rmm::device_uvector<int>(h_fn_to_nested.size(), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_fn_to_nested.data(),
                                    h_fn_to_nested.data(),
                                    h_fn_to_nested.size() * sizeof(int),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
    }

    count_repeated_fields_kernel<<<blocks, threads, 0, stream.value()>>>(
      *d_in,
      d_schema.data(),
      num_fields,
      0,
      d_repeated_info.data(),
      num_repeated,
      d_repeated_indices.data(),
      d_nested_locations.data(),
      num_nested,
      d_nested_indices.data(),
      d_error.data(),
      d_fn_to_rep.data(),
      static_cast<int>(d_fn_to_rep.size()),
      d_fn_to_nested.data(),
      static_cast<int>(d_fn_to_nested.size()));
  }

  // Store decoded columns by schema index for ordered assembly at the end.
  std::vector<std::unique_ptr<cudf::column>> column_map(num_fields);

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

    // Required-field validation applies to all scalar leaves, not just top-level numerics.
    maybe_check_required_fields(
      d_locations.data(), scalar_field_indices, schema, num_rows, d_error.data(), stream, mr);

    // Batched scalar extraction: group non-special fixed-width fields by extraction
    // category and extract all fields of each category with a single 2D kernel launch.
    {
      struct scalar_buf_pair {
        rmm::device_uvector<uint8_t> out_bytes;
        rmm::device_uvector<bool> valid;
        scalar_buf_pair(rmm::cuda_stream_view s, rmm::device_async_resource_ref m)
          : out_bytes(0, s, m), valid(0, s, m)
        {
        }
      };

      // Classify each scalar field
      // 0=I32, 1=U32, 2=I64, 3=U64, 4=BOOL, 5=I32zz, 6=I64zz, 7=F32, 8=F64,
      // 9=I32fixed, 10=I64fixed, 11=fallback
      constexpr int NUM_GROUPS   = 12;
      constexpr int GRP_FALLBACK = 11;
      std::vector<int> group_lists[NUM_GROUPS];

      for (int i = 0; i < num_scalar; i++) {
        int si   = scalar_field_indices[i];
        auto tid = schema_output_types[si].id();
        int enc  = schema[si].encoding;
        bool zz  = (enc == spark_rapids_jni::ENC_ZIGZAG);

        // STRING, LIST, and enum-as-string go to per-field path
        if (tid == cudf::type_id::STRING || tid == cudf::type_id::LIST) continue;

        bool is_fixed = (enc == spark_rapids_jni::ENC_FIXED);

        // INT32 with enum validation goes to fallback
        if (tid == cudf::type_id::INT32 && !zz && !is_fixed &&
            si < static_cast<int>(enum_valid_values.size()) && !enum_valid_values[si].empty()) {
          group_lists[GRP_FALLBACK].push_back(i);
          continue;
        }

        int g = GRP_FALLBACK;
        if (tid == cudf::type_id::INT32 && is_fixed)
          g = 9;
        else if (tid == cudf::type_id::INT64 && is_fixed)
          g = 10;
        else if (tid == cudf::type_id::UINT32 && is_fixed)
          g = 9;
        else if (tid == cudf::type_id::UINT64 && is_fixed)
          g = 10;
        else if (tid == cudf::type_id::INT32 && !zz)
          g = 0;
        else if (tid == cudf::type_id::UINT32)
          g = 1;
        else if (tid == cudf::type_id::INT64 && !zz)
          g = 2;
        else if (tid == cudf::type_id::UINT64)
          g = 3;
        else if (tid == cudf::type_id::BOOL8)
          g = 4;
        else if (tid == cudf::type_id::INT32 && zz)
          g = 5;
        else if (tid == cudf::type_id::INT64 && zz)
          g = 6;
        else if (tid == cudf::type_id::FLOAT32)
          g = 7;
        else if (tid == cudf::type_id::FLOAT64)
          g = 8;
        group_lists[g].push_back(i);
      }

      // Helper: batch-extract one group using a 2D kernel, then build columns.
      auto do_batch = [&](std::vector<int> const& idxs, auto kernel_launcher) {
        int nf = static_cast<int>(idxs.size());
        if (nf == 0) return;

        std::vector<std::unique_ptr<scalar_buf_pair>> bufs;
        bufs.reserve(nf);
        std::vector<protobuf_detail::batched_scalar_desc> h_descs(nf);

        for (int j = 0; j < nf; j++) {
          int li   = idxs[j];
          int si   = scalar_field_indices[li];
          bool hd  = schema[si].has_default_value;
          auto& bp = *bufs.emplace_back(std::make_unique<scalar_buf_pair>(stream, mr));
          bp.valid = rmm::device_uvector<bool>(std::max(1, num_rows), stream, mr);
          // BOOL8 default comes from default_bools (converted to 0/1 int)
          bool is_bool  = (schema_output_types[si].id() == cudf::type_id::BOOL8);
          int64_t def_i = hd ? (is_bool ? (default_bools[si] ? 1 : 0) : default_ints[si]) : 0;
          h_descs[j]    = {li, nullptr, bp.valid.data(), hd, def_i, hd ? default_floats[si] : 0.0};
        }

        // kernel_launcher allocates out_bytes, sets h_descs[j].output, and launches kernel
        kernel_launcher(nf, h_descs, bufs);

        // Build columns
        for (int j = 0; j < nf; j++) {
          int si   = scalar_field_indices[idxs[j]];
          auto dt  = schema_output_types[si];
          auto& bp = *bufs[j];
          auto [mask, null_count] =
            protobuf_detail::make_null_mask_from_valid(bp.valid, stream, mr);
          column_map[si] = std::make_unique<cudf::column>(
            dt, num_rows, bp.out_bytes.release(), std::move(mask), null_count);
        }
      };

      // Varint launcher for type T with zigzag ZZ
      auto varint_launch = [&](int nf,
                               std::vector<protobuf_detail::batched_scalar_desc>& h_descs,
                               std::vector<std::unique_ptr<scalar_buf_pair>>& bufs,
                               size_t elem_size,
                               auto kernel_fn) {
        for (int j = 0; j < nf; j++) {
          bufs[j]->out_bytes = rmm::device_uvector<uint8_t>(num_rows * elem_size, stream, mr);
          h_descs[j].output  = bufs[j]->out_bytes.data();
        }
        rmm::device_uvector<protobuf_detail::batched_scalar_desc> d_descs(nf, stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(d_descs.data(),
                                      h_descs.data(),
                                      nf * sizeof(h_descs[0]),
                                      cudaMemcpyHostToDevice,
                                      stream.value()));
        dim3 grid((num_rows + threads - 1u) / threads, nf);
        kernel_fn(grid,
                  threads,
                  stream.value(),
                  message_data,
                  list_offsets,
                  base_offset,
                  d_locations.data(),
                  num_scalar,
                  d_descs.data(),
                  nf,
                  num_rows,
                  d_error.data());
      };

// Dispatch groups 0-8 as batched
#define LAUNCH_VARINT_BATCH(GROUP, TYPE, ZZ)                                                  \
  do_batch(group_lists[GROUP], [&](int nf, auto& hd, auto& bf) {                              \
    varint_launch(nf, hd, bf, sizeof(TYPE), [](dim3 g, int t, cudaStream_t s, auto... args) { \
      protobuf_detail::extract_varint_batched_kernel<TYPE, ZZ><<<g, t, 0, s>>>(args...);      \
    });                                                                                       \
  })

#define LAUNCH_FIXED_BATCH(GROUP, TYPE, WT_VAL)                                               \
  do_batch(group_lists[GROUP], [&](int nf, auto& hd, auto& bf) {                              \
    varint_launch(nf, hd, bf, sizeof(TYPE), [](dim3 g, int t, cudaStream_t s, auto... args) { \
      protobuf_detail::extract_fixed_batched_kernel<TYPE, WT_VAL><<<g, t, 0, s>>>(args...);   \
    });                                                                                       \
  })

      LAUNCH_VARINT_BATCH(0, int32_t, false);
      LAUNCH_VARINT_BATCH(1, uint32_t, false);
      LAUNCH_VARINT_BATCH(2, int64_t, false);
      LAUNCH_VARINT_BATCH(3, uint64_t, false);
      LAUNCH_VARINT_BATCH(4, uint8_t, false);
      LAUNCH_VARINT_BATCH(5, int32_t, true);
      LAUNCH_VARINT_BATCH(6, int64_t, true);
      LAUNCH_FIXED_BATCH(7, float, WT_32BIT);
      LAUNCH_FIXED_BATCH(8, double, WT_64BIT);
      LAUNCH_FIXED_BATCH(9, int32_t, WT_32BIT);
      LAUNCH_FIXED_BATCH(10, int64_t, WT_64BIT);

#undef LAUNCH_VARINT_BATCH
#undef LAUNCH_FIXED_BATCH

      // Per-field fallback (INT32 with enum, etc.)
      for (int i : group_lists[GRP_FALLBACK]) {
        int schema_idx        = scalar_field_indices[i];
        auto const field_meta = make_field_meta_view(context, schema_idx);
        auto const dt         = field_meta.output_type;
        auto const enc        = field_meta.schema.encoding;
        bool has_def          = field_meta.schema.has_default_value;
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
                                                      has_def ? field_meta.default_int : 0,
                                                      has_def ? field_meta.default_float : 0.0,
                                                      has_def ? field_meta.default_bool : false,
                                                      field_meta.default_string,
                                                      schema_idx,
                                                      enum_valid_values,
                                                      enum_names,
                                                      d_row_has_invalid_enum,
                                                      d_error,
                                                      stream,
                                                      mr);
      }
    }

    // Per-field extraction for STRING and LIST types
    for (int i = 0; i < num_scalar; i++) {
      int schema_idx        = scalar_field_indices[i];
      auto const field_meta = make_field_meta_view(context, schema_idx);
      auto const dt         = field_meta.output_type;
      if (dt.id() != cudf::type_id::STRING && dt.id() != cudf::type_id::LIST) continue;
      auto const enc = field_meta.schema.encoding;
      bool has_def   = field_meta.schema.has_default_value;

      switch (dt.id()) {
        case cudf::type_id::STRING: {
          if (enc == spark_rapids_jni::ENC_ENUM_STRING) {
            // ENUM-as-string path:
            // 1. Decode enum numeric value as INT32 varint.
            // 2. Validate against enum_valid_values.
            // 3. Convert INT32 -> UTF-8 enum name bytes.
            rmm::device_uvector<int32_t> out(num_rows, stream, mr);
            rmm::device_uvector<bool> valid((num_rows > 0 ? num_rows : 1), stream, mr);
            int64_t def_int = has_def ? field_meta.default_int : 0;
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
                {
                  int err_val = ERR_MISSING_ENUM_META;
                  CUDF_CUDA_TRY(cudaMemcpyAsync(
                    d_error.data(), &err_val, sizeof(int), cudaMemcpyHostToDevice, stream.value()));
                }
                column_map[schema_idx] = make_null_column(dt, num_rows, stream, mr);
              }
            } else {
              {
                int err_val = ERR_MISSING_ENUM_META;
                CUDF_CUDA_TRY(cudaMemcpyAsync(
                  d_error.data(), &err_val, sizeof(int), cudaMemcpyHostToDevice, stream.value()));
              }
              column_map[schema_idx] = make_null_column(dt, num_rows, stream, mr);
            }
          } else {
            // Regular protobuf STRING (length-delimited)
            bool has_def_str    = has_def;
            auto const& def_str = field_meta.default_string;
            TopLevelLocationProvider len_provider{
              list_offsets, base_offset, d_locations.data(), i, num_scalar};
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
          auto const& def_bytes = field_meta.default_string;
          TopLevelLocationProvider len_provider{
            list_offsets, base_offset, d_locations.data(), i, num_scalar};
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

  // Required top-level nested messages are tracked in d_nested_locations during the scan/count
  // pass.
  maybe_check_required_fields(
    d_nested_locations.data(), nested_field_indices, schema, num_rows, d_error.data(), stream, mr);

  // Process repeated fields (three-phase: offsets → combined scan → build columns)
  if (num_repeated > 0) {
    // Phase A: Compute per-row offsets for each repeated field.
    struct repeated_field_work {
      int schema_idx;
      int32_t total_count{0};
      rmm::device_uvector<int32_t> counts;
      rmm::device_uvector<int32_t> offsets;
      std::unique_ptr<rmm::device_uvector<protobuf_detail::repeated_occurrence>> occurrences;

      repeated_field_work(int si,
                          cudf::size_type n,
                          rmm::cuda_stream_view s,
                          rmm::device_async_resource_ref m)
        : schema_idx(si), counts(n, s, m), offsets(n + 1, s, m)
      {
      }
    };

    std::vector<std::unique_ptr<repeated_field_work>> rep_work;
    rep_work.reserve(num_repeated);

    for (int ri = 0; ri < num_repeated; ri++) {
      int schema_idx = repeated_field_indices[ri];
      auto& w        = *rep_work.emplace_back(
        std::make_unique<repeated_field_work>(schema_idx, num_rows, stream, mr));

      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(num_rows),
                        w.counts.data(),
                        extract_strided_count{d_repeated_info.data(), ri, num_repeated});

      CUDF_CUDA_TRY(cudaMemsetAsync(w.offsets.data(), 0, sizeof(int32_t), stream.value()));
      thrust::inclusive_scan(
        rmm::exec_policy(stream), w.counts.begin(), w.counts.end(), w.offsets.data() + 1);

      CUDF_CUDA_TRY(cudaMemcpyAsync(&w.total_count,
                                    w.offsets.data() + num_rows,
                                    sizeof(int32_t),
                                    cudaMemcpyDeviceToHost,
                                    stream.value()));
    }
    stream.synchronize();

    // Phase B: Allocate occurrence buffers and launch ONE combined scan kernel.
    std::vector<protobuf_detail::repeated_field_scan_desc> h_scan_descs;
    h_scan_descs.reserve(num_repeated);

    for (auto& wp : rep_work) {
      if (wp->total_count > 0) {
        wp->occurrences =
          std::make_unique<rmm::device_uvector<protobuf_detail::repeated_occurrence>>(
            wp->total_count, stream, mr);
        h_scan_descs.push_back({schema[wp->schema_idx].field_number,
                                schema[wp->schema_idx].wire_type,
                                wp->offsets.data(),
                                wp->occurrences->data()});
      }
    }

    if (!h_scan_descs.empty()) {
      rmm::device_uvector<protobuf_detail::repeated_field_scan_desc> d_scan_descs(
        h_scan_descs.size(), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_scan_descs.data(),
                                    h_scan_descs.data(),
                                    h_scan_descs.size() * sizeof(h_scan_descs[0]),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));

      // Build field_number -> scan_desc_index lookup for the combined kernel
      int max_scan_fn = 0;
      for (auto const& sd : h_scan_descs) {
        max_scan_fn = std::max(max_scan_fn, sd.field_number);
      }
      rmm::device_uvector<int> d_fn_to_scan(0, stream, mr);
      int fn_to_scan_size = 0;
      if (max_scan_fn <= protobuf_detail::FIELD_LOOKUP_TABLE_MAX) {
        std::vector<int> h_fn_to_scan(max_scan_fn + 1, -1);
        for (int i = 0; i < static_cast<int>(h_scan_descs.size()); i++) {
          h_fn_to_scan[h_scan_descs[i].field_number] = i;
        }
        d_fn_to_scan = rmm::device_uvector<int>(h_fn_to_scan.size(), stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(d_fn_to_scan.data(),
                                      h_fn_to_scan.data(),
                                      h_fn_to_scan.size() * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      stream.value()));
        fn_to_scan_size = static_cast<int>(h_fn_to_scan.size());
      }

      scan_all_repeated_occurrences_kernel<<<blocks, threads, 0, stream.value()>>>(
        *d_in,
        d_scan_descs.data(),
        static_cast<int>(h_scan_descs.size()),
        d_error.data(),
        d_fn_to_scan.data(),
        fn_to_scan_size);
    }

    // Phase C: Build columns per field.
    for (int ri = 0; ri < num_repeated; ri++) {
      auto& w              = *rep_work[ri];
      int schema_idx       = w.schema_idx;
      auto element_type    = schema_output_types[schema_idx];
      int32_t total_count  = w.total_count;
      auto& d_field_counts = w.counts;

      if (total_count > 0) {
        auto& d_occurrences = *w.occurrences;

        // Build the appropriate column type based on element type
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
                                                    d_error,
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
                                                    d_error,
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
                                                     d_error,
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
                                                     d_error,
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
                                                  d_error,
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
                                                   d_error,
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
                                                    d_error,
                                                    stream,
                                                    mr);
            break;
          case cudf::type_id::STRING: {
            auto const field_meta = make_field_meta_view(context, schema_idx);
            auto enc              = field_meta.schema.encoding;
            if (enc == spark_rapids_jni::ENC_ENUM_STRING) {
              if (!field_meta.enum_valid_values.empty() &&
                  field_meta.enum_valid_values.size() == field_meta.enum_names.size()) {
                column_map[schema_idx] =
                  build_repeated_enum_string_column(binary_input,
                                                    message_data,
                                                    list_offsets,
                                                    base_offset,
                                                    d_field_counts,
                                                    d_occurrences,
                                                    total_count,
                                                    num_rows,
                                                    field_meta.enum_valid_values,
                                                    field_meta.enum_names,
                                                    d_row_has_invalid_enum,
                                                    d_error,
                                                    stream,
                                                    mr);
              } else {
                int err_val = ERR_MISSING_ENUM_META;
                CUDF_CUDA_TRY(cudaMemcpyAsync(
                  d_error.data(), &err_val, sizeof(int), cudaMemcpyHostToDevice, stream.value()));
                column_map[schema_idx] =
                  make_null_column(schema_output_types[schema_idx], num_rows, stream, mr);
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
                                                                    d_error,
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
                                                                  d_error,
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
                                                                    message_data_size,
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
                                                           std::move(null_mask));
        } else {
          column_map[schema_idx] = cudf::make_lists_column(
            num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{});
        }
      }
    }
  }

  // Process nested struct fields (Phase 2)
  if (num_nested > 0) {
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
                                                                 message_data_size,
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
                                                                 nullptr,
                                                                 0);
    }
  }

  // Assemble top_level_children in schema order (not processing order)
  std::vector<std::unique_ptr<cudf::column>> top_level_children;
  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == -1) {  // Top-level field
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

  CUDF_CUDA_TRY(cudaPeekAtLastError());
  int h_error = 0;
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(&h_error, d_error.data(), sizeof(int), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  if (fail_on_errors && h_error != 0) { throw cudf::logic_error(error_message(h_error)); }

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
