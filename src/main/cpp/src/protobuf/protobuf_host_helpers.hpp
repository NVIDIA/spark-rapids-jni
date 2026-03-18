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

#include "protobuf/protobuf_kernels.cuh"

namespace spark_rapids_jni::protobuf::detail {

// ============================================================================
// Field number lookup table helpers
// ============================================================================

/**
 * Build a host-side direct-mapped lookup table: field_number -> local_index,
 * given an array of schema indices and the schema itself.
 * Returns an empty vector if the max field number exceeds the threshold.
 */
inline std::vector<int> build_index_lookup_table(nested_field_descriptor const* schema,
                                                 int const* field_indices,
                                                 int num_indices)
{
  int max_fn = 0;
  for (int i = 0; i < num_indices; i++) {
    max_fn = std::max(max_fn, schema[field_indices[i]].field_number);
  }
  if (max_fn > FIELD_LOOKUP_TABLE_MAX) return {};
  std::vector<int> table(max_fn + 1, -1);
  for (int i = 0; i < num_indices; i++) {
    table[schema[field_indices[i]].field_number] = i;
  }
  return table;
}

/**
 * Build a host-side direct-mapped lookup table: field_number -> field_index.
 * Returns an empty vector if the max field number exceeds the threshold.
 */
inline std::vector<int> build_field_lookup_table(field_descriptor const* descs, int num_fields)
{
  int max_fn = 0;
  for (int i = 0; i < num_fields; i++) {
    max_fn = std::max(max_fn, descs[i].field_number);
  }
  if (max_fn > FIELD_LOOKUP_TABLE_MAX) return {};
  std::vector<int> table(max_fn + 1, -1);
  for (int i = 0; i < num_fields; i++) {
    table[descs[i].field_number] = i;
  }
  return table;
}

template <typename T>
inline std::pair<rmm::device_buffer, cudf::size_type> make_null_mask_from_valid(
  rmm::device_uvector<T> const& valid,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end   = begin + valid.size();
  auto pred  = [ptr = valid.data()] __device__(cudf::size_type i) {
    return static_cast<bool>(ptr[i]);
  };
  return cudf::detail::valid_if(begin, end, pred, stream, mr);
}

template <typename T, typename LaunchFn>
std::unique_ptr<cudf::column> extract_and_build_scalar_column(cudf::data_type dt,
                                                              int num_rows,
                                                              LaunchFn&& launch_extract,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<T> out(num_rows, stream, mr);
  rmm::device_uvector<bool> valid((num_rows > 0 ? num_rows : 1), stream, mr);
  if (num_rows == 0) {
    return std::make_unique<cudf::column>(dt, 0, out.release(), rmm::device_buffer{}, 0);
  }
  launch_extract(out.data(), valid.data());
  auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
  return std::make_unique<cudf::column>(dt, num_rows, out.release(), std::move(mask), null_count);
}

template <typename T, typename LocationProvider>
// Shared integer extractor for INT32/INT64/UINT32/UINT64 decode paths.
inline void extract_integer_into_buffers(uint8_t const* message_data,
                                         LocationProvider const& loc_provider,
                                         int num_rows,
                                         int blocks,
                                         int threads,
                                         bool has_default,
                                         int64_t default_value,
                                         int encoding,
                                         bool enable_zigzag,
                                         T* out_ptr,
                                         bool* valid_ptr,
                                         int* error_ptr,
                                         rmm::cuda_stream_view stream)
{
  if (enable_zigzag && encoding == spark_rapids_jni::protobuf::encoding_value(
                                     spark_rapids_jni::protobuf::proto_encoding::ZIGZAG)) {
    extract_varint_kernel<T, true, LocationProvider>
      <<<blocks, threads, 0, stream.value()>>>(message_data,
                                               loc_provider,
                                               num_rows,
                                               out_ptr,
                                               valid_ptr,
                                               error_ptr,
                                               has_default,
                                               default_value);
  } else if (encoding == spark_rapids_jni::protobuf::encoding_value(
                           spark_rapids_jni::protobuf::proto_encoding::FIXED)) {
    if constexpr (sizeof(T) == 4) {
      extract_fixed_kernel<T,
                           spark_rapids_jni::protobuf::wire_type_value(
                             spark_rapids_jni::protobuf::proto_wire_type::I32BIT),
                           LocationProvider>
        <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                 loc_provider,
                                                 num_rows,
                                                 out_ptr,
                                                 valid_ptr,
                                                 error_ptr,
                                                 has_default,
                                                 static_cast<T>(default_value));
    } else {
      static_assert(sizeof(T) == 8, "extract_integer_into_buffers only supports 32/64-bit");
      extract_fixed_kernel<T,
                           spark_rapids_jni::protobuf::wire_type_value(
                             spark_rapids_jni::protobuf::proto_wire_type::I64BIT),
                           LocationProvider>
        <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                 loc_provider,
                                                 num_rows,
                                                 out_ptr,
                                                 valid_ptr,
                                                 error_ptr,
                                                 has_default,
                                                 static_cast<T>(default_value));
    }
  } else {
    extract_varint_kernel<T, false, LocationProvider>
      <<<blocks, threads, 0, stream.value()>>>(message_data,
                                               loc_provider,
                                               num_rows,
                                               out_ptr,
                                               valid_ptr,
                                               error_ptr,
                                               has_default,
                                               default_value);
  }
}

template <typename T, typename LocationProvider>
// Builds a scalar column for integer-like protobuf fields.
std::unique_ptr<cudf::column> extract_and_build_integer_column(cudf::data_type dt,
                                                               uint8_t const* message_data,
                                                               LocationProvider const& loc_provider,
                                                               int num_rows,
                                                               int blocks,
                                                               int threads,
                                                               rmm::device_uvector<int>& d_error,
                                                               bool has_default,
                                                               int64_t default_value,
                                                               int encoding,
                                                               bool enable_zigzag,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::device_async_resource_ref mr)
{
  return extract_and_build_scalar_column<T>(
    dt,
    num_rows,
    [&](T* out_ptr, bool* valid_ptr) {
      extract_integer_into_buffers<T, LocationProvider>(message_data,
                                                        loc_provider,
                                                        num_rows,
                                                        blocks,
                                                        threads,
                                                        has_default,
                                                        default_value,
                                                        encoding,
                                                        enable_zigzag,
                                                        out_ptr,
                                                        valid_ptr,
                                                        d_error.data(),
                                                        stream);
    },
    stream,
    mr);
}

struct extract_strided_count {
  repeated_field_info const* info;
  int field_idx;
  int num_fields;

  __device__ int32_t operator()(int row) const
  {
    return info[flat_index(static_cast<size_t>(row),
                           static_cast<size_t>(num_fields),
                           static_cast<size_t>(field_idx))]
      .count;
  }
};

/**
 * Find all child field indices for a given parent index in the schema.
 * This is a commonly used pattern throughout the codebase.
 *
 * @param schema The schema vector (either nested_field_descriptor or
 * device_nested_field_descriptor)
 * @param num_fields Number of fields in the schema
 * @param parent_idx The parent index to search for
 * @return Vector of child field indices
 */
template <typename SchemaT>
std::vector<int> find_child_field_indices(SchemaT const& schema, int num_fields, int parent_idx)
{
  std::vector<int> child_indices;
  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == parent_idx) { child_indices.push_back(i); }
  }
  return child_indices;
}

// Forward declarations needed by make_empty_struct_column_with_schema
std::unique_ptr<cudf::column> make_empty_column_safe(cudf::data_type dtype,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> make_empty_list_column(std::unique_ptr<cudf::column> element_col,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

template <typename SchemaT>
std::unique_ptr<cudf::column> make_empty_struct_column_with_schema(
  SchemaT const& schema,
  std::vector<cudf::data_type> const& schema_output_types,
  int parent_idx,
  int num_fields,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto child_indices = find_child_field_indices(schema, num_fields, parent_idx);

  std::vector<std::unique_ptr<cudf::column>> children;
  for (int child_idx : child_indices) {
    auto child_type = schema_output_types[child_idx];

    std::unique_ptr<cudf::column> child_col;
    if (child_type.id() == cudf::type_id::STRUCT) {
      child_col = make_empty_struct_column_with_schema(
        schema, schema_output_types, child_idx, num_fields, stream, mr);
    } else {
      child_col = make_empty_column_safe(child_type, stream, mr);
    }

    if (schema[child_idx].is_repeated) {
      child_col = make_empty_list_column(std::move(child_col), stream, mr);
    }

    children.push_back(std::move(child_col));
  }

  return cudf::make_structs_column(0, std::move(children), 0, rmm::device_buffer{}, stream, mr);
}

inline void maybe_check_required_fields(field_location const* locations,
                                        std::vector<int> const& field_indices,
                                        std::vector<nested_field_descriptor> const& schema,
                                        int num_rows,
                                        cudf::bitmask_type const* input_null_mask,
                                        cudf::size_type input_offset,
                                        field_location const* parent_locs,
                                        bool* row_force_null,
                                        int32_t const* top_row_indices,
                                        int* error_flag,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  if (num_rows == 0 || field_indices.empty()) { return; }

  bool has_required = false;
  std::vector<uint8_t> h_is_required(field_indices.size());
  for (size_t i = 0; i < field_indices.size(); ++i) {
    h_is_required[i] = schema[field_indices[i]].is_required ? 1 : 0;
    has_required |= (h_is_required[i] != 0);
  }
  if (!has_required) { return; }

  rmm::device_uvector<uint8_t> d_is_required(field_indices.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_is_required.data(),
                                h_is_required.data(),
                                h_is_required.size() * sizeof(uint8_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  check_required_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    locations,
    d_is_required.data(),
    static_cast<int>(field_indices.size()),
    num_rows,
    input_null_mask,
    input_offset,
    parent_locs,
    row_force_null,
    top_row_indices,
    error_flag);
}

__global__ void validate_enum_values_kernel(int32_t const* values,
                                            bool* valid,
                                            bool* row_has_invalid_enum,
                                            int32_t const* valid_enum_values,
                                            int num_valid_values,
                                            int num_rows);

__global__ void compute_enum_string_lengths_kernel(int32_t const* values,
                                                   bool const* valid,
                                                   int32_t const* valid_enum_values,
                                                   int32_t const* enum_name_offsets,
                                                   int num_valid_values,
                                                   int32_t* lengths,
                                                   int num_rows);

__global__ void copy_enum_string_chars_kernel(int32_t const* values,
                                              bool const* valid,
                                              int32_t const* valid_enum_values,
                                              int32_t const* enum_name_offsets,
                                              uint8_t const* enum_name_chars,
                                              int num_valid_values,
                                              int32_t const* output_offsets,
                                              char* out_chars,
                                              int num_rows);

inline void propagate_invalid_enum_flags_to_rows(rmm::device_uvector<bool> const& item_invalid,
                                                 rmm::device_uvector<bool>& row_invalid,
                                                 int num_items,
                                                 int32_t const* top_row_indices,
                                                 bool propagate_to_rows,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  if (num_items == 0 || row_invalid.size() == 0 || !propagate_to_rows) { return; }

  if (top_row_indices == nullptr) {
    CUDF_EXPECTS(static_cast<size_t>(num_items) <= row_invalid.size(),
                 "enum invalid-row propagation exceeded row buffer");
    thrust::transform(rmm::exec_policy(stream),
                      row_invalid.begin(),
                      row_invalid.begin() + num_items,
                      item_invalid.begin(),
                      row_invalid.begin(),
                      [] __device__(bool row_is_invalid, bool item_is_invalid) {
                        return row_is_invalid || item_is_invalid;
                      });
    return;
  }

  rmm::device_uvector<int32_t> invalid_rows(num_items, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(num_items),
                    invalid_rows.begin(),
                    [item_invalid = item_invalid.data(), top_row_indices] __device__(int idx) {
                      return item_invalid[idx] ? top_row_indices[idx] : -1;
                    });

  auto valid_end =
    thrust::remove(rmm::exec_policy(stream), invalid_rows.begin(), invalid_rows.end(), -1);
  thrust::sort(rmm::exec_policy(stream), invalid_rows.begin(), valid_end);
  auto unique_end = thrust::unique(rmm::exec_policy(stream), invalid_rows.begin(), valid_end);
  thrust::for_each(rmm::exec_policy(stream),
                   invalid_rows.begin(),
                   unique_end,
                   [row_invalid = row_invalid.data()] __device__(int32_t row_idx) {
                     row_invalid[row_idx] = true;
                   });
}

inline void validate_enum_and_propagate_rows(rmm::device_uvector<int32_t> const& values,
                                             rmm::device_uvector<bool>& valid,
                                             std::vector<int32_t> const& valid_enums,
                                             rmm::device_uvector<bool>& row_invalid,
                                             int num_items,
                                             int32_t const* top_row_indices,
                                             bool propagate_to_rows,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  if (num_items == 0 || valid_enums.empty()) { return; }

  auto const blocks = static_cast<int>((num_items + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  rmm::device_uvector<int32_t> d_valid_enums(valid_enums.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_valid_enums.data(),
                                valid_enums.data(),
                                valid_enums.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  rmm::device_uvector<bool> item_invalid(num_items, stream, mr);
  thrust::fill(rmm::exec_policy(stream), item_invalid.begin(), item_invalid.end(), false);
  validate_enum_values_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    values.data(),
    valid.data(),
    item_invalid.data(),
    d_valid_enums.data(),
    static_cast<int>(valid_enums.size()),
    num_items);

  propagate_invalid_enum_flags_to_rows(
    item_invalid, row_invalid, num_items, top_row_indices, propagate_to_rows, stream, mr);
}

// ============================================================================
// Forward declarations of builder/utility functions
// ============================================================================

std::unique_ptr<cudf::column> make_null_column(cudf::data_type dtype,
                                               cudf::size_type num_rows,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> make_null_list_column_with_child(
  std::unique_ptr<cudf::column> child_col,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_enum_string_column(
  rmm::device_uvector<int32_t>& enum_values,
  rmm::device_uvector<bool>& valid,
  std::vector<int32_t> const& valid_enums,
  std::vector<std::vector<uint8_t>> const& enum_name_bytes,
  rmm::device_uvector<bool>& d_row_force_null,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices = nullptr,
  bool propagate_invalid_rows    = true);

// Complex builder forward declarations
std::unique_ptr<cudf::column> build_repeated_enum_string_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  rmm::device_uvector<int32_t> const& d_field_counts,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  std::vector<int32_t> const& valid_enums,
  std::vector<std::vector<uint8_t>> const& enum_name_bytes,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_repeated_string_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  device_nested_field_descriptor const& field_desc,
  rmm::device_uvector<int32_t> const& d_field_counts,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  bool is_bytes,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> build_nested_struct_column(
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  rmm::device_uvector<field_location> const& d_parent_locs,
  std::vector<int> const& child_field_indices,
  std::vector<nested_field_descriptor> const& schema,
  int num_fields,
  std::vector<cudf::data_type> const& schema_output_types,
  std::vector<int64_t> const& default_ints,
  std::vector<double> const& default_floats,
  std::vector<bool> const& default_bools,
  std::vector<std::vector<uint8_t>> const& default_strings,
  std::vector<std::vector<int32_t>> const& enum_valid_values,
  std::vector<std::vector<std::vector<uint8_t>>> const& enum_names,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<int>& d_error,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices,
  int depth,
  bool propagate_invalid_rows = true);

std::unique_ptr<cudf::column> build_repeated_child_list_column(
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  field_location const* parent_locs,
  int num_parent_rows,
  int child_schema_idx,
  std::vector<nested_field_descriptor> const& schema,
  int num_fields,
  std::vector<cudf::data_type> const& schema_output_types,
  std::vector<int64_t> const& default_ints,
  std::vector<double> const& default_floats,
  std::vector<bool> const& default_bools,
  std::vector<std::vector<uint8_t>> const& default_strings,
  std::vector<std::vector<int32_t>> const& enum_valid_values,
  std::vector<std::vector<std::vector<uint8_t>>> const& enum_names,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices,
  int depth,
  bool propagate_invalid_rows = true);

std::unique_ptr<cudf::column> build_repeated_struct_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
  cudf::size_type message_data_size,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  device_nested_field_descriptor const& field_desc,
  rmm::device_uvector<int32_t> const& d_field_counts,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  std::vector<device_nested_field_descriptor> const& h_device_schema,
  std::vector<int> const& child_field_indices,
  std::vector<cudf::data_type> const& schema_output_types,
  std::vector<int64_t> const& default_ints,
  std::vector<double> const& default_floats,
  std::vector<bool> const& default_bools,
  std::vector<std::vector<uint8_t>> const& default_strings,
  std::vector<nested_field_descriptor> const& schema,
  std::vector<std::vector<int32_t>> const& enum_valid_values,
  std::vector<std::vector<std::vector<uint8_t>>> const& enum_names,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<int>& d_error_top,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template <typename LengthProvider, typename CopyProvider, typename ValidityFn>
inline std::unique_ptr<cudf::column> extract_and_build_string_or_bytes_column(
  bool as_bytes,
  uint8_t const* message_data,
  int num_rows,
  LengthProvider const& length_provider,
  CopyProvider const& copy_provider,
  ValidityFn validity_fn,
  bool has_default,
  std::vector<uint8_t> const& default_bytes,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  int32_t def_len = has_default ? static_cast<int32_t>(default_bytes.size()) : 0;
  rmm::device_uvector<uint8_t> d_default(def_len, stream, mr);
  if (has_default && def_len > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_default.data(), default_bytes.data(), def_len, cudaMemcpyHostToDevice, stream.value()));
  }

  rmm::device_uvector<int32_t> lengths(num_rows, stream, mr);
  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((num_rows + threads - 1u) / threads);
  extract_lengths_kernel<LengthProvider><<<blocks, threads, 0, stream.value()>>>(
    length_provider, num_rows, lengths.data(), has_default, def_len);

  auto [offsets_col, total_size] =
    cudf::strings::detail::make_offsets_child_column(lengths.begin(), lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_size, stream, mr);
  if (total_size > 0) {
    copy_varlen_data_kernel<CopyProvider>
      <<<blocks, threads, 0, stream.value()>>>(message_data,
                                               copy_provider,
                                               num_rows,
                                               offsets_col->view().data<int32_t>(),
                                               chars.data(),
                                               d_error.data(),
                                               has_default,
                                               d_default.data(),
                                               def_len);
  }

  if (num_rows == 0) {
    if (as_bytes) {
      auto bytes_child = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::UINT8}, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
      return cudf::make_lists_column(
        0, std::move(offsets_col), std::move(bytes_child), 0, rmm::device_buffer{});
    }
    return cudf::make_strings_column(
      0, std::move(offsets_col), chars.release(), 0, rmm::device_buffer{});
  }

  rmm::device_uvector<bool> valid(num_rows, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(num_rows),
                    valid.data(),
                    validity_fn);
  auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
  if (as_bytes) {
    auto bytes_child =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                     total_size,
                                     rmm::device_buffer(chars.data(), total_size, stream, mr),
                                     rmm::device_buffer{},
                                     0);
    return cudf::make_lists_column(
      num_rows, std::move(offsets_col), std::move(bytes_child), null_count, std::move(mask));
  }

  return cudf::make_strings_column(
    num_rows, std::move(offsets_col), chars.release(), null_count, std::move(mask));
}

template <typename LocationProvider>
inline std::unique_ptr<cudf::column> extract_typed_column(
  cudf::data_type dt,
  int encoding,
  uint8_t const* message_data,
  LocationProvider const& loc_provider,
  int num_items,
  int blocks,
  int threads_per_block,
  bool has_default,
  int64_t default_int,
  double default_float,
  bool default_bool,
  std::vector<uint8_t> const& default_string,
  int schema_idx,
  std::vector<std::vector<int32_t>> const& enum_valid_values,
  std::vector<std::vector<std::vector<uint8_t>>> const& enum_names,
  rmm::device_uvector<bool>& d_row_force_null,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int32_t const* top_row_indices = nullptr,
  bool propagate_invalid_rows    = true)
{
  switch (dt.id()) {
    case cudf::type_id::BOOL8: {
      int64_t def_val = has_default ? (default_bool ? 1 : 0) : 0;
      return extract_and_build_scalar_column<uint8_t>(
        dt,
        num_items,
        [&](uint8_t* out_ptr, bool* valid_ptr) {
          extract_varint_kernel<uint8_t, false, LocationProvider>
            <<<blocks, threads_per_block, 0, stream.value()>>>(message_data,
                                                               loc_provider,
                                                               num_items,
                                                               out_ptr,
                                                               valid_ptr,
                                                               d_error.data(),
                                                               has_default,
                                                               def_val);
        },
        stream,
        mr);
    }
    case cudf::type_id::INT32: {
      if (num_items == 0) {
        return std::make_unique<cudf::column>(dt, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
      }
      rmm::device_uvector<int32_t> out(num_items, stream, mr);
      rmm::device_uvector<bool> valid(num_items, stream, mr);
      extract_integer_into_buffers<int32_t, LocationProvider>(message_data,
                                                              loc_provider,
                                                              num_items,
                                                              blocks,
                                                              threads_per_block,
                                                              has_default,
                                                              default_int,
                                                              encoding,
                                                              true,
                                                              out.data(),
                                                              valid.data(),
                                                              d_error.data(),
                                                              stream);
      if (schema_idx < static_cast<int>(enum_valid_values.size())) {
        auto const& valid_enums = enum_valid_values[schema_idx];
        if (!valid_enums.empty()) {
          validate_enum_and_propagate_rows(out,
                                           valid,
                                           valid_enums,
                                           d_row_force_null,
                                           num_items,
                                           top_row_indices,
                                           propagate_invalid_rows,
                                           stream,
                                           mr);
        }
      }
      auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
      return std::make_unique<cudf::column>(
        dt, num_items, out.release(), std::move(mask), null_count);
    }
    case cudf::type_id::UINT32:
      return extract_and_build_integer_column<uint32_t>(dt,
                                                        message_data,
                                                        loc_provider,
                                                        num_items,
                                                        blocks,
                                                        threads_per_block,
                                                        d_error,
                                                        has_default,
                                                        default_int,
                                                        encoding,
                                                        false,
                                                        stream,
                                                        mr);
    case cudf::type_id::INT64:
      return extract_and_build_integer_column<int64_t>(dt,
                                                       message_data,
                                                       loc_provider,
                                                       num_items,
                                                       blocks,
                                                       threads_per_block,
                                                       d_error,
                                                       has_default,
                                                       default_int,
                                                       encoding,
                                                       true,
                                                       stream,
                                                       mr);
    case cudf::type_id::UINT64:
      return extract_and_build_integer_column<uint64_t>(dt,
                                                        message_data,
                                                        loc_provider,
                                                        num_items,
                                                        blocks,
                                                        threads_per_block,
                                                        d_error,
                                                        has_default,
                                                        default_int,
                                                        encoding,
                                                        false,
                                                        stream,
                                                        mr);
    case cudf::type_id::FLOAT32: {
      float def_float_val = has_default ? static_cast<float>(default_float) : 0.0f;
      return extract_and_build_scalar_column<float>(
        dt,
        num_items,
        [&](float* out_ptr, bool* valid_ptr) {
          extract_fixed_kernel<float,
                               spark_rapids_jni::protobuf::wire_type_value(
                                 spark_rapids_jni::protobuf::proto_wire_type::I32BIT),
                               LocationProvider>
            <<<blocks, threads_per_block, 0, stream.value()>>>(message_data,
                                                               loc_provider,
                                                               num_items,
                                                               out_ptr,
                                                               valid_ptr,
                                                               d_error.data(),
                                                               has_default,
                                                               def_float_val);
        },
        stream,
        mr);
    }
    case cudf::type_id::FLOAT64: {
      double def_double = has_default ? default_float : 0.0;
      return extract_and_build_scalar_column<double>(
        dt,
        num_items,
        [&](double* out_ptr, bool* valid_ptr) {
          extract_fixed_kernel<double,
                               spark_rapids_jni::protobuf::wire_type_value(
                                 spark_rapids_jni::protobuf::proto_wire_type::I64BIT),
                               LocationProvider>
            <<<blocks, threads_per_block, 0, stream.value()>>>(message_data,
                                                               loc_provider,
                                                               num_items,
                                                               out_ptr,
                                                               valid_ptr,
                                                               d_error.data(),
                                                               has_default,
                                                               def_double);
        },
        stream,
        mr);
    }
    default: return make_null_column(dt, num_items, stream, mr);
  }
}

template <typename T>
inline std::unique_ptr<cudf::column> build_repeated_scalar_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
  cudf::size_type const* list_offsets,
  cudf::size_type base_offset,
  device_nested_field_descriptor const& field_desc,
  rmm::device_uvector<int32_t> const& d_field_counts,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const input_null_count = binary_input.null_count();

  if (total_count == 0) {
    // All rows have count=0, but we still need to check input nulls
    rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
    thrust::fill(rmm::exec_policy(stream), offsets.begin(), offsets.end(), 0);
    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      num_rows + 1,
                                                      offsets.release(),
                                                      rmm::device_buffer{},
                                                      0);
    auto elem_type   = field_desc.output_type_id == static_cast<int>(cudf::type_id::LIST)
                         ? cudf::type_id::UINT8
                         : static_cast<cudf::type_id>(field_desc.output_type_id);
    auto child_col   = make_empty_column_safe(cudf::data_type{elem_type}, stream, mr);

    if (input_null_count > 0) {
      // Copy input null mask - only input nulls produce output nulls
      auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
      return cudf::make_lists_column(num_rows,
                                     std::move(offsets_col),
                                     std::move(child_col),
                                     input_null_count,
                                     std::move(null_mask));
    } else {
      // No input nulls, all rows get empty arrays []
      return cudf::make_lists_column(
        num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{});
    }
  }

  rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_field_counts.begin(), d_field_counts.end(), list_offs.begin(), 0);

  int32_t total_count_i32 = static_cast<int32_t>(total_count);
  thrust::fill_n(rmm::exec_policy(stream), list_offs.data() + num_rows, 1, total_count_i32);

  rmm::device_uvector<T> values(total_count, stream, mr);

  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((total_count + threads - 1u) / threads);

  int encoding = field_desc.encoding;
  bool zigzag  = (encoding == spark_rapids_jni::protobuf::encoding_value(
                               spark_rapids_jni::protobuf::proto_encoding::ZIGZAG));

  // For float/double types, always use fixed kernel (they use wire type 32BIT/64BIT)
  // For integer types, use fixed kernel only if encoding is
  // spark_rapids_jni::protobuf::encoding_value(spark_rapids_jni::protobuf::proto_encoding::FIXED)
  constexpr bool is_floating_point = std::is_same_v<T, float> || std::is_same_v<T, double>;
  bool use_fixed_kernel =
    is_floating_point || (encoding == spark_rapids_jni::protobuf::encoding_value(
                                        spark_rapids_jni::protobuf::proto_encoding::FIXED));

  RepeatedLocationProvider loc_provider{list_offsets, base_offset, d_occurrences.data()};
  if (use_fixed_kernel) {
    if constexpr (sizeof(T) == 4) {
      extract_fixed_kernel<T,
                           spark_rapids_jni::protobuf::wire_type_value(
                             spark_rapids_jni::protobuf::proto_wire_type::I32BIT),
                           RepeatedLocationProvider><<<blocks, threads, 0, stream.value()>>>(
        message_data, loc_provider, total_count, values.data(), nullptr, d_error.data());
    } else {
      extract_fixed_kernel<T,
                           spark_rapids_jni::protobuf::wire_type_value(
                             spark_rapids_jni::protobuf::proto_wire_type::I64BIT),
                           RepeatedLocationProvider><<<blocks, threads, 0, stream.value()>>>(
        message_data, loc_provider, total_count, values.data(), nullptr, d_error.data());
    }
  } else if (zigzag) {
    extract_varint_kernel<T, true, RepeatedLocationProvider>
      <<<blocks, threads, 0, stream.value()>>>(
        message_data, loc_provider, total_count, values.data(), nullptr, d_error.data());
  } else {
    extract_varint_kernel<T, false, RepeatedLocationProvider>
      <<<blocks, threads, 0, stream.value()>>>(
        message_data, loc_provider, total_count, values.data(), nullptr, d_error.data());
  }

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    num_rows + 1,
                                                    list_offs.release(),
                                                    rmm::device_buffer{},
                                                    0);
  auto child_col   = std::make_unique<cudf::column>(
    cudf::data_type{static_cast<cudf::type_id>(field_desc.output_type_id)},
    total_count,
    values.release(),
    rmm::device_buffer{},
    0);

  // Only rows where INPUT is null should produce null output
  // Rows with valid input but count=0 should produce empty array []
  if (input_null_count > 0) {
    // Copy input null mask - only input nulls produce output nulls
    auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
    return cudf::make_lists_column(num_rows,
                                   std::move(offsets_col),
                                   std::move(child_col),
                                   input_null_count,
                                   std::move(null_mask));
  }

  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{});
}

}  // namespace spark_rapids_jni::protobuf::detail
