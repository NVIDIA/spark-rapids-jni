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

#include <cudf/lists/detail/lists_column_factories.hpp>

namespace spark_rapids_jni::protobuf_detail {

/**
 * Helper to build string or bytes column for repeated message child fields.
 * When as_bytes=false, builds a STRING column. When as_bytes=true, builds LIST<UINT8>.
 * Uses GPU kernels for parallel extraction (critical performance fix!).
 */
inline std::unique_ptr<cudf::column> build_repeated_msg_child_varlen_column(
  uint8_t const* message_data,
  rmm::device_uvector<int32_t> const& d_msg_row_offsets,
  rmm::device_uvector<field_location> const& d_msg_locs,
  rmm::device_uvector<field_location> const& d_child_locs,
  int child_idx,
  int num_child_fields,
  int total_count,
  rmm::device_uvector<int>& d_error,
  bool as_bytes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (total_count == 0) {
    if (as_bytes) {
      return make_empty_column_safe(cudf::data_type{cudf::type_id::LIST}, stream, mr);
    }
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }

  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = (total_count + threads - 1) / threads;

  rmm::device_uvector<int32_t> d_lengths(total_count, stream, mr);
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(total_count),
    d_lengths.data(),
    [child_locs = d_child_locs.data(), ci = child_idx, ncf = num_child_fields] __device__(int idx) {
      auto const& loc = child_locs[idx * ncf + ci];
      return loc.offset >= 0 ? loc.length : 0;
    });

  auto [offsets_col, total_data] = cudf::strings::detail::make_offsets_child_column(
    d_lengths.begin(), d_lengths.end(), stream, mr);

  rmm::device_uvector<char> d_data(total_data, stream, mr);
  rmm::device_uvector<bool> d_valid((total_count > 0 ? total_count : 1), stream, mr);

  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(total_count),
    d_valid.data(),
    [child_locs = d_child_locs.data(), ci = child_idx, ncf = num_child_fields] __device__(int idx) {
      return child_locs[idx * ncf + ci].offset >= 0;
    });

  if (total_data > 0) {
    RepeatedMsgChildLocationProvider loc_provider{d_msg_row_offsets.data(),
                                                  0,
                                                  d_msg_locs.data(),
                                                  d_child_locs.data(),
                                                  child_idx,
                                                  num_child_fields};
    copy_varlen_data_kernel<RepeatedMsgChildLocationProvider>
      <<<blocks, threads, 0, stream.value()>>>(message_data,
                                               loc_provider,
                                               total_count,
                                               offsets_col->view().data<int32_t>(),
                                               d_data.data(),
                                               d_error.data());
  }

  auto [mask, null_count] = make_null_mask_from_valid(d_valid, stream, mr);

  if (as_bytes) {
    auto bytes_child =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                     total_data,
                                     rmm::device_buffer(d_data.data(), total_data, stream, mr),
                                     rmm::device_buffer{},
                                     0);
    return cudf::make_lists_column(total_count,
                                   std::move(offsets_col),
                                   std::move(bytes_child),
                                   null_count,
                                   std::move(mask),
                                   stream,
                                   mr);
  }

  return cudf::make_strings_column(
    total_count, std::move(offsets_col), d_data.release(), null_count, std::move(mask));
}

// ============================================================================
// Utility functions
// ============================================================================

// Note: make_null_mask_from_valid is defined earlier in the file (before
// scan_repeated_message_children_kernel)

/**
 * Create an all-null column of the specified type.
 */
std::unique_ptr<cudf::column> make_null_column(cudf::data_type dtype,
                                               cudf::size_type num_rows,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  if (num_rows == 0) { return cudf::make_empty_column(dtype); }

  switch (dtype.id()) {
    case cudf::type_id::BOOL8:
    case cudf::type_id::INT8:
    case cudf::type_id::UINT8:
    case cudf::type_id::INT16:
    case cudf::type_id::UINT16:
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64:
      return cudf::make_fixed_width_column(dtype, num_rows, cudf::mask_state::ALL_NULL, stream, mr);
    case cudf::type_id::STRING: {
      rmm::device_uvector<cudf::strings::detail::string_index_pair> pairs(num_rows, stream, mr);
      thrust::fill(rmm::exec_policy(stream),
                   pairs.data(),
                   pairs.end(),
                   cudf::strings::detail::string_index_pair{nullptr, 0});
      return cudf::strings::detail::make_strings_column(pairs.data(), pairs.end(), stream, mr);
    }
    case cudf::type_id::LIST:
      return cudf::lists::detail::make_all_nulls_lists_column(
        num_rows, cudf::data_type{cudf::type_id::UINT8}, stream, mr);
    case cudf::type_id::STRUCT: {
      std::vector<std::unique_ptr<cudf::column>> empty_children;
      auto null_mask = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
      return cudf::make_structs_column(
        num_rows, std::move(empty_children), num_rows, std::move(null_mask), stream, mr);
    }
    default: CUDF_FAIL("Unsupported type for null column creation");
  }
}

/**
 * Create an empty column (0 rows) of the specified type.
 * This handles nested types (LIST, STRUCT) that cudf::make_empty_column doesn't support.
 */
std::unique_ptr<cudf::column> make_empty_column_safe(cudf::data_type dtype,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  switch (dtype.id()) {
    case cudf::type_id::LIST: {
      // Create empty list column with empty UINT8 child (Spark BinaryType maps to LIST<UINT8>)
      auto offsets_col =
        std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                       1,
                                       rmm::device_buffer(sizeof(int32_t), stream, mr),
                                       rmm::device_buffer{},
                                       0);
      // Initialize offset to 0
      int32_t zero = 0;
      CUDF_CUDA_TRY(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(),
                                    &zero,
                                    sizeof(int32_t),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      auto child_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::UINT8}, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
      return cudf::make_lists_column(
        0, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{}, stream, mr);
    }
    case cudf::type_id::STRUCT: {
      // Create empty struct column with no children
      std::vector<std::unique_ptr<cudf::column>> empty_children;
      return cudf::make_structs_column(
        0, std::move(empty_children), 0, rmm::device_buffer{}, stream, mr);
    }
    default:
      // For non-nested types, use cudf's make_empty_column
      return cudf::make_empty_column(dtype);
  }
}

/**
 * Create an all-null LIST column with the provided child column.
 * The child column is expected to have 0 rows.
 */
std::unique_ptr<cudf::column> make_null_list_column_with_child(
  std::unique_ptr<cudf::column> child_col,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
  thrust::fill(rmm::exec_policy(stream), offsets.begin(), offsets.end(), 0);
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    num_rows + 1,
                                                    offsets.release(),
                                                    rmm::device_buffer{},
                                                    0);
  auto null_mask   = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
  return cudf::make_lists_column(num_rows,
                                 std::move(offsets_col),
                                 std::move(child_col),
                                 num_rows,
                                 std::move(null_mask),
                                 stream,
                                 mr);
}

/**
 * Wrap a 0-row element column into a 0-row LIST column.
 */
std::unique_ptr<cudf::column> make_empty_list_column(std::unique_ptr<cudf::column> element_col,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    1,
                                                    rmm::device_buffer(sizeof(int32_t), stream, mr),
                                                    rmm::device_buffer{},
                                                    0);
  int32_t zero     = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(),
                                &zero,
                                sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  return cudf::make_lists_column(
    0, std::move(offsets_col), std::move(element_col), 0, rmm::device_buffer{}, stream, mr);
}


std::unique_ptr<cudf::column> build_enum_string_column(
  rmm::device_uvector<int32_t>& enum_values,
  rmm::device_uvector<bool>& valid,
  std::vector<int32_t> const& valid_enums,
  std::vector<std::vector<uint8_t>> const& enum_name_bytes,
  rmm::device_uvector<bool>& d_row_has_invalid_enum,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((num_rows + threads - 1) / threads);

  rmm::device_uvector<int32_t> d_valid_enums(valid_enums.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_valid_enums.data(),
                                valid_enums.data(),
                                valid_enums.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  validate_enum_values_kernel<<<blocks, threads, 0, stream.value()>>>(
    enum_values.data(),
    valid.data(),
    d_row_has_invalid_enum.data(),
    d_valid_enums.data(),
    static_cast<int>(valid_enums.size()),
    num_rows);

  std::vector<int32_t> h_name_offsets(valid_enums.size() + 1, 0);
  int32_t total_name_chars = 0;
  for (size_t k = 0; k < enum_name_bytes.size(); ++k) {
    total_name_chars += static_cast<int32_t>(enum_name_bytes[k].size());
    h_name_offsets[k + 1] = total_name_chars;
  }
  std::vector<uint8_t> h_name_chars(total_name_chars);
  int32_t cursor = 0;
  for (auto const& name : enum_name_bytes) {
    if (!name.empty()) {
      std::copy(name.data(), name.data() + name.size(), h_name_chars.data() + cursor);
      cursor += static_cast<int32_t>(name.size());
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

  rmm::device_uvector<int32_t> lengths(num_rows, stream, mr);
  compute_enum_string_lengths_kernel<<<blocks, threads, 0, stream.value()>>>(
    enum_values.data(),
    valid.data(),
    d_valid_enums.data(),
    d_name_offsets.data(),
    static_cast<int>(valid_enums.size()),
    lengths.data(),
    num_rows);

  auto [offsets_col, total_chars] = cudf::strings::detail::make_offsets_child_column(
    lengths.begin(), lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_chars, stream, mr);
  if (total_chars > 0) {
    copy_enum_string_chars_kernel<<<blocks, threads, 0, stream.value()>>>(
      enum_values.data(),
      valid.data(),
      d_valid_enums.data(),
      d_name_offsets.data(),
      d_name_chars.data(),
      static_cast<int>(valid_enums.size()),
      offsets_col->view().data<int32_t>(),
      chars.data(),
      num_rows);
  }

  auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
  return cudf::make_strings_column(
    num_rows, std::move(offsets_col), chars.release(), null_count, std::move(mask));
}

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
    auto child_col   = is_bytes ? make_empty_column_safe(
                                  cudf::data_type{cudf::type_id::LIST}, stream, mr)  // LIST<UINT8>
                                : cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

    if (input_null_count > 0) {
      // Copy input null mask - only input nulls produce output nulls
      auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
      return cudf::make_lists_column(num_rows,
                                     std::move(offsets_col),
                                     std::move(child_col),
                                     input_null_count,
                                     std::move(null_mask),
                                     stream,
                                     mr);
    } else {
      // No input nulls, all rows get empty arrays []
      return cudf::make_lists_column(num_rows,
                                     std::move(offsets_col),
                                     std::move(child_col),
                                     0,
                                     rmm::device_buffer{},
                                     stream,
                                     mr);
    }
  }

  rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_field_counts.begin(), d_field_counts.end(), list_offs.begin(), 0);

  CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data() + num_rows,
                                &total_count,
                                sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  // Extract string lengths from occurrences
  rmm::device_uvector<int32_t> str_lengths(total_count, stream, mr);
  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = (total_count + threads - 1) / threads;
  RepeatedLocationProvider loc_provider{nullptr, 0, d_occurrences.data()};
  extract_lengths_kernel<RepeatedLocationProvider>
    <<<blocks, threads, 0, stream.value()>>>(loc_provider, total_count, str_lengths.data());

  auto [str_offsets_col, total_chars] = cudf::strings::detail::make_offsets_child_column(
    str_lengths.begin(), str_lengths.end(), stream, mr);

  rmm::device_uvector<char> chars(total_chars, stream, mr);
  rmm::device_uvector<int> d_error(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 0, sizeof(int), stream.value()));
  if (total_chars > 0) {
    RepeatedLocationProvider loc_provider{list_offsets, base_offset, d_occurrences.data()};
    copy_varlen_data_kernel<RepeatedLocationProvider><<<blocks, threads, 0, stream.value()>>>(
      message_data,
      loc_provider,
      total_count,
      str_offsets_col->view().data<int32_t>(),
      chars.data(),
      d_error.data());
  }

  std::unique_ptr<cudf::column> child_col;
  if (is_bytes) {
    auto bytes_child =
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                     total_chars,
                                     rmm::device_buffer(chars.data(), total_chars, stream, mr),
                                     rmm::device_buffer{},
                                     0);
    child_col = cudf::make_lists_column(total_count,
                                        std::move(str_offsets_col),
                                        std::move(bytes_child),
                                        0,
                                        rmm::device_buffer{},
                                        stream,
                                        mr);
  } else {
    child_col = cudf::make_strings_column(
      total_count, std::move(str_offsets_col), chars.release(), 0, rmm::device_buffer{});
  }

  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    num_rows + 1,
                                                    list_offs.release(),
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
                                   std::move(null_mask),
                                   stream,
                                   mr);
  }

  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{}, stream, mr);
}

// Forward declaration -- build_nested_struct_column is defined after build_repeated_struct_column
// but the latter's STRUCT-child case needs to call it.
std::unique_ptr<cudf::column> build_nested_struct_column(
  uint8_t const* message_data,
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
  rmm::device_uvector<bool>& d_row_has_invalid_enum,
  rmm::device_uvector<int>& d_error,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int depth);

// Forward declaration -- build_repeated_child_list_column is defined after
// build_nested_struct_column but both build_repeated_struct_column and build_nested_struct_column
// need to call it.
std::unique_ptr<cudf::column> build_repeated_child_list_column(
  uint8_t const* message_data,
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
  rmm::device_uvector<bool>& d_row_has_invalid_enum,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int depth);

std::unique_ptr<cudf::column> build_repeated_struct_column(
  cudf::column_view const& binary_input,
  uint8_t const* message_data,
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
  rmm::device_uvector<bool>& d_row_has_invalid_enum,
  rmm::device_uvector<int>& d_error_top,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const input_null_count = binary_input.null_count();
  int num_child_fields        = static_cast<int>(child_field_indices.size());

  if (total_count == 0 || num_child_fields == 0) {
    // All rows have count=0 or no child fields - return list of empty structs
    rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
    thrust::fill(rmm::exec_policy(stream), offsets.begin(), offsets.end(), 0);
    auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      num_rows + 1,
                                                      offsets.release(),
                                                      rmm::device_buffer{},
                                                      0);

    // Build empty struct child column with proper nested structure
    int num_schema_fields = static_cast<int>(h_device_schema.size());
    std::vector<std::unique_ptr<cudf::column>> empty_struct_children;
    for (int child_schema_idx : child_field_indices) {
      auto child_type = schema_output_types[child_schema_idx];
      std::unique_ptr<cudf::column> child_col;
      if (child_type.id() == cudf::type_id::STRUCT) {
        child_col = make_empty_struct_column_with_schema(
          h_device_schema, schema_output_types, child_schema_idx, num_schema_fields, stream, mr);
      } else {
        child_col = make_empty_column_safe(child_type, stream, mr);
      }
      if (h_device_schema[child_schema_idx].is_repeated) {
        child_col = make_empty_list_column(std::move(child_col), stream, mr);
      }
      empty_struct_children.push_back(std::move(child_col));
    }
    auto empty_struct = cudf::make_structs_column(
      0, std::move(empty_struct_children), 0, rmm::device_buffer{}, stream, mr);

    if (input_null_count > 0) {
      auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
      return cudf::make_lists_column(num_rows,
                                     std::move(offsets_col),
                                     std::move(empty_struct),
                                     input_null_count,
                                     std::move(null_mask),
                                     stream,
                                     mr);
    } else {
      return cudf::make_lists_column(num_rows,
                                     std::move(offsets_col),
                                     std::move(empty_struct),
                                     0,
                                     rmm::device_buffer{},
                                     stream,
                                     mr);
    }
  }

  rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_field_counts.begin(), d_field_counts.end(), list_offs.begin(), 0);

  CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data() + num_rows,
                                &total_count,
                                sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  // Build child field descriptors for scanning within each message occurrence
  std::vector<field_descriptor> h_child_descs(num_child_fields);
  for (int ci = 0; ci < num_child_fields; ci++) {
    int child_schema_idx                 = child_field_indices[ci];
    h_child_descs[ci].field_number       = h_device_schema[child_schema_idx].field_number;
    h_child_descs[ci].expected_wire_type = h_device_schema[child_schema_idx].wire_type;
  }
  rmm::device_uvector<field_descriptor> d_child_descs(num_child_fields, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_child_descs.data(),
                                h_child_descs.data(),
                                num_child_fields * sizeof(field_descriptor),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  // For each occurrence, we need to scan for child fields
  // Create "virtual" parent locations from the occurrences using GPU kernel
  // This replaces the host-side loop with D->H->D copy pattern (critical performance fix!)
  rmm::device_uvector<field_location> d_msg_locs(total_count, stream, mr);
  rmm::device_uvector<int32_t> d_msg_row_offsets(total_count, stream, mr);
  rmm::device_uvector<cudf::size_type> d_msg_row_offsets_size(total_count, stream, mr);
  {
    auto const occ_threads = THREADS_PER_BLOCK;
    auto const occ_blocks  = (total_count + occ_threads - 1) / occ_threads;
    compute_msg_locations_from_occurrences_kernel<<<occ_blocks, occ_threads, 0, stream.value()>>>(
      d_occurrences.data(),
      list_offsets,
      base_offset,
      d_msg_locs.data(),
      d_msg_row_offsets.data(),
      total_count);
  }
  thrust::transform(rmm::exec_policy(stream),
                    d_msg_row_offsets.data(),
                    d_msg_row_offsets.end(),
                    d_msg_row_offsets_size.data(),
                    [] __device__(int32_t v) { return static_cast<cudf::size_type>(v); });

  // Scan for child fields within each message occurrence
  rmm::device_uvector<field_location> d_child_locs(total_count * num_child_fields, stream, mr);
  // Reuse top-level error flag so failfast can observe nested repeated-message failures.
  auto& d_error = d_error_top;

  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = (total_count + threads - 1) / threads;

  // Use a custom kernel to scan child fields within message occurrences
  // This is similar to scan_nested_message_fields_kernel but operates on occurrences
  scan_repeated_message_children_kernel<<<blocks, threads, 0, stream.value()>>>(
    message_data,
    d_msg_row_offsets.data(),
    d_msg_locs.data(),
    total_count,
    d_child_descs.data(),
    num_child_fields,
    d_child_locs.data(),
    d_error.data());

  // Note: We no longer need to copy child_locs to host because:
  // 1. All scalar extraction kernels access d_child_locs directly on device
  // 2. String extraction uses GPU kernels
  // 3. Nested struct locations are computed on GPU via compute_nested_struct_locations_kernel

  // Extract child field values - build one column per child field
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  int num_schema_fields = static_cast<int>(h_device_schema.size());
  for (int ci = 0; ci < num_child_fields; ci++) {
    int child_schema_idx   = child_field_indices[ci];
    auto const dt          = schema_output_types[child_schema_idx];
    auto const enc         = h_device_schema[child_schema_idx].encoding;
    bool has_def           = h_device_schema[child_schema_idx].has_default_value;
    bool child_is_repeated = h_device_schema[child_schema_idx].is_repeated;

    if (child_is_repeated) {
      struct_children.push_back(build_repeated_child_list_column(message_data,
                                                                 d_msg_row_offsets_size.data(),
                                                                 0,
                                                                 d_msg_locs.data(),
                                                                 total_count,
                                                                 child_schema_idx,
                                                                 schema,
                                                                 num_schema_fields,
                                                                 schema_output_types,
                                                                 default_ints,
                                                                 default_floats,
                                                                 default_bools,
                                                                 default_strings,
                                                                 enum_valid_values,
                                                                 enum_names,
                                                                 d_row_has_invalid_enum,
                                                                 d_error_top,
                                                                 stream,
                                                                 mr,
                                                                 1));
      continue;
    }

    switch (dt.id()) {
      case cudf::type_id::BOOL8:
      case cudf::type_id::INT32:
      case cudf::type_id::UINT32:
      case cudf::type_id::INT64:
      case cudf::type_id::UINT64:
      case cudf::type_id::FLOAT32:
      case cudf::type_id::FLOAT64: {
        RepeatedMsgChildLocationProvider loc_provider{d_msg_row_offsets.data(),
                                                      0,
                                                      d_msg_locs.data(),
                                                      d_child_locs.data(),
                                                      ci,
                                                      num_child_fields};
        struct_children.push_back(
          extract_typed_column(dt,
                               enc,
                               message_data,
                               loc_provider,
                               total_count,
                               blocks,
                               threads,
                               has_def,
                               has_def ? default_ints[child_schema_idx] : 0,
                               has_def ? default_floats[child_schema_idx] : 0.0,
                               has_def ? default_bools[child_schema_idx] : false,
                               default_strings[child_schema_idx],
                               child_schema_idx,
                               enum_valid_values,
                               enum_names,
                               d_row_has_invalid_enum,
                               d_error,
                               stream,
                               mr));
        break;
      }
      case cudf::type_id::STRING: {
        struct_children.push_back(build_repeated_msg_child_varlen_column(message_data,
                                                                         d_msg_row_offsets,
                                                                         d_msg_locs,
                                                                         d_child_locs,
                                                                         ci,
                                                                         num_child_fields,
                                                                         total_count,
                                                                         d_error,
                                                                         false,
                                                                         stream,
                                                                         mr));
        break;
      }
      case cudf::type_id::LIST: {
        struct_children.push_back(build_repeated_msg_child_varlen_column(message_data,
                                                                         d_msg_row_offsets,
                                                                         d_msg_locs,
                                                                         d_child_locs,
                                                                         ci,
                                                                         num_child_fields,
                                                                         total_count,
                                                                         d_error,
                                                                         true,
                                                                         stream,
                                                                         mr));
        break;
      }
      case cudf::type_id::STRUCT: {
        // Nested struct inside repeated message - use recursive build_nested_struct_column
        int num_schema_fields = static_cast<int>(h_device_schema.size());
        auto grandchild_indices =
          find_child_field_indices(h_device_schema, num_schema_fields, child_schema_idx);

        if (grandchild_indices.empty()) {
          struct_children.push_back(
            cudf::make_structs_column(total_count,
                                      std::vector<std::unique_ptr<cudf::column>>{},
                                      0,
                                      rmm::device_buffer{},
                                      stream,
                                      mr));
        } else {
          // Compute virtual parent locations for each occurrence's nested struct child
          rmm::device_uvector<field_location> d_nested_locs(total_count, stream, mr);
          rmm::device_uvector<cudf::size_type> d_nested_row_offsets(total_count, stream, mr);
          {
            // Convert int32_t row offsets to cudf::size_type and compute nested struct locations
            rmm::device_uvector<int32_t> d_nested_row_offsets_i32(total_count, stream, mr);
            compute_nested_struct_locations_kernel<<<blocks, threads, 0, stream.value()>>>(
              d_child_locs.data(),
              d_msg_locs.data(),
              d_msg_row_offsets.data(),
              ci,
              num_child_fields,
              d_nested_locs.data(),
              d_nested_row_offsets_i32.data(),
              total_count);
            // Add base_offset back so build_nested_struct_column can subtract it
            thrust::transform(rmm::exec_policy(stream),
                              d_nested_row_offsets_i32.data(),
                              d_nested_row_offsets_i32.end(),
                              d_nested_row_offsets.data(),
                              [base_offset] __device__(int32_t v) {
                                return static_cast<cudf::size_type>(v) + base_offset;
                              });
          }

          struct_children.push_back(build_nested_struct_column(message_data,
                                                               d_nested_row_offsets.data(),
                                                               base_offset,
                                                               d_nested_locs,
                                                               grandchild_indices,
                                                               schema,
                                                               num_schema_fields,
                                                               schema_output_types,
                                                               default_ints,
                                                               default_floats,
                                                               default_bools,
                                                               default_strings,
                                                               enum_valid_values,
                                                               enum_names,
                                                               d_row_has_invalid_enum,
                                                               d_error_top,
                                                               total_count,
                                                               stream,
                                                               mr,
                                                               0));
        }
        break;
      }
      default:
        // Unsupported child type - create null column
        struct_children.push_back(make_null_column(dt, total_count, stream, mr));
        break;
    }
  }

  // Build the struct column from child columns
  auto struct_col = cudf::make_structs_column(
    total_count, std::move(struct_children), 0, rmm::device_buffer{}, stream, mr);

  // Build the list offsets column
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    num_rows + 1,
                                                    list_offs.release(),
                                                    rmm::device_buffer{},
                                                    0);

  // Build the final LIST column
  if (input_null_count > 0) {
    auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
    return cudf::make_lists_column(num_rows,
                                   std::move(offsets_col),
                                   std::move(struct_col),
                                   input_null_count,
                                   std::move(null_mask),
                                   stream,
                                   mr);
  }

  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(struct_col), 0, rmm::device_buffer{}, stream, mr);
}

std::unique_ptr<cudf::column> build_nested_struct_column(
  uint8_t const* message_data,
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
  rmm::device_uvector<bool>& d_row_has_invalid_enum,
  rmm::device_uvector<int>& d_error,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int depth)
{
  CUDF_EXPECTS(depth < MAX_NESTED_STRUCT_DECODE_DEPTH,
               "Nested protobuf struct depth exceeds supported decode recursion limit");

  if (num_rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_children;
    for (int child_schema_idx : child_field_indices) {
      auto child_type = schema_output_types[child_schema_idx];
      std::unique_ptr<cudf::column> child_col;
      if (child_type.id() == cudf::type_id::STRUCT) {
        child_col = make_empty_struct_column_with_schema(
          schema, schema_output_types, child_schema_idx, num_fields, stream, mr);
      } else {
        child_col = make_empty_column_safe(child_type, stream, mr);
      }
      if (schema[child_schema_idx].is_repeated) {
        child_col = make_empty_list_column(std::move(child_col), stream, mr);
      }
      empty_children.push_back(std::move(child_col));
    }
    return cudf::make_structs_column(
      0, std::move(empty_children), 0, rmm::device_buffer{}, stream, mr);
  }

  auto const threads   = THREADS_PER_BLOCK;
  auto const blocks    = static_cast<int>((num_rows + threads - 1) / threads);
  int num_child_fields = static_cast<int>(child_field_indices.size());

  std::vector<field_descriptor> h_child_field_descs(num_child_fields);
  for (int i = 0; i < num_child_fields; i++) {
    int child_idx                             = child_field_indices[i];
    h_child_field_descs[i].field_number       = schema[child_idx].field_number;
    h_child_field_descs[i].expected_wire_type = schema[child_idx].wire_type;
  }

  rmm::device_uvector<field_descriptor> d_child_field_descs(num_child_fields, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_child_field_descs.data(),
                                h_child_field_descs.data(),
                                num_child_fields * sizeof(field_descriptor),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  rmm::device_uvector<field_location> d_child_locations(
    static_cast<size_t>(num_rows) * num_child_fields, stream, mr);
  scan_nested_message_fields_kernel<<<blocks, threads, 0, stream.value()>>>(
    message_data,
    list_offsets,
    base_offset,
    d_parent_locs.data(),
    num_rows,
    d_child_field_descs.data(),
    num_child_fields,
    d_child_locations.data(),
    d_error.data());

  std::vector<std::unique_ptr<cudf::column>> struct_children;
  for (int ci = 0; ci < num_child_fields; ci++) {
    int child_schema_idx = child_field_indices[ci];
    auto const dt        = schema_output_types[child_schema_idx];
    auto const enc       = schema[child_schema_idx].encoding;
    bool has_def         = schema[child_schema_idx].has_default_value;
    bool is_repeated     = schema[child_schema_idx].is_repeated;

    if (is_repeated) {
      struct_children.push_back(build_repeated_child_list_column(message_data,
                                                                 list_offsets,
                                                                 base_offset,
                                                                 d_parent_locs.data(),
                                                                 num_rows,
                                                                 child_schema_idx,
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
                                                                 stream,
                                                                 mr,
                                                                 depth));
      continue;
    }

    switch (dt.id()) {
      case cudf::type_id::BOOL8:
      case cudf::type_id::INT32:
      case cudf::type_id::UINT32:
      case cudf::type_id::INT64:
      case cudf::type_id::UINT64:
      case cudf::type_id::FLOAT32:
      case cudf::type_id::FLOAT64: {
        NestedLocationProvider loc_provider{list_offsets,
                                            base_offset,
                                            d_parent_locs.data(),
                                            d_child_locations.data(),
                                            ci,
                                            num_child_fields};
        struct_children.push_back(
          extract_typed_column(dt,
                               enc,
                               message_data,
                               loc_provider,
                               num_rows,
                               blocks,
                               threads,
                               has_def,
                               has_def ? default_ints[child_schema_idx] : 0,
                               has_def ? default_floats[child_schema_idx] : 0.0,
                               has_def ? default_bools[child_schema_idx] : false,
                               default_strings[child_schema_idx],
                               child_schema_idx,
                               enum_valid_values,
                               enum_names,
                               d_row_has_invalid_enum,
                               d_error,
                               stream,
                               mr));
        break;
      }
      case cudf::type_id::STRING: {
        if (enc == spark_rapids_jni::ENC_ENUM_STRING) {
          rmm::device_uvector<int32_t> out(num_rows, stream, mr);
          rmm::device_uvector<bool> valid((num_rows > 0 ? num_rows : 1), stream, mr);
          int64_t def_int = has_def ? default_ints[child_schema_idx] : 0;
          NestedLocationProvider loc_provider{list_offsets,
                                              base_offset,
                                              d_parent_locs.data(),
                                              d_child_locations.data(),
                                              ci,
                                              num_child_fields};
          extract_varint_kernel<int32_t, false, NestedLocationProvider>
            <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                     loc_provider,
                                                     num_rows,
                                                     out.data(),
                                                     valid.data(),
                                                     d_error.data(),
                                                     has_def,
                                                     def_int);

          if (child_schema_idx < static_cast<int>(enum_valid_values.size()) &&
              child_schema_idx < static_cast<int>(enum_names.size())) {
            auto const& valid_enums     = enum_valid_values[child_schema_idx];
            auto const& enum_name_bytes = enum_names[child_schema_idx];
            if (!valid_enums.empty() && valid_enums.size() == enum_name_bytes.size()) {
              struct_children.push_back(build_enum_string_column(out,
                                                                 valid,
                                                                 valid_enums,
                                                                 enum_name_bytes,
                                                                 d_row_has_invalid_enum,
                                                                 num_rows,
                                                                 stream,
                                                                 mr));
            } else {
              CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 1, sizeof(int), stream.value()));
              struct_children.push_back(make_null_column(dt, num_rows, stream, mr));
            }
          } else {
            CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 1, sizeof(int), stream.value()));
            struct_children.push_back(make_null_column(dt, num_rows, stream, mr));
          }
        } else {
          bool has_def_str    = has_def;
          auto const& def_str = default_strings[child_schema_idx];
          NestedLocationProvider len_provider{
            nullptr, 0, d_parent_locs.data(), d_child_locations.data(), ci, num_child_fields};
          NestedLocationProvider copy_provider{list_offsets,
                                               base_offset,
                                               d_parent_locs.data(),
                                               d_child_locations.data(),
                                               ci,
                                               num_child_fields};
          auto valid_fn = [plocs = d_parent_locs.data(),
                           flocs = d_child_locations.data(),
                           ci,
                           num_child_fields,
                           has_def_str] __device__(cudf::size_type row) {
            return (plocs[row].offset >= 0 && flocs[row * num_child_fields + ci].offset >= 0) ||
                   has_def_str;
          };
          struct_children.push_back(extract_and_build_string_or_bytes_column(false,
                                                                             message_data,
                                                                             num_rows,
                                                                             len_provider,
                                                                             copy_provider,
                                                                             valid_fn,
                                                                             has_def_str,
                                                                             def_str,
                                                                             d_error,
                                                                             stream,
                                                                             mr));
        }
        break;
      }
      case cudf::type_id::LIST: {
        // bytes (BinaryType) represented as LIST<UINT8>
        bool has_def_bytes    = has_def;
        auto const& def_bytes = default_strings[child_schema_idx];
        NestedLocationProvider len_provider{
          nullptr, 0, d_parent_locs.data(), d_child_locations.data(), ci, num_child_fields};
        NestedLocationProvider copy_provider{list_offsets,
                                             base_offset,
                                             d_parent_locs.data(),
                                             d_child_locations.data(),
                                             ci,
                                             num_child_fields};
        auto valid_fn = [plocs = d_parent_locs.data(),
                         flocs = d_child_locations.data(),
                         ci,
                         num_child_fields,
                         has_def_bytes] __device__(cudf::size_type row) {
          return (plocs[row].offset >= 0 && flocs[row * num_child_fields + ci].offset >= 0) ||
                 has_def_bytes;
        };
        struct_children.push_back(extract_and_build_string_or_bytes_column(true,
                                                                           message_data,
                                                                           num_rows,
                                                                           len_provider,
                                                                           copy_provider,
                                                                           valid_fn,
                                                                           has_def_bytes,
                                                                           def_bytes,
                                                                           d_error,
                                                                           stream,
                                                                           mr));
        break;
      }
      case cudf::type_id::STRUCT: {
        auto gc_indices = find_child_field_indices(schema, num_fields, child_schema_idx);
        if (gc_indices.empty()) {
          struct_children.push_back(make_null_column(dt, num_rows, stream, mr));
          break;
        }
        rmm::device_uvector<field_location> d_gc_parent(num_rows, stream, mr);
        compute_grandchild_parent_locations_kernel<<<blocks, threads, 0, stream.value()>>>(
          d_parent_locs.data(),
          d_child_locations.data(),
          ci,
          num_child_fields,
          d_gc_parent.data(),
          num_rows);
        struct_children.push_back(build_nested_struct_column(message_data,
                                                             list_offsets,
                                                             base_offset,
                                                             d_gc_parent,
                                                             gc_indices,
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
                                                             depth + 1));
        break;
      }
      default: struct_children.push_back(make_null_column(dt, num_rows, stream, mr)); break;
    }
  }

  rmm::device_uvector<bool> struct_valid((num_rows > 0 ? num_rows : 1), stream, mr);
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(num_rows),
    struct_valid.data(),
    [plocs = d_parent_locs.data()] __device__(auto row) { return plocs[row].offset >= 0; });
  auto [struct_mask, struct_null_count] = make_null_mask_from_valid(struct_valid, stream, mr);
  return cudf::make_structs_column(
    num_rows, std::move(struct_children), struct_null_count, std::move(struct_mask), stream, mr);
}

/**
 * Build a LIST column for a repeated child field inside a parent message.
 * Shared between build_nested_struct_column and build_repeated_struct_column.
 */
std::unique_ptr<cudf::column> build_repeated_child_list_column(
  uint8_t const* message_data,
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
  rmm::device_uvector<bool>& d_row_has_invalid_enum,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  int depth)
{
  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = static_cast<int>((num_parent_rows + threads - 1) / threads);

  auto elem_type_id = schema[child_schema_idx].output_type;
  rmm::device_uvector<repeated_field_info> d_rep_info(num_parent_rows, stream, mr);

  std::vector<int> rep_indices = {0};
  rmm::device_uvector<int> d_rep_indices(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_rep_indices.data(), rep_indices.data(), sizeof(int), cudaMemcpyHostToDevice, stream.value()));

  device_nested_field_descriptor rep_desc;
  rep_desc.field_number      = schema[child_schema_idx].field_number;
  rep_desc.wire_type         = schema[child_schema_idx].wire_type;
  rep_desc.output_type_id    = static_cast<int>(schema[child_schema_idx].output_type);
  rep_desc.is_repeated       = true;
  rep_desc.parent_idx        = -1;
  rep_desc.depth             = 0;
  rep_desc.encoding          = 0;
  rep_desc.is_required       = false;
  rep_desc.has_default_value = false;

  std::vector<device_nested_field_descriptor> h_rep_schema = {rep_desc};
  rmm::device_uvector<device_nested_field_descriptor> d_rep_schema(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_rep_schema.data(),
                                h_rep_schema.data(),
                                sizeof(device_nested_field_descriptor),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  count_repeated_in_nested_kernel<<<blocks, threads, 0, stream.value()>>>(message_data,
                                                                          row_offsets,
                                                                          base_offset,
                                                                          parent_locs,
                                                                          num_parent_rows,
                                                                          d_rep_schema.data(),
                                                                          1,
                                                                          d_rep_info.data(),
                                                                          1,
                                                                          d_rep_indices.data(),
                                                                          d_error.data());

  rmm::device_uvector<int32_t> d_rep_counts(num_parent_rows, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    d_rep_info.data(),
                    d_rep_info.end(),
                    d_rep_counts.data(),
                    [] __device__(repeated_field_info const& info) { return info.count; });
  int total_rep_count =
    thrust::reduce(rmm::exec_policy(stream), d_rep_counts.data(), d_rep_counts.end(), 0);

  if (total_rep_count == 0) {
    rmm::device_uvector<int32_t> list_offsets_vec(num_parent_rows + 1, stream, mr);
    thrust::fill(rmm::exec_policy(stream), list_offsets_vec.data(), list_offsets_vec.end(), 0);
    auto list_offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                           num_parent_rows + 1,
                                                           list_offsets_vec.release(),
                                                           rmm::device_buffer{},
                                                           0);
    std::unique_ptr<cudf::column> child_col;
    if (elem_type_id == cudf::type_id::STRUCT) {
      child_col = make_empty_struct_column_with_schema(
        schema, schema_output_types, child_schema_idx, num_fields, stream, mr);
    } else {
      child_col = make_empty_column_safe(cudf::data_type{elem_type_id}, stream, mr);
    }
    return cudf::make_lists_column(num_parent_rows,
                                   std::move(list_offsets_col),
                                   std::move(child_col),
                                   0,
                                   rmm::device_buffer{},
                                   stream,
                                   mr);
  }

  rmm::device_uvector<int32_t> list_offs(num_parent_rows + 1, stream, mr);
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_rep_counts.data(), d_rep_counts.end(), list_offs.begin(), 0);
  CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data() + num_parent_rows,
                                &total_rep_count,
                                sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  rmm::device_uvector<repeated_occurrence> d_rep_occs(total_rep_count, stream, mr);
  scan_repeated_in_nested_kernel<<<blocks, threads, 0, stream.value()>>>(message_data,
                                                                         row_offsets,
                                                                         base_offset,
                                                                         parent_locs,
                                                                         num_parent_rows,
                                                                         d_rep_schema.data(),
                                                                         1,
                                                                         list_offs.data(),
                                                                         1,
                                                                         d_rep_indices.data(),
                                                                         d_rep_occs.data(),
                                                                         d_error.data());

  std::unique_ptr<cudf::column> child_values;
  auto const rep_blocks =
    static_cast<int>((total_rep_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  NestedRepeatedLocationProvider nr_loc{
    row_offsets, base_offset, parent_locs, d_rep_occs.data()};

  if (elem_type_id == cudf::type_id::BOOL8 || elem_type_id == cudf::type_id::INT32 ||
      elem_type_id == cudf::type_id::UINT32 || elem_type_id == cudf::type_id::INT64 ||
      elem_type_id == cudf::type_id::UINT64 || elem_type_id == cudf::type_id::FLOAT32 ||
      elem_type_id == cudf::type_id::FLOAT64) {
    child_values =
      extract_typed_column(cudf::data_type{elem_type_id},
                           schema[child_schema_idx].encoding,
                           message_data,
                           nr_loc,
                           total_rep_count,
                           rep_blocks,
                           THREADS_PER_BLOCK,
                           false,
                           0,
                           0.0,
                           false,
                           std::vector<uint8_t>{},
                           child_schema_idx,
                           enum_valid_values,
                           enum_names,
                           d_row_has_invalid_enum,
                           d_error,
                           stream,
                           mr);
  } else if (elem_type_id == cudf::type_id::STRING || elem_type_id == cudf::type_id::LIST) {
    bool as_bytes = (elem_type_id == cudf::type_id::LIST);
    auto valid_fn = [] __device__(cudf::size_type) { return true; };
    std::vector<uint8_t> empty_default;
    child_values = extract_and_build_string_or_bytes_column(as_bytes,
                                                            message_data,
                                                            total_rep_count,
                                                            nr_loc,
                                                            nr_loc,
                                                            valid_fn,
                                                            false,
                                                            empty_default,
                                                            d_error,
                                                            stream,
                                                            mr);
  } else if (elem_type_id == cudf::type_id::STRUCT) {
    auto gc_indices = find_child_field_indices(schema, num_fields, child_schema_idx);
    if (gc_indices.empty()) {
      child_values = cudf::make_structs_column(total_rep_count,
                                               std::vector<std::unique_ptr<cudf::column>>{},
                                               0,
                                               rmm::device_buffer{},
                                               stream,
                                               mr);
    } else {
      rmm::device_uvector<cudf::size_type> d_virtual_row_offsets(total_rep_count, stream, mr);
      rmm::device_uvector<field_location> d_virtual_parent_locs(total_rep_count, stream, mr);
      auto const rep_blk = (total_rep_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      compute_virtual_parents_for_nested_repeated_kernel<<<rep_blk,
                                                           THREADS_PER_BLOCK,
                                                           0,
                                                           stream.value()>>>(
        d_rep_occs.data(),
        row_offsets,
        parent_locs,
        d_virtual_row_offsets.data(),
        d_virtual_parent_locs.data(),
        total_rep_count);

      child_values = build_nested_struct_column(message_data,
                                                d_virtual_row_offsets.data(),
                                                base_offset,
                                                d_virtual_parent_locs,
                                                gc_indices,
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
                                                total_rep_count,
                                                stream,
                                                mr,
                                                depth + 1);
    }
  } else {
    child_values = make_empty_column_safe(cudf::data_type{elem_type_id}, stream, mr);
  }

  auto list_offs_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                      num_parent_rows + 1,
                                                      list_offs.release(),
                                                      rmm::device_buffer{},
                                                      0);
  return cudf::make_lists_column(num_parent_rows,
                                 std::move(list_offs_col),
                                 std::move(child_values),
                                 0,
                                 rmm::device_buffer{},
                                 stream,
                                 mr);
}

}  // namespace spark_rapids_jni::protobuf_detail
