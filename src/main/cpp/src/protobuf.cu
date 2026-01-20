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

#include "protobuf.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

namespace {

// Wire type constants
constexpr int WT_VARINT = 0;
constexpr int WT_64BIT  = 1;
constexpr int WT_LEN    = 2;
constexpr int WT_32BIT  = 5;

}  // namespace

namespace {

/**
 * Structure to record field location within a message.
 * offset < 0 means field was not found.
 */
struct field_location {
  int32_t offset;  // Offset of field data within the message (-1 if not found)
  int32_t length;  // Length of field data in bytes
};

/**
 * Field descriptor passed to the scanning kernel.
 */
struct field_descriptor {
  int field_number;     // Protobuf field number
  int expected_wire_type;  // Expected wire type for this field
};

// ============================================================================
// Device helper functions
// ============================================================================

__device__ inline bool read_varint(uint8_t const* cur,
                                   uint8_t const* end,
                                   uint64_t& out,
                                   int& bytes)
{
  out   = 0;
  bytes = 0;
  int shift = 0;
  while (cur < end && bytes < 10) {
    uint8_t b = *cur++;
    // For the 10th byte (bytes == 9, shift == 63), only the lowest bit is valid
    if (bytes == 9 && (b & 0xFE) != 0) {
      return false;  // Invalid: 10th byte has more than 1 significant bit
    }
    out |= (static_cast<uint64_t>(b & 0x7Fu) << shift);
    bytes++;
    if ((b & 0x80u) == 0) { return true; }
    shift += 7;
  }
  return false;
}

__device__ inline int get_wire_type_size(int wt, uint8_t const* cur, uint8_t const* end)
{
  switch (wt) {
    case WT_VARINT: {
      // Need to scan to find the end of varint
      int count = 0;
      while (cur < end && count < 10) {
        if ((*cur++ & 0x80u) == 0) { return count + 1; }
        count++;
      }
      return -1;  // Invalid varint
    }
    case WT_64BIT:
      // Check if there's enough data for 8 bytes
      if (end - cur < 8) return -1;
      return 8;
    case WT_32BIT:
      // Check if there's enough data for 4 bytes
      if (end - cur < 4) return -1;
      return 4;
    case WT_LEN: {
      uint64_t len;
      int n;
      if (!read_varint(cur, end, len, n)) return -1;
      if (len > static_cast<uint64_t>(end - cur - n) || len > static_cast<uint64_t>(INT_MAX))
        return -1;
      return n + static_cast<int>(len);
    }
    default: return -1;
  }
}

__device__ inline bool skip_field(uint8_t const* cur,
                                  uint8_t const* end,
                                  int wt,
                                  uint8_t const*& out_cur)
{
  int size = get_wire_type_size(wt, cur, end);
  if (size < 0) return false;
  // Ensure we don't skip past the end of the buffer
  if (cur + size > end) return false;
  out_cur = cur + size;
  return true;
}

template <typename T>
__device__ inline T load_le(uint8_t const* p);

template <>
__device__ inline uint32_t load_le<uint32_t>(uint8_t const* p)
{
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

template <>
__device__ inline uint64_t load_le<uint64_t>(uint8_t const* p)
{
  uint64_t v = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v |= (static_cast<uint64_t>(p[i]) << (8 * i));
  }
  return v;
}

// ============================================================================
// Pass 1: Scan all fields kernel - records (offset, length) for each field
// ============================================================================

/**
 * Fused scanning kernel: scans each message once and records the location
 * of all requested fields.
 *
 * For "last one wins" semantics (protobuf standard for repeated scalars),
 * we continue scanning even after finding a field.
 */
__global__ void scan_all_fields_kernel(
  cudf::column_device_view const d_in,
  field_descriptor const* field_descs,  // [num_fields]
  int num_fields,
  field_location* locations,  // [num_rows * num_fields] row-major
  int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::detail::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  // Initialize all field locations to "not found"
  for (int f = 0; f < num_fields; f++) {
    locations[row * num_fields + f] = {-1, 0};
  }

  if (in.nullable() && in.is_null(row)) {
    return;  // Null input row - all fields remain "not found"
  }

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  auto start        = in.offset_at(row) - base;
  auto end          = in.offset_at(row + 1) - base;

  // Bounds check
  if (start < 0 || end < start || end > child.size()) {
    atomicExch(error_flag, 1);
    return;
  }

  uint8_t const* cur  = bytes + start;
  uint8_t const* stop = bytes + end;

  // Scan the message once, recording locations of all target fields
  while (cur < stop) {
    uint64_t key;
    int key_bytes;
    if (!read_varint(cur, stop, key, key_bytes)) {
      atomicExch(error_flag, 1);
      return;
    }
    cur += key_bytes;

    int fn = static_cast<int>(key >> 3);
    int wt = static_cast<int>(key & 0x7);

    if (fn == 0) {
      atomicExch(error_flag, 1);
      return;
    }

    // Check if this field is one we're looking for
    for (int f = 0; f < num_fields; f++) {
      if (field_descs[f].field_number == fn) {
        // Check wire type matches
        if (wt != field_descs[f].expected_wire_type) {
          atomicExch(error_flag, 1);
          return;
        }

        // Record the location (relative to message start)
        int data_offset = static_cast<int>(cur - bytes - start);

        if (wt == WT_LEN) {
          // For length-delimited, record offset after length prefix and the data length
          uint64_t len;
          int len_bytes;
          if (!read_varint(cur, stop, len, len_bytes)) {
            atomicExch(error_flag, 1);
            return;
          }
          if (len > static_cast<uint64_t>(stop - cur - len_bytes) ||
              len > static_cast<uint64_t>(INT_MAX)) {
            atomicExch(error_flag, 1);
            return;
          }
          // Record offset pointing to the actual data (after length prefix)
          locations[row * num_fields + f] = {data_offset + len_bytes,
                                             static_cast<int32_t>(len)};
        } else {
          // For fixed-size and varint fields, record offset and compute length
          int field_size = get_wire_type_size(wt, cur, stop);
          if (field_size < 0) {
            atomicExch(error_flag, 1);
            return;
          }
          locations[row * num_fields + f] = {data_offset, field_size};
        }
        // Don't break - continue to support "last one wins" semantics
      }
    }

    // Skip to next field
    uint8_t const* next;
    if (!skip_field(cur, stop, wt, next)) {
      atomicExch(error_flag, 1);
      return;
    }
    cur = next;
  }
}

// ============================================================================
// Pass 2: Extract data kernels
// ============================================================================

/**
 * Extract varint field data using pre-recorded locations.
 */
template <typename OutT, bool ZigZag = false>
__global__ void extract_varint_from_locations_kernel(
  uint8_t const* message_data,
  cudf::size_type const* offsets,  // List offsets for each row
  cudf::size_type base_offset,
  field_location const* locations,  // [num_rows * num_fields]
  int field_idx,
  int num_fields,
  OutT* out,
  bool* valid,
  int num_rows,
  int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto loc = locations[row * num_fields + field_idx];
  if (loc.offset < 0) {
    valid[row] = false;
    return;
  }

  // Calculate absolute offset in the message data
  auto row_start         = offsets[row] - base_offset;
  uint8_t const* cur     = message_data + row_start + loc.offset;
  uint8_t const* cur_end = cur + loc.length;

  uint64_t v;
  int n;
  if (!read_varint(cur, cur_end, v, n)) {
    atomicExch(error_flag, 1);
    valid[row] = false;
    return;
  }

  if constexpr (ZigZag) { v = (v >> 1) ^ (-(v & 1)); }
  out[row]   = static_cast<OutT>(v);
  valid[row] = true;
}

/**
 * Extract fixed-size field data (fixed32, fixed64, float, double).
 */
template <typename OutT, int WT>
__global__ void extract_fixed_from_locations_kernel(
  uint8_t const* message_data,
  cudf::size_type const* offsets,
  cudf::size_type base_offset,
  field_location const* locations,
  int field_idx,
  int num_fields,
  OutT* out,
  bool* valid,
  int num_rows,
  int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto loc = locations[row * num_fields + field_idx];
  if (loc.offset < 0) {
    valid[row] = false;
    return;
  }

  auto row_start     = offsets[row] - base_offset;
  uint8_t const* cur = message_data + row_start + loc.offset;

  OutT value;
  if constexpr (WT == WT_32BIT) {
    if (loc.length < 4) {
      atomicExch(error_flag, 1);
      valid[row] = false;
      return;
    }
    uint32_t raw = load_le<uint32_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  } else {
    if (loc.length < 8) {
      atomicExch(error_flag, 1);
      valid[row] = false;
      return;
    }
    uint64_t raw = load_le<uint64_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  }

  out[row]   = value;
  valid[row] = true;
}

/**
 * Kernel to copy variable-length data (string/bytes) to output buffer.
 * Uses pre-computed output offsets from prefix sum.
 */
__global__ void copy_varlen_data_kernel(
  uint8_t const* message_data,
  cudf::size_type const* input_offsets,  // List offsets for input rows
  cudf::size_type base_offset,
  field_location const* locations,
  int field_idx,
  int num_fields,
  int32_t const* output_offsets,  // Pre-computed output offsets (prefix sum)
  char* output_data,
  int num_rows)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto loc = locations[row * num_fields + field_idx];
  if (loc.offset < 0 || loc.length == 0) return;

  auto row_start       = input_offsets[row] - base_offset;
  uint8_t const* src   = message_data + row_start + loc.offset;
  char* dst            = output_data + output_offsets[row];

  // Copy data
  for (int i = 0; i < loc.length; i++) {
    dst[i] = static_cast<char>(src[i]);
  }
}

/**
 * Kernel to extract lengths from locations for prefix sum.
 */
__global__ void extract_lengths_kernel(
  field_location const* locations,
  int field_idx,
  int num_fields,
  int32_t* lengths,
  int num_rows)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto loc     = locations[row * num_fields + field_idx];
  lengths[row] = (loc.offset >= 0) ? loc.length : 0;
}

// ============================================================================
// Utility functions
// ============================================================================

inline std::pair<rmm::device_buffer, cudf::size_type> make_null_mask_from_valid(
  rmm::device_uvector<bool> const& valid,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end   = begin + valid.size();
  auto pred  = [ptr = valid.data()] __device__(cudf::size_type i) { return ptr[i]; };
  return cudf::detail::valid_if(begin, end, pred, stream, mr);
}

/**
 * Get the expected wire type for a given cudf type and encoding.
 */
int get_expected_wire_type(cudf::type_id type_id, int encoding)
{
  switch (type_id) {
    case cudf::type_id::BOOL8:
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT64:
      if (encoding == spark_rapids_jni::ENC_FIXED) {
        return (type_id == cudf::type_id::INT32 || type_id == cudf::type_id::UINT32) ? WT_32BIT
                                                                                      : WT_64BIT;
      }
      return WT_VARINT;
    case cudf::type_id::FLOAT32: return WT_32BIT;
    case cudf::type_id::FLOAT64: return WT_64BIT;
    case cudf::type_id::STRING:
    case cudf::type_id::LIST: return WT_LEN;
    default: CUDF_FAIL("Unsupported type for protobuf decoding");
  }
}

/**
 * Create an all-null column of the specified type.
 */
std::unique_ptr<cudf::column> make_null_column(
  cudf::data_type dtype,
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
    case cudf::type_id::FLOAT64: {
      auto data      = rmm::device_buffer(cudf::size_of(dtype) * num_rows, stream, mr);
      auto null_mask = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);
      return std::make_unique<cudf::column>(
        dtype, num_rows, std::move(data), std::move(null_mask), num_rows);
    }
    case cudf::type_id::STRING: {
      // Create empty strings column with all nulls
      rmm::device_uvector<cudf::strings::detail::string_index_pair> pairs(num_rows, stream, mr);
      thrust::fill(rmm::exec_policy(stream),
                   pairs.begin(),
                   pairs.end(),
                   cudf::strings::detail::string_index_pair{nullptr, 0});
      return cudf::strings::detail::make_strings_column(pairs.begin(), pairs.end(), stream, mr);
    }
    case cudf::type_id::LIST: {
      // Create LIST<INT8> with all nulls
      // Offsets: all zeros
      rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
      thrust::fill(rmm::exec_policy(stream), offsets.begin(), offsets.end(), 0);
      auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        num_rows + 1,
        offsets.release(),
        rmm::device_buffer{},
        0);

      // Empty child
      auto child_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT8},
        0,
        rmm::device_buffer{},
        rmm::device_buffer{},
        0);

      // All null mask
      auto null_mask = cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL, stream, mr);

      return cudf::make_lists_column(num_rows,
                                     std::move(offsets_col),
                                     std::move(child_col),
                                     num_rows,
                                     std::move(null_mask),
                                     stream,
                                     mr);
    }
    default: CUDF_FAIL("Unsupported type for null column creation");
  }
}

}  // namespace

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> decode_protobuf_to_struct(
  cudf::column_view const& binary_input,
  int total_num_fields,
  std::vector<int> const& decoded_field_indices,
  std::vector<int> const& field_numbers,
  std::vector<cudf::data_type> const& all_types,
  std::vector<int> const& encodings,
  bool fail_on_errors)
{
  CUDF_EXPECTS(binary_input.type().id() == cudf::type_id::LIST,
               "binary_input must be a LIST<INT8/UINT8> column");
  cudf::lists_column_view const in_list(binary_input);
  auto const child_type = in_list.child().type().id();
  CUDF_EXPECTS(child_type == cudf::type_id::INT8 || child_type == cudf::type_id::UINT8,
               "binary_input must be a LIST<INT8/UINT8> column");
  CUDF_EXPECTS(static_cast<int>(all_types.size()) == total_num_fields,
               "all_types size must equal total_num_fields");
  CUDF_EXPECTS(decoded_field_indices.size() == field_numbers.size(),
               "decoded_field_indices and field_numbers must have the same length");
  CUDF_EXPECTS(encodings.size() == field_numbers.size(),
               "encodings and field_numbers must have the same length");

  auto const stream = cudf::get_default_stream();
  auto mr           = cudf::get_current_device_resource_ref();
  auto rows         = binary_input.size();
  auto num_decoded_fields = static_cast<int>(field_numbers.size());

  // Handle zero-row case
  if (rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_children;
    empty_children.reserve(total_num_fields);
    for (auto const& dt : all_types) {
      empty_children.push_back(cudf::make_empty_column(dt));
    }
    return cudf::make_structs_column(
      0, std::move(empty_children), 0, rmm::device_buffer{}, stream, mr);
  }

  // Handle case with no fields to decode
  if (num_decoded_fields == 0) {
    std::vector<std::unique_ptr<cudf::column>> null_children;
    null_children.reserve(total_num_fields);
    for (auto const& dt : all_types) {
      null_children.push_back(make_null_column(dt, rows, stream, mr));
    }
    return cudf::make_structs_column(
      rows, std::move(null_children), 0, rmm::device_buffer{}, stream, mr);
  }

  auto d_in = cudf::column_device_view::create(binary_input, stream);

  // Prepare field descriptors for the scanning kernel
  std::vector<field_descriptor> h_field_descs(num_decoded_fields);
  for (int i = 0; i < num_decoded_fields; i++) {
    int schema_idx                      = decoded_field_indices[i];
    h_field_descs[i].field_number       = field_numbers[i];
    h_field_descs[i].expected_wire_type =
      get_expected_wire_type(all_types[schema_idx].id(), encodings[i]);
  }

  rmm::device_uvector<field_descriptor> d_field_descs(num_decoded_fields, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_field_descs.data(),
                                h_field_descs.data(),
                                num_decoded_fields * sizeof(field_descriptor),
                                cudaMemcpyHostToDevice,
                                stream.value()));

  // Allocate field locations array: [rows * num_decoded_fields]
  rmm::device_uvector<field_location> d_locations(
    static_cast<size_t>(rows) * num_decoded_fields, stream, mr);

  // Track errors
  rmm::device_uvector<int> d_error(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 0, sizeof(int), stream.value()));

  auto const threads = 256;
  auto const blocks  = static_cast<int>((rows + threads - 1) / threads);

  // =========================================================================
  // Pass 1: Scan all messages and record field locations
  // =========================================================================
  scan_all_fields_kernel<<<blocks, threads, 0, stream.value()>>>(
    *d_in, d_field_descs.data(), num_decoded_fields, d_locations.data(), d_error.data());

  // Get message data pointer and offsets for pass 2
  auto const* message_data =
    reinterpret_cast<uint8_t const*>(in_list.child().data<int8_t>());
  auto const* list_offsets = in_list.offsets().data<cudf::size_type>();
  // Get the base offset by copying from device to host
  cudf::size_type base_offset = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&base_offset,
                                list_offsets,
                                sizeof(cudf::size_type),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();

  // =========================================================================
  // Pass 2: Extract data for each field
  // =========================================================================
  std::vector<std::unique_ptr<cudf::column>> all_children(total_num_fields);
  int decoded_idx = 0;

  for (int schema_idx = 0; schema_idx < total_num_fields; schema_idx++) {
    if (decoded_idx < num_decoded_fields &&
        decoded_field_indices[decoded_idx] == schema_idx) {
      // This field needs to be decoded
      auto const dt  = all_types[schema_idx];
      auto const enc = encodings[decoded_idx];

      switch (dt.id()) {
        case cudf::type_id::BOOL8: {
          rmm::device_uvector<uint8_t> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          extract_varint_from_locations_kernel<uint8_t><<<blocks, threads, 0, stream.value()>>>(
            message_data,
            list_offsets,
            base_offset,
            d_locations.data(),
            decoded_idx,
            num_decoded_fields,
            out.data(),
            valid.data(),
            rows,
            d_error.data());
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::INT32: {
          rmm::device_uvector<int32_t> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          if (enc == spark_rapids_jni::ENC_ZIGZAG) {
            extract_varint_from_locations_kernel<int32_t, true><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
              num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          } else if (enc == spark_rapids_jni::ENC_FIXED) {
            extract_fixed_from_locations_kernel<int32_t, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
              num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          } else {
            extract_varint_from_locations_kernel<int32_t, false><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
              num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          }
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::UINT32: {
          rmm::device_uvector<uint32_t> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          if (enc == spark_rapids_jni::ENC_FIXED) {
            extract_fixed_from_locations_kernel<uint32_t, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
              num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          } else {
            extract_varint_from_locations_kernel<uint32_t><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
              num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          }
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::INT64: {
          rmm::device_uvector<int64_t> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          if (enc == spark_rapids_jni::ENC_ZIGZAG) {
            extract_varint_from_locations_kernel<int64_t, true><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
              num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          } else if (enc == spark_rapids_jni::ENC_FIXED) {
            extract_fixed_from_locations_kernel<int64_t, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
              num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          } else {
            extract_varint_from_locations_kernel<int64_t, false><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
              num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          }
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::UINT64: {
          rmm::device_uvector<uint64_t> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          if (enc == spark_rapids_jni::ENC_FIXED) {
            extract_fixed_from_locations_kernel<uint64_t, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
              num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          } else {
            extract_varint_from_locations_kernel<uint64_t><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
              num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          }
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::FLOAT32: {
          rmm::device_uvector<float> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          extract_fixed_from_locations_kernel<float, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
            message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
            num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::FLOAT64: {
          rmm::device_uvector<double> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          extract_fixed_from_locations_kernel<double, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
            message_data, list_offsets, base_offset, d_locations.data(), decoded_idx,
            num_decoded_fields, out.data(), valid.data(), rows, d_error.data());
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::STRING: {
          // Extract lengths and compute output offsets via prefix sum
          rmm::device_uvector<int32_t> lengths(rows, stream, mr);
          extract_lengths_kernel<<<blocks, threads, 0, stream.value()>>>(
            d_locations.data(), decoded_idx, num_decoded_fields, lengths.data(), rows);

          rmm::device_uvector<int32_t> output_offsets(rows + 1, stream, mr);
          thrust::exclusive_scan(
            rmm::exec_policy(stream), lengths.begin(), lengths.end(), output_offsets.begin(), 0);

          // Get total size
          int32_t total_chars = 0;
          CUDF_CUDA_TRY(cudaMemcpyAsync(&total_chars,
                                        output_offsets.data() + rows - 1,
                                        sizeof(int32_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.value()));
          int32_t last_len = 0;
          CUDF_CUDA_TRY(cudaMemcpyAsync(&last_len,
                                        lengths.data() + rows - 1,
                                        sizeof(int32_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.value()));
          stream.synchronize();
          total_chars += last_len;

          // Set the final offset
          CUDF_CUDA_TRY(cudaMemcpyAsync(output_offsets.data() + rows,
                                        &total_chars,
                                        sizeof(int32_t),
                                        cudaMemcpyHostToDevice,
                                        stream.value()));

          // Allocate and copy character data
          rmm::device_uvector<char> chars(total_chars, stream, mr);
          if (total_chars > 0) {
            copy_varlen_data_kernel<<<blocks, threads, 0, stream.value()>>>(
              message_data,
              list_offsets,
              base_offset,
              d_locations.data(),
              decoded_idx,
              num_decoded_fields,
              output_offsets.data(),
              chars.data(),
              rows);
          }

          // Create validity mask (field found = valid)
          rmm::device_uvector<bool> valid(rows, stream, mr);
          thrust::transform(
            rmm::exec_policy(stream),
            thrust::make_counting_iterator<cudf::size_type>(0),
            thrust::make_counting_iterator<cudf::size_type>(rows),
            valid.begin(),
            [locs = d_locations.data(), decoded_idx, num_decoded_fields] __device__(auto row) {
              return locs[row * num_decoded_fields + decoded_idx].offset >= 0;
            });
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);

          // Create offsets column
          auto offsets_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::INT32},
            rows + 1,
            output_offsets.release(),
            rmm::device_buffer{},
            0);

          // Create strings column using offsets + chars buffer
          all_children[schema_idx] = cudf::make_strings_column(
            rows, std::move(offsets_col), chars.release(), null_count, std::move(mask));
          break;
        }

        case cudf::type_id::LIST: {
          // For protobuf bytes: create LIST<INT8> directly (optimization #2)
          // Extract lengths and compute output offsets via prefix sum
          rmm::device_uvector<int32_t> lengths(rows, stream, mr);
          extract_lengths_kernel<<<blocks, threads, 0, stream.value()>>>(
            d_locations.data(), decoded_idx, num_decoded_fields, lengths.data(), rows);

          rmm::device_uvector<int32_t> output_offsets(rows + 1, stream, mr);
          thrust::exclusive_scan(
            rmm::exec_policy(stream), lengths.begin(), lengths.end(), output_offsets.begin(), 0);

          // Get total size
          int32_t total_bytes = 0;
          CUDF_CUDA_TRY(cudaMemcpyAsync(&total_bytes,
                                        output_offsets.data() + rows - 1,
                                        sizeof(int32_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.value()));
          int32_t last_len = 0;
          CUDF_CUDA_TRY(cudaMemcpyAsync(&last_len,
                                        lengths.data() + rows - 1,
                                        sizeof(int32_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.value()));
          stream.synchronize();
          total_bytes += last_len;

          // Set the final offset
          CUDF_CUDA_TRY(cudaMemcpyAsync(output_offsets.data() + rows,
                                        &total_bytes,
                                        sizeof(int32_t),
                                        cudaMemcpyHostToDevice,
                                        stream.value()));

          // Allocate and copy byte data directly to INT8 buffer
          rmm::device_uvector<int8_t> child_data(total_bytes, stream, mr);
          if (total_bytes > 0) {
            copy_varlen_data_kernel<<<blocks, threads, 0, stream.value()>>>(
              message_data,
              list_offsets,
              base_offset,
              d_locations.data(),
              decoded_idx,
              num_decoded_fields,
              output_offsets.data(),
              reinterpret_cast<char*>(child_data.data()),
              rows);
          }

          // Create validity mask
          rmm::device_uvector<bool> valid(rows, stream, mr);
          thrust::transform(
            rmm::exec_policy(stream),
            thrust::make_counting_iterator<cudf::size_type>(0),
            thrust::make_counting_iterator<cudf::size_type>(rows),
            valid.begin(),
            [locs = d_locations.data(), decoded_idx, num_decoded_fields] __device__(auto row) {
              return locs[row * num_decoded_fields + decoded_idx].offset >= 0;
            });
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);

          // Create offsets column
          auto offsets_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::INT32},
            rows + 1,
            output_offsets.release(),
            rmm::device_buffer{},
            0);

          // Create INT8 child column directly (no intermediate strings column!)
          auto child_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::INT8},
            total_bytes,
            child_data.release(),
            rmm::device_buffer{},
            0);

          all_children[schema_idx] = cudf::make_lists_column(rows,
                                                             std::move(offsets_col),
                                                             std::move(child_col),
                                                             null_count,
                                                             std::move(mask),
                                                             stream,
                                                             mr);
          break;
        }

        default: CUDF_FAIL("Unsupported output type for protobuf decoder");
      }

      decoded_idx++;
    } else {
      // This field is not decoded - create null column
      all_children[schema_idx] = make_null_column(all_types[schema_idx], rows, stream, mr);
    }
  }

  // Check for errors
  CUDF_CUDA_TRY(cudaPeekAtLastError());

  // Check for any parse errors.
  // Note: We check errors after all kernels complete rather than between kernel launches
  // to avoid expensive synchronization overhead. If fail_on_errors is true and an error
  // occurred, all kernels will have executed but we throw an exception here.
  int h_error = 0;
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(&h_error, d_error.data(), sizeof(int), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  if (fail_on_errors) {
    CUDF_EXPECTS(h_error == 0, "Malformed protobuf message or unsupported wire type");
  }

  // Build the final struct
  rmm::device_buffer struct_mask{0, stream, mr};
  return cudf::make_structs_column(
    rows, std::move(all_children), 0, std::move(struct_mask), stream, mr);
}

}  // namespace spark_rapids_jni
