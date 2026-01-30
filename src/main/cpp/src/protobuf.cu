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

#include <map>
#include <iostream>
#include <cstdlib>
#include <type_traits>

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
  int field_number;        // Protobuf field number
  int expected_wire_type;  // Expected wire type for this field
};

/**
 * Information about repeated field occurrences in a row.
 */
struct repeated_field_info {
  int32_t count;           // Number of occurrences in this row
  int32_t total_length;    // Total bytes for all occurrences (for varlen fields)
};

/**
 * Location of a single occurrence of a repeated field.
 */
struct repeated_occurrence {
  int32_t row_idx;         // Which row this occurrence belongs to
  int32_t offset;          // Offset within the message
  int32_t length;          // Length of the field data
};

/**
 * Device-side descriptor for nested schema fields.
 */
struct device_nested_field_descriptor {
  int field_number;
  int parent_idx;
  int depth;
  int wire_type;
  int output_type_id;
  int encoding;
  bool is_repeated;
  bool is_required;
  bool has_default_value;
};

// ============================================================================
// Device helper functions
// ============================================================================

__device__ inline bool read_varint(uint8_t const* cur,
                                   uint8_t const* end,
                                   uint64_t& out,
                                   int& bytes)
{
  out       = 0;
  bytes     = 0;
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

/**
 * Get the data offset and length for a field at current position.
 * Returns true on success, false on error.
 */
__device__ inline bool get_field_data_location(uint8_t const* cur,
                                               uint8_t const* end,
                                               int wt,
                                               int32_t& data_offset,
                                               int32_t& data_length)
{
  if (wt == WT_LEN) {
    // For length-delimited, read the length prefix
    uint64_t len;
    int len_bytes;
    if (!read_varint(cur, end, len, len_bytes)) return false;
    if (len > static_cast<uint64_t>(end - cur - len_bytes) ||
        len > static_cast<uint64_t>(INT_MAX)) {
      return false;
    }
    data_offset = len_bytes;  // offset past the length prefix
    data_length = static_cast<int32_t>(len);
  } else {
    // For fixed-size and varint fields
    int field_size = get_wire_type_size(wt, cur, end);
    if (field_size < 0) return false;
    data_offset = 0;
    data_length = field_size;
  }
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
  field_location* locations,            // [num_rows * num_fields] row-major
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
          locations[row * num_fields + f] = {data_offset + len_bytes, static_cast<int32_t>(len)};
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
// Pass 1b: Count repeated fields kernel
// ============================================================================

/**
 * Count occurrences of repeated fields in each row.
 * Also records locations of nested message fields for hierarchical processing.
 */
__global__ void count_repeated_fields_kernel(
  cudf::column_device_view const d_in,
  device_nested_field_descriptor const* schema,
  int num_fields,
  int depth_level,                    // Which depth level we're processing
  repeated_field_info* repeated_info, // [num_rows * num_repeated_fields_at_this_depth]
  int num_repeated_fields,            // Number of repeated fields at this depth
  int const* repeated_field_indices,  // Indices into schema for repeated fields at this depth
  field_location* nested_locations,   // Locations of nested messages for next depth [num_rows * num_nested]
  int num_nested_fields,              // Number of nested message fields at this depth
  int const* nested_field_indices,    // Indices into schema for nested message fields
  int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::detail::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  // Initialize repeated counts to 0
  for (int f = 0; f < num_repeated_fields; f++) {
    repeated_info[row * num_repeated_fields + f] = {0, 0};
  }

  // Initialize nested locations to not found
  for (int f = 0; f < num_nested_fields; f++) {
    nested_locations[row * num_nested_fields + f] = {-1, 0};
  }

  if (in.nullable() && in.is_null(row)) {
    return;
  }

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  auto start        = in.offset_at(row) - base;
  auto end          = in.offset_at(row + 1) - base;

  if (start < 0 || end < start || end > child.size()) {
    atomicExch(error_flag, 1);
    return;
  }

  uint8_t const* cur  = bytes + start;
  uint8_t const* stop = bytes + end;

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

    // Check repeated fields at this depth
    for (int i = 0; i < num_repeated_fields; i++) {
      int schema_idx = repeated_field_indices[i];
      if (schema[schema_idx].field_number == fn && schema[schema_idx].depth == depth_level) {
        int expected_wt = schema[schema_idx].wire_type;
        
        // Handle both packed and unpacked encoding for repeated fields
        // Packed encoding uses wire type LEN (2) even for scalar types
        bool is_packed = (wt == WT_LEN && expected_wt != WT_LEN);
        
        if (!is_packed && wt != expected_wt) {
          atomicExch(error_flag, 1);
          return;
        }
        
        if (is_packed) {
          // Packed encoding: read length, then count elements inside
          uint64_t packed_len;
          int len_bytes;
          if (!read_varint(cur, stop, packed_len, len_bytes)) {
            atomicExch(error_flag, 1);
            return;
          }
          
          // Count elements based on type
          uint8_t const* packed_start = cur + len_bytes;
          uint8_t const* packed_end = packed_start + packed_len;
          if (packed_end > stop) {
            atomicExch(error_flag, 1);
            return;
          }
          
          int count = 0;
          if (expected_wt == WT_VARINT) {
            // Count varints in the packed data
            uint8_t const* p = packed_start;
            while (p < packed_end) {
              uint64_t dummy;
              int vbytes;
              if (!read_varint(p, packed_end, dummy, vbytes)) {
                atomicExch(error_flag, 1);
                return;
              }
              p += vbytes;
              count++;
            }
          } else if (expected_wt == WT_32BIT) {
            count = static_cast<int>(packed_len) / 4;
          } else if (expected_wt == WT_64BIT) {
            count = static_cast<int>(packed_len) / 8;
          }
          
          repeated_info[row * num_repeated_fields + i].count += count;
          repeated_info[row * num_repeated_fields + i].total_length += static_cast<int32_t>(packed_len);
        } else {
          // Non-packed encoding: single element
          int32_t data_offset, data_length;
          if (!get_field_data_location(cur, stop, wt, data_offset, data_length)) {
            atomicExch(error_flag, 1);
            return;
          }
          
          repeated_info[row * num_repeated_fields + i].count++;
          repeated_info[row * num_repeated_fields + i].total_length += data_length;
        }
      }
    }

    // Check nested message fields at this depth (last one wins for non-repeated)
    for (int i = 0; i < num_nested_fields; i++) {
      int schema_idx = nested_field_indices[i];
      if (schema[schema_idx].field_number == fn && schema[schema_idx].depth == depth_level) {
        if (wt != WT_LEN) {
          atomicExch(error_flag, 1);
          return;
        }
        
        uint64_t len;
        int len_bytes;
        if (!read_varint(cur, stop, len, len_bytes)) {
          atomicExch(error_flag, 1);
          return;
        }
        
        int32_t msg_offset = static_cast<int32_t>(cur - bytes - start) + len_bytes;
        nested_locations[row * num_nested_fields + i] = {msg_offset, static_cast<int32_t>(len)};
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

/**
 * Scan and record all occurrences of repeated fields.
 * Called after count_repeated_fields_kernel to fill in actual locations.
 */
__global__ void scan_repeated_field_occurrences_kernel(
  cudf::column_device_view const d_in,
  device_nested_field_descriptor const* schema,
  int schema_idx,                     // Which field in schema we're scanning
  int depth_level,
  int32_t const* output_offsets,      // Pre-computed offsets from prefix sum [num_rows + 1]
  repeated_occurrence* occurrences,   // Output: all occurrences [total_count]
  int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::detail::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  if (in.nullable() && in.is_null(row)) {
    return;
  }

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  auto start        = in.offset_at(row) - base;
  auto end          = in.offset_at(row + 1) - base;

  if (start < 0 || end < start || end > child.size()) {
    atomicExch(error_flag, 1);
    return;
  }

  uint8_t const* cur  = bytes + start;
  uint8_t const* stop = bytes + end;

  int target_fn = schema[schema_idx].field_number;
  int target_wt = schema[schema_idx].wire_type;
  int write_idx = output_offsets[row];

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

    if (fn == target_fn) {
      // Check for packed encoding: wire type LEN but expected non-LEN
      bool is_packed = (wt == WT_LEN && target_wt != WT_LEN);
      
      if (is_packed) {
        // Packed encoding: multiple elements in a length-delimited blob
        uint64_t packed_len;
        int len_bytes;
        if (!read_varint(cur, stop, packed_len, len_bytes)) {
          atomicExch(error_flag, 1);
          return;
        }
        
        uint8_t const* packed_start = cur + len_bytes;
        uint8_t const* packed_end = packed_start + packed_len;
        if (packed_end > stop) {
          atomicExch(error_flag, 1);
          return;
        }
        
        // Record each element in the packed blob
        if (target_wt == WT_VARINT) {
          // Varints: parse each one
          uint8_t const* p = packed_start;
          while (p < packed_end) {
            int32_t elem_offset = static_cast<int32_t>(p - bytes - start);
            uint64_t dummy;
            int vbytes;
            if (!read_varint(p, packed_end, dummy, vbytes)) {
              atomicExch(error_flag, 1);
              return;
            }
            occurrences[write_idx] = {static_cast<int32_t>(row), elem_offset, vbytes};
            write_idx++;
            p += vbytes;
          }
        } else if (target_wt == WT_32BIT) {
          // Fixed 32-bit: each element is 4 bytes
          uint8_t const* p = packed_start;
          while (p + 4 <= packed_end) {
            int32_t elem_offset = static_cast<int32_t>(p - bytes - start);
            occurrences[write_idx] = {static_cast<int32_t>(row), elem_offset, 4};
            write_idx++;
            p += 4;
          }
        } else if (target_wt == WT_64BIT) {
          // Fixed 64-bit: each element is 8 bytes
          uint8_t const* p = packed_start;
          while (p + 8 <= packed_end) {
            int32_t elem_offset = static_cast<int32_t>(p - bytes - start);
            occurrences[write_idx] = {static_cast<int32_t>(row), elem_offset, 8};
            write_idx++;
            p += 8;
          }
        }
      } else if (wt == target_wt) {
        // Non-packed encoding: single element
        int32_t data_offset, data_length;
        if (!get_field_data_location(cur, stop, wt, data_offset, data_length)) {
          atomicExch(error_flag, 1);
          return;
        }
        
        int32_t abs_offset = static_cast<int32_t>(cur - bytes - start) + data_offset;
        occurrences[write_idx] = {static_cast<int32_t>(row), abs_offset, data_length};
        write_idx++;
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
 * Supports default values for missing fields.
 */
template <typename OutT, bool ZigZag = false>
__global__ void extract_varint_from_locations_kernel(
  uint8_t const* message_data,
  cudf::size_type const* offsets,   // List offsets for each row
  cudf::size_type base_offset,
  field_location const* locations,  // [num_rows * num_fields]
  int field_idx,
  int num_fields,
  OutT* out,
  bool* valid,
  int num_rows,
  int* error_flag,
  bool has_default      = false,
  int64_t default_value = 0)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto loc = locations[row * num_fields + field_idx];
  if (loc.offset < 0) {
    // Field not found - use default value if available
    if (has_default) {
      out[row]   = static_cast<OutT>(default_value);
      valid[row] = true;
    } else {
      valid[row] = false;
    }
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
 * Supports default values for missing fields.
 */
template <typename OutT, int WT>
__global__ void extract_fixed_from_locations_kernel(uint8_t const* message_data,
                                                    cudf::size_type const* offsets,
                                                    cudf::size_type base_offset,
                                                    field_location const* locations,
                                                    int field_idx,
                                                    int num_fields,
                                                    OutT* out,
                                                    bool* valid,
                                                    int num_rows,
                                                    int* error_flag,
                                                    bool has_default   = false,
                                                    OutT default_value = OutT{})
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto loc = locations[row * num_fields + field_idx];
  if (loc.offset < 0) {
    // Field not found - use default value if available
    if (has_default) {
      out[row]   = default_value;
      valid[row] = true;
    } else {
      valid[row] = false;
    }
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
 * Supports default values for missing fields.
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
  int num_rows,
  bool has_default            = false,
  uint8_t const* default_data = nullptr,
  int32_t default_length      = 0)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto loc  = locations[row * num_fields + field_idx];
  char* dst = output_data + output_offsets[row];

  if (loc.offset < 0) {
    // Field not found - use default if available
    if (has_default && default_length > 0) {
      for (int i = 0; i < default_length; i++) {
        dst[i] = static_cast<char>(default_data[i]);
      }
    }
    return;
  }

  if (loc.length == 0) return;

  auto row_start     = input_offsets[row] - base_offset;
  uint8_t const* src = message_data + row_start + loc.offset;

  // Copy data
  for (int i = 0; i < loc.length; i++) {
    dst[i] = static_cast<char>(src[i]);
  }
}

/**
 * Kernel to extract lengths from locations for prefix sum.
 * Supports default values for missing fields.
 */
__global__ void extract_lengths_kernel(field_location const* locations,
                                       int field_idx,
                                       int num_fields,
                                       int32_t* lengths,
                                       int num_rows,
                                       bool has_default       = false,
                                       int32_t default_length = 0)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto loc = locations[row * num_fields + field_idx];
  if (loc.offset >= 0) {
    lengths[row] = loc.length;
  } else if (has_default) {
    lengths[row] = default_length;
  } else {
    lengths[row] = 0;
  }
}

// ============================================================================
// Repeated field extraction kernels
// ============================================================================

/**
 * Extract repeated varint values using pre-recorded occurrences.
 */
template <typename OutT, bool ZigZag = false>
__global__ void extract_repeated_varint_kernel(
  uint8_t const* message_data,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  repeated_occurrence const* occurrences,
  int total_occurrences,
  OutT* out,
  int* error_flag)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_occurrences) return;

  auto const& occ = occurrences[idx];
  auto row_start = row_offsets[occ.row_idx] - base_offset;
  uint8_t const* cur = message_data + row_start + occ.offset;
  uint8_t const* cur_end = cur + occ.length;

  uint64_t v;
  int n;
  if (!read_varint(cur, cur_end, v, n)) {
    atomicExch(error_flag, 1);
    out[idx] = OutT{};
    return;
  }

  if constexpr (ZigZag) { v = (v >> 1) ^ (-(v & 1)); }
  out[idx] = static_cast<OutT>(v);
}

/**
 * Extract repeated fixed-size values using pre-recorded occurrences.
 */
template <typename OutT, int WT>
__global__ void extract_repeated_fixed_kernel(
  uint8_t const* message_data,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  repeated_occurrence const* occurrences,
  int total_occurrences,
  OutT* out,
  int* error_flag)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_occurrences) return;

  auto const& occ = occurrences[idx];
  auto row_start = row_offsets[occ.row_idx] - base_offset;
  uint8_t const* cur = message_data + row_start + occ.offset;

  OutT value;
  if constexpr (WT == WT_32BIT) {
    if (occ.length < 4) {
      atomicExch(error_flag, 1);
      out[idx] = OutT{};
      return;
    }
    uint32_t raw = load_le<uint32_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  } else {
    if (occ.length < 8) {
      atomicExch(error_flag, 1);
      out[idx] = OutT{};
      return;
    }
    uint64_t raw = load_le<uint64_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  }

  out[idx] = value;
}

/**
 * Copy repeated variable-length data (string/bytes) using pre-recorded occurrences.
 */
__global__ void copy_repeated_varlen_data_kernel(
  uint8_t const* message_data,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  repeated_occurrence const* occurrences,
  int total_occurrences,
  int32_t const* output_offsets,  // Pre-computed output offsets for strings
  char* output_data)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_occurrences) return;

  auto const& occ = occurrences[idx];
  if (occ.length == 0) return;

  auto row_start = row_offsets[occ.row_idx] - base_offset;
  uint8_t const* src = message_data + row_start + occ.offset;
  char* dst = output_data + output_offsets[idx];

  for (int i = 0; i < occ.length; i++) {
    dst[i] = static_cast<char>(src[i]);
  }
}

/**
 * Extract lengths from repeated occurrences for prefix sum.
 */
__global__ void extract_repeated_lengths_kernel(
  repeated_occurrence const* occurrences,
  int total_occurrences,
  int32_t* lengths)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_occurrences) return;

  lengths[idx] = occurrences[idx].length;
}

// ============================================================================
// Nested message scanning kernels
// ============================================================================

/**
 * Scan nested message fields.
 * Each row represents a nested message at a specific parent location.
 * This kernel finds fields within the nested message bytes.
 */
__global__ void scan_nested_message_fields_kernel(
  uint8_t const* message_data,
  cudf::size_type const* parent_row_offsets,
  cudf::size_type parent_base_offset,
  field_location const* parent_locations,
  int num_parent_rows,
  field_descriptor const* field_descs,
  int num_fields,
  field_location* output_locations,
  int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_parent_rows) return;

  for (int f = 0; f < num_fields; f++) {
    output_locations[row * num_fields + f] = {-1, 0};
  }

  auto const& parent_loc = parent_locations[row];
  if (parent_loc.offset < 0) {
    return;
  }

  auto parent_row_start = parent_row_offsets[row] - parent_base_offset;
  uint8_t const* nested_start = message_data + parent_row_start + parent_loc.offset;
  uint8_t const* nested_end = nested_start + parent_loc.length;

  uint8_t const* cur = nested_start;

  while (cur < nested_end) {
    uint64_t key;
    int key_bytes;
    if (!read_varint(cur, nested_end, key, key_bytes)) {
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

    for (int f = 0; f < num_fields; f++) {
      if (field_descs[f].field_number == fn) {
        if (wt != field_descs[f].expected_wire_type) {
          atomicExch(error_flag, 1);
          return;
        }

        int data_offset = static_cast<int>(cur - nested_start);

        if (wt == WT_LEN) {
          uint64_t len;
          int len_bytes;
          if (!read_varint(cur, nested_end, len, len_bytes)) {
            atomicExch(error_flag, 1);
            return;
          }
          if (len > static_cast<uint64_t>(nested_end - cur - len_bytes) ||
              len > static_cast<uint64_t>(INT_MAX)) {
            atomicExch(error_flag, 1);
            return;
          }
          output_locations[row * num_fields + f] = {data_offset + len_bytes, static_cast<int32_t>(len)};
        } else {
          int field_size = get_wire_type_size(wt, cur, nested_end);
          if (field_size < 0) {
            atomicExch(error_flag, 1);
            return;
          }
          output_locations[row * num_fields + f] = {data_offset, field_size};
        }
      }
    }

    uint8_t const* next;
    if (!skip_field(cur, nested_end, wt, next)) {
      atomicExch(error_flag, 1);
      return;
    }
    cur = next;
  }
}

// Utility function: make_null_mask_from_valid
// (Moved here to be available for repeated message child extraction)
template <typename T>
inline std::pair<rmm::device_buffer, cudf::size_type> make_null_mask_from_valid(
  rmm::device_uvector<T> const& valid,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto begin = thrust::make_counting_iterator<cudf::size_type>(0);
  auto end   = begin + valid.size();
  auto pred  = [ptr = valid.data()] __device__(cudf::size_type i) { return static_cast<bool>(ptr[i]); };
  return cudf::detail::valid_if(begin, end, pred, stream, mr);
}

/**
 * Scan for child fields within repeated message occurrences.
 * Each occurrence is a protobuf message, and we need to find child field locations within it.
 */
__global__ void scan_repeated_message_children_kernel(
  uint8_t const* message_data,
  int32_t const* msg_row_offsets,    // Row offset for each occurrence
  field_location const* msg_locs,    // Location of each message occurrence (offset within row, length)
  int num_occurrences,
  field_descriptor const* child_descs,
  int num_child_fields,
  field_location* child_locs,        // Output: [num_occurrences * num_child_fields]
  int* error_flag)
{
  auto occ_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (occ_idx >= num_occurrences) return;

  // Initialize child locations to not found
  for (int f = 0; f < num_child_fields; f++) {
    child_locs[occ_idx * num_child_fields + f] = {-1, 0};
  }

  auto const& msg_loc = msg_locs[occ_idx];
  if (msg_loc.offset < 0) return;

  // Calculate absolute position of this message in the data
  int32_t row_offset = msg_row_offsets[occ_idx];
  uint8_t const* msg_start = message_data + row_offset + msg_loc.offset;
  uint8_t const* msg_end = msg_start + msg_loc.length;

  uint8_t const* cur = msg_start;

  while (cur < msg_end) {
    uint64_t key;
    int key_bytes;
    if (!read_varint(cur, msg_end, key, key_bytes)) {
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

    // Check against child field descriptors
    for (int f = 0; f < num_child_fields; f++) {
      if (child_descs[f].field_number == fn) {
        if (wt != child_descs[f].expected_wire_type) {
          // Wire type mismatch - could be OK for some cases (e.g., packed vs unpacked)
          // For now, just continue
          continue;
        }

        int data_offset = static_cast<int>(cur - msg_start);

        if (wt == WT_LEN) {
          uint64_t len;
          int len_bytes;
          if (!read_varint(cur, msg_end, len, len_bytes)) {
            atomicExch(error_flag, 1);
            return;
          }
          // Store offset (after length prefix) and length
          child_locs[occ_idx * num_child_fields + f] = {data_offset + len_bytes, static_cast<int32_t>(len)};
        } else {
          // For varint/fixed types, store offset and estimated length
          int32_t data_length = 0;
          if (wt == WT_VARINT) {
            uint64_t dummy;
            int vbytes;
            if (read_varint(cur, msg_end, dummy, vbytes)) {
              data_length = vbytes;
            }
          } else if (wt == WT_32BIT) {
            data_length = 4;
          } else if (wt == WT_64BIT) {
            data_length = 8;
          }
          child_locs[occ_idx * num_child_fields + f] = {data_offset, data_length};
        }
        // Don't break - last occurrence wins (protobuf semantics)
      }
    }

    // Skip to next field
    uint8_t const* next;
    if (!skip_field(cur, msg_end, wt, next)) {
      atomicExch(error_flag, 1);
      return;
    }
    cur = next;
  }
}

/**
 * Count repeated field occurrences within nested messages.
 * Similar to count_repeated_fields_kernel but operates on nested message locations.
 */
__global__ void count_repeated_in_nested_kernel(
  uint8_t const* message_data,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  field_location const* parent_locs,
  int num_rows,
  device_nested_field_descriptor const* schema,
  int num_fields,
  repeated_field_info* repeated_info,
  int num_repeated,
  int const* repeated_indices,
  int* error_flag)
{
  auto row_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;

  // Initialize counts
  for (int ri = 0; ri < num_repeated; ri++) {
    repeated_info[row_idx * num_repeated + ri] = {0, 0};
  }

  auto const& parent_loc = parent_locs[row_idx];
  if (parent_loc.offset < 0) return;

  cudf::size_type row_off;
  row_off = row_offsets[row_idx] - base_offset;
  
  uint8_t const* msg_start = message_data + row_off + parent_loc.offset;
  uint8_t const* msg_end = msg_start + parent_loc.length;
  uint8_t const* cur = msg_start;

  while (cur < msg_end) {
    uint64_t key;
    int key_bytes;
    if (!read_varint(cur, msg_end, key, key_bytes)) {
      atomicExch(error_flag, 1);
      return;
    }
    cur += key_bytes;

    int fn = static_cast<int>(key >> 3);
    int wt = static_cast<int>(key & 0x7);

    // Check if this is one of our repeated fields
    for (int ri = 0; ri < num_repeated; ri++) {
      int schema_idx = repeated_indices[ri];
      if (schema[schema_idx].field_number == fn && schema[schema_idx].is_repeated) {
        int data_len = 0;
        if (wt == WT_LEN) {
          uint64_t len;
          int len_bytes;
          if (!read_varint(cur, msg_end, len, len_bytes)) {
            atomicExch(error_flag, 1);
            return;
          }
          data_len = static_cast<int>(len);
        }
        repeated_info[row_idx * num_repeated + ri].count++;
        repeated_info[row_idx * num_repeated + ri].total_length += data_len;
      }
    }

    uint8_t const* next;
    if (!skip_field(cur, msg_end, wt, next)) {
      atomicExch(error_flag, 1);
      return;
    }
    cur = next;
  }
}

/**
 * Scan for repeated field occurrences within nested messages.
 */
__global__ void scan_repeated_in_nested_kernel(
  uint8_t const* message_data,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  field_location const* parent_locs,
  int num_rows,
  device_nested_field_descriptor const* schema,
  int num_fields,
  repeated_field_info const* repeated_info,
  int num_repeated,
  int const* repeated_indices,
  repeated_occurrence* occurrences,
  int* error_flag)
{
  auto row_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row_idx >= num_rows) return;

  auto const& parent_loc = parent_locs[row_idx];
  if (parent_loc.offset < 0) return;

  // Calculate output offset for this row
  int occ_offset = 0;
  for (int r = 0; r < row_idx; r++) {
    occ_offset += repeated_info[r * num_repeated].count;
  }

  cudf::size_type row_off = row_offsets[row_idx] - base_offset;
  
  uint8_t const* msg_start = message_data + row_off + parent_loc.offset;
  uint8_t const* msg_end = msg_start + parent_loc.length;
  uint8_t const* cur = msg_start;

  int occ_idx = 0;

  while (cur < msg_end) {
    uint64_t key;
    int key_bytes;
    if (!read_varint(cur, msg_end, key, key_bytes)) {
      atomicExch(error_flag, 1);
      return;
    }
    cur += key_bytes;

    int fn = static_cast<int>(key >> 3);
    int wt = static_cast<int>(key & 0x7);

    // Check if this is our repeated field (assuming single repeated field for simplicity)
    int schema_idx = repeated_indices[0];
    if (schema[schema_idx].field_number == fn && schema[schema_idx].is_repeated) {
      int32_t data_offset = static_cast<int32_t>(cur - msg_start);
      int32_t data_len = 0;
      
      if (wt == WT_LEN) {
        uint64_t len;
        int len_bytes;
        if (!read_varint(cur, msg_end, len, len_bytes)) {
          atomicExch(error_flag, 1);
          return;
        }
        data_offset += len_bytes;
        data_len = static_cast<int32_t>(len);
      } else if (wt == WT_VARINT) {
        uint64_t dummy;
        int vbytes;
        if (read_varint(cur, msg_end, dummy, vbytes)) {
          data_len = vbytes;
        }
      } else if (wt == WT_32BIT) {
        data_len = 4;
      } else if (wt == WT_64BIT) {
        data_len = 8;
      }
      
      occurrences[occ_offset + occ_idx] = {row_idx, data_offset, data_len};
      occ_idx++;
    }

    uint8_t const* next;
    if (!skip_field(cur, msg_end, wt, next)) {
      atomicExch(error_flag, 1);
      return;
    }
    cur = next;
  }
}

/**
 * Extract varint values from repeated field occurrences within nested messages.
 */
template <typename OutT, bool ZigZag = false>
__global__ void extract_repeated_in_nested_varint_kernel(
  uint8_t const* message_data,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  field_location const* parent_locs,
  repeated_occurrence const* occurrences,
  int total_count,
  OutT* out,
  int* error_flag)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_count) return;

  auto const& occ = occurrences[idx];
  auto const& parent_loc = parent_locs[occ.row_idx];
  
  cudf::size_type row_off = row_offsets[occ.row_idx] - base_offset;
  uint8_t const* data_ptr = message_data + row_off + parent_loc.offset + occ.offset;

  uint64_t val;
  int vbytes;
  if (!read_varint(data_ptr, data_ptr + 10, val, vbytes)) {
    atomicExch(error_flag, 1);
    return;
  }

  if constexpr (ZigZag) {
    val = (val >> 1) ^ (~(val & 1) + 1);
  }

  out[idx] = static_cast<OutT>(val);
}

/**
 * Extract string values from repeated field occurrences within nested messages.
 */
__global__ void extract_repeated_in_nested_string_kernel(
  uint8_t const* message_data,
  cudf::size_type const* row_offsets,
  cudf::size_type base_offset,
  field_location const* parent_locs,
  repeated_occurrence const* occurrences,
  int total_count,
  int32_t const* str_offsets,
  char* chars,
  int* error_flag)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_count) return;

  auto const& occ = occurrences[idx];
  auto const& parent_loc = parent_locs[occ.row_idx];
  
  cudf::size_type row_off = row_offsets[occ.row_idx] - base_offset;
  uint8_t const* data_ptr = message_data + row_off + parent_loc.offset + occ.offset;
  
  int32_t out_offset = str_offsets[idx];
  for (int32_t i = 0; i < occ.length; i++) {
    chars[out_offset + i] = static_cast<char>(data_ptr[i]);
  }
}

/**
 * Extract varint child fields from repeated message occurrences.
 */
template <typename OutT, bool ZigZag = false>
__global__ void extract_repeated_msg_child_varint_kernel(
  uint8_t const* message_data,
  int32_t const* msg_row_offsets,
  field_location const* msg_locs,
  field_location const* child_locs,
  int child_idx,
  int num_child_fields,
  OutT* out,
  bool* valid,
  int num_occurrences,
  int* error_flag,
  bool has_default = false,
  int64_t default_value = 0)
{
  auto occ_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (occ_idx >= num_occurrences) return;

  auto const& msg_loc = msg_locs[occ_idx];
  auto const& field_loc = child_locs[occ_idx * num_child_fields + child_idx];

  if (msg_loc.offset < 0 || field_loc.offset < 0) {
    if (has_default) {
      out[occ_idx] = static_cast<OutT>(default_value);
      valid[occ_idx] = true;
    } else {
      valid[occ_idx] = false;
    }
    return;
  }

  int32_t row_offset = msg_row_offsets[occ_idx];
  uint8_t const* msg_start = message_data + row_offset + msg_loc.offset;
  uint8_t const* cur = msg_start + field_loc.offset;

  uint64_t val;
  int vbytes;
  if (!read_varint(cur, cur + 10, val, vbytes)) {
    atomicExch(error_flag, 1);
    valid[occ_idx] = false;
    return;
  }

  if constexpr (ZigZag) {
    val = (val >> 1) ^ (~(val & 1) + 1);
  }

  out[occ_idx] = static_cast<OutT>(val);
  valid[occ_idx] = true;
}

/**
 * Extract fixed-size child fields from repeated message occurrences.
 */
template <typename OutT, int WT>
__global__ void extract_repeated_msg_child_fixed_kernel(
  uint8_t const* message_data,
  int32_t const* msg_row_offsets,
  field_location const* msg_locs,
  field_location const* child_locs,
  int child_idx,
  int num_child_fields,
  OutT* out,
  bool* valid,
  int num_occurrences,
  int* error_flag,
  bool has_default = false,
  OutT default_value = OutT{})
{
  auto occ_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (occ_idx >= num_occurrences) return;

  auto const& msg_loc = msg_locs[occ_idx];
  auto const& field_loc = child_locs[occ_idx * num_child_fields + child_idx];

  if (msg_loc.offset < 0 || field_loc.offset < 0) {
    if (has_default) {
      out[occ_idx] = default_value;
      valid[occ_idx] = true;
    } else {
      valid[occ_idx] = false;
    }
    return;
  }

  int32_t row_offset = msg_row_offsets[occ_idx];
  uint8_t const* msg_start = message_data + row_offset + msg_loc.offset;
  uint8_t const* cur = msg_start + field_loc.offset;

  OutT value;
  if constexpr (WT == WT_32BIT) {
    uint32_t raw = load_le<uint32_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  } else {
    uint64_t raw = load_le<uint64_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  }

  out[occ_idx] = value;
  valid[occ_idx] = true;
}

/**
 * Helper to build string column for repeated message child fields.
 */
inline std::unique_ptr<cudf::column> build_repeated_msg_child_string_column(
  uint8_t const* message_data,
  rmm::device_uvector<int32_t> const& d_msg_row_offsets,
  rmm::device_uvector<field_location> const& d_msg_locs,
  rmm::device_uvector<field_location> const& d_child_locs,
  int child_idx,
  int num_child_fields,
  int total_count,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (total_count == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }

  // Get string lengths from child_locs
  std::vector<field_location> h_child_locs(total_count * num_child_fields);
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_child_locs.data(), d_child_locs.data(),
                                h_child_locs.size() * sizeof(field_location),
                                cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();

  std::vector<int32_t> h_lengths(total_count);
  int32_t total_chars = 0;
  for (int i = 0; i < total_count; i++) {
    auto const& loc = h_child_locs[i * num_child_fields + child_idx];
    if (loc.offset >= 0) {
      h_lengths[i] = loc.length;
      total_chars += loc.length;
    } else {
      h_lengths[i] = 0;
    }
  }

  // Build string offsets
  rmm::device_uvector<int32_t> str_offsets(total_count + 1, stream, mr);
  std::vector<int32_t> h_offsets(total_count + 1);
  h_offsets[0] = 0;
  for (int i = 0; i < total_count; i++) {
    h_offsets[i + 1] = h_offsets[i] + h_lengths[i];
  }
  CUDF_CUDA_TRY(cudaMemcpyAsync(str_offsets.data(), h_offsets.data(),
                                (total_count + 1) * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream.value()));

  // Copy string data
  rmm::device_uvector<char> chars(total_chars, stream, mr);
  if (total_chars > 0) {
    std::vector<field_location> h_msg_locs(total_count);
    std::vector<int32_t> h_row_offsets(total_count);
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_msg_locs.data(), d_msg_locs.data(),
                                  total_count * sizeof(field_location),
                                  cudaMemcpyDeviceToHost, stream.value()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_row_offsets.data(), d_msg_row_offsets.data(),
                                  total_count * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();

    // Copy each string on host (not ideal but works)
    std::vector<char> h_chars(total_chars);
    int char_idx = 0;
    for (int i = 0; i < total_count; i++) {
      auto const& field_loc = h_child_locs[i * num_child_fields + child_idx];
      if (field_loc.offset >= 0 && field_loc.length > 0) {
        int32_t row_offset = h_row_offsets[i];
        int32_t msg_offset = h_msg_locs[i].offset;
        uint8_t const* str_ptr = message_data + row_offset + msg_offset + field_loc.offset;
        // Need to copy from device - use cudaMemcpy
        CUDF_CUDA_TRY(cudaMemcpy(h_chars.data() + char_idx, str_ptr,
                                 field_loc.length, cudaMemcpyDeviceToHost));
        char_idx += field_loc.length;
      }
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(chars.data(), h_chars.data(),
                                  total_chars, cudaMemcpyHostToDevice, stream.value()));
  }

  // Build validity mask
  rmm::device_uvector<bool> valid(total_count, stream, mr);
  std::vector<uint8_t> h_valid(total_count);
  for (int i = 0; i < total_count; i++) {
    h_valid[i] = (h_child_locs[i * num_child_fields + child_idx].offset >= 0) ? 1 : 0;
  }
  rmm::device_uvector<uint8_t> d_valid_u8(total_count, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_valid_u8.data(), h_valid.data(),
                                total_count * sizeof(uint8_t),
                                cudaMemcpyHostToDevice, stream.value()));

  auto [mask, null_count] = make_null_mask_from_valid(d_valid_u8, stream, mr);

  auto str_offsets_col = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, total_count + 1, str_offsets.release(), rmm::device_buffer{}, 0);
  return cudf::make_strings_column(total_count, std::move(str_offsets_col), chars.release(), null_count, std::move(mask));
}

/**
 * Extract varint from nested message locations.
 */
template <typename OutT, bool ZigZag = false>
__global__ void extract_nested_varint_kernel(
  uint8_t const* message_data,
  cudf::size_type const* parent_row_offsets,
  cudf::size_type parent_base_offset,
  field_location const* parent_locations,
  field_location const* field_locations,
  int field_idx,
  int num_fields,
  OutT* out,
  bool* valid,
  int num_rows,
  int* error_flag,
  bool has_default = false,
  int64_t default_value = 0)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto const& parent_loc = parent_locations[row];
  auto const& field_loc = field_locations[row * num_fields + field_idx];

  if (parent_loc.offset < 0 || field_loc.offset < 0) {
    if (has_default) {
      out[row] = static_cast<OutT>(default_value);
      valid[row] = true;
    } else {
      valid[row] = false;
    }
    return;
  }

  auto parent_row_start = parent_row_offsets[row] - parent_base_offset;
  uint8_t const* cur = message_data + parent_row_start + parent_loc.offset + field_loc.offset;
  uint8_t const* cur_end = cur + field_loc.length;

  uint64_t v;
  int n;
  if (!read_varint(cur, cur_end, v, n)) {
    atomicExch(error_flag, 1);
    valid[row] = false;
    return;
  }

  if constexpr (ZigZag) { v = (v >> 1) ^ (-(v & 1)); }
  out[row] = static_cast<OutT>(v);
  valid[row] = true;
}

/**
 * Extract fixed-size from nested message locations.
 */
template <typename OutT, int WT>
__global__ void extract_nested_fixed_kernel(
  uint8_t const* message_data,
  cudf::size_type const* parent_row_offsets,
  cudf::size_type parent_base_offset,
  field_location const* parent_locations,
  field_location const* field_locations,
  int field_idx,
  int num_fields,
  OutT* out,
  bool* valid,
  int num_rows,
  int* error_flag,
  bool has_default = false,
  OutT default_value = OutT{})
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto const& parent_loc = parent_locations[row];
  auto const& field_loc = field_locations[row * num_fields + field_idx];

  if (parent_loc.offset < 0 || field_loc.offset < 0) {
    if (has_default) {
      out[row] = default_value;
      valid[row] = true;
    } else {
      valid[row] = false;
    }
    return;
  }

  auto parent_row_start = parent_row_offsets[row] - parent_base_offset;
  uint8_t const* cur = message_data + parent_row_start + parent_loc.offset + field_loc.offset;

  OutT value;
  if constexpr (WT == WT_32BIT) {
    if (field_loc.length < 4) {
      atomicExch(error_flag, 1);
      valid[row] = false;
      return;
    }
    uint32_t raw = load_le<uint32_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  } else {
    if (field_loc.length < 8) {
      atomicExch(error_flag, 1);
      valid[row] = false;
      return;
    }
    uint64_t raw = load_le<uint64_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  }

  out[row] = value;
  valid[row] = true;
}

/**
 * Copy nested variable-length data (string/bytes).
 */
__global__ void copy_nested_varlen_data_kernel(
  uint8_t const* message_data,
  cudf::size_type const* parent_row_offsets,
  cudf::size_type parent_base_offset,
  field_location const* parent_locations,
  field_location const* field_locations,
  int field_idx,
  int num_fields,
  int32_t const* output_offsets,
  char* output_data,
  int num_rows,
  bool has_default = false,
  uint8_t const* default_data = nullptr,
  int32_t default_length = 0)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto const& parent_loc = parent_locations[row];
  auto const& field_loc = field_locations[row * num_fields + field_idx];

  char* dst = output_data + output_offsets[row];

  if (parent_loc.offset < 0 || field_loc.offset < 0) {
    if (has_default && default_length > 0) {
      for (int i = 0; i < default_length; i++) {
        dst[i] = static_cast<char>(default_data[i]);
      }
    }
    return;
  }

  if (field_loc.length == 0) return;

  auto parent_row_start = parent_row_offsets[row] - parent_base_offset;
  uint8_t const* src = message_data + parent_row_start + parent_loc.offset + field_loc.offset;

  for (int i = 0; i < field_loc.length; i++) {
    dst[i] = static_cast<char>(src[i]);
  }
}

/**
 * Extract nested field lengths for prefix sum.
 */
__global__ void extract_nested_lengths_kernel(
  field_location const* parent_locations,
  field_location const* field_locations,
  int field_idx,
  int num_fields,
  int32_t* lengths,
  int num_rows,
  bool has_default = false,
  int32_t default_length = 0)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto const& parent_loc = parent_locations[row];
  auto const& field_loc = field_locations[row * num_fields + field_idx];

  if (parent_loc.offset >= 0 && field_loc.offset >= 0) {
    lengths[row] = field_loc.length;
  } else if (has_default) {
    lengths[row] = default_length;
  } else {
    lengths[row] = 0;
  }
}

/**
 * Extract scalar string field lengths for prefix sum.
 * For top-level STRING fields (not nested within a struct).
 */
__global__ void extract_scalar_string_lengths_kernel(
  field_location const* field_locations,
  int field_idx,
  int num_fields,
  int32_t* lengths,
  int num_rows,
  bool has_default = false,
  int32_t default_length = 0)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto const& loc = field_locations[row * num_fields + field_idx];

  if (loc.offset >= 0) {
    lengths[row] = loc.length;
  } else if (has_default) {
    lengths[row] = default_length;
  } else {
    lengths[row] = 0;
  }
}

/**
 * Copy scalar string field data.
 * For top-level STRING fields (not nested within a struct).
 */
__global__ void copy_scalar_string_data_kernel(
  uint8_t const* message_data,
  cudf::size_type const* row_offsets,
  cudf::size_type row_base_offset,
  field_location const* field_locations,
  int field_idx,
  int num_fields,
  int32_t const* output_offsets,
  char* output_data,
  int num_rows,
  bool has_default = false,
  uint8_t const* default_data = nullptr,
  int32_t default_length = 0)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto const& loc = field_locations[row * num_fields + field_idx];

  char* dst = output_data + output_offsets[row];

  if (loc.offset < 0) {
    // Field not found - use default if available
    if (has_default && default_length > 0) {
      for (int i = 0; i < default_length; i++) {
        dst[i] = static_cast<char>(default_data[i]);
      }
    }
    return;
  }

  if (loc.length == 0) return;

  auto row_start = row_offsets[row] - row_base_offset;
  uint8_t const* src = message_data + row_start + loc.offset;

  for (int i = 0; i < loc.length; i++) {
    dst[i] = static_cast<char>(src[i]);
  }
}

// ============================================================================
// Utility functions
// ============================================================================

// Note: make_null_mask_from_valid is defined earlier in the file (before scan_repeated_message_children_kernel)

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
      // Create LIST with all nulls
      // Offsets: all zeros (empty lists)
      rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
      thrust::fill(rmm::exec_policy(stream), offsets.begin(), offsets.end(), 0);
      auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                        num_rows + 1,
                                                        offsets.release(),
                                                        rmm::device_buffer{},
                                                        0);

      // Empty child column - use INT8 as default element type
      // This works because the list has 0 elements, so the child type doesn't matter for nulls
      auto child_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT8}, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);

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
    case cudf::type_id::STRUCT: {
      // Create STRUCT with all nulls and no children
      // Note: This is a workaround. Proper nested struct handling requires recursive processing
      // with full schema information. An empty struct with no children won't match expected
      // schema for deeply nested types, but prevents crashes for unprocessed struct fields.
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
      auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, 1, rmm::device_buffer(sizeof(int32_t), stream, mr),
        rmm::device_buffer{}, 0);
      // Initialize offset to 0
      int32_t zero = 0;
      CUDF_CUDA_TRY(cudaMemcpyAsync(offsets_col->mutable_view().data<int32_t>(), &zero,
                                    sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));
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
 * Find all child field indices for a given parent index in the schema.
 * This is a commonly used pattern throughout the codebase.
 * 
 * @param schema The schema vector (either nested_field_descriptor or device_nested_field_descriptor)
 * @param num_fields Number of fields in the schema
 * @param parent_idx The parent index to search for
 * @return Vector of child field indices
 */
template <typename SchemaT>
std::vector<int> find_child_field_indices(
  SchemaT const& schema,
  int num_fields,
  int parent_idx)
{
  std::vector<int> child_indices;
  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == parent_idx) {
      child_indices.push_back(i);
    }
  }
  return child_indices;
}

/**
 * Recursively create an empty struct column with proper nested structure based on schema.
 * This handles STRUCT children that contain their own grandchildren.
 * 
 * @param schema The schema vector
 * @param schema_output_types Output types for each schema field
 * @param parent_idx Index of the parent field (whose children we want to create)
 * @param num_fields Total number of fields in schema
 * @param stream CUDA stream
 * @param mr Memory resource
 * @return Empty struct column with proper nested structure
 */
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
    
    // Recursively handle nested struct children
    if (child_type.id() == cudf::type_id::STRUCT) {
      children.push_back(make_empty_struct_column_with_schema(
        schema, schema_output_types, child_idx, num_fields, stream, mr));
    } else {
      children.push_back(make_empty_column_safe(child_type, stream, mr));
    }
  }
  
  return cudf::make_structs_column(0, std::move(children), 0, rmm::device_buffer{}, stream, mr);
}

}  // namespace

// ============================================================================
// Kernel to check required fields after scan pass
// ============================================================================

/**
 * Check if any required fields are missing (offset < 0) and set error flag.
 * This is called after the scan pass to validate required field constraints.
 */
__global__ void check_required_fields_kernel(
  field_location const* locations,  // [num_rows * num_fields]
  uint8_t const* is_required,       // [num_fields] (1 = required, 0 = optional)
  int num_fields,
  int num_rows,
  int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  for (int f = 0; f < num_fields; f++) {
    if (is_required[f] != 0 && locations[row * num_fields + f].offset < 0) {
      // Required field is missing - set error flag
      atomicExch(error_flag, 1);
      return;  // No need to check other fields for this row
    }
  }
}

/**
 * Validate enum values against a set of valid values.
 * If a value is not in the valid set:
 * 1. Mark the field as invalid (valid[row] = false)
 * 2. Mark the row as having an invalid enum (row_has_invalid_enum[row] = true)
 *
 * This matches Spark CPU PERMISSIVE mode behavior: when an unknown enum value is
 * encountered, the entire struct row is set to null (not just the enum field).
 *
 * The valid_values array must be sorted for binary search.
 */
__global__ void validate_enum_values_kernel(
  int32_t const* values,             // [num_rows] extracted enum values
  bool* valid,                       // [num_rows] field validity flags (will be modified)
  bool* row_has_invalid_enum,        // [num_rows] row-level invalid enum flag (will be set to true)
  int32_t const* valid_enum_values,  // sorted array of valid enum values
  int num_valid_values,              // size of valid_enum_values
  int num_rows)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  // Skip if already invalid (field was missing) - missing field is not an enum error
  if (!valid[row]) return;

  int32_t val = values[row];

  // Binary search for the value in valid_enum_values
  int left   = 0;
  int right  = num_valid_values - 1;
  bool found = false;

  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (valid_enum_values[mid] == val) {
      found = true;
      break;
    } else if (valid_enum_values[mid] < val) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  // If not found, mark as invalid
  if (!found) {
    valid[row] = false;
    // Also mark the row as having an invalid enum - this will null the entire struct row
    row_has_invalid_enum[row] = true;
  }
}

namespace spark_rapids_jni {

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
  CUDF_EXPECTS(is_required.size() == field_numbers.size(),
               "is_required and field_numbers must have the same length");
  CUDF_EXPECTS(has_default_value.size() == field_numbers.size(),
               "has_default_value and field_numbers must have the same length");
  CUDF_EXPECTS(default_ints.size() == field_numbers.size(),
               "default_ints and field_numbers must have the same length");
  CUDF_EXPECTS(default_floats.size() == field_numbers.size(),
               "default_floats and field_numbers must have the same length");
  CUDF_EXPECTS(default_bools.size() == field_numbers.size(),
               "default_bools and field_numbers must have the same length");
  CUDF_EXPECTS(default_strings.size() == field_numbers.size(),
               "default_strings and field_numbers must have the same length");

  auto const stream       = cudf::get_default_stream();
  auto mr                 = cudf::get_current_device_resource_ref();
  auto rows               = binary_input.size();
  auto num_decoded_fields = static_cast<int>(field_numbers.size());

  // Handle zero-row case
  if (rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_children;
    empty_children.reserve(total_num_fields);
    for (auto const& dt : all_types) {
      empty_children.push_back(make_empty_column_safe(dt, stream, mr));
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
    int schema_idx                = decoded_field_indices[i];
    h_field_descs[i].field_number = field_numbers[i];
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

  // Check if any field has enum validation
  bool has_enum_fields = std::any_of(
    enum_valid_values.begin(), enum_valid_values.end(), [](auto const& v) { return !v.empty(); });

  // Track rows with invalid enum values (used to null entire struct row)
  // This matches Spark CPU PERMISSIVE mode behavior
  rmm::device_uvector<bool> d_row_has_invalid_enum(has_enum_fields ? rows : 0, stream, mr);
  if (has_enum_fields) {
    // Initialize all to false (no invalid enums yet)
    CUDF_CUDA_TRY(
      cudaMemsetAsync(d_row_has_invalid_enum.data(), 0, rows * sizeof(bool), stream.value()));
  }

  auto const threads = 256;
  auto const blocks  = static_cast<int>((rows + threads - 1) / threads);

  // =========================================================================
  // Pass 1: Scan all messages and record field locations
  // =========================================================================
  scan_all_fields_kernel<<<blocks, threads, 0, stream.value()>>>(
    *d_in, d_field_descs.data(), num_decoded_fields, d_locations.data(), d_error.data());

  // =========================================================================
  // Check required fields (after scan pass)
  // =========================================================================
  // Only check if any field is required to avoid unnecessary kernel launch
  bool has_required_fields =
    std::any_of(is_required.begin(), is_required.end(), [](bool b) { return b; });
  if (has_required_fields) {
    // Copy is_required flags to device
    // Note: std::vector<bool> is special (bitfield), so we convert to uint8_t
    rmm::device_uvector<uint8_t> d_is_required(num_decoded_fields, stream, mr);
    std::vector<uint8_t> h_is_required_vec(num_decoded_fields);
    for (int i = 0; i < num_decoded_fields; i++) {
      h_is_required_vec[i] = is_required[i] ? 1 : 0;
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_is_required.data(),
                                  h_is_required_vec.data(),
                                  num_decoded_fields * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice,
                                  stream.value()));

    check_required_fields_kernel<<<blocks, threads, 0, stream.value()>>>(
      d_locations.data(), d_is_required.data(), num_decoded_fields, rows, d_error.data());
  }

  // Get message data pointer and offsets for pass 2
  auto const* message_data = reinterpret_cast<uint8_t const*>(in_list.child().data<int8_t>());
  auto const* list_offsets = in_list.offsets().data<cudf::size_type>();
  // Get the base offset by copying from device to host
  cudf::size_type base_offset = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &base_offset, list_offsets, sizeof(cudf::size_type), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();

  // =========================================================================
  // Pass 2: Extract data for each field
  // =========================================================================
  std::vector<std::unique_ptr<cudf::column>> all_children(total_num_fields);
  int decoded_idx = 0;

  for (int schema_idx = 0; schema_idx < total_num_fields; schema_idx++) {
    if (decoded_idx < num_decoded_fields && decoded_field_indices[decoded_idx] == schema_idx) {
      // This field needs to be decoded
      auto const dt  = all_types[schema_idx];
      auto const enc = encodings[decoded_idx];

      switch (dt.id()) {
        case cudf::type_id::BOOL8: {
          rmm::device_uvector<uint8_t> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          bool has_def    = has_default_value[decoded_idx];
          int64_t def_val = has_def ? (default_bools[decoded_idx] ? 1 : 0) : 0;
          extract_varint_from_locations_kernel<uint8_t>
            <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                     list_offsets,
                                                     base_offset,
                                                     d_locations.data(),
                                                     decoded_idx,
                                                     num_decoded_fields,
                                                     out.data(),
                                                     valid.data(),
                                                     rows,
                                                     d_error.data(),
                                                     has_def,
                                                     def_val);
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::INT32: {
          rmm::device_uvector<int32_t> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          bool has_def      = has_default_value[decoded_idx];
          int64_t def_int   = has_def ? default_ints[decoded_idx] : 0;
          int32_t def_fixed = static_cast<int32_t>(def_int);
          if (enc == spark_rapids_jni::ENC_ZIGZAG) {
            extract_varint_from_locations_kernel<int32_t, true>
              <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                       list_offsets,
                                                       base_offset,
                                                       d_locations.data(),
                                                       decoded_idx,
                                                       num_decoded_fields,
                                                       out.data(),
                                                       valid.data(),
                                                       rows,
                                                       d_error.data(),
                                                       has_def,
                                                       def_int);
          } else if (enc == spark_rapids_jni::ENC_FIXED) {
            extract_fixed_from_locations_kernel<int32_t, WT_32BIT>
              <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                       list_offsets,
                                                       base_offset,
                                                       d_locations.data(),
                                                       decoded_idx,
                                                       num_decoded_fields,
                                                       out.data(),
                                                       valid.data(),
                                                       rows,
                                                       d_error.data(),
                                                       has_def,
                                                       def_fixed);
          } else {
            extract_varint_from_locations_kernel<int32_t, false>
              <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                       list_offsets,
                                                       base_offset,
                                                       d_locations.data(),
                                                       decoded_idx,
                                                       num_decoded_fields,
                                                       out.data(),
                                                       valid.data(),
                                                       rows,
                                                       d_error.data(),
                                                       has_def,
                                                       def_int);
          }

          // Validate enum values if this is an enum field
          // enum_valid_values[decoded_idx] is non-empty for enum fields
          auto const& valid_enums = enum_valid_values[decoded_idx];
          if (!valid_enums.empty()) {
            // Copy valid enum values to device
            rmm::device_uvector<int32_t> d_valid_enums(valid_enums.size(), stream, mr);
            CUDF_CUDA_TRY(cudaMemcpyAsync(d_valid_enums.data(),
                                          valid_enums.data(),
                                          valid_enums.size() * sizeof(int32_t),
                                          cudaMemcpyHostToDevice,
                                          stream.value()));

            // Validate enum values - unknown values will null the entire row
            validate_enum_values_kernel<<<blocks, threads, 0, stream.value()>>>(
              out.data(),
              valid.data(),
              d_row_has_invalid_enum.data(),
              d_valid_enums.data(),
              static_cast<int>(valid_enums.size()),
              rows);
          }

          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::UINT32: {
          rmm::device_uvector<uint32_t> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          bool has_def       = has_default_value[decoded_idx];
          int64_t def_int    = has_def ? default_ints[decoded_idx] : 0;
          uint32_t def_fixed = static_cast<uint32_t>(def_int);
          if (enc == spark_rapids_jni::ENC_FIXED) {
            extract_fixed_from_locations_kernel<uint32_t, WT_32BIT>
              <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                       list_offsets,
                                                       base_offset,
                                                       d_locations.data(),
                                                       decoded_idx,
                                                       num_decoded_fields,
                                                       out.data(),
                                                       valid.data(),
                                                       rows,
                                                       d_error.data(),
                                                       has_def,
                                                       def_fixed);
          } else {
            extract_varint_from_locations_kernel<uint32_t>
              <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                       list_offsets,
                                                       base_offset,
                                                       d_locations.data(),
                                                       decoded_idx,
                                                       num_decoded_fields,
                                                       out.data(),
                                                       valid.data(),
                                                       rows,
                                                       d_error.data(),
                                                       has_def,
                                                       def_int);
          }
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::INT64: {
          rmm::device_uvector<int64_t> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          bool has_def    = has_default_value[decoded_idx];
          int64_t def_int = has_def ? default_ints[decoded_idx] : 0;
          if (enc == spark_rapids_jni::ENC_ZIGZAG) {
            extract_varint_from_locations_kernel<int64_t, true>
              <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                       list_offsets,
                                                       base_offset,
                                                       d_locations.data(),
                                                       decoded_idx,
                                                       num_decoded_fields,
                                                       out.data(),
                                                       valid.data(),
                                                       rows,
                                                       d_error.data(),
                                                       has_def,
                                                       def_int);
          } else if (enc == spark_rapids_jni::ENC_FIXED) {
            extract_fixed_from_locations_kernel<int64_t, WT_64BIT>
              <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                       list_offsets,
                                                       base_offset,
                                                       d_locations.data(),
                                                       decoded_idx,
                                                       num_decoded_fields,
                                                       out.data(),
                                                       valid.data(),
                                                       rows,
                                                       d_error.data(),
                                                       has_def,
                                                       def_int);
          } else {
            extract_varint_from_locations_kernel<int64_t, false>
              <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                       list_offsets,
                                                       base_offset,
                                                       d_locations.data(),
                                                       decoded_idx,
                                                       num_decoded_fields,
                                                       out.data(),
                                                       valid.data(),
                                                       rows,
                                                       d_error.data(),
                                                       has_def,
                                                       def_int);
          }
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::UINT64: {
          rmm::device_uvector<uint64_t> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          bool has_def       = has_default_value[decoded_idx];
          int64_t def_int    = has_def ? default_ints[decoded_idx] : 0;
          uint64_t def_fixed = static_cast<uint64_t>(def_int);
          if (enc == spark_rapids_jni::ENC_FIXED) {
            extract_fixed_from_locations_kernel<uint64_t, WT_64BIT>
              <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                       list_offsets,
                                                       base_offset,
                                                       d_locations.data(),
                                                       decoded_idx,
                                                       num_decoded_fields,
                                                       out.data(),
                                                       valid.data(),
                                                       rows,
                                                       d_error.data(),
                                                       has_def,
                                                       def_fixed);
          } else {
            extract_varint_from_locations_kernel<uint64_t>
              <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                       list_offsets,
                                                       base_offset,
                                                       d_locations.data(),
                                                       decoded_idx,
                                                       num_decoded_fields,
                                                       out.data(),
                                                       valid.data(),
                                                       rows,
                                                       d_error.data(),
                                                       has_def,
                                                       def_int);
          }
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::FLOAT32: {
          rmm::device_uvector<float> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          bool has_def    = has_default_value[decoded_idx];
          float def_float = has_def ? static_cast<float>(default_floats[decoded_idx]) : 0.0f;
          extract_fixed_from_locations_kernel<float, WT_32BIT>
            <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                     list_offsets,
                                                     base_offset,
                                                     d_locations.data(),
                                                     decoded_idx,
                                                     num_decoded_fields,
                                                     out.data(),
                                                     valid.data(),
                                                     rows,
                                                     d_error.data(),
                                                     has_def,
                                                     def_float);
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::FLOAT64: {
          rmm::device_uvector<double> out(rows, stream, mr);
          rmm::device_uvector<bool> valid(rows, stream, mr);
          bool has_def      = has_default_value[decoded_idx];
          double def_double = has_def ? default_floats[decoded_idx] : 0.0;
          extract_fixed_from_locations_kernel<double, WT_64BIT>
            <<<blocks, threads, 0, stream.value()>>>(message_data,
                                                     list_offsets,
                                                     base_offset,
                                                     d_locations.data(),
                                                     decoded_idx,
                                                     num_decoded_fields,
                                                     out.data(),
                                                     valid.data(),
                                                     rows,
                                                     d_error.data(),
                                                     has_def,
                                                     def_double);
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          all_children[schema_idx] =
            std::make_unique<cudf::column>(dt, rows, out.release(), std::move(mask), null_count);
          break;
        }

        case cudf::type_id::STRING: {
          // Check for default value
          bool has_def        = has_default_value[decoded_idx];
          auto const& def_str = default_strings[decoded_idx];
          int32_t def_len     = has_def ? static_cast<int32_t>(def_str.size()) : 0;

          // Copy default string to device if needed
          rmm::device_uvector<uint8_t> d_default_str(def_len, stream, mr);
          if (has_def && def_len > 0) {
            CUDF_CUDA_TRY(cudaMemcpyAsync(d_default_str.data(),
                                          def_str.data(),
                                          def_len,
                                          cudaMemcpyHostToDevice,
                                          stream.value()));
          }

          // Extract lengths and compute output offsets via prefix sum
          rmm::device_uvector<int32_t> lengths(rows, stream, mr);
          extract_lengths_kernel<<<blocks, threads, 0, stream.value()>>>(d_locations.data(),
                                                                         decoded_idx,
                                                                         num_decoded_fields,
                                                                         lengths.data(),
                                                                         rows,
                                                                         has_def,
                                                                         def_len);

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
            copy_varlen_data_kernel<<<blocks, threads, 0, stream.value()>>>(message_data,
                                                                            list_offsets,
                                                                            base_offset,
                                                                            d_locations.data(),
                                                                            decoded_idx,
                                                                            num_decoded_fields,
                                                                            output_offsets.data(),
                                                                            chars.data(),
                                                                            rows,
                                                                            has_def,
                                                                            d_default_str.data(),
                                                                            def_len);
          }

          // Create validity mask (field found OR has default = valid)
          rmm::device_uvector<bool> valid(rows, stream, mr);
          thrust::transform(
            rmm::exec_policy(stream),
            thrust::make_counting_iterator<cudf::size_type>(0),
            thrust::make_counting_iterator<cudf::size_type>(rows),
            valid.begin(),
            [locs = d_locations.data(), decoded_idx, num_decoded_fields, has_def] __device__(
              auto row) {
              return locs[row * num_decoded_fields + decoded_idx].offset >= 0 || has_def;
            });
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);

          // Create offsets column
          auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
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
          // Check for default value
          bool has_def          = has_default_value[decoded_idx];
          auto const& def_bytes = default_strings[decoded_idx];
          int32_t def_len       = has_def ? static_cast<int32_t>(def_bytes.size()) : 0;

          // Copy default bytes to device if needed
          rmm::device_uvector<uint8_t> d_default_bytes(def_len, stream, mr);
          if (has_def && def_len > 0) {
            CUDF_CUDA_TRY(cudaMemcpyAsync(d_default_bytes.data(),
                                          def_bytes.data(),
                                          def_len,
                                          cudaMemcpyHostToDevice,
                                          stream.value()));
          }

          // Extract lengths and compute output offsets via prefix sum
          rmm::device_uvector<int32_t> lengths(rows, stream, mr);
          extract_lengths_kernel<<<blocks, threads, 0, stream.value()>>>(d_locations.data(),
                                                                         decoded_idx,
                                                                         num_decoded_fields,
                                                                         lengths.data(),
                                                                         rows,
                                                                         has_def,
                                                                         def_len);

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
              rows,
              has_def,
              d_default_bytes.data(),
              def_len);
          }

          // Create validity mask (field found OR has default = valid)
          rmm::device_uvector<bool> valid(rows, stream, mr);
          thrust::transform(
            rmm::exec_policy(stream),
            thrust::make_counting_iterator<cudf::size_type>(0),
            thrust::make_counting_iterator<cudf::size_type>(rows),
            valid.begin(),
            [locs = d_locations.data(), decoded_idx, num_decoded_fields, has_def] __device__(
              auto row) {
              return locs[row * num_decoded_fields + decoded_idx].offset >= 0 || has_def;
            });
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);

          // Create offsets column
          auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                            rows + 1,
                                                            output_offsets.release(),
                                                            rmm::device_buffer{},
                                                            0);

          // Create INT8 child column directly (no intermediate strings column!)
          auto child_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT8},
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

  // Check for any parse errors or missing required fields.
  // Note: We check errors after all kernels complete rather than between kernel launches
  // to avoid expensive synchronization overhead. If fail_on_errors is true and an error
  // occurred, all kernels will have executed but we throw an exception here.
  int h_error = 0;
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(&h_error, d_error.data(), sizeof(int), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  if (fail_on_errors) {
    CUDF_EXPECTS(h_error == 0,
                 "Malformed protobuf message, unsupported wire type, or missing required field");
  }

  // Build the final struct
  // If any rows have invalid enum values, create a null mask for the struct
  // This matches Spark CPU PERMISSIVE mode: unknown enum values null the entire row
  cudf::size_type struct_null_count = 0;
  rmm::device_buffer struct_mask{0, stream, mr};

  if (has_enum_fields) {
    // Create struct null mask: row is valid if it has NO invalid enums
    auto [mask, null_count] = cudf::detail::valid_if(
      thrust::make_counting_iterator<cudf::size_type>(0),
      thrust::make_counting_iterator<cudf::size_type>(rows),
      [row_invalid = d_row_has_invalid_enum.data()] __device__(cudf::size_type row) {
        return !row_invalid[row];  // valid if NOT invalid
      },
      stream,
      mr);
    struct_mask       = std::move(mask);
    struct_null_count = null_count;
  }

  return cudf::make_structs_column(
    rows, std::move(all_children), struct_null_count, std::move(struct_mask), stream, mr);
}

// ============================================================================
// Nested protobuf decoding implementation
// ============================================================================

namespace {

/**
 * Helper to build a repeated scalar column (LIST of scalar type).
 */
template <typename T>
std::unique_ptr<cudf::column> build_repeated_scalar_column(
  cudf::column_view const& binary_input,
  device_nested_field_descriptor const& field_desc,
  std::vector<repeated_field_info> const& h_repeated_info,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Get input column's null mask to determine which output rows should be null
  // Only rows where INPUT is null should produce null output
  // Rows with valid input but count=0 should produce empty array []
  cudf::lists_column_view const in_list(binary_input);
  auto const input_null_count = binary_input.null_count();

  if (total_count == 0) {
    // All rows have count=0, but we still need to check input nulls
    rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
    thrust::fill(rmm::exec_policy(stream), offsets.begin(), offsets.end(), 0);
    auto offsets_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, num_rows + 1, offsets.release(), rmm::device_buffer{}, 0);
    auto elem_type = field_desc.output_type_id == static_cast<int>(cudf::type_id::LIST)
                     ? cudf::type_id::UINT8
                     : static_cast<cudf::type_id>(field_desc.output_type_id);
    auto child_col = make_empty_column_safe(cudf::data_type{elem_type}, stream, mr);
    
    if (input_null_count > 0) {
      // Copy input null mask - only input nulls produce output nulls
      auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
      return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(child_col),
                                     input_null_count, std::move(null_mask), stream, mr);
    } else {
      // No input nulls, all rows get empty arrays []
      return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(child_col),
                                     0, rmm::device_buffer{}, stream, mr);
    }
  }

  auto const* message_data = reinterpret_cast<uint8_t const*>(in_list.child().data<int8_t>());
  auto const* list_offsets = in_list.offsets().data<cudf::size_type>();

  cudf::size_type base_offset = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&base_offset, list_offsets, sizeof(cudf::size_type),
                                cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();

  // Build list offsets from counts
  rmm::device_uvector<int32_t> counts(num_rows, stream, mr);
  std::vector<int32_t> h_counts(num_rows);
  for (int i = 0; i < num_rows; i++) {
    h_counts[i] = h_repeated_info[i].count;
  }
  CUDF_CUDA_TRY(cudaMemcpyAsync(counts.data(), h_counts.data(), num_rows * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream.value()));

  rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy(stream), counts.begin(), counts.end(), list_offs.begin(), 0);
  
  int32_t last_offset_h = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&last_offset_h, list_offs.data() + num_rows - 1, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream.value()));
  int32_t last_count_h = h_counts[num_rows - 1];
  stream.synchronize();
  last_offset_h += last_count_h;
  CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data() + num_rows, &last_offset_h, sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream.value()));

  // Extract values
  rmm::device_uvector<T> values(total_count, stream, mr);
  rmm::device_uvector<int> d_error(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 0, sizeof(int), stream.value()));

  auto const threads = 256;
  auto const blocks = (total_count + threads - 1) / threads;

  int encoding = field_desc.encoding;
  bool zigzag = (encoding == spark_rapids_jni::ENC_ZIGZAG);
  
  // For float/double types, always use fixed kernel (they use wire type 32BIT/64BIT)
  // For integer types, use fixed kernel only if encoding is ENC_FIXED
  constexpr bool is_floating_point = std::is_same_v<T, float> || std::is_same_v<T, double>;
  bool use_fixed_kernel = is_floating_point || (encoding == spark_rapids_jni::ENC_FIXED);

  if (use_fixed_kernel) {
    if constexpr (sizeof(T) == 4) {
      extract_repeated_fixed_kernel<T, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
        message_data, list_offsets, base_offset, d_occurrences.data(), total_count, values.data(), d_error.data());
    } else {
      extract_repeated_fixed_kernel<T, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
        message_data, list_offsets, base_offset, d_occurrences.data(), total_count, values.data(), d_error.data());
    }
  } else if (zigzag) {
    extract_repeated_varint_kernel<T, true><<<blocks, threads, 0, stream.value()>>>(
      message_data, list_offsets, base_offset, d_occurrences.data(), total_count, values.data(), d_error.data());
  } else {
    extract_repeated_varint_kernel<T, false><<<blocks, threads, 0, stream.value()>>>(
      message_data, list_offsets, base_offset, d_occurrences.data(), total_count, values.data(), d_error.data());
  }

  auto offsets_col = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, list_offs.release(), rmm::device_buffer{}, 0);
  auto child_col = std::make_unique<cudf::column>(
    cudf::data_type{static_cast<cudf::type_id>(field_desc.output_type_id)}, total_count, values.release(), rmm::device_buffer{}, 0);

  // Only rows where INPUT is null should produce null output
  // Rows with valid input but count=0 should produce empty array []
  if (input_null_count > 0) {
    // Copy input null mask - only input nulls produce output nulls
    auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
    return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(child_col),
                                   input_null_count, std::move(null_mask), stream, mr);
  }

  return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{}, stream, mr);
}

/**
 * Build a repeated string/bytes column (LIST of STRING or LIST<UINT8>).
 */
std::unique_ptr<cudf::column> build_repeated_string_column(
  cudf::column_view const& binary_input,
  device_nested_field_descriptor const& field_desc,
  std::vector<repeated_field_info> const& h_repeated_info,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  bool is_bytes,  // true for bytes (LIST<UINT8>), false for string
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Get input column's null mask to determine which output rows should be null
  // Only rows where INPUT is null should produce null output
  // Rows with valid input but count=0 should produce empty array []
  auto const input_null_count = binary_input.null_count();

  if (total_count == 0) {
    // All rows have count=0, but we still need to check input nulls
    rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
    thrust::fill(rmm::exec_policy(stream), offsets.begin(), offsets.end(), 0);
    auto offsets_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, num_rows + 1, offsets.release(), rmm::device_buffer{}, 0);
    auto child_col = is_bytes
      ? make_empty_column_safe(cudf::data_type{cudf::type_id::LIST}, stream, mr)  // LIST<UINT8>
      : cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
    
    if (input_null_count > 0) {
      // Copy input null mask - only input nulls produce output nulls
      auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
      return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(child_col),
                                     input_null_count, std::move(null_mask), stream, mr);
    } else {
      // No input nulls, all rows get empty arrays []
      return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(child_col),
                                     0, rmm::device_buffer{}, stream, mr);
    }
  }

  cudf::lists_column_view const in_list(binary_input);
  auto const* message_data = reinterpret_cast<uint8_t const*>(in_list.child().data<int8_t>());
  auto const* list_offsets = in_list.offsets().data<cudf::size_type>();

  cudf::size_type base_offset = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&base_offset, list_offsets, sizeof(cudf::size_type),
                                cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();

  // Build list offsets from counts
  rmm::device_uvector<int32_t> counts(num_rows, stream, mr);
  std::vector<int32_t> h_counts(num_rows);
  for (int i = 0; i < num_rows; i++) {
    h_counts[i] = h_repeated_info[i].count;
  }
  CUDF_CUDA_TRY(cudaMemcpyAsync(counts.data(), h_counts.data(), num_rows * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream.value()));

  rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy(stream), counts.begin(), counts.end(), list_offs.begin(), 0);

  int32_t last_offset_h = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&last_offset_h, list_offs.data() + num_rows - 1, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream.value()));
  int32_t last_count_h = h_counts[num_rows - 1];
  stream.synchronize();
  last_offset_h += last_count_h;
  CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data() + num_rows, &last_offset_h, sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream.value()));

  // Extract string lengths from occurrences
  rmm::device_uvector<int32_t> str_lengths(total_count, stream, mr);
  auto const threads = 256;
  auto const blocks = (total_count + threads - 1) / threads;
  extract_repeated_lengths_kernel<<<blocks, threads, 0, stream.value()>>>(
    d_occurrences.data(), total_count, str_lengths.data());

  // Compute string offsets via prefix sum
  rmm::device_uvector<int32_t> str_offsets(total_count + 1, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy(stream), str_lengths.begin(), str_lengths.end(), str_offsets.begin(), 0);

  int32_t total_chars = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&total_chars, str_offsets.data() + total_count - 1, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream.value()));
  int32_t last_len = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&last_len, str_lengths.data() + total_count - 1, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  total_chars += last_len;
  CUDF_CUDA_TRY(cudaMemcpyAsync(str_offsets.data() + total_count, &total_chars, sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream.value()));

  // Copy string data
  rmm::device_uvector<char> chars(total_chars, stream, mr);
  if (total_chars > 0) {
    copy_repeated_varlen_data_kernel<<<blocks, threads, 0, stream.value()>>>(
      message_data, list_offsets, base_offset, d_occurrences.data(), total_count,
      str_offsets.data(), chars.data());
  }

  // Build the child column (either STRING or LIST<UINT8>)
  std::unique_ptr<cudf::column> child_col;
  if (is_bytes) {
    // Build LIST<UINT8> for bytes (Spark BinaryType maps to LIST<UINT8>)
    auto str_offsets_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, total_count + 1, str_offsets.release(), rmm::device_buffer{}, 0);
    auto bytes_child = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::UINT8}, total_chars,
      rmm::device_buffer(chars.data(), total_chars, stream, mr), rmm::device_buffer{}, 0);
    child_col = cudf::make_lists_column(total_count, std::move(str_offsets_col), std::move(bytes_child),
                                        0, rmm::device_buffer{}, stream, mr);
  } else {
    // Build STRING column
    auto str_offsets_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, total_count + 1, str_offsets.release(), rmm::device_buffer{}, 0);
    child_col = cudf::make_strings_column(total_count, std::move(str_offsets_col), chars.release(), 0, rmm::device_buffer{});
  }

  auto offsets_col = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, list_offs.release(), rmm::device_buffer{}, 0);

  // Only rows where INPUT is null should produce null output
  // Rows with valid input but count=0 should produce empty array []
  if (input_null_count > 0) {
    // Copy input null mask - only input nulls produce output nulls
    auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
    return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(child_col),
                                   input_null_count, std::move(null_mask), stream, mr);
  }

  return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{}, stream, mr);
}

/**
 * Build a repeated struct column (LIST of STRUCT).
 * This handles repeated message fields like: repeated Item items = 2;
 * The output is ArrayType(StructType(...))
 */
std::unique_ptr<cudf::column> build_repeated_struct_column(
  cudf::column_view const& binary_input,
  device_nested_field_descriptor const& field_desc,
  std::vector<repeated_field_info> const& h_repeated_info,
  rmm::device_uvector<repeated_occurrence>& d_occurrences,
  int total_count,
  int num_rows,
  // Child field information
  std::vector<device_nested_field_descriptor> const& h_device_schema,
  std::vector<int> const& child_field_indices,  // Indices of child fields in schema
  std::vector<cudf::data_type> const& schema_output_types,
  std::vector<int64_t> const& default_ints,
  std::vector<double> const& default_floats,
  std::vector<bool> const& default_bools,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const input_null_count = binary_input.null_count();
  int num_child_fields = static_cast<int>(child_field_indices.size());

  if (total_count == 0 || num_child_fields == 0) {
    // All rows have count=0 or no child fields - return list of empty structs
    rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
    thrust::fill(rmm::exec_policy(stream), offsets.begin(), offsets.end(), 0);
    auto offsets_col = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, num_rows + 1, offsets.release(), rmm::device_buffer{}, 0);
    
    // Build empty struct child column with proper nested structure
    int num_schema_fields = static_cast<int>(h_device_schema.size());
    std::vector<std::unique_ptr<cudf::column>> empty_struct_children;
    for (int child_schema_idx : child_field_indices) {
      auto child_type = schema_output_types[child_schema_idx];
      if (child_type.id() == cudf::type_id::STRUCT) {
        // Use helper to recursively build nested struct
        empty_struct_children.push_back(make_empty_struct_column_with_schema(
          h_device_schema, schema_output_types, child_schema_idx, num_schema_fields, stream, mr));
      } else {
        empty_struct_children.push_back(make_empty_column_safe(child_type, stream, mr));
      }
    }
    auto empty_struct = cudf::make_structs_column(0, std::move(empty_struct_children), 0, rmm::device_buffer{}, stream, mr);
    
    if (input_null_count > 0) {
      auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
      return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(empty_struct),
                                     input_null_count, std::move(null_mask), stream, mr);
    } else {
      return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(empty_struct),
                                     0, rmm::device_buffer{}, stream, mr);
    }
  }

  cudf::lists_column_view const in_list(binary_input);
  auto const* message_data = reinterpret_cast<uint8_t const*>(in_list.child().data<int8_t>());
  auto const* list_offsets = in_list.offsets().data<cudf::size_type>();

  cudf::size_type base_offset = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&base_offset, list_offsets, sizeof(cudf::size_type),
                                cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();

  // Build list offsets from counts (for the outer LIST column)
  rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
  std::vector<int32_t> h_counts(num_rows);
  for (int i = 0; i < num_rows; i++) {
    h_counts[i] = h_repeated_info[i].count;
  }
  rmm::device_uvector<int32_t> counts(num_rows, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(counts.data(), h_counts.data(), num_rows * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream.value()));
  thrust::exclusive_scan(rmm::exec_policy(stream), counts.begin(), counts.end(), list_offs.begin(), 0);
  
  int32_t last_offset_h = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&last_offset_h, list_offs.data() + num_rows - 1, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream.value()));
  int32_t last_count_h = h_counts[num_rows - 1];
  stream.synchronize();
  last_offset_h += last_count_h;
  CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data() + num_rows, &last_offset_h, sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream.value()));

  // Copy occurrences to host for processing
  std::vector<repeated_occurrence> h_occurrences(total_count);
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_occurrences.data(), d_occurrences.data(),
                                total_count * sizeof(repeated_occurrence),
                                cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();

  // Build child field descriptors for scanning within each message occurrence
  std::vector<field_descriptor> h_child_descs(num_child_fields);
  for (int ci = 0; ci < num_child_fields; ci++) {
    int child_schema_idx = child_field_indices[ci];
    h_child_descs[ci].field_number = h_device_schema[child_schema_idx].field_number;
    h_child_descs[ci].expected_wire_type = h_device_schema[child_schema_idx].wire_type;
  }
  rmm::device_uvector<field_descriptor> d_child_descs(num_child_fields, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_child_descs.data(), h_child_descs.data(),
                                num_child_fields * sizeof(field_descriptor),
                                cudaMemcpyHostToDevice, stream.value()));

  // For each occurrence, we need to scan for child fields
  // Create "virtual" parent locations from the occurrences
  // Each occurrence becomes a "parent" message for child field scanning
  std::vector<field_location> h_msg_locs(total_count);
  std::vector<int32_t> h_msg_row_offsets(total_count);
  for (int i = 0; i < total_count; i++) {
    auto const& occ = h_occurrences[i];
    // Get the row's start offset in the binary column
    cudf::size_type row_offset;
    CUDF_CUDA_TRY(cudaMemcpyAsync(&row_offset, list_offsets + occ.row_idx, sizeof(cudf::size_type),
                                  cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
    h_msg_row_offsets[i] = static_cast<int32_t>(row_offset - base_offset);
    h_msg_locs[i] = {occ.offset, occ.length};
  }

  rmm::device_uvector<field_location> d_msg_locs(total_count, stream, mr);
  rmm::device_uvector<int32_t> d_msg_row_offsets(total_count, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_msg_locs.data(), h_msg_locs.data(),
                                total_count * sizeof(field_location),
                                cudaMemcpyHostToDevice, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_msg_row_offsets.data(), h_msg_row_offsets.data(),
                                total_count * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream.value()));

  // Scan for child fields within each message occurrence
  rmm::device_uvector<field_location> d_child_locs(total_count * num_child_fields, stream, mr);
  rmm::device_uvector<int> d_error(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 0, sizeof(int), stream.value()));

  auto const threads = 256;
  auto const blocks = (total_count + threads - 1) / threads;

  // Use a custom kernel to scan child fields within message occurrences
  // This is similar to scan_nested_message_fields_kernel but operates on occurrences
  scan_repeated_message_children_kernel<<<blocks, threads, 0, stream.value()>>>(
    message_data, d_msg_row_offsets.data(), d_msg_locs.data(), total_count,
    d_child_descs.data(), num_child_fields, d_child_locs.data(), d_error.data());

  // Copy child locations to host
  std::vector<field_location> h_child_locs(total_count * num_child_fields);
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_child_locs.data(), d_child_locs.data(),
                                h_child_locs.size() * sizeof(field_location),
                                cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();

  // Extract child field values - build one column per child field
  std::vector<std::unique_ptr<cudf::column>> struct_children;
  for (int ci = 0; ci < num_child_fields; ci++) {
    int child_schema_idx = child_field_indices[ci];
    auto const dt = schema_output_types[child_schema_idx];
    auto const enc = h_device_schema[child_schema_idx].encoding;
    bool has_def = h_device_schema[child_schema_idx].has_default_value;

    switch (dt.id()) {
      case cudf::type_id::BOOL8: {
        rmm::device_uvector<uint8_t> out(total_count, stream, mr);
        rmm::device_uvector<bool> valid(total_count, stream, mr);
        int64_t def_val = has_def ? (default_bools[child_schema_idx] ? 1 : 0) : 0;
        extract_repeated_msg_child_varint_kernel<uint8_t><<<blocks, threads, 0, stream.value()>>>(
          message_data, d_msg_row_offsets.data(), d_msg_locs.data(),
          d_child_locs.data(), ci, num_child_fields, out.data(), valid.data(),
          total_count, d_error.data(), has_def, def_val);
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        struct_children.push_back(std::make_unique<cudf::column>(
          dt, total_count, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::INT32: {
        rmm::device_uvector<int32_t> out(total_count, stream, mr);
        rmm::device_uvector<bool> valid(total_count, stream, mr);
        int64_t def_int = has_def ? default_ints[child_schema_idx] : 0;
        if (enc == spark_rapids_jni::ENC_ZIGZAG) {
          extract_repeated_msg_child_varint_kernel<int32_t, true><<<blocks, threads, 0, stream.value()>>>(
            message_data, d_msg_row_offsets.data(), d_msg_locs.data(),
            d_child_locs.data(), ci, num_child_fields, out.data(), valid.data(),
            total_count, d_error.data(), has_def, def_int);
        } else if (enc == spark_rapids_jni::ENC_FIXED) {
          extract_repeated_msg_child_fixed_kernel<int32_t, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
            message_data, d_msg_row_offsets.data(), d_msg_locs.data(),
            d_child_locs.data(), ci, num_child_fields, out.data(), valid.data(),
            total_count, d_error.data(), has_def, static_cast<int32_t>(def_int));
        } else {
          extract_repeated_msg_child_varint_kernel<int32_t, false><<<blocks, threads, 0, stream.value()>>>(
            message_data, d_msg_row_offsets.data(), d_msg_locs.data(),
            d_child_locs.data(), ci, num_child_fields, out.data(), valid.data(),
            total_count, d_error.data(), has_def, def_int);
        }
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        struct_children.push_back(std::make_unique<cudf::column>(
          dt, total_count, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::INT64: {
        rmm::device_uvector<int64_t> out(total_count, stream, mr);
        rmm::device_uvector<bool> valid(total_count, stream, mr);
        int64_t def_int = has_def ? default_ints[child_schema_idx] : 0;
        if (enc == spark_rapids_jni::ENC_ZIGZAG) {
          extract_repeated_msg_child_varint_kernel<int64_t, true><<<blocks, threads, 0, stream.value()>>>(
            message_data, d_msg_row_offsets.data(), d_msg_locs.data(),
            d_child_locs.data(), ci, num_child_fields, out.data(), valid.data(),
            total_count, d_error.data(), has_def, def_int);
        } else if (enc == spark_rapids_jni::ENC_FIXED) {
          extract_repeated_msg_child_fixed_kernel<int64_t, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
            message_data, d_msg_row_offsets.data(), d_msg_locs.data(),
            d_child_locs.data(), ci, num_child_fields, out.data(), valid.data(),
            total_count, d_error.data(), has_def, def_int);
        } else {
          extract_repeated_msg_child_varint_kernel<int64_t, false><<<blocks, threads, 0, stream.value()>>>(
            message_data, d_msg_row_offsets.data(), d_msg_locs.data(),
            d_child_locs.data(), ci, num_child_fields, out.data(), valid.data(),
            total_count, d_error.data(), has_def, def_int);
        }
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        struct_children.push_back(std::make_unique<cudf::column>(
          dt, total_count, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::FLOAT32: {
        rmm::device_uvector<float> out(total_count, stream, mr);
        rmm::device_uvector<bool> valid(total_count, stream, mr);
        float def_float = has_def ? static_cast<float>(default_floats[child_schema_idx]) : 0.0f;
        extract_repeated_msg_child_fixed_kernel<float, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
          message_data, d_msg_row_offsets.data(), d_msg_locs.data(),
          d_child_locs.data(), ci, num_child_fields, out.data(), valid.data(),
          total_count, d_error.data(), has_def, def_float);
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        struct_children.push_back(std::make_unique<cudf::column>(
          dt, total_count, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::FLOAT64: {
        rmm::device_uvector<double> out(total_count, stream, mr);
        rmm::device_uvector<bool> valid(total_count, stream, mr);
        double def_double = has_def ? default_floats[child_schema_idx] : 0.0;
        extract_repeated_msg_child_fixed_kernel<double, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
          message_data, d_msg_row_offsets.data(), d_msg_locs.data(),
          d_child_locs.data(), ci, num_child_fields, out.data(), valid.data(),
          total_count, d_error.data(), has_def, def_double);
        auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
        struct_children.push_back(std::make_unique<cudf::column>(
          dt, total_count, out.release(), std::move(mask), null_count));
        break;
      }
      case cudf::type_id::STRING: {
        // For strings, we need a two-pass approach: first get lengths, then copy data
        struct_children.push_back(
          build_repeated_msg_child_string_column(
            message_data, d_msg_row_offsets, d_msg_locs,
            d_child_locs, ci, num_child_fields, total_count, d_error, stream, mr));
        break;
      }
      case cudf::type_id::STRUCT: {
        // Nested struct inside repeated message - need to extract grandchild fields
        int num_schema_fields = static_cast<int>(h_device_schema.size());
        auto grandchild_indices = find_child_field_indices(h_device_schema, num_schema_fields, child_schema_idx);
        
        if (grandchild_indices.empty()) {
          // No grandchildren - create empty struct column
          struct_children.push_back(cudf::make_structs_column(
            total_count, std::vector<std::unique_ptr<cudf::column>>{}, 0, rmm::device_buffer{}, stream, mr));
        } else {
          // Build grandchild columns
          // For each occurrence, the nested struct location is in child_locs[occ * num_child_fields + ci]
          // We need to scan within each nested struct for grandchild fields
          
          // Build grandchild field descriptors
          int num_grandchildren = static_cast<int>(grandchild_indices.size());
          std::vector<field_descriptor> h_gc_descs(num_grandchildren);
          for (int gci = 0; gci < num_grandchildren; gci++) {
            int gc_schema_idx = grandchild_indices[gci];
            h_gc_descs[gci].field_number = h_device_schema[gc_schema_idx].field_number;
            h_gc_descs[gci].expected_wire_type = h_device_schema[gc_schema_idx].wire_type;
          }
          rmm::device_uvector<field_descriptor> d_gc_descs(num_grandchildren, stream, mr);
          CUDF_CUDA_TRY(cudaMemcpyAsync(d_gc_descs.data(), h_gc_descs.data(),
                                        num_grandchildren * sizeof(field_descriptor),
                                        cudaMemcpyHostToDevice, stream.value()));
          
          // Create nested struct locations from child_locs
          // Each occurrence's nested struct is at child_locs[occ * num_child_fields + ci]
          std::vector<field_location> h_nested_locs(total_count);
          std::vector<int32_t> h_nested_row_offsets(total_count);
          for (int occ = 0; occ < total_count; occ++) {
            auto const& nested_loc = h_child_locs[occ * num_child_fields + ci];
            auto const& msg_loc = h_msg_locs[occ];
            h_nested_row_offsets[occ] = h_msg_row_offsets[occ] + msg_loc.offset;
            h_nested_locs[occ] = nested_loc;
          }
          
          rmm::device_uvector<field_location> d_nested_locs(total_count, stream, mr);
          rmm::device_uvector<int32_t> d_nested_row_offsets(total_count, stream, mr);
          CUDF_CUDA_TRY(cudaMemcpyAsync(d_nested_locs.data(), h_nested_locs.data(),
                                        total_count * sizeof(field_location),
                                        cudaMemcpyHostToDevice, stream.value()));
          CUDF_CUDA_TRY(cudaMemcpyAsync(d_nested_row_offsets.data(), h_nested_row_offsets.data(),
                                        total_count * sizeof(int32_t),
                                        cudaMemcpyHostToDevice, stream.value()));
          
          // Scan for grandchild fields
          rmm::device_uvector<field_location> d_gc_locs(total_count * num_grandchildren, stream, mr);
          scan_repeated_message_children_kernel<<<blocks, threads, 0, stream.value()>>>(
            message_data, d_nested_row_offsets.data(), d_nested_locs.data(), total_count,
            d_gc_descs.data(), num_grandchildren, d_gc_locs.data(), d_error.data());
          
          // Copy grandchild locations to host
          std::vector<field_location> h_gc_locs(total_count * num_grandchildren);
          CUDF_CUDA_TRY(cudaMemcpyAsync(h_gc_locs.data(), d_gc_locs.data(),
                                        h_gc_locs.size() * sizeof(field_location),
                                        cudaMemcpyDeviceToHost, stream.value()));
          stream.synchronize();
          
          // Extract grandchild values
          std::vector<std::unique_ptr<cudf::column>> grandchild_cols;
          for (int gci = 0; gci < num_grandchildren; gci++) {
            int gc_schema_idx = grandchild_indices[gci];
            auto const gc_dt = schema_output_types[gc_schema_idx];
            auto const gc_enc = h_device_schema[gc_schema_idx].encoding;
            bool gc_has_def = h_device_schema[gc_schema_idx].has_default_value;
            
            switch (gc_dt.id()) {
              case cudf::type_id::INT32: {
                rmm::device_uvector<int32_t> out(total_count, stream, mr);
                rmm::device_uvector<bool> valid(total_count, stream, mr);
                int64_t def_val = gc_has_def ? default_ints[gc_schema_idx] : 0;
                if (gc_enc == spark_rapids_jni::ENC_ZIGZAG) {
                  extract_repeated_msg_child_varint_kernel<int32_t, true><<<blocks, threads, 0, stream.value()>>>(
                    message_data, d_nested_row_offsets.data(), d_nested_locs.data(),
                    d_gc_locs.data(), gci, num_grandchildren, out.data(), valid.data(),
                    total_count, d_error.data(), gc_has_def, def_val);
                } else {
                  extract_repeated_msg_child_varint_kernel<int32_t, false><<<blocks, threads, 0, stream.value()>>>(
                    message_data, d_nested_row_offsets.data(), d_nested_locs.data(),
                    d_gc_locs.data(), gci, num_grandchildren, out.data(), valid.data(),
                    total_count, d_error.data(), gc_has_def, def_val);
                }
                auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
                grandchild_cols.push_back(std::make_unique<cudf::column>(
                  gc_dt, total_count, out.release(), std::move(mask), null_count));
                break;
              }
              case cudf::type_id::INT64: {
                rmm::device_uvector<int64_t> out(total_count, stream, mr);
                rmm::device_uvector<bool> valid(total_count, stream, mr);
                int64_t def_val = gc_has_def ? default_ints[gc_schema_idx] : 0;
                extract_repeated_msg_child_varint_kernel<int64_t, false><<<blocks, threads, 0, stream.value()>>>(
                  message_data, d_nested_row_offsets.data(), d_nested_locs.data(),
                  d_gc_locs.data(), gci, num_grandchildren, out.data(), valid.data(),
                  total_count, d_error.data(), gc_has_def, def_val);
                auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
                grandchild_cols.push_back(std::make_unique<cudf::column>(
                  gc_dt, total_count, out.release(), std::move(mask), null_count));
                break;
              }
              case cudf::type_id::STRING: {
                grandchild_cols.push_back(
                  build_repeated_msg_child_string_column(
                    message_data, d_nested_row_offsets, d_nested_locs,
                    d_gc_locs, gci, num_grandchildren, total_count, d_error, stream, mr));
                break;
              }
              default:
                // Unsupported grandchild type - create null column
                grandchild_cols.push_back(make_null_column(gc_dt, total_count, stream, mr));
                break;
            }
          }
          
          // Build the nested struct column
          auto nested_struct_col = cudf::make_structs_column(
            total_count, std::move(grandchild_cols), 0, rmm::device_buffer{}, stream, mr);
          struct_children.push_back(std::move(nested_struct_col));
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
  auto struct_col = cudf::make_structs_column(total_count, std::move(struct_children), 0, rmm::device_buffer{}, stream, mr);

  // Build the list offsets column
  auto offsets_col = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, num_rows + 1, list_offs.release(), rmm::device_buffer{}, 0);

  // Build the final LIST column
  if (input_null_count > 0) {
    auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
    return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(struct_col),
                                   input_null_count, std::move(null_mask), stream, mr);
  }

  return cudf::make_lists_column(num_rows, std::move(offsets_col), std::move(struct_col), 0, rmm::device_buffer{}, stream, mr);
}

}  // anonymous namespace

std::unique_ptr<cudf::column> decode_nested_protobuf_to_struct(
  cudf::column_view const& binary_input,
  std::vector<nested_field_descriptor> const& schema,
  std::vector<cudf::data_type> const& schema_output_types,
  std::vector<int64_t> const& default_ints,
  std::vector<double> const& default_floats,
  std::vector<bool> const& default_bools,
  std::vector<std::vector<uint8_t>> const& default_strings,
  std::vector<std::vector<int32_t>> const& enum_valid_values,
  bool fail_on_errors)
{
  CUDF_EXPECTS(binary_input.type().id() == cudf::type_id::LIST,
               "binary_input must be a LIST<INT8/UINT8> column");
  cudf::lists_column_view const in_list(binary_input);
  auto const child_type = in_list.child().type().id();
  CUDF_EXPECTS(child_type == cudf::type_id::INT8 || child_type == cudf::type_id::UINT8,
               "binary_input must be a LIST<INT8/UINT8> column");

  auto const stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto num_rows = binary_input.size();
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
          CUDF_CUDA_TRY(cudaMemcpyAsync(offsets.data(), &zero, sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));
          auto offsets_col = std::make_unique<cudf::column>(
            cudf::data_type{cudf::type_id::INT32}, 1, offsets.release(), rmm::device_buffer{}, 0);
          auto empty_struct = make_empty_struct_column_with_schema(
            schema, schema_output_types, i, num_fields, stream, mr);
          empty_children.push_back(cudf::make_lists_column(0, std::move(offsets_col), std::move(empty_struct),
                                                           0, rmm::device_buffer{}, stream, mr));
        } else if (field_type.id() == cudf::type_id::STRUCT && !schema[i].is_repeated) {
          // Non-repeated nested message field
          empty_children.push_back(make_empty_struct_column_with_schema(
            schema, schema_output_types, i, num_fields, stream, mr));
        } else {
          empty_children.push_back(make_empty_column_safe(field_type, stream, mr));
        }
      }
    }
    return cudf::make_structs_column(0, std::move(empty_children), 0, rmm::device_buffer{}, stream, mr);
  }

  // Copy schema to device
  std::vector<device_nested_field_descriptor> h_device_schema(num_fields);
  for (int i = 0; i < num_fields; i++) {
    h_device_schema[i] = {
      schema[i].field_number,
      schema[i].parent_idx,
      schema[i].depth,
      schema[i].wire_type,
      static_cast<int>(schema[i].output_type),
      schema[i].encoding,
      schema[i].is_repeated,
      schema[i].is_required,
      schema[i].has_default_value
    };
  }

  rmm::device_uvector<device_nested_field_descriptor> d_schema(num_fields, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_schema.data(), h_device_schema.data(),
                                num_fields * sizeof(device_nested_field_descriptor),
                                cudaMemcpyHostToDevice, stream.value()));

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
  int num_nested = static_cast<int>(nested_field_indices.size());
  int num_scalar = static_cast<int>(scalar_field_indices.size());

  // Error flag
  rmm::device_uvector<int> d_error(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 0, sizeof(int), stream.value()));

  auto const threads = 256;
  auto const blocks = static_cast<int>((num_rows + threads - 1) / threads);

  // Allocate for counting repeated fields
  rmm::device_uvector<repeated_field_info> d_repeated_info(
    num_repeated > 0 ? static_cast<size_t>(num_rows) * num_repeated : 1, stream, mr);
  rmm::device_uvector<field_location> d_nested_locations(
    num_nested > 0 ? static_cast<size_t>(num_rows) * num_nested : 1, stream, mr);

  rmm::device_uvector<int> d_repeated_indices(num_repeated > 0 ? num_repeated : 1, stream, mr);
  rmm::device_uvector<int> d_nested_indices(num_nested > 0 ? num_nested : 1, stream, mr);

  if (num_repeated > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_repeated_indices.data(), repeated_field_indices.data(),
                                  num_repeated * sizeof(int), cudaMemcpyHostToDevice, stream.value()));
  }
  if (num_nested > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_nested_indices.data(), nested_field_indices.data(),
                                  num_nested * sizeof(int), cudaMemcpyHostToDevice, stream.value()));
  }

  // Count repeated fields at depth 0
  if (num_repeated > 0 || num_nested > 0) {
    count_repeated_fields_kernel<<<blocks, threads, 0, stream.value()>>>(
      *d_in,
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
  }

  // For scalar fields at depth 0, use the existing scan_all_fields_kernel
  // Use a map to store columns by schema index, then assemble in order at the end
  std::map<int, std::unique_ptr<cudf::column>> column_map;
  
  // Process scalar fields using existing infrastructure
  if (num_scalar > 0) {
    std::vector<field_descriptor> h_field_descs(num_scalar);
    for (int i = 0; i < num_scalar; i++) {
      int schema_idx = scalar_field_indices[i];
      h_field_descs[i].field_number = schema[schema_idx].field_number;
      h_field_descs[i].expected_wire_type = schema[schema_idx].wire_type;
    }

    rmm::device_uvector<field_descriptor> d_field_descs(num_scalar, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_field_descs.data(), h_field_descs.data(),
                                  num_scalar * sizeof(field_descriptor),
                                  cudaMemcpyHostToDevice, stream.value()));

    rmm::device_uvector<field_location> d_locations(
      static_cast<size_t>(num_rows) * num_scalar, stream, mr);

    scan_all_fields_kernel<<<blocks, threads, 0, stream.value()>>>(
      *d_in, d_field_descs.data(), num_scalar, d_locations.data(), d_error.data());

    // Extract scalar values (reusing existing extraction logic)
    cudf::lists_column_view const in_list_view(binary_input);
    auto const* message_data = reinterpret_cast<uint8_t const*>(in_list_view.child().data<int8_t>());
    auto const* list_offsets = in_list_view.offsets().data<cudf::size_type>();

    cudf::size_type base_offset = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(&base_offset, list_offsets, sizeof(cudf::size_type),
                                  cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();

    for (int i = 0; i < num_scalar; i++) {
      int schema_idx = scalar_field_indices[i];
      auto const dt = schema_output_types[schema_idx];
      auto const enc = schema[schema_idx].encoding;
      bool has_def = schema[schema_idx].has_default_value;

      switch (dt.id()) {
        case cudf::type_id::BOOL8: {
          rmm::device_uvector<uint8_t> out(num_rows, stream, mr);
          rmm::device_uvector<bool> valid(num_rows, stream, mr);
          int64_t def_val = has_def ? (default_bools[schema_idx] ? 1 : 0) : 0;
          extract_varint_from_locations_kernel<uint8_t><<<blocks, threads, 0, stream.value()>>>(
            message_data, list_offsets, base_offset, d_locations.data(), i, num_scalar,
            out.data(), valid.data(), num_rows, d_error.data(), has_def, def_val);
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          column_map[schema_idx] = std::make_unique<cudf::column>(
            dt, num_rows, out.release(), std::move(mask), null_count);
          break;
        }
        case cudf::type_id::INT32: {
          rmm::device_uvector<int32_t> out(num_rows, stream, mr);
          rmm::device_uvector<bool> valid(num_rows, stream, mr);
          int64_t def_int = has_def ? default_ints[schema_idx] : 0;
          if (enc == spark_rapids_jni::ENC_ZIGZAG) {
            extract_varint_from_locations_kernel<int32_t, true><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), i, num_scalar,
              out.data(), valid.data(), num_rows, d_error.data(), has_def, def_int);
          } else if (enc == spark_rapids_jni::ENC_FIXED) {
            extract_fixed_from_locations_kernel<int32_t, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), i, num_scalar,
              out.data(), valid.data(), num_rows, d_error.data(), has_def, static_cast<int32_t>(def_int));
          } else {
            extract_varint_from_locations_kernel<int32_t, false><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), i, num_scalar,
              out.data(), valid.data(), num_rows, d_error.data(), has_def, def_int);
          }
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          column_map[schema_idx] = std::make_unique<cudf::column>(
            dt, num_rows, out.release(), std::move(mask), null_count);
          break;
        }
        case cudf::type_id::INT64: {
          rmm::device_uvector<int64_t> out(num_rows, stream, mr);
          rmm::device_uvector<bool> valid(num_rows, stream, mr);
          int64_t def_int = has_def ? default_ints[schema_idx] : 0;
          if (enc == spark_rapids_jni::ENC_ZIGZAG) {
            extract_varint_from_locations_kernel<int64_t, true><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), i, num_scalar,
              out.data(), valid.data(), num_rows, d_error.data(), has_def, def_int);
          } else if (enc == spark_rapids_jni::ENC_FIXED) {
            extract_fixed_from_locations_kernel<int64_t, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), i, num_scalar,
              out.data(), valid.data(), num_rows, d_error.data(), has_def, def_int);
          } else {
            extract_varint_from_locations_kernel<int64_t, false><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), i, num_scalar,
              out.data(), valid.data(), num_rows, d_error.data(), has_def, def_int);
          }
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          column_map[schema_idx] = std::make_unique<cudf::column>(
            dt, num_rows, out.release(), std::move(mask), null_count);
          break;
        }
        case cudf::type_id::FLOAT32: {
          rmm::device_uvector<float> out(num_rows, stream, mr);
          rmm::device_uvector<bool> valid(num_rows, stream, mr);
          float def_float = has_def ? static_cast<float>(default_floats[schema_idx]) : 0.0f;
          extract_fixed_from_locations_kernel<float, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
            message_data, list_offsets, base_offset, d_locations.data(), i, num_scalar,
            out.data(), valid.data(), num_rows, d_error.data(), has_def, def_float);
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          column_map[schema_idx] = std::make_unique<cudf::column>(
            dt, num_rows, out.release(), std::move(mask), null_count);
          break;
        }
        case cudf::type_id::FLOAT64: {
          rmm::device_uvector<double> out(num_rows, stream, mr);
          rmm::device_uvector<bool> valid(num_rows, stream, mr);
          double def_double = has_def ? default_floats[schema_idx] : 0.0;
          extract_fixed_from_locations_kernel<double, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
            message_data, list_offsets, base_offset, d_locations.data(), i, num_scalar,
            out.data(), valid.data(), num_rows, d_error.data(), has_def, def_double);
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
          column_map[schema_idx] = std::make_unique<cudf::column>(
            dt, num_rows, out.release(), std::move(mask), null_count);
          break;
        }
        case cudf::type_id::STRING: {
          // Extract top-level STRING scalar field
          bool has_def_str = has_def && !default_strings[schema_idx].empty();
          auto const& def_str = default_strings[schema_idx];
          int32_t def_len = has_def_str ? static_cast<int32_t>(def_str.size()) : 0;

          rmm::device_uvector<uint8_t> d_default_str(def_len, stream, mr);
          if (has_def_str && def_len > 0) {
            CUDF_CUDA_TRY(cudaMemcpyAsync(d_default_str.data(), def_str.data(), def_len,
                                          cudaMemcpyHostToDevice, stream.value()));
          }

          // Extract string lengths
          rmm::device_uvector<int32_t> lengths(num_rows, stream, mr);
          extract_scalar_string_lengths_kernel<<<blocks, threads, 0, stream.value()>>>(
            d_locations.data(), i, num_scalar, lengths.data(), num_rows, has_def_str, def_len);

          // Compute offsets via prefix sum
          rmm::device_uvector<int32_t> output_offsets(num_rows + 1, stream, mr);
          thrust::exclusive_scan(rmm::exec_policy(stream), lengths.begin(), lengths.end(),
                                 output_offsets.begin(), 0);

          int32_t total_chars = 0;
          CUDF_CUDA_TRY(cudaMemcpyAsync(&total_chars, output_offsets.data() + num_rows - 1,
                                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream.value()));
          int32_t last_len = 0;
          CUDF_CUDA_TRY(cudaMemcpyAsync(&last_len, lengths.data() + num_rows - 1,
                                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream.value()));
          stream.synchronize();
          total_chars += last_len;
          CUDF_CUDA_TRY(cudaMemcpyAsync(output_offsets.data() + num_rows, &total_chars,
                                        sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));

          // Copy string data
          rmm::device_uvector<char> chars(total_chars, stream, mr);
          if (total_chars > 0) {
            copy_scalar_string_data_kernel<<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_locations.data(), i, num_scalar,
              output_offsets.data(), chars.data(), num_rows, has_def_str,
              d_default_str.data(), def_len);
          }

          // Build validity mask
          rmm::device_uvector<bool> valid(num_rows, stream, mr);
          thrust::transform(rmm::exec_policy(stream),
                            thrust::make_counting_iterator<cudf::size_type>(0),
                            thrust::make_counting_iterator<cudf::size_type>(num_rows),
                            valid.begin(),
                            [locs = d_locations.data(), i, num_scalar, has_def_str] __device__(auto row) {
                              return locs[row * num_scalar + i].offset >= 0 || has_def_str;
                            });
          auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);

          auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                            num_rows + 1, output_offsets.release(),
                                                            rmm::device_buffer{}, 0);
          column_map[schema_idx] = cudf::make_strings_column(
            num_rows, std::move(offsets_col), chars.release(), null_count, std::move(mask));
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
    std::vector<repeated_field_info> h_repeated_info(static_cast<size_t>(num_rows) * num_repeated);
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_repeated_info.data(), d_repeated_info.data(),
                                  h_repeated_info.size() * sizeof(repeated_field_info),
                                  cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();

    cudf::lists_column_view const in_list_view(binary_input);
    auto const* list_offsets = in_list_view.offsets().data<cudf::size_type>();

    for (int ri = 0; ri < num_repeated; ri++) {
      int schema_idx = repeated_field_indices[ri];
      auto element_type = schema_output_types[schema_idx];

      // Get per-row info for this repeated field
      std::vector<repeated_field_info> field_info(num_rows);
      int total_count = 0;
      for (int row = 0; row < num_rows; row++) {
        field_info[row] = h_repeated_info[row * num_repeated + ri];
        total_count += field_info[row].count;
      }

      if (total_count > 0) {
        // Build offsets for occurrence scanning
        rmm::device_uvector<int32_t> d_occ_offsets(num_rows + 1, stream, mr);
        std::vector<int32_t> h_occ_offsets(num_rows + 1);
        h_occ_offsets[0] = 0;
        for (int row = 0; row < num_rows; row++) {
          h_occ_offsets[row + 1] = h_occ_offsets[row] + field_info[row].count;
        }
        CUDF_CUDA_TRY(cudaMemcpyAsync(d_occ_offsets.data(), h_occ_offsets.data(),
                                      (num_rows + 1) * sizeof(int32_t),
                                      cudaMemcpyHostToDevice, stream.value()));

        // Scan for all occurrences
        rmm::device_uvector<repeated_occurrence> d_occurrences(total_count, stream, mr);
        scan_repeated_field_occurrences_kernel<<<blocks, threads, 0, stream.value()>>>(
          *d_in, d_schema.data(), schema_idx, 0, d_occ_offsets.data(),
          d_occurrences.data(), d_error.data());

        // Build the appropriate column type based on element type
        // For now, support scalar repeated fields
        auto child_type_id = static_cast<cudf::type_id>(h_device_schema[schema_idx].output_type_id);
        
        // The output_type in schema is the LIST type, but we need element type
        // For repeated int32, output_type should indicate the element is INT32
        switch (child_type_id) {
          case cudf::type_id::INT32:
            column_map[schema_idx] = build_repeated_scalar_column<int32_t>(
              binary_input, h_device_schema[schema_idx], field_info, d_occurrences,
              total_count, num_rows, stream, mr);
            break;
          case cudf::type_id::INT64:
            column_map[schema_idx] = build_repeated_scalar_column<int64_t>(
              binary_input, h_device_schema[schema_idx], field_info, d_occurrences,
              total_count, num_rows, stream, mr);
            break;
          case cudf::type_id::FLOAT32:
            column_map[schema_idx] = build_repeated_scalar_column<float>(
              binary_input, h_device_schema[schema_idx], field_info, d_occurrences,
              total_count, num_rows, stream, mr);
            break;
          case cudf::type_id::FLOAT64:
            column_map[schema_idx] = build_repeated_scalar_column<double>(
              binary_input, h_device_schema[schema_idx], field_info, d_occurrences,
              total_count, num_rows, stream, mr);
            break;
          case cudf::type_id::BOOL8:
            column_map[schema_idx] = build_repeated_scalar_column<uint8_t>(
              binary_input, h_device_schema[schema_idx], field_info, d_occurrences,
              total_count, num_rows, stream, mr);
            break;
          case cudf::type_id::STRING:
            column_map[schema_idx] = build_repeated_string_column(
              binary_input, h_device_schema[schema_idx], field_info, d_occurrences,
              total_count, num_rows, false, stream, mr);
            break;
          case cudf::type_id::LIST:  // bytes as LIST<INT8>
            column_map[schema_idx] = build_repeated_string_column(
              binary_input, h_device_schema[schema_idx], field_info, d_occurrences,
              total_count, num_rows, true, stream, mr);
            break;
          case cudf::type_id::STRUCT: {
            // Repeated message field - ArrayType(StructType)
            auto child_field_indices = find_child_field_indices(schema, num_fields, schema_idx);
            if (child_field_indices.empty()) {
              // No child fields - create null column
              column_map[schema_idx] = make_null_column(element_type, num_rows, stream, mr);
            } else {
              column_map[schema_idx] = build_repeated_struct_column(
                binary_input, h_device_schema[schema_idx], field_info, d_occurrences,
                total_count, num_rows, h_device_schema, child_field_indices,
                schema_output_types, default_ints, default_floats, default_bools,
                stream, mr);
            }
            break;
          }
          default:
            // Unsupported element type - create null column
            column_map[schema_idx] = make_null_column(element_type, num_rows, stream, mr);
            break;
        }
      } else {
        // All rows have count=0 - create list of empty elements
        rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
        thrust::fill(rmm::exec_policy(stream), offsets.begin(), offsets.end(), 0);
        auto offsets_col = std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT32}, num_rows + 1, offsets.release(), rmm::device_buffer{}, 0);
        
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
          auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
          column_map[schema_idx] = cudf::make_lists_column(
            num_rows, std::move(offsets_col), std::move(child_col), input_null_count, std::move(null_mask), stream, mr);
        } else {
          column_map[schema_idx] = cudf::make_lists_column(
            num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{}, stream, mr);
        }
      }
    }
  }

  // Process nested struct fields (Phase 2)
  if (num_nested > 0) {
    // Copy nested locations to host for processing
    std::vector<field_location> h_nested_locations(static_cast<size_t>(num_rows) * num_nested);
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_nested_locations.data(), d_nested_locations.data(),
                                  h_nested_locations.size() * sizeof(field_location),
                                  cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();

    cudf::lists_column_view const in_list_view(binary_input);
    auto const* message_data = reinterpret_cast<uint8_t const*>(in_list_view.child().data<int8_t>());
    auto const* list_offsets = in_list_view.offsets().data<cudf::size_type>();

    cudf::size_type base_offset = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(&base_offset, list_offsets, sizeof(cudf::size_type),
                                  cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();

    for (int ni = 0; ni < num_nested; ni++) {
      int parent_schema_idx = nested_field_indices[ni];

      // Find child fields of this nested message
      auto child_field_indices = find_child_field_indices(schema, num_fields, parent_schema_idx);

      if (child_field_indices.empty()) {
        // No child fields - create empty struct
        column_map[parent_schema_idx] = make_null_column(
          schema_output_types[parent_schema_idx], num_rows, stream, mr);
        continue;
      }

      int num_child_fields = static_cast<int>(child_field_indices.size());

      // Build field descriptors for child fields
      std::vector<field_descriptor> h_child_field_descs(num_child_fields);
      for (int i = 0; i < num_child_fields; i++) {
        int child_idx = child_field_indices[i];
        h_child_field_descs[i].field_number = schema[child_idx].field_number;
        h_child_field_descs[i].expected_wire_type = schema[child_idx].wire_type;
      }

      rmm::device_uvector<field_descriptor> d_child_field_descs(num_child_fields, stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_child_field_descs.data(), h_child_field_descs.data(),
                                    num_child_fields * sizeof(field_descriptor),
                                    cudaMemcpyHostToDevice, stream.value()));

      // Prepare parent locations for this nested field
      rmm::device_uvector<field_location> d_parent_locs(num_rows, stream, mr);
      std::vector<field_location> h_parent_locs(num_rows);
      for (int row = 0; row < num_rows; row++) {
        h_parent_locs[row] = h_nested_locations[row * num_nested + ni];
      }
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_parent_locs.data(), h_parent_locs.data(),
                                    num_rows * sizeof(field_location),
                                    cudaMemcpyHostToDevice, stream.value()));

      // Scan for child fields within nested messages
      rmm::device_uvector<field_location> d_child_locations(
        static_cast<size_t>(num_rows) * num_child_fields, stream, mr);

      scan_nested_message_fields_kernel<<<blocks, threads, 0, stream.value()>>>(
        message_data, list_offsets, base_offset, d_parent_locs.data(), num_rows,
        d_child_field_descs.data(), num_child_fields, d_child_locations.data(), d_error.data());

      // Extract child field values
      std::vector<std::unique_ptr<cudf::column>> struct_children;
      for (int ci = 0; ci < num_child_fields; ci++) {
        int child_schema_idx = child_field_indices[ci];
        auto const dt = schema_output_types[child_schema_idx];
        auto const enc = schema[child_schema_idx].encoding;
        bool has_def = schema[child_schema_idx].has_default_value;
        bool is_repeated = schema[child_schema_idx].is_repeated;

        // Check if this is a repeated field (ArrayType)
        if (is_repeated) {
          // Handle repeated field inside nested message
          auto elem_type_id = schema[child_schema_idx].output_type;
          
          // Copy child locations to host
          std::vector<field_location> h_rep_parent_locs(num_rows);
          CUDF_CUDA_TRY(cudaMemcpyAsync(h_rep_parent_locs.data(), d_parent_locs.data(),
            num_rows * sizeof(field_location), cudaMemcpyDeviceToHost, stream.value()));
          stream.synchronize();
          
          // Count repeated field occurrences for each row
          rmm::device_uvector<repeated_field_info> d_rep_info(num_rows, stream, mr);
          
          std::vector<int> rep_indices = {0};
          rmm::device_uvector<int> d_rep_indices(1, stream, mr);
          CUDF_CUDA_TRY(cudaMemcpyAsync(d_rep_indices.data(), rep_indices.data(),
            sizeof(int), cudaMemcpyHostToDevice, stream.value()));
          
          device_nested_field_descriptor rep_desc;
          rep_desc.field_number = schema[child_schema_idx].field_number;
          rep_desc.wire_type = schema[child_schema_idx].wire_type;
          rep_desc.output_type_id = static_cast<int>(schema[child_schema_idx].output_type);
          rep_desc.is_repeated = true;
          
          std::vector<device_nested_field_descriptor> h_rep_schema = {rep_desc};
          rmm::device_uvector<device_nested_field_descriptor> d_rep_schema(1, stream, mr);
          CUDF_CUDA_TRY(cudaMemcpyAsync(d_rep_schema.data(), h_rep_schema.data(),
            sizeof(device_nested_field_descriptor), cudaMemcpyHostToDevice, stream.value()));
          
          count_repeated_in_nested_kernel<<<blocks, threads, 0, stream.value()>>>(
            message_data, list_offsets, base_offset, d_parent_locs.data(), num_rows,
            d_rep_schema.data(), 1, d_rep_info.data(), 1, d_rep_indices.data(), d_error.data());
          
          std::vector<repeated_field_info> h_rep_info(num_rows);
          CUDF_CUDA_TRY(cudaMemcpyAsync(h_rep_info.data(), d_rep_info.data(),
            num_rows * sizeof(repeated_field_info), cudaMemcpyDeviceToHost, stream.value()));
          stream.synchronize();
          
          int total_rep_count = 0;
          for (int row = 0; row < num_rows; row++) {
            total_rep_count += h_rep_info[row].count;
          }
          
          if (total_rep_count == 0) {
            rmm::device_uvector<int32_t> list_offsets_vec(num_rows + 1, stream, mr);
            thrust::fill(rmm::exec_policy(stream), list_offsets_vec.begin(), list_offsets_vec.end(), 0);
            auto list_offsets_col = std::make_unique<cudf::column>(
              cudf::data_type{cudf::type_id::INT32}, num_rows + 1, list_offsets_vec.release(), rmm::device_buffer{}, 0);
            auto child_col = make_empty_column_safe(cudf::data_type{elem_type_id}, stream, mr);
            struct_children.push_back(cudf::make_lists_column(
              num_rows, std::move(list_offsets_col), std::move(child_col), 0, rmm::device_buffer{}, stream, mr));
          } else {
            rmm::device_uvector<repeated_occurrence> d_rep_occs(total_rep_count, stream, mr);
            scan_repeated_in_nested_kernel<<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_parent_locs.data(), num_rows,
              d_rep_schema.data(), 1, d_rep_info.data(), 1, d_rep_indices.data(),
              d_rep_occs.data(), d_error.data());
            
            rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
            std::vector<int32_t> h_list_offs(num_rows + 1);
            h_list_offs[0] = 0;
            for (int row = 0; row < num_rows; row++) {
              h_list_offs[row + 1] = h_list_offs[row] + h_rep_info[row].count;
            }
            CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data(), h_list_offs.data(),
              (num_rows + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));
            
            std::unique_ptr<cudf::column> child_values;
            if (elem_type_id == cudf::type_id::INT32) {
              rmm::device_uvector<int32_t> values(total_rep_count, stream, mr);
              extract_repeated_in_nested_varint_kernel<int32_t, false><<<(total_rep_count + 255) / 256, 256, 0, stream.value()>>>(
                message_data, list_offsets, base_offset, d_parent_locs.data(),
                d_rep_occs.data(), total_rep_count, values.data(), d_error.data());
              child_values = std::make_unique<cudf::column>(
                cudf::data_type{cudf::type_id::INT32}, total_rep_count, values.release(), rmm::device_buffer{}, 0);
            } else if (elem_type_id == cudf::type_id::INT64) {
              rmm::device_uvector<int64_t> values(total_rep_count, stream, mr);
              extract_repeated_in_nested_varint_kernel<int64_t, false><<<(total_rep_count + 255) / 256, 256, 0, stream.value()>>>(
                message_data, list_offsets, base_offset, d_parent_locs.data(),
                d_rep_occs.data(), total_rep_count, values.data(), d_error.data());
              child_values = std::make_unique<cudf::column>(
                cudf::data_type{cudf::type_id::INT64}, total_rep_count, values.release(), rmm::device_buffer{}, 0);
            } else if (elem_type_id == cudf::type_id::STRING) {
              std::vector<repeated_occurrence> h_rep_occs(total_rep_count);
              CUDF_CUDA_TRY(cudaMemcpyAsync(h_rep_occs.data(), d_rep_occs.data(),
                total_rep_count * sizeof(repeated_occurrence), cudaMemcpyDeviceToHost, stream.value()));
              stream.synchronize();
              
              int32_t total_chars = 0;
              std::vector<int32_t> h_str_offs(total_rep_count + 1);
              h_str_offs[0] = 0;
              for (int i = 0; i < total_rep_count; i++) {
                h_str_offs[i + 1] = h_str_offs[i] + h_rep_occs[i].length;
                total_chars += h_rep_occs[i].length;
              }
              
              rmm::device_uvector<int32_t> str_offs(total_rep_count + 1, stream, mr);
              CUDF_CUDA_TRY(cudaMemcpyAsync(str_offs.data(), h_str_offs.data(),
                (total_rep_count + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));
              
              rmm::device_uvector<char> chars(total_chars, stream, mr);
              if (total_chars > 0) {
                extract_repeated_in_nested_string_kernel<<<(total_rep_count + 255) / 256, 256, 0, stream.value()>>>(
                  message_data, list_offsets, base_offset, d_parent_locs.data(),
                  d_rep_occs.data(), total_rep_count, str_offs.data(), chars.data(), d_error.data());
              }
              
              auto str_offs_col = std::make_unique<cudf::column>(
                cudf::data_type{cudf::type_id::INT32}, total_rep_count + 1, str_offs.release(), rmm::device_buffer{}, 0);
              child_values = cudf::make_strings_column(total_rep_count, std::move(str_offs_col), chars.release(), 0, rmm::device_buffer{});
            } else {
              child_values = make_empty_column_safe(cudf::data_type{elem_type_id}, stream, mr);
            }
            
            auto list_offs_col = std::make_unique<cudf::column>(
              cudf::data_type{cudf::type_id::INT32}, num_rows + 1, list_offs.release(), rmm::device_buffer{}, 0);
            struct_children.push_back(cudf::make_lists_column(
              num_rows, std::move(list_offs_col), std::move(child_values), 0, rmm::device_buffer{}, stream, mr));
          }
          continue;  // Skip the switch statement below
        }

        switch (dt.id()) {
          case cudf::type_id::BOOL8: {
            rmm::device_uvector<uint8_t> out(num_rows, stream, mr);
            rmm::device_uvector<bool> valid(num_rows, stream, mr);
            int64_t def_val = has_def ? (default_bools[child_schema_idx] ? 1 : 0) : 0;
            extract_nested_varint_kernel<uint8_t><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_parent_locs.data(),
              d_child_locations.data(), ci, num_child_fields, out.data(), valid.data(),
              num_rows, d_error.data(), has_def, def_val);
            auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
            struct_children.push_back(std::make_unique<cudf::column>(
              dt, num_rows, out.release(), std::move(mask), null_count));
            break;
          }
          case cudf::type_id::INT32: {
            rmm::device_uvector<int32_t> out(num_rows, stream, mr);
            rmm::device_uvector<bool> valid(num_rows, stream, mr);
            int64_t def_int = has_def ? default_ints[child_schema_idx] : 0;
            if (enc == spark_rapids_jni::ENC_ZIGZAG) {
              extract_nested_varint_kernel<int32_t, true><<<blocks, threads, 0, stream.value()>>>(
                message_data, list_offsets, base_offset, d_parent_locs.data(),
                d_child_locations.data(), ci, num_child_fields, out.data(), valid.data(),
                num_rows, d_error.data(), has_def, def_int);
            } else if (enc == spark_rapids_jni::ENC_FIXED) {
              extract_nested_fixed_kernel<int32_t, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
                message_data, list_offsets, base_offset, d_parent_locs.data(),
                d_child_locations.data(), ci, num_child_fields, out.data(), valid.data(),
                num_rows, d_error.data(), has_def, static_cast<int32_t>(def_int));
            } else {
              extract_nested_varint_kernel<int32_t, false><<<blocks, threads, 0, stream.value()>>>(
                message_data, list_offsets, base_offset, d_parent_locs.data(),
                d_child_locations.data(), ci, num_child_fields, out.data(), valid.data(),
                num_rows, d_error.data(), has_def, def_int);
            }
            auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
            struct_children.push_back(std::make_unique<cudf::column>(
              dt, num_rows, out.release(), std::move(mask), null_count));
            break;
          }
          case cudf::type_id::INT64: {
            rmm::device_uvector<int64_t> out(num_rows, stream, mr);
            rmm::device_uvector<bool> valid(num_rows, stream, mr);
            int64_t def_int = has_def ? default_ints[child_schema_idx] : 0;
            if (enc == spark_rapids_jni::ENC_ZIGZAG) {
              extract_nested_varint_kernel<int64_t, true><<<blocks, threads, 0, stream.value()>>>(
                message_data, list_offsets, base_offset, d_parent_locs.data(),
                d_child_locations.data(), ci, num_child_fields, out.data(), valid.data(),
                num_rows, d_error.data(), has_def, def_int);
            } else if (enc == spark_rapids_jni::ENC_FIXED) {
              extract_nested_fixed_kernel<int64_t, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
                message_data, list_offsets, base_offset, d_parent_locs.data(),
                d_child_locations.data(), ci, num_child_fields, out.data(), valid.data(),
                num_rows, d_error.data(), has_def, def_int);
            } else {
              extract_nested_varint_kernel<int64_t, false><<<blocks, threads, 0, stream.value()>>>(
                message_data, list_offsets, base_offset, d_parent_locs.data(),
                d_child_locations.data(), ci, num_child_fields, out.data(), valid.data(),
                num_rows, d_error.data(), has_def, def_int);
            }
            auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
            struct_children.push_back(std::make_unique<cudf::column>(
              dt, num_rows, out.release(), std::move(mask), null_count));
            break;
          }
          case cudf::type_id::FLOAT32: {
            rmm::device_uvector<float> out(num_rows, stream, mr);
            rmm::device_uvector<bool> valid(num_rows, stream, mr);
            float def_float = has_def ? static_cast<float>(default_floats[child_schema_idx]) : 0.0f;
            extract_nested_fixed_kernel<float, WT_32BIT><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_parent_locs.data(),
              d_child_locations.data(), ci, num_child_fields, out.data(), valid.data(),
              num_rows, d_error.data(), has_def, def_float);
            auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
            struct_children.push_back(std::make_unique<cudf::column>(
              dt, num_rows, out.release(), std::move(mask), null_count));
            break;
          }
          case cudf::type_id::FLOAT64: {
            rmm::device_uvector<double> out(num_rows, stream, mr);
            rmm::device_uvector<bool> valid(num_rows, stream, mr);
            double def_double = has_def ? default_floats[child_schema_idx] : 0.0;
            extract_nested_fixed_kernel<double, WT_64BIT><<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_parent_locs.data(),
              d_child_locations.data(), ci, num_child_fields, out.data(), valid.data(),
              num_rows, d_error.data(), has_def, def_double);
            auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);
            struct_children.push_back(std::make_unique<cudf::column>(
              dt, num_rows, out.release(), std::move(mask), null_count));
            break;
          }
          case cudf::type_id::STRING: {
            bool has_def_str = has_def && !default_strings[child_schema_idx].empty();
            auto const& def_str = default_strings[child_schema_idx];
            int32_t def_len = has_def_str ? static_cast<int32_t>(def_str.size()) : 0;

            rmm::device_uvector<uint8_t> d_default_str(def_len, stream, mr);
            if (has_def_str && def_len > 0) {
              CUDF_CUDA_TRY(cudaMemcpyAsync(d_default_str.data(), def_str.data(), def_len,
                                            cudaMemcpyHostToDevice, stream.value()));
            }

            rmm::device_uvector<int32_t> lengths(num_rows, stream, mr);
            extract_nested_lengths_kernel<<<blocks, threads, 0, stream.value()>>>(
              d_parent_locs.data(), d_child_locations.data(), ci, num_child_fields,
              lengths.data(), num_rows, has_def_str, def_len);

            rmm::device_uvector<int32_t> output_offsets(num_rows + 1, stream, mr);
            thrust::exclusive_scan(rmm::exec_policy(stream), lengths.begin(), lengths.end(),
                                   output_offsets.begin(), 0);

            int32_t total_chars = 0;
            CUDF_CUDA_TRY(cudaMemcpyAsync(&total_chars, output_offsets.data() + num_rows - 1,
                                          sizeof(int32_t), cudaMemcpyDeviceToHost, stream.value()));
            int32_t last_len = 0;
            CUDF_CUDA_TRY(cudaMemcpyAsync(&last_len, lengths.data() + num_rows - 1,
                                          sizeof(int32_t), cudaMemcpyDeviceToHost, stream.value()));
            stream.synchronize();
            total_chars += last_len;
            CUDF_CUDA_TRY(cudaMemcpyAsync(output_offsets.data() + num_rows, &total_chars,
                                          sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));

            rmm::device_uvector<char> chars(total_chars, stream, mr);
            if (total_chars > 0) {
              copy_nested_varlen_data_kernel<<<blocks, threads, 0, stream.value()>>>(
                message_data, list_offsets, base_offset, d_parent_locs.data(),
                d_child_locations.data(), ci, num_child_fields, output_offsets.data(),
                chars.data(), num_rows, has_def_str, d_default_str.data(), def_len);
            }

            rmm::device_uvector<bool> valid(num_rows, stream, mr);
            thrust::transform(rmm::exec_policy(stream),
                              thrust::make_counting_iterator<cudf::size_type>(0),
                              thrust::make_counting_iterator<cudf::size_type>(num_rows),
                              valid.begin(),
                              [plocs = d_parent_locs.data(),
                               flocs = d_child_locations.data(),
                               ci, num_child_fields, has_def_str] __device__(auto row) {
                                return (plocs[row].offset >= 0 &&
                                        flocs[row * num_child_fields + ci].offset >= 0) || has_def_str;
                              });
            auto [mask, null_count] = make_null_mask_from_valid(valid, stream, mr);

            auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                              num_rows + 1, output_offsets.release(),
                                                              rmm::device_buffer{}, 0);
            struct_children.push_back(cudf::make_strings_column(
              num_rows, std::move(offsets_col), chars.release(), null_count, std::move(mask)));
            break;
          }
          case cudf::type_id::STRUCT: {
            // Recursively process nested struct (depth > 1)
            auto gc_indices = find_child_field_indices(schema, num_fields, child_schema_idx);
            if (gc_indices.empty()) {
              struct_children.push_back(make_null_column(dt, num_rows, stream, mr));
              break;
            }
            int num_gc = static_cast<int>(gc_indices.size());

            // Get child struct locations for grandchild scanning
            // IMPORTANT: Need to compute ABSOLUTE offsets (relative to row start)
            // d_child_locations contains offsets relative to parent message (Middle)
            // We need: child_offset_in_row = parent_offset_in_row + child_offset_in_parent
            std::vector<field_location> h_parent_locs(num_rows);
            std::vector<field_location> h_child_locs_rel(num_rows);
            CUDF_CUDA_TRY(cudaMemcpyAsync(h_parent_locs.data(), d_parent_locs.data(),
              num_rows * sizeof(field_location), cudaMemcpyDeviceToHost, stream.value()));
            for (int row = 0; row < num_rows; row++) {
              CUDF_CUDA_TRY(cudaMemcpyAsync(&h_child_locs_rel[row],
                d_child_locations.data() + row * num_child_fields + ci,
                sizeof(field_location), cudaMemcpyDeviceToHost, stream.value()));
            }
            stream.synchronize();
            
            // Compute absolute offsets
            std::vector<field_location> h_gc_parent_abs(num_rows);
            for (int row = 0; row < num_rows; row++) {
              if (h_parent_locs[row].offset >= 0 && h_child_locs_rel[row].offset >= 0) {
                // Absolute offset = parent offset + child's relative offset
                h_gc_parent_abs[row].offset = h_parent_locs[row].offset + h_child_locs_rel[row].offset;
                h_gc_parent_abs[row].length = h_child_locs_rel[row].length;
              } else {
                h_gc_parent_abs[row] = {-1, 0};
              }
            }
            
            rmm::device_uvector<field_location> d_gc_parent(num_rows, stream, mr);
            CUDF_CUDA_TRY(cudaMemcpyAsync(d_gc_parent.data(), h_gc_parent_abs.data(),
              num_rows * sizeof(field_location), cudaMemcpyHostToDevice, stream.value()));

            // Build grandchild field descriptors
            std::vector<field_descriptor> h_gc_descs(num_gc);
            for (int gi = 0; gi < num_gc; gi++) {
              h_gc_descs[gi].field_number = schema[gc_indices[gi]].field_number;
              h_gc_descs[gi].expected_wire_type = schema[gc_indices[gi]].wire_type;
            }
            rmm::device_uvector<field_descriptor> d_gc_descs(num_gc, stream, mr);
            CUDF_CUDA_TRY(cudaMemcpyAsync(d_gc_descs.data(), h_gc_descs.data(),
              num_gc * sizeof(field_descriptor), cudaMemcpyHostToDevice, stream.value()));

            // Scan for grandchild fields
            rmm::device_uvector<field_location> d_gc_locs(num_rows * num_gc, stream, mr);
            scan_nested_message_fields_kernel<<<blocks, threads, 0, stream.value()>>>(
              message_data, list_offsets, base_offset, d_gc_parent.data(), num_rows,
              d_gc_descs.data(), num_gc, d_gc_locs.data(), d_error.data());

            // Extract grandchild values (handle scalar types only)
            std::vector<std::unique_ptr<cudf::column>> gc_cols;
            for (int gi = 0; gi < num_gc; gi++) {
              int gc_idx = gc_indices[gi];
              auto gc_dt = schema_output_types[gc_idx];
              bool gc_def = schema[gc_idx].has_default_value;
              if (gc_dt.id() == cudf::type_id::INT32) {
                rmm::device_uvector<int32_t> out(num_rows, stream, mr);
                rmm::device_uvector<bool> val(num_rows, stream, mr);
                int64_t dv = gc_def ? default_ints[gc_idx] : 0;
                extract_nested_varint_kernel<int32_t, false><<<blocks, threads, 0, stream.value()>>>(
                  message_data, list_offsets, base_offset, d_gc_parent.data(),
                  d_gc_locs.data(), gi, num_gc, out.data(), val.data(), num_rows, d_error.data(), gc_def, dv);
                auto [m, nc] = make_null_mask_from_valid(val, stream, mr);
                gc_cols.push_back(std::make_unique<cudf::column>(gc_dt, num_rows, out.release(), std::move(m), nc));
              } else {
                gc_cols.push_back(make_null_column(gc_dt, num_rows, stream, mr));
              }
            }

            // Build nested struct validity
            rmm::device_uvector<bool> ns_valid(num_rows, stream, mr);
            thrust::transform(rmm::exec_policy(stream),
              thrust::make_counting_iterator<cudf::size_type>(0),
              thrust::make_counting_iterator<cudf::size_type>(num_rows), ns_valid.begin(),
              [p = d_parent_locs.data(), c = d_child_locations.data(), ci, ncf = num_child_fields] __device__(auto r) {
                return p[r].offset >= 0 && c[r * ncf + ci].offset >= 0;
              });
            auto [ns_mask, ns_nc] = make_null_mask_from_valid(ns_valid, stream, mr);
            struct_children.push_back(cudf::make_structs_column(num_rows, std::move(gc_cols), ns_nc, std::move(ns_mask), stream, mr));
            break;
          }
          default:
            // For unsupported types, create null columns
            struct_children.push_back(make_null_column(dt, num_rows, stream, mr));
            break;
        }
      }

      // Build struct validity based on parent location
      rmm::device_uvector<bool> struct_valid(num_rows, stream, mr);
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator<cudf::size_type>(0),
                        thrust::make_counting_iterator<cudf::size_type>(num_rows),
                        struct_valid.begin(),
                        [plocs = d_parent_locs.data()] __device__(auto row) {
                          return plocs[row].offset >= 0;
                        });
      auto [struct_mask, struct_null_count] = make_null_mask_from_valid(struct_valid, stream, mr);

      column_map[parent_schema_idx] = cudf::make_structs_column(
        num_rows, std::move(struct_children), struct_null_count, std::move(struct_mask), stream, mr);
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
        // Field not processed - create null column
        top_level_children.push_back(make_null_column(schema_output_types[i], num_rows, stream, mr));
      }
    }
  }

  // Check for errors
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  int h_error = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&h_error, d_error.data(), sizeof(int), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  if (fail_on_errors) {
    CUDF_EXPECTS(h_error == 0, "Malformed protobuf message or unsupported wire type");
  }

  return cudf::make_structs_column(num_rows, std::move(top_level_children), 0, rmm::device_buffer{}, stream, mr);
}

}  // namespace spark_rapids_jni
