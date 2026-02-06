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
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
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
// Single-pass decoder data structures
// ============================================================================

/// Maximum nesting depth for single-pass decoder
constexpr int SP_MAX_DEPTH = 10;

/// Maximum number of counted columns (repeated fields at all depths)
constexpr int SP_MAX_COUNTED = 128;

/// Maximum number of output columns
constexpr int SP_MAX_OUTPUT_COLS = 512;

/// Message type descriptor: groups fields belonging to the same protobuf message
struct sp_msg_type {
  int first_field_idx;          // Start index in the global sp_field_entry array
  int num_fields;               // Number of direct child fields
  int lookup_offset;            // Offset into d_field_lookup table (-1 if not using lookup)
  int max_field_number;         // Max field number + 1 (size of lookup region)
};

/// Field entry for single-pass decoder (device-side, sorted by field_number per msg type)
struct sp_field_entry {
  int field_number;             // Protobuf field number
  int wire_type;                // Expected wire type
  int output_type_id;           // cudf type_id cast to int (-1 for struct containers)
  int encoding;                 // ENC_DEFAULT / ENC_FIXED / ENC_ZIGZAG
  int child_msg_type;           // For nested messages: index into sp_msg_type (-1 otherwise)
  int col_idx;                  // Index into output column descriptors (-1 for containers)
  int count_idx;                // For repeated fields: index into per-row count array (-1 if not)
  bool is_repeated;             // Whether this field is repeated
  bool has_default;             // Whether this field has a default value
  int64_t default_int;          // Default value for int/long/bool
  double default_float;         // Default value for float/double
};

/// Stack entry for nested message parsing within a kernel thread
struct sp_stack_entry {
  int parent_end_offset;        // End offset of parent message (relative to row start)
  int msg_type_idx;             // Saved message type index
  int write_base;               // Saved write base for non-repeated children
};

/// Output column descriptor (device-side, used during Pass 2)
struct sp_col_desc {
  void* data;                   // Typed data buffer (or string_index_pair* for strings)
  bool* validity;               // Validity buffer (one bool per element)
};

/// Pair for zero-copy string references (device-side)
struct sp_string_pair {
  char const* ptr;              // Pointer into message data (null if not found)
  int32_t length;               // String length in bytes (0 if not found)
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
 * Kernel to extract string data from repeated message child fields.
 * Copies all strings in parallel on the GPU instead of per-string host copies.
 */
__global__ void extract_repeated_msg_child_strings_kernel(
  uint8_t const* message_data,
  int32_t const* msg_row_offsets,
  field_location const* msg_locs,
  field_location const* child_locs,
  int child_idx,
  int num_child_fields,
  int32_t const* string_offsets,  // Output offsets (exclusive scan of lengths)
  char* output_chars,
  bool* valid,
  int total_count)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_count) return;

  auto const& field_loc = child_locs[idx * num_child_fields + child_idx];
  
  if (field_loc.offset < 0 || field_loc.length == 0) {
    valid[idx] = false;
    return;
  }
  
  valid[idx] = true;
  
  int32_t row_offset = msg_row_offsets[idx];
  int32_t msg_offset = msg_locs[idx].offset;
  uint8_t const* str_src = message_data + row_offset + msg_offset + field_loc.offset;
  char* str_dst = output_chars + string_offsets[idx];
  
  // Copy string data
  for (int i = 0; i < field_loc.length; i++) {
    str_dst[i] = static_cast<char>(str_src[i]);
  }
}

/**
 * Kernel to compute string lengths from child field locations.
 */
__global__ void compute_string_lengths_kernel(
  field_location const* child_locs,
  int child_idx,
  int num_child_fields,
  int32_t* lengths,
  int total_count)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_count) return;

  auto const& loc = child_locs[idx * num_child_fields + child_idx];
  lengths[idx] = (loc.offset >= 0) ? loc.length : 0;
}

/**
 * Helper to build string column for repeated message child fields.
 * Uses GPU kernels for parallel string extraction (critical performance fix!).
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

  auto const threads = 256;
  auto const blocks = (total_count + threads - 1) / threads;

  // Compute string lengths on GPU
  rmm::device_uvector<int32_t> d_lengths(total_count, stream, mr);
  compute_string_lengths_kernel<<<blocks, threads, 0, stream.value()>>>(
    d_child_locs.data(), child_idx, num_child_fields, d_lengths.data(), total_count);

  // Compute offsets via exclusive scan
  rmm::device_uvector<int32_t> d_str_offsets(total_count + 1, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy(stream), 
                         d_lengths.begin(), d_lengths.end(), 
                         d_str_offsets.begin(), 0);
  
  // Get total chars count
  int32_t total_chars = 0;
  int32_t last_len = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&total_chars, d_str_offsets.data() + total_count - 1,
                                sizeof(int32_t), cudaMemcpyDeviceToHost, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(&last_len, d_lengths.data() + total_count - 1,
                                sizeof(int32_t), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  total_chars += last_len;
  
  // Set final offset
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_str_offsets.data() + total_count, &total_chars,
                                sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));

  // Allocate output chars and validity
  rmm::device_uvector<char> d_chars(total_chars, stream, mr);
  rmm::device_uvector<bool> d_valid(total_count, stream, mr);

  // Extract all strings in parallel on GPU (critical performance fix!)
  if (total_chars > 0) {
    extract_repeated_msg_child_strings_kernel<<<blocks, threads, 0, stream.value()>>>(
      message_data, d_msg_row_offsets.data(), d_msg_locs.data(),
      d_child_locs.data(), child_idx, num_child_fields,
      d_str_offsets.data(), d_chars.data(), d_valid.data(), total_count);
  } else {
    // No strings, just set validity
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(total_count),
                      d_valid.begin(),
                      [child_locs = d_child_locs.data(), ci = child_idx, ncf = num_child_fields] __device__(int idx) {
                        return child_locs[idx * ncf + ci].offset >= 0;
                      });
  }

  auto [mask, null_count] = make_null_mask_from_valid(d_valid, stream, mr);

  auto str_offsets_col = std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32}, total_count + 1, d_str_offsets.release(), rmm::device_buffer{}, 0);
  return cudf::make_strings_column(total_count, std::move(str_offsets_col), d_chars.release(), null_count, std::move(mask));
}

/**
 * Kernel to compute nested struct locations from child field locations.
 * Replaces host-side loop that was copying data D->H, processing, then H->D.
 * This is a critical performance optimization.
 */
__global__ void compute_nested_struct_locations_kernel(
  field_location const* child_locs,     // Child field locations from parent scan
  field_location const* msg_locs,       // Parent message locations
  int32_t const* msg_row_offsets,       // Parent message row offsets
  int child_idx,                        // Which child field is the nested struct
  int num_child_fields,                 // Total number of child fields per occurrence
  field_location* nested_locs,          // Output: nested struct locations
  int32_t* nested_row_offsets,          // Output: nested struct row offsets
  int total_count)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_count) return;
  
  // Get the nested struct location from child_locs
  nested_locs[idx] = child_locs[idx * num_child_fields + child_idx];
  // Compute absolute row offset = msg_row_offset + msg_offset
  nested_row_offsets[idx] = msg_row_offsets[idx] + msg_locs[idx].offset;
}

/**
 * Kernel to compute absolute grandchild parent locations from parent and child locations.
 * Computes: gc_parent_abs[i] = parent[i].offset + child[i * ncf + ci].offset
 * This replaces host-side loop with D->H->D copy pattern.
 */
__global__ void compute_grandchild_parent_locations_kernel(
  field_location const* parent_locs,     // Parent locations (row count)
  field_location const* child_locs,      // Child locations (row * num_child_fields)
  int child_idx,                         // Which child field
  int num_child_fields,                  // Total child fields per row
  field_location* gc_parent_abs,         // Output: absolute grandchild parent locations
  int num_rows)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  
  auto const& parent_loc = parent_locs[row];
  auto const& child_loc = child_locs[row * num_child_fields + child_idx];
  
  if (parent_loc.offset >= 0 && child_loc.offset >= 0) {
    // Absolute offset = parent offset + child's relative offset
    gc_parent_abs[row].offset = parent_loc.offset + child_loc.offset;
    gc_parent_abs[row].length = child_loc.length;
  } else {
    gc_parent_abs[row] = {-1, 0};
  }
}

/**
 * Kernel to compute message locations and row offsets from repeated occurrences.
 * Replaces host-side loop that processed occurrences.
 */
__global__ void compute_msg_locations_from_occurrences_kernel(
  repeated_occurrence const* occurrences,  // Repeated field occurrences
  cudf::size_type const* list_offsets,     // List offsets for rows
  cudf::size_type base_offset,             // Base offset to subtract
  field_location* msg_locs,                // Output: message locations
  int32_t* msg_row_offsets,                // Output: message row offsets
  int total_count)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_count) return;
  
  auto const& occ = occurrences[idx];
  msg_row_offsets[idx] = static_cast<int32_t>(list_offsets[occ.row_idx] - base_offset);
  msg_locs[idx] = {occ.offset, occ.length};
}

/**
 * Functor to extract count from repeated_field_info with strided access.
 * Used for extracting counts for a specific repeated field from 2D array.
 */
struct extract_strided_count {
  repeated_field_info const* info;
  int field_idx;
  int num_fields;
  
  __device__ int32_t operator()(int row) const {
    return info[row * num_fields + field_idx].count;
  }
};

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

  // Build list offsets from counts entirely on GPU (performance fix!)
  // Copy h_repeated_info to device and use thrust::transform to extract counts
  rmm::device_uvector<repeated_field_info> d_rep_info(num_rows, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_rep_info.data(), h_repeated_info.data(),
                                num_rows * sizeof(repeated_field_info),
                                cudaMemcpyHostToDevice, stream.value()));
  
  rmm::device_uvector<int32_t> counts(num_rows, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    d_rep_info.begin(), d_rep_info.end(),
                    counts.begin(),
                    [] __device__(repeated_field_info const& info) { return info.count; });

  rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy(stream), counts.begin(), counts.end(), list_offs.begin(), 0);
  
  // Set last offset = total_count
  CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data() + num_rows, &total_count,
                                sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));

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

  // Build list offsets from counts entirely on GPU (performance fix!)
  // Copy h_repeated_info to device and use thrust::transform to extract counts
  rmm::device_uvector<repeated_field_info> d_rep_info(num_rows, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_rep_info.data(), h_repeated_info.data(),
                                num_rows * sizeof(repeated_field_info),
                                cudaMemcpyHostToDevice, stream.value()));
  
  rmm::device_uvector<int32_t> counts(num_rows, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    d_rep_info.begin(), d_rep_info.end(),
                    counts.begin(),
                    [] __device__(repeated_field_info const& info) { return info.count; });

  rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy(stream), counts.begin(), counts.end(), list_offs.begin(), 0);
  
  // Set last offset = total_count
  CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data() + num_rows, &total_count,
                                sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));

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

  // Build list offsets from counts entirely on GPU (performance fix!)
  // Copy repeated_info to device and use thrust::transform to extract counts
  rmm::device_uvector<repeated_field_info> d_rep_info(num_rows, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_rep_info.data(), h_repeated_info.data(),
                                num_rows * sizeof(repeated_field_info),
                                cudaMemcpyHostToDevice, stream.value()));
  
  rmm::device_uvector<int32_t> counts(num_rows, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    d_rep_info.begin(), d_rep_info.end(),
                    counts.begin(),
                    [] __device__(repeated_field_info const& info) { return info.count; });
  
  rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy(stream), counts.begin(), counts.end(), list_offs.begin(), 0);
  
  // Set last offset = total_count (already computed on caller side)
  CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data() + num_rows, &total_count,
                                sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));

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
  // Create "virtual" parent locations from the occurrences using GPU kernel
  // This replaces the host-side loop with D->H->D copy pattern (critical performance fix!)
  rmm::device_uvector<field_location> d_msg_locs(total_count, stream, mr);
  rmm::device_uvector<int32_t> d_msg_row_offsets(total_count, stream, mr);
  {
    auto const occ_threads = 256;
    auto const occ_blocks = (total_count + occ_threads - 1) / occ_threads;
    compute_msg_locations_from_occurrences_kernel<<<occ_blocks, occ_threads, 0, stream.value()>>>(
      d_occurrences.data(), list_offsets, base_offset,
      d_msg_locs.data(), d_msg_row_offsets.data(), total_count);
  }

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

  // Note: We no longer need to copy child_locs to host because:
  // 1. All scalar extraction kernels access d_child_locs directly on device
  // 2. String extraction uses GPU kernels
  // 3. Nested struct locations are computed on GPU via compute_nested_struct_locations_kernel

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
          
          // Create nested struct locations from child_locs using GPU kernel
          // This eliminates the D->H->D copy pattern (critical performance optimization)
          rmm::device_uvector<field_location> d_nested_locs(total_count, stream, mr);
          rmm::device_uvector<int32_t> d_nested_row_offsets(total_count, stream, mr);
          compute_nested_struct_locations_kernel<<<blocks, threads, 0, stream.value()>>>(
            d_child_locs.data(), d_msg_locs.data(), d_msg_row_offsets.data(),
            ci, num_child_fields, d_nested_locs.data(), d_nested_row_offsets.data(), total_count);
          
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

// ============================================================================
// Single-Pass Decoder Implementation
// ============================================================================

/**
 * O(1) field lookup using direct-mapped table.
 * d_field_lookup[msg_type.lookup_offset + field_number] = field_entry index, or -1.
 */
__device__ inline int sp_lookup_field(
  sp_msg_type const* msg_types,
  sp_field_entry const* /*field_entries*/,
  int const* d_field_lookup,
  int msg_type_idx,
  int field_number)
{
  auto const& mt = msg_types[msg_type_idx];
  if (field_number < 0 || field_number >= mt.max_field_number) return -1;
  return d_field_lookup[mt.lookup_offset + field_number];
}

/**
 * Write an extracted scalar value to the output column.
 * cur is advanced past the consumed bytes.
 */
__device__ inline void sp_write_scalar(
  uint8_t const*& cur,
  uint8_t const* end,
  sp_field_entry const& fe,
  sp_col_desc* col_descs,
  int write_pos)
{
  if (fe.col_idx < 0) return;
  auto& cd = col_descs[fe.col_idx];

  if (fe.wire_type == WT_VARINT) {
    uint64_t val; int vb;
    if (!read_varint(cur, end, val, vb)) return;
    cur += vb;
    if (fe.encoding == spark_rapids_jni::ENC_ZIGZAG) {
      val = (val >> 1) ^ (-(val & 1));
    }
    int tid = fe.output_type_id;
    if (tid == static_cast<int>(cudf::type_id::BOOL8))
      reinterpret_cast<uint8_t*>(cd.data)[write_pos] = val ? 1 : 0;
    else if (tid == static_cast<int>(cudf::type_id::INT32))
      reinterpret_cast<int32_t*>(cd.data)[write_pos] = static_cast<int32_t>(val);
    else if (tid == static_cast<int>(cudf::type_id::UINT32))
      reinterpret_cast<uint32_t*>(cd.data)[write_pos] = static_cast<uint32_t>(val);
    else if (tid == static_cast<int>(cudf::type_id::INT64))
      reinterpret_cast<int64_t*>(cd.data)[write_pos] = static_cast<int64_t>(val);
    else if (tid == static_cast<int>(cudf::type_id::UINT64))
      reinterpret_cast<uint64_t*>(cd.data)[write_pos] = val;
    cd.validity[write_pos] = true;

  } else if (fe.wire_type == WT_32BIT) {
    if (end - cur < 4) return;
    uint32_t raw = load_le<uint32_t>(cur);
    cur += 4;
    int tid = fe.output_type_id;
    if (tid == static_cast<int>(cudf::type_id::FLOAT32)) {
      float f; memcpy(&f, &raw, 4);
      reinterpret_cast<float*>(cd.data)[write_pos] = f;
    } else {
      reinterpret_cast<int32_t*>(cd.data)[write_pos] = static_cast<int32_t>(raw);
    }
    cd.validity[write_pos] = true;

  } else if (fe.wire_type == WT_64BIT) {
    if (end - cur < 8) return;
    uint64_t raw = load_le<uint64_t>(cur);
    cur += 8;
    int tid = fe.output_type_id;
    if (tid == static_cast<int>(cudf::type_id::FLOAT64)) {
      double d; memcpy(&d, &raw, 8);
      reinterpret_cast<double*>(cd.data)[write_pos] = d;
    } else {
      reinterpret_cast<int64_t*>(cd.data)[write_pos] = static_cast<int64_t>(raw);
    }
    cd.validity[write_pos] = true;

  } else if (fe.wire_type == WT_LEN) {
    // String / bytes
    uint64_t len; int lb;
    if (!read_varint(cur, end, len, lb)) return;
    auto* pairs = reinterpret_cast<sp_string_pair*>(cd.data);
    pairs[write_pos].ptr = reinterpret_cast<char const*>(cur + lb);
    pairs[write_pos].length = static_cast<int32_t>(len);
    cd.validity[write_pos] = true;
    cur += lb + static_cast<int>(len);
  }
}

/**
 * Count the number of packed elements in a length-delimited blob for a given element wire type.
 */
__device__ inline int sp_count_packed(
  uint8_t const* data, int data_len, int elem_wire_type)
{
  if (elem_wire_type == WT_VARINT) {
    int count = 0;
    uint8_t const* p = data;
    uint8_t const* pe = data + data_len;
    while (p < pe) {
      while (p < pe && (*p & 0x80u)) p++;
      if (p < pe) { p++; count++; }
    }
    return count;
  } else if (elem_wire_type == WT_32BIT) {
    return data_len / 4;
  } else if (elem_wire_type == WT_64BIT) {
    return data_len / 8;
  }
  return 0;
}

// ============================================================================
// Pass 1: Unified Count Kernel
// Walks each message once, counting all repeated fields at all depths.
// ============================================================================

__global__ void sp_unified_count_kernel(
  cudf::column_device_view const d_in,
  sp_msg_type const* msg_types,
  sp_field_entry const* fields,
  int const* d_field_lookup,
  int32_t* d_counts,            // [num_rows * num_count_cols]
  int num_count_cols,
  int* error_flag)
{
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= d_in.size()) return;

  auto const in = cudf::detail::lists_column_device_view(d_in);
  auto const& child = in.child();
  auto const base = in.offsets().element<cudf::size_type>(in.offset());
  auto const start = in.offset_at(row) - base;
  auto const stop  = in.offset_at(row + 1) - base;
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());

  // Local counters for each counted column (repeated fields)
  int32_t local_counts[SP_MAX_COUNTED];
  for (int i = 0; i < num_count_cols && i < SP_MAX_COUNTED; i++) local_counts[i] = 0;

  // Stack for nested message parsing
  sp_stack_entry stack[SP_MAX_DEPTH];
  int depth = 0;
  int msg_type = 0;  // Root message type

  uint8_t const* cur = bytes + start;
  uint8_t const* end_ptr = bytes + stop;

  while (cur < end_ptr || depth > 0) {
    if (cur >= end_ptr) {
      if (depth <= 0) break;
      depth--;
      cur = end_ptr;
      end_ptr = bytes + start + stack[depth].parent_end_offset;
      msg_type = stack[depth].msg_type_idx;
      continue;
    }

    // Read tag
    uint64_t key; int kb;
    if (!read_varint(cur, end_ptr, key, kb)) { atomicExch(error_flag, 1); break; }
    cur += kb;
    int fn = static_cast<int>(key >> 3);
    int wt = static_cast<int>(key & 0x7);

    int fi = sp_lookup_field(msg_types, fields, d_field_lookup, msg_type, fn);

    if (fi < 0) {
      // Unknown field - skip
      uint8_t const* next;
      if (!skip_field(cur, end_ptr, wt, next)) { atomicExch(error_flag, 1); break; }
      cur = next;
      continue;
    }

    auto const& fe = fields[fi];

    // Check for packed encoding (repeated + WT_LEN but element is not LEN)
    if (fe.is_repeated && wt == WT_LEN && fe.wire_type != WT_LEN && fe.count_idx >= 0) {
      uint64_t len; int lb;
      if (!read_varint(cur, end_ptr, len, lb)) { atomicExch(error_flag, 1); break; }
      int packed_len = static_cast<int>(len);
      local_counts[fe.count_idx] += sp_count_packed(cur + lb, packed_len, fe.wire_type);
      cur += lb + packed_len;
      continue;
    }

    // Wire type mismatch - skip
    if (wt != fe.wire_type) {
      uint8_t const* next;
      if (!skip_field(cur, end_ptr, wt, next)) { atomicExch(error_flag, 1); break; }
      cur = next;
      continue;
    }

    // Nested message field
    if (fe.child_msg_type >= 0 && wt == WT_LEN) {
      uint64_t len; int lb;
      if (!read_varint(cur, end_ptr, len, lb)) { atomicExch(error_flag, 1); break; }
      cur += lb;
      int sub_end = static_cast<int>((cur + static_cast<int>(len)) - (bytes + start));

      if (fe.is_repeated && fe.count_idx >= 0) {
        local_counts[fe.count_idx]++;
      }

      if (depth < SP_MAX_DEPTH) {
        stack[depth] = {static_cast<int>(end_ptr - (bytes + start)), msg_type, 0};
        depth++;
        end_ptr = bytes + start + sub_end;
        msg_type = fe.child_msg_type;
      } else {
        // Max depth exceeded - skip sub-message
        cur += static_cast<int>(len);
      }
      continue;
    }

    // Repeated non-message field
    if (fe.is_repeated && fe.count_idx >= 0) {
      local_counts[fe.count_idx]++;
    }

    // Skip field value
    uint8_t const* next;
    if (!skip_field(cur, end_ptr, wt, next)) { atomicExch(error_flag, 1); break; }
    cur = next;
  }

  // Write counts to global memory
  for (int i = 0; i < num_count_cols && i < SP_MAX_COUNTED; i++) {
    d_counts[static_cast<size_t>(row) * num_count_cols + i] = local_counts[i];
  }
}

// ============================================================================
// Pass 2: Unified Extract Kernel
// Walks each message once, extracting all field values at all depths.
// ============================================================================

__global__ void sp_unified_extract_kernel(
  cudf::column_device_view const d_in,
  sp_msg_type const* msg_types,
  sp_field_entry const* fields,
  int const* d_field_lookup,
  sp_col_desc* col_descs,
  int32_t const* d_row_offsets,   // [num_rows * num_count_cols] - per-row write offsets
  int32_t* const* d_parent_bufs,  // [num_count_cols] - parent index buffers (null if not inner)
  int num_count_cols,
  int* error_flag)
{
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= d_in.size()) return;

  auto const in = cudf::detail::lists_column_device_view(d_in);
  auto const& child = in.child();
  auto const base = in.offsets().element<cudf::size_type>(in.offset());
  auto const start = in.offset_at(row) - base;
  auto const stop  = in.offset_at(row + 1) - base;
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());

  // Local write counters (initialized from row offsets)
  int32_t local_counter[SP_MAX_COUNTED];
  for (int i = 0; i < num_count_cols && i < SP_MAX_COUNTED; i++) {
    local_counter[i] = d_row_offsets[static_cast<size_t>(row) * num_count_cols + i];
  }

  sp_stack_entry stack[SP_MAX_DEPTH];
  int depth = 0;
  int msg_type = 0;
  int write_base = row;   // Write position for non-repeated children

  uint8_t const* cur = bytes + start;
  uint8_t const* end_ptr = bytes + stop;

  while (cur < end_ptr || depth > 0) {
    if (cur >= end_ptr) {
      if (depth <= 0) break;
      depth--;
      cur = end_ptr;
      end_ptr = bytes + start + stack[depth].parent_end_offset;
      msg_type = stack[depth].msg_type_idx;
      write_base = stack[depth].write_base;
      continue;
    }

    // Read tag
    uint64_t key; int kb;
    if (!read_varint(cur, end_ptr, key, kb)) { atomicExch(error_flag, 1); break; }
    cur += kb;
    int fn = static_cast<int>(key >> 3);
    int wt = static_cast<int>(key & 0x7);

    int fi = sp_lookup_field(msg_types, fields, d_field_lookup, msg_type, fn);

    if (fi < 0) {
      uint8_t const* next;
      if (!skip_field(cur, end_ptr, wt, next)) { atomicExch(error_flag, 1); break; }
      cur = next;
      continue;
    }

    auto const& fe = fields[fi];

    // Packed encoding for repeated scalars
    if (fe.is_repeated && wt == WT_LEN && fe.wire_type != WT_LEN && fe.count_idx >= 0) {
      uint64_t len; int lb;
      if (!read_varint(cur, end_ptr, len, lb)) { atomicExch(error_flag, 1); break; }
      uint8_t const* pstart = cur + lb;
      uint8_t const* pend = pstart + static_cast<int>(len);
      uint8_t const* p = pstart;

      while (p < pend) {
        int pos = local_counter[fe.count_idx]++;
        if (d_parent_bufs && d_parent_bufs[fe.count_idx]) {
          d_parent_bufs[fe.count_idx][pos] = write_base;
        }
        if (fe.col_idx >= 0) {
          auto& cd = col_descs[fe.col_idx];
          if (fe.wire_type == WT_VARINT) {
            uint64_t val; int vb;
            if (!read_varint(p, pend, val, vb)) break;
            p += vb;
            if (fe.encoding == spark_rapids_jni::ENC_ZIGZAG) val = (val >> 1) ^ (-(val & 1));
            int tid = fe.output_type_id;
            if (tid == static_cast<int>(cudf::type_id::BOOL8))
              reinterpret_cast<uint8_t*>(cd.data)[pos] = val ? 1 : 0;
            else if (tid == static_cast<int>(cudf::type_id::INT32))
              reinterpret_cast<int32_t*>(cd.data)[pos] = static_cast<int32_t>(val);
            else if (tid == static_cast<int>(cudf::type_id::UINT32))
              reinterpret_cast<uint32_t*>(cd.data)[pos] = static_cast<uint32_t>(val);
            else if (tid == static_cast<int>(cudf::type_id::INT64))
              reinterpret_cast<int64_t*>(cd.data)[pos] = static_cast<int64_t>(val);
            else if (tid == static_cast<int>(cudf::type_id::UINT64))
              reinterpret_cast<uint64_t*>(cd.data)[pos] = val;
            cd.validity[pos] = true;
          } else if (fe.wire_type == WT_32BIT) {
            if (pend - p < 4) break;
            uint32_t raw = load_le<uint32_t>(p); p += 4;
            if (fe.output_type_id == static_cast<int>(cudf::type_id::FLOAT32)) {
              float f; memcpy(&f, &raw, 4);
              reinterpret_cast<float*>(cd.data)[pos] = f;
            } else {
              reinterpret_cast<int32_t*>(cd.data)[pos] = static_cast<int32_t>(raw);
            }
            cd.validity[pos] = true;
          } else if (fe.wire_type == WT_64BIT) {
            if (pend - p < 8) break;
            uint64_t raw = load_le<uint64_t>(p); p += 8;
            if (fe.output_type_id == static_cast<int>(cudf::type_id::FLOAT64)) {
              double d; memcpy(&d, &raw, 8);
              reinterpret_cast<double*>(cd.data)[pos] = d;
            } else {
              reinterpret_cast<int64_t*>(cd.data)[pos] = static_cast<int64_t>(raw);
            }
            cd.validity[pos] = true;
          }
        }
      }
      cur = pend;
      continue;
    }

    // Wire type mismatch - skip
    if (wt != fe.wire_type) {
      uint8_t const* next;
      if (!skip_field(cur, end_ptr, wt, next)) { atomicExch(error_flag, 1); break; }
      cur = next;
      continue;
    }

    // Nested message
    if (fe.child_msg_type >= 0 && wt == WT_LEN) {
      uint64_t len; int lb;
      if (!read_varint(cur, end_ptr, len, lb)) { atomicExch(error_flag, 1); break; }
      cur += lb;
      int sub_end = static_cast<int>((cur + static_cast<int>(len)) - (bytes + start));

      int new_write_base = write_base;
      if (fe.is_repeated && fe.count_idx >= 0) {
        int p_pos = local_counter[fe.count_idx]++;
        if (d_parent_bufs && d_parent_bufs[fe.count_idx]) {
          d_parent_bufs[fe.count_idx][p_pos] = write_base;
        }
        new_write_base = p_pos;
      }
      // Set struct validity if we have a col_idx
      if (fe.col_idx >= 0) {
        col_descs[fe.col_idx].validity[new_write_base] = true;
      }

      if (depth < SP_MAX_DEPTH) {
        stack[depth] = {static_cast<int>(end_ptr - (bytes + start)), msg_type, write_base};
        depth++;
        end_ptr = bytes + start + sub_end;
        msg_type = fe.child_msg_type;
        write_base = new_write_base;
      } else {
        cur += static_cast<int>(len);
      }
      continue;
    }

    // Non-message field: extract value
    if (fe.is_repeated && fe.count_idx >= 0) {
      int pos = local_counter[fe.count_idx]++;
      if (d_parent_bufs && d_parent_bufs[fe.count_idx]) {
        d_parent_bufs[fe.count_idx][pos] = write_base;
      }
      sp_write_scalar(cur, end_ptr, fe, col_descs, pos);
    } else {
      // Non-repeated: write at write_base (last one wins on overwrite)
      sp_write_scalar(cur, end_ptr, fe, col_descs, write_base);
    }
  }
}

// ============================================================================
// Fused prefix sum + list offsets kernels (replaces per-column thrust loops)
// ============================================================================

/**
 * Compute exclusive prefix sums for ALL count columns in a single kernel launch.
 * One thread per count column - each thread serially scans its column.
 * Also writes the per-column totals and builds list offsets (num_rows+1).
 */
__global__ void sp_compute_offsets_kernel(
  int32_t const* d_counts,       // [num_rows × num_count_cols] row-major
  int32_t* d_row_offsets,        // [num_rows × num_count_cols] row-major output
  int32_t* d_totals,             // [num_count_cols] output
  int32_t** d_list_offs_ptrs,    // [num_count_cols] pointers to list offset buffers (num_rows+1 each)
  int num_rows,
  int num_count_cols)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_count_cols) return;

  int32_t* list_offs = d_list_offs_ptrs[c];
  int32_t sum = 0;
  for (int r = 0; r < num_rows; r++) {
    auto idx = static_cast<size_t>(r) * num_count_cols + c;
    int32_t val = d_counts[idx];
    d_row_offsets[idx] = sum;
    if (list_offs) list_offs[r] = sum;
    sum += val;
  }
  d_totals[c] = sum;
  if (list_offs) list_offs[num_rows] = sum;
}

// ============================================================================
// Host-side helpers for single-pass decoder
// ============================================================================

/// Host-side column info for assembly
struct sp_host_col_info {
  int schema_idx;
  int col_idx;             // col_idx in sp_col_desc (-1 for repeated struct containers)
  int count_idx;           // For repeated fields (-1 otherwise)
  int parent_count_idx;    // count_idx of nearest repeated ancestor (-1 for top-level)
  cudf::type_id type_id;
  bool is_repeated;
  bool is_string;
  int parent_schema_idx;   // -1 for top-level
};

/**
 * Build single-pass schema from nested_field_descriptor arrays.
 * Produces message type tables, field entries, and column info.
 */
void build_single_pass_schema(
  std::vector<nested_field_descriptor> const& schema,
  std::vector<cudf::data_type> const& schema_output_types,
  std::vector<int64_t> const& default_ints,
  std::vector<double> const& default_floats,
  std::vector<bool> const& default_bools,
  // Outputs:
  std::vector<sp_msg_type>& msg_types,
  std::vector<sp_field_entry>& field_entries,
  std::vector<sp_host_col_info>& col_infos,
  std::vector<int>& field_lookup_table,
  int& num_count_cols,
  int& num_output_cols)
{
  int num_fields = static_cast<int>(schema.size());

  // Group children by parent_idx
  std::map<int, std::vector<int>> parent_to_children;
  for (int i = 0; i < num_fields; i++) {
    parent_to_children[schema[i].parent_idx].push_back(i);
  }

  // Assign message type indices: root first, then each struct parent
  std::map<int, int> parent_to_msg_type;
  int msg_type_counter = 0;
  parent_to_msg_type[-1] = msg_type_counter++;

  for (int i = 0; i < num_fields; i++) {
    auto type_id = schema_output_types[i].id();
    if (type_id == cudf::type_id::STRUCT && parent_to_children.count(i) > 0) {
      parent_to_msg_type[i] = msg_type_counter++;
    }
  }

  // Assign col_idx and count_idx via DFS
  int col_counter = 0;
  int count_counter = 0;
  std::map<int, int> schema_to_col_idx;
  std::map<int, int> schema_to_count_idx;

  std::function<void(int, int)> assign_indices = [&](int parent_idx, int parent_count_idx) {
    auto it = parent_to_children.find(parent_idx);
    if (it == parent_to_children.end()) return;

    for (int si : it->second) {
      auto type_id = schema_output_types[si].id();
      bool is_repeated = schema[si].is_repeated;
      bool is_struct = (type_id == cudf::type_id::STRUCT);
      // STRING and LIST (bytes) are both length-delimited and stored as sp_string_pair
      bool is_string = (type_id == cudf::type_id::STRING || type_id == cudf::type_id::LIST);

      int my_count_idx = -1;
      if (is_repeated) {
        my_count_idx = count_counter++;
        schema_to_count_idx[si] = my_count_idx;
      }

      int my_col_idx = -1;
      // All non-repeated-struct fields get a col_idx for data writing.
      // Non-repeated struct containers also get one for validity tracking.
      if (is_struct && !is_repeated) {
        my_col_idx = col_counter++;
        schema_to_col_idx[si] = my_col_idx;
      } else if (!is_struct) {
        my_col_idx = col_counter++;
        schema_to_col_idx[si] = my_col_idx;
      }
      // Repeated structs: no col_idx (list offsets from count, struct from children)

      sp_host_col_info info{};
      info.schema_idx = si;
      info.col_idx = my_col_idx;
      info.count_idx = my_count_idx;
      info.parent_count_idx = parent_count_idx;
      info.type_id = type_id;
      info.is_repeated = is_repeated;
      info.is_string = is_string;
      info.parent_schema_idx = parent_idx;
      col_infos.push_back(info);

      if (is_struct) {
        int child_parent_count = is_repeated ? my_count_idx : parent_count_idx;
        assign_indices(si, child_parent_count);
      }
    }
  };

  assign_indices(-1, -1);
  num_count_cols = count_counter;
  num_output_cols = col_counter;

  // Build sp_msg_type and sp_field_entry arrays
  msg_types.resize(msg_type_counter);
  for (auto& [pidx, mt_idx] : parent_to_msg_type) {
    auto it = parent_to_children.find(pidx);
    if (it == parent_to_children.end()) {
      msg_types[mt_idx] = {static_cast<int>(field_entries.size()), 0, -1, 0};
      continue;
    }
    auto children = it->second;
    std::sort(children.begin(), children.end(), [&](int a, int b) {
      return schema[a].field_number < schema[b].field_number;
    });

    int first_idx = static_cast<int>(field_entries.size());
    for (int si : children) {
      sp_field_entry e{};
      e.field_number = schema[si].field_number;
      e.wire_type = schema[si].wire_type;
      e.output_type_id = static_cast<int>(schema_output_types[si].id());
      e.encoding = schema[si].encoding;
      e.is_repeated = schema[si].is_repeated;
      e.has_default = schema[si].has_default_value;
      e.default_int = e.has_default ? default_ints[si] : 0;
      e.default_float = e.has_default ? default_floats[si] : 0.0;

      auto type_id = schema_output_types[si].id();
      if (type_id == cudf::type_id::STRUCT) {
        auto mt_it = parent_to_msg_type.find(si);
        e.child_msg_type = (mt_it != parent_to_msg_type.end()) ? mt_it->second : -1;
      } else {
        e.child_msg_type = -1;
      }

      auto col_it = schema_to_col_idx.find(si);
      e.col_idx = (col_it != schema_to_col_idx.end()) ? col_it->second : -1;
      auto cnt_it = schema_to_count_idx.find(si);
      e.count_idx = (cnt_it != schema_to_count_idx.end()) ? cnt_it->second : -1;

      field_entries.push_back(e);
    }
    msg_types[mt_idx] = {first_idx, static_cast<int>(children.size()), -1, 0};
  }

  // Build direct-mapped field lookup table for O(1) field lookup
  // For each message type, allocate a region of [0..max_field_number) in the table.
  // table[offset + field_number] = index into field_entries, or -1 if not found.
  int lookup_offset = 0;
  for (int mt = 0; mt < msg_type_counter; mt++) {
    auto& mtype = msg_types[mt];
    if (mtype.num_fields == 0) {
      mtype.lookup_offset = lookup_offset;
      mtype.max_field_number = 1;  // at least 1 to avoid zero-size
      field_lookup_table.push_back(-1);
      lookup_offset += 1;
      continue;
    }
    // Find max field number in this message type
    int max_fn = 0;
    for (int f = mtype.first_field_idx; f < mtype.first_field_idx + mtype.num_fields; f++) {
      max_fn = std::max(max_fn, field_entries[f].field_number);
    }
    int table_size = max_fn + 1;
    mtype.lookup_offset = lookup_offset;
    mtype.max_field_number = table_size;

    // Fill with -1 (not found)
    int base = static_cast<int>(field_lookup_table.size());
    field_lookup_table.resize(base + table_size, -1);
    // Set entries for known fields
    for (int f = mtype.first_field_idx; f < mtype.first_field_idx + mtype.num_fields; f++) {
      field_lookup_table[base + field_entries[f].field_number] = f;
    }
    lookup_offset += table_size;
  }
}

/**
 * Recursively build a cudf column for a field in the schema.
 * Returns the assembled column.
 */
std::unique_ptr<cudf::column> sp_build_column_recursive(
  std::vector<nested_field_descriptor> const& schema,
  std::vector<cudf::data_type> const& schema_output_types,
  std::vector<sp_host_col_info> const& col_infos,
  std::map<int, sp_host_col_info const*> const& schema_idx_to_info,
  // Buffers (bulk-allocated):
  std::vector<uint8_t*>& col_data_ptrs,                        // col_idx -> data pointer in bulk buffer
  std::vector<bool*>& col_validity_ptrs,                        // col_idx -> validity pointer in bulk buffer
  std::vector<rmm::device_uvector<int32_t>>& list_offsets_bufs, // count_idx -> offsets (top-level)
  std::vector<int32_t*>& inner_offs_ptrs,                       // count_idx -> inner offsets pointer (or null)
  std::vector<int>& inner_buf_sizes,                            // count_idx -> inner offsets size (0 if not inner)
  std::vector<int32_t>& col_sizes,                             // col_idx -> element count
  std::vector<int32_t>& count_totals,                          // count_idx -> total count
  std::vector<size_t>& col_elem_bytes,                         // col_idx -> element byte size
  int schema_idx,
  int num_fields,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto it = schema_idx_to_info.find(schema_idx);
  if (it == schema_idx_to_info.end()) {
    return make_null_column(schema_output_types[schema_idx], num_rows, stream, mr);
  }
  auto const& info = *(it->second);
  auto type_id = info.type_id;
  bool is_repeated = info.is_repeated;
  bool is_string = info.is_string;

  // Determine element count for this column
  int elem_count = num_rows;
  if (info.parent_count_idx >= 0) {
    elem_count = count_totals[info.parent_count_idx];
  }

  if (type_id == cudf::type_id::STRUCT) {
    // Find children of this struct
    std::vector<int> child_schema_indices;
    for (int i = 0; i < num_fields; i++) {
      if (schema[i].parent_idx == schema_idx) child_schema_indices.push_back(i);
    }

    if (is_repeated) {
      // LIST<STRUCT>: build struct children, then wrap in list
      int total = count_totals[info.count_idx];
      std::vector<std::unique_ptr<cudf::column>> struct_children;
      for (int child_si : child_schema_indices) {
        struct_children.push_back(sp_build_column_recursive(
          schema, schema_output_types, col_infos, schema_idx_to_info,
          col_data_ptrs, col_validity_ptrs, list_offsets_bufs, inner_offs_ptrs, inner_buf_sizes,
          col_sizes, count_totals, col_elem_bytes, child_si, num_fields, num_rows, stream, mr));
      }
      auto struct_col = cudf::make_structs_column(
        total, std::move(struct_children), 0, rmm::device_buffer{}, stream, mr);

      // List offsets: use inner offsets for nested repeated fields, top-level offsets otherwise
      std::unique_ptr<cudf::column> offsets_col;
      if (inner_offs_ptrs[info.count_idx] != nullptr) {
        int sz = inner_buf_sizes[info.count_idx];
        auto buf = rmm::device_buffer(sz * sizeof(int32_t), stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(buf.data(), inner_offs_ptrs[info.count_idx],
          sz * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream.value()));
        offsets_col = std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT32}, sz, std::move(buf), rmm::device_buffer{}, 0);
      } else {
        auto& offs = list_offsets_bufs[info.count_idx];
        auto const offs_size = static_cast<cudf::size_type>(offs.size());
        offsets_col = std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT32}, offs_size, offs.release(), rmm::device_buffer{}, 0);
      }

      return cudf::make_lists_column(
        elem_count, std::move(offsets_col), std::move(struct_col),
        0, rmm::device_buffer{}, stream, mr);
    } else {
      // Non-repeated struct
      std::vector<std::unique_ptr<cudf::column>> struct_children;
      for (int child_si : child_schema_indices) {
        struct_children.push_back(sp_build_column_recursive(
          schema, schema_output_types, col_infos, schema_idx_to_info,
          col_data_ptrs, col_validity_ptrs, list_offsets_bufs, inner_offs_ptrs, inner_buf_sizes,
          col_sizes, count_totals, col_elem_bytes, child_si, num_fields, num_rows, stream, mr));
      }
      // Struct validity from col_idx
      int ci = info.col_idx;
      if (ci >= 0 && col_validity_ptrs[ci] != nullptr && col_sizes[ci] > 0) {
        auto [mask, null_count] = cudf::detail::valid_if(
          thrust::make_counting_iterator<cudf::size_type>(0),
          thrust::make_counting_iterator<cudf::size_type>(elem_count),
          [vld = col_validity_ptrs[ci]] __device__ (cudf::size_type i) { return vld[i]; },
          stream, mr);
        return cudf::make_structs_column(
          elem_count, std::move(struct_children), null_count, std::move(mask), stream, mr);
      }
      return cudf::make_structs_column(
        elem_count, std::move(struct_children), 0, rmm::device_buffer{}, stream, mr);
    }
  }

  // Leaf field (scalar or string)
  int ci = info.col_idx;
  if (ci < 0) {
    return make_null_column(schema_output_types[schema_idx], elem_count, stream, mr);
  }

  // Helper lambda: build a STRING column from sp_string_pair data
  auto build_string_col = [&](int col_idx, int count, bool use_validity) -> std::unique_ptr<cudf::column> {
    auto* pairs = reinterpret_cast<sp_string_pair*>(col_data_ptrs[col_idx]);
    rmm::device_uvector<cudf::strings::detail::string_index_pair> str_pairs(count, stream, mr);
    if (use_validity) {
      thrust::transform(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(count),
        str_pairs.begin(),
        [pairs, vld = col_validity_ptrs[col_idx]] __device__ (int i) -> cudf::strings::detail::string_index_pair {
          if (vld[i]) return {pairs[i].ptr, pairs[i].length};
          return {nullptr, 0};
        });
    } else {
      thrust::transform(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(count),
        str_pairs.begin(),
        [pairs] __device__ (int i) -> cudf::strings::detail::string_index_pair {
          return {pairs[i].ptr, pairs[i].length};
        });
    }
    return cudf::strings::detail::make_strings_column(
      str_pairs.begin(), str_pairs.end(), stream, mr);
  };

  // Helper lambda: build a LIST<UINT8> (bytes/binary) column from sp_string_pair data
  auto build_bytes_col = [&](int col_idx, int count) -> std::unique_ptr<cudf::column> {
    auto* pairs = reinterpret_cast<sp_string_pair*>(col_data_ptrs[col_idx]);
    auto* vld = col_validity_ptrs[col_idx];
    // Compute lengths and prefix sum -> offsets (inclusive scan then shift)
    rmm::device_uvector<int32_t> byte_offs(count + 1, stream, mr);
    if (count > 0) {
      // Compute lengths directly into offsets[1..count], then exclusive_scan
      thrust::transform(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(count),
        byte_offs.begin(),  // write to [0..count-1]
        [pairs, vld] __device__ (int i) -> int32_t { return vld[i] ? pairs[i].length : 0; });
      thrust::exclusive_scan(rmm::exec_policy(stream),
        byte_offs.begin(), byte_offs.begin() + count, byte_offs.begin(), 0);
      // Total bytes via transform_reduce (avoids D->H sync for last_off + last_len)
      int32_t total_bytes = thrust::transform_reduce(rmm::exec_policy(stream),
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(count),
        [pairs, vld] __device__ (int i) -> int32_t {
          return vld[i] ? pairs[i].length : 0;
        }, 0, cuda::std::plus<int32_t>{});
      CUDF_CUDA_TRY(cudaMemcpyAsync(byte_offs.data() + count, &total_bytes,
        sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));
      // Copy binary data
      rmm::device_uvector<uint8_t> child_data(total_bytes > 0 ? total_bytes : 0, stream, mr);
      if (total_bytes > 0) {
        thrust::for_each(rmm::exec_policy(stream),
          thrust::make_counting_iterator(0), thrust::make_counting_iterator(count),
          [pairs, offs = byte_offs.data(), out = child_data.data(), vld] __device__ (int i) {
            if (vld[i] && pairs[i].ptr && pairs[i].length > 0) {
              memcpy(out + offs[i], pairs[i].ptr, pairs[i].length);
            }
          });
      }
      auto off_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32}, count + 1, byte_offs.release(), rmm::device_buffer{}, 0);
      auto ch_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8}, total_bytes, child_data.release(), rmm::device_buffer{}, 0);
      auto [mask, null_count] = cudf::detail::valid_if(
        thrust::make_counting_iterator<cudf::size_type>(0),
        thrust::make_counting_iterator<cudf::size_type>(count),
        [v = vld] __device__ (cudf::size_type i) { return v[i]; }, stream, mr);
      return cudf::make_lists_column(count, std::move(off_col), std::move(ch_col), null_count, std::move(mask), stream, mr);
    } else {
      // Empty bytes column
      thrust::fill(rmm::exec_policy(stream), byte_offs.begin(), byte_offs.end(), 0);
      auto off_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32}, count + 1, byte_offs.release(), rmm::device_buffer{}, 0);
      auto ch_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8}, 0, rmm::device_buffer{}, rmm::device_buffer{}, 0);
      return cudf::make_lists_column(count, std::move(off_col), std::move(ch_col), 0, rmm::device_buffer{}, stream, mr);
    }
  };

  bool is_bytes = (type_id == cudf::type_id::LIST);

  if (is_repeated) {
    // LIST<scalar/string/bytes>: build child column then wrap in list
    int total = count_totals[info.count_idx];
    std::unique_ptr<cudf::column> child_col;

    if (is_bytes) {
      // repeated bytes -> LIST<LIST<UINT8>>: build inner LIST<UINT8> then wrap in outer list
      child_col = build_bytes_col(ci, total);
    } else if (is_string) {
      child_col = build_string_col(ci, total, false);
    } else {
      auto dt = schema_output_types[schema_idx];
      auto [mask, null_count] = cudf::detail::valid_if(
        thrust::make_counting_iterator<cudf::size_type>(0),
        thrust::make_counting_iterator<cudf::size_type>(total),
        [v = col_validity_ptrs[ci]] __device__ (cudf::size_type i) { return v[i]; }, stream, mr);
      // Copy data from bulk buffer into a new device_buffer for cudf::column ownership
      auto data_buf = rmm::device_buffer(total * col_elem_bytes[ci], stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(data_buf.data(), col_data_ptrs[ci],
        total * col_elem_bytes[ci], cudaMemcpyDeviceToDevice, stream.value()));
      child_col = std::make_unique<cudf::column>(
        dt, total, std::move(data_buf), std::move(mask), null_count);
    }

    // Use inner offsets if available, else use top-level list offsets
    std::unique_ptr<cudf::column> offsets_col;
    if (inner_offs_ptrs[info.count_idx] != nullptr) {
      int sz = inner_buf_sizes[info.count_idx];
      auto buf = rmm::device_buffer(sz * sizeof(int32_t), stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(buf.data(), inner_offs_ptrs[info.count_idx],
        sz * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream.value()));
      offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, sz, std::move(buf), rmm::device_buffer{}, 0);
    } else {
      auto& offs = list_offsets_bufs[info.count_idx];
      auto const offs_size = static_cast<cudf::size_type>(offs.size());
      offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32}, offs_size, offs.release(), rmm::device_buffer{}, 0);
    }

    return cudf::make_lists_column(
      elem_count, std::move(offsets_col), std::move(child_col),
      0, rmm::device_buffer{}, stream, mr);
  }

  // Non-repeated leaf
  if (is_bytes) {
    return build_bytes_col(ci, elem_count);
  }
  if (is_string) {
    return build_string_col(ci, elem_count, true);
  }

  // Non-repeated non-string scalar
  auto dt = schema_output_types[schema_idx];
  auto [mask, null_count] = cudf::detail::valid_if(
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(elem_count),
    [v = col_validity_ptrs[ci]] __device__ (cudf::size_type i) { return v[i]; }, stream, mr);
  // Copy data from bulk buffer into a new device_buffer for cudf::column ownership
  auto data_buf = rmm::device_buffer(elem_count * col_elem_bytes[ci], stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(data_buf.data(), col_data_ptrs[ci],
    elem_count * col_elem_bytes[ci], cudaMemcpyDeviceToDevice, stream.value()));
  return std::make_unique<cudf::column>(
    dt, elem_count, std::move(data_buf), std::move(mask), null_count);
}

/**
 * Main single-pass decoder orchestration.
 */
std::unique_ptr<cudf::column> decode_nested_protobuf_single_pass(
  cudf::column_view const& binary_input,
  std::vector<nested_field_descriptor> const& schema,
  std::vector<cudf::data_type> const& schema_output_types,
  std::vector<int64_t> const& default_ints,
  std::vector<double> const& default_floats,
  std::vector<bool> const& default_bools,
  std::vector<std::vector<uint8_t>> const& default_strings,
  bool fail_on_errors)
{
  auto const stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  auto num_rows = binary_input.size();
  auto num_fields = static_cast<int>(schema.size());

  // Timing instrumentation (enabled by PROTOBUF_SP_TIMING=1)
  static bool sp_timing_enabled = (std::getenv("PROTOBUF_SP_TIMING") != nullptr &&
                                    std::string(std::getenv("PROTOBUF_SP_TIMING")) == "1");
  static int sp_call_count = 0;
  static double sp_phase_totals[8] = {};  // accumulate across calls
  cudaEvent_t t_start, t1, t2, t3, t4, t5, t6, t7;
  if (sp_timing_enabled) {
    cudaEventCreate(&t_start); cudaEventCreate(&t1); cudaEventCreate(&t2);
    cudaEventCreate(&t3); cudaEventCreate(&t4); cudaEventCreate(&t5);
    cudaEventCreate(&t6); cudaEventCreate(&t7);
    cudaEventRecord(t_start, stream.value());
  }

  // === Phase 1: Schema Preprocessing ===
  std::vector<sp_msg_type> h_msg_types;
  std::vector<sp_field_entry> h_field_entries;
  std::vector<sp_host_col_info> col_infos;
  std::vector<int> h_field_lookup;
  int num_count_cols = 0;
  int num_output_cols = 0;

  build_single_pass_schema(schema, schema_output_types,
    default_ints, default_floats, default_bools,
    h_msg_types, h_field_entries, col_infos, h_field_lookup,
    num_count_cols, num_output_cols);

  // Check limits
  if (num_count_cols > SP_MAX_COUNTED || num_output_cols > SP_MAX_OUTPUT_COLS) {
    return nullptr;  // Signal caller to fall back to old decoder
  }

  // Copy schema to device
  rmm::device_uvector<sp_msg_type> d_msg_types(h_msg_types.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_msg_types.data(), h_msg_types.data(),
    h_msg_types.size() * sizeof(sp_msg_type), cudaMemcpyHostToDevice, stream.value()));

  rmm::device_uvector<sp_field_entry> d_field_entries(h_field_entries.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_field_entries.data(), h_field_entries.data(),
    h_field_entries.size() * sizeof(sp_field_entry), cudaMemcpyHostToDevice, stream.value()));

  // Copy O(1) field lookup table to device
  rmm::device_uvector<int> d_field_lookup(h_field_lookup.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_field_lookup.data(), h_field_lookup.data(),
    h_field_lookup.size() * sizeof(int), cudaMemcpyHostToDevice, stream.value()));

  auto d_in = cudf::column_device_view::create(binary_input, stream);

  rmm::device_uvector<int> d_error(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 0, sizeof(int), stream.value()));

  int const threads = 256;
  int const blocks = (num_rows + threads - 1) / threads;

  if (sp_timing_enabled) cudaEventRecord(t1, stream.value());  // end schema prep + alloc

  // === Phase 2: Pass 1 - Count ===
  rmm::device_uvector<int32_t> d_counts(
    num_count_cols > 0 ? static_cast<size_t>(num_rows) * num_count_cols : 1, stream, mr);
  if (num_count_cols > 0) {
    CUDF_CUDA_TRY(cudaMemsetAsync(d_counts.data(), 0,
      static_cast<size_t>(num_rows) * num_count_cols * sizeof(int32_t), stream.value()));
  }

  sp_unified_count_kernel<<<blocks, threads, 0, stream.value()>>>(
    *d_in, d_msg_types.data(), d_field_entries.data(), d_field_lookup.data(),
    d_counts.data(), num_count_cols, d_error.data());

  if (sp_timing_enabled) cudaEventRecord(t2, stream.value());  // end count kernel

  // === Phase 3: Compute Offsets and Allocate Buffers ===
  // Fused: compute all prefix sums + list offsets in a SINGLE kernel launch.
  // Replaces ~50 syncs + ~200 kernel launches with 1 kernel + 1 sync.
  rmm::device_uvector<int32_t> d_row_offsets(
    num_count_cols > 0 ? static_cast<size_t>(num_rows) * num_count_cols : 1, stream, mr);

  std::vector<int32_t> count_totals(num_count_cols, 0);
  std::vector<rmm::device_uvector<int32_t>> list_offsets_bufs;
  list_offsets_bufs.reserve(num_count_cols);

  // Pre-allocate all list offset buffers and collect device pointers
  std::vector<int32_t*> h_list_offs_ptrs(num_count_cols, nullptr);
  for (int c = 0; c < num_count_cols; c++) {
    list_offsets_bufs.emplace_back(num_rows + 1, stream, mr);
    h_list_offs_ptrs[c] = list_offsets_bufs.back().data();
  }

  if (num_count_cols > 0) {
    // Copy list offset pointers to device
    rmm::device_uvector<int32_t*> d_list_offs_ptrs(num_count_cols, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_list_offs_ptrs.data(), h_list_offs_ptrs.data(),
      num_count_cols * sizeof(int32_t*), cudaMemcpyHostToDevice, stream.value()));

    // Device buffer for totals
    rmm::device_uvector<int32_t> d_totals(num_count_cols, stream, mr);

    // Single fused kernel: prefix sums + totals + list offsets for all columns
    int const off_threads = std::min(num_count_cols, 256);
    int const off_blocks = (num_count_cols + off_threads - 1) / off_threads;
    sp_compute_offsets_kernel<<<off_blocks, off_threads, 0, stream.value()>>>(
      d_counts.data(), d_row_offsets.data(), d_totals.data(),
      d_list_offs_ptrs.data(), num_rows, num_count_cols);

    // Single D->H copy for all totals + single sync
    CUDF_CUDA_TRY(cudaMemcpyAsync(count_totals.data(), d_totals.data(),
      num_count_cols * sizeof(int32_t), cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
  }

  // Build schema_idx -> col_info lookup
  std::map<int, sp_host_col_info const*> schema_idx_to_info;
  for (auto const& ci : col_infos) {
    schema_idx_to_info[ci.schema_idx] = &ci;
  }

  // Determine buffer sizes for each col_idx
  std::vector<int32_t> col_sizes(num_output_cols, 0);
  for (auto const& ci : col_infos) {
    if (ci.col_idx < 0) continue;
    if (ci.is_repeated && ci.count_idx >= 0) {
      col_sizes[ci.col_idx] = count_totals[ci.count_idx];
    } else if (ci.parent_count_idx >= 0) {
      col_sizes[ci.col_idx] = count_totals[ci.parent_count_idx];
    } else {
      col_sizes[ci.col_idx] = num_rows;
    }
  }

  // Build col_idx -> col_info lookup (avoids O(N^2) inner loop)
  std::vector<sp_host_col_info const*> col_idx_to_info(num_output_cols, nullptr);
  for (auto const& c : col_infos) {
    if (c.col_idx >= 0) col_idx_to_info[c.col_idx] = &c;
  }

  // Compute per-column element sizes and total buffer sizes for BULK allocation.
  // Replaces ~992 individual RMM allocations + ~992 memsets with 2 allocs + 2 memsets.
  std::vector<size_t> col_elem_bytes(num_output_cols, 0);
  std::vector<size_t> col_data_offsets(num_output_cols, 0);
  std::vector<size_t> col_validity_offsets(num_output_cols, 0);
  size_t total_data_bytes = 0;
  size_t total_validity_elems = 0;

  for (int ci_idx = 0; ci_idx < num_output_cols; ci_idx++) {
    auto const* cinfo = col_idx_to_info[ci_idx];
    size_t eb = 0;
    if (cinfo) {
      auto tid = cinfo->type_id;
      if (tid == cudf::type_id::STRING || tid == cudf::type_id::LIST) {
        eb = sizeof(sp_string_pair);
      } else if (tid == cudf::type_id::STRUCT) {
        eb = 0;
      } else {
        eb = cudf::size_of(cudf::data_type{tid});
      }
    }
    col_elem_bytes[ci_idx] = eb;
    int32_t sz = col_sizes[ci_idx];
    // Align data offset to 16 bytes for coalesced GPU access
    col_data_offsets[ci_idx] = total_data_bytes;
    total_data_bytes += (sz > 0 ? sz * eb : 0);
    total_data_bytes = (total_data_bytes + 15) & ~size_t{15};  // 16-byte align

    col_validity_offsets[ci_idx] = total_validity_elems;
    total_validity_elems += (sz > 0 ? sz : 0);
  }

  // TWO bulk allocations instead of ~992 individual ones
  rmm::device_uvector<uint8_t> bulk_data(total_data_bytes > 0 ? total_data_bytes : 1, stream, mr);
  rmm::device_uvector<bool> bulk_validity(total_validity_elems > 0 ? total_validity_elems : 1, stream, mr);

  // TWO bulk memsets instead of ~992 individual ones
  if (total_data_bytes > 0) {
    CUDF_CUDA_TRY(cudaMemsetAsync(bulk_data.data(), 0, total_data_bytes, stream.value()));
  }
  if (total_validity_elems > 0) {
    CUDF_CUDA_TRY(cudaMemsetAsync(bulk_validity.data(), 0, total_validity_elems * sizeof(bool), stream.value()));
  }

  // Per-column pointers into the bulk buffers
  std::vector<uint8_t*> col_data_ptrs(num_output_cols, nullptr);
  std::vector<bool*> col_validity_ptrs(num_output_cols, nullptr);

  for (int ci_idx = 0; ci_idx < num_output_cols; ci_idx++) {
    int32_t sz = col_sizes[ci_idx];
    if (sz > 0) {
      col_data_ptrs[ci_idx] = bulk_data.data() + col_data_offsets[ci_idx];
      col_validity_ptrs[ci_idx] = bulk_validity.data() + col_validity_offsets[ci_idx];
    }
  }

  // Fill non-zero defaults (rare - proto3 defaults are all 0)
  for (int ci_idx = 0; ci_idx < num_output_cols; ci_idx++) {
    auto const* cinfo = col_idx_to_info[ci_idx];
    int32_t sz = col_sizes[ci_idx];
    if (!cinfo || sz <= 0) continue;
    if (cinfo->type_id == cudf::type_id::STRING ||
        cinfo->type_id == cudf::type_id::LIST ||
        cinfo->type_id == cudf::type_id::STRUCT) continue;
    int si = cinfo->schema_idx;
    if (!schema[si].has_default_value) continue;

    auto tid = cinfo->type_id;
    bool non_zero = false;
    if (tid == cudf::type_id::BOOL8) non_zero = default_bools[si];
    else if (tid == cudf::type_id::FLOAT32 || tid == cudf::type_id::FLOAT64)
      non_zero = (default_floats[si] != 0.0);
    else non_zero = (default_ints[si] != 0);

    if (non_zero) {
      thrust::fill_n(rmm::exec_policy(stream), col_validity_ptrs[ci_idx], sz, true);
      auto* dp = col_data_ptrs[ci_idx];
      if (tid == cudf::type_id::BOOL8)
        thrust::fill_n(rmm::exec_policy(stream), dp, sz, static_cast<uint8_t>(1));
      else if (tid == cudf::type_id::INT32 || tid == cudf::type_id::UINT32)
        thrust::fill_n(rmm::exec_policy(stream), reinterpret_cast<int32_t*>(dp), sz, static_cast<int32_t>(default_ints[si]));
      else if (tid == cudf::type_id::INT64 || tid == cudf::type_id::UINT64)
        thrust::fill_n(rmm::exec_policy(stream), reinterpret_cast<int64_t*>(dp), sz, default_ints[si]);
      else if (tid == cudf::type_id::FLOAT32)
        thrust::fill_n(rmm::exec_policy(stream), reinterpret_cast<float*>(dp), sz, static_cast<float>(default_floats[si]));
      else if (tid == cudf::type_id::FLOAT64)
        thrust::fill_n(rmm::exec_policy(stream), reinterpret_cast<double*>(dp), sz, default_floats[si]);
    }
  }

  // Build device-side column descriptors (using bulk buffer pointers)
  std::vector<sp_col_desc> h_col_descs(num_output_cols);
  for (int i = 0; i < num_output_cols; i++) {
    h_col_descs[i].data = col_data_ptrs[i];
    h_col_descs[i].validity = col_validity_ptrs[i];
  }
  rmm::device_uvector<sp_col_desc> d_col_descs(num_output_cols, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_col_descs.data(), h_col_descs.data(),
    num_output_cols * sizeof(sp_col_desc), cudaMemcpyHostToDevice, stream.value()));

  // Allocate parent index buffers for inner repeated fields
  std::vector<rmm::device_uvector<int32_t>> parent_idx_storage;
  std::vector<int32_t*> h_parent_bufs(num_count_cols, nullptr);
  parent_idx_storage.reserve(num_count_cols);

  for (auto const& ci : col_infos) {
    if (ci.count_idx >= 0 && ci.parent_count_idx >= 0) {
      // Inner repeated field: needs parent index buffer
      int total = count_totals[ci.count_idx];
      parent_idx_storage.emplace_back(total > 0 ? total : 0, stream, mr);
      h_parent_bufs[ci.count_idx] = parent_idx_storage.back().data();
    }
  }

  rmm::device_uvector<int32_t*> d_parent_bufs_arr(num_count_cols, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_parent_bufs_arr.data(), h_parent_bufs.data(),
    num_count_cols * sizeof(int32_t*), cudaMemcpyHostToDevice, stream.value()));

  if (sp_timing_enabled) cudaEventRecord(t3, stream.value());  // end offsets + buffer alloc

  // === Phase 4: Pass 2 - Extract ===
  sp_unified_extract_kernel<<<blocks, threads, 0, stream.value()>>>(
    *d_in, d_msg_types.data(), d_field_entries.data(), d_field_lookup.data(),
    d_col_descs.data(), d_row_offsets.data(),
    d_parent_bufs_arr.data(), num_count_cols, d_error.data());

  if (sp_timing_enabled) cudaEventRecord(t4, stream.value());  // end extract kernel

  // === Phase 5: Compute Inner List Offsets ===
  // Pre-compute which count columns are inner (parent_count_idx >= 0) and their sizes.
  // Bulk-allocate a single buffer for all inner offsets to avoid memory pool fragmentation.
  struct inner_info_t { int count_idx; int parent_count_idx; int total_child; int total_parent; };
  std::vector<inner_info_t> inner_infos;
  size_t total_inner_elems = 0;
  std::vector<int> inner_buf_offsets(num_count_cols, -1);  // offset into bulk inner buffer
  std::vector<int> inner_buf_sizes(num_count_cols, 0);

  for (int c = 0; c < num_count_cols; c++) {
    sp_host_col_info const* cinfo_ptr = nullptr;
    for (auto const& ci : col_infos) {
      if (ci.count_idx == c) { cinfo_ptr = &ci; break; }
    }
    if (cinfo_ptr && cinfo_ptr->parent_count_idx >= 0) {
      int total_child = count_totals[c];
      int total_parent = count_totals[cinfo_ptr->parent_count_idx];
      int sz = total_parent + 1;
      inner_buf_offsets[c] = static_cast<int>(total_inner_elems);
      inner_buf_sizes[c] = sz;
      total_inner_elems += sz;
      inner_infos.push_back({c, cinfo_ptr->parent_count_idx, total_child, total_parent});
    }
  }

  // Single bulk allocation for all inner offsets
  rmm::device_uvector<int32_t> bulk_inner_offsets(
    total_inner_elems > 0 ? total_inner_elems : 1, stream, mr);
  if (total_inner_elems > 0) {
    CUDF_CUDA_TRY(cudaMemsetAsync(bulk_inner_offsets.data(), 0,
      total_inner_elems * sizeof(int32_t), stream.value()));
  }

  // Compute inner offsets via lower_bound (only for non-empty inner fields)
  for (auto const& ii : inner_infos) {
    int c = ii.count_idx;
    int32_t* out = bulk_inner_offsets.data() + inner_buf_offsets[c];
    if (ii.total_child > 0 && ii.total_parent > 0 && h_parent_bufs[c] != nullptr) {
      thrust::lower_bound(rmm::exec_policy(stream),
        h_parent_bufs[c], h_parent_bufs[c] + ii.total_child,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(ii.total_parent + 1),
        out);
    }
    // else: already zeroed by memset
  }

  // Build inner offset pointer array (points into bulk buffer, no per-column allocation)
  std::vector<int32_t*> inner_offs_ptrs(num_count_cols, nullptr);
  for (int c = 0; c < num_count_cols; c++) {
    if (inner_buf_offsets[c] >= 0) {
      inner_offs_ptrs[c] = bulk_inner_offsets.data() + inner_buf_offsets[c];
    }
  }

  if (sp_timing_enabled) cudaEventRecord(t5, stream.value());  // end inner offsets

  // === Phase 6: Column Assembly ===
  // Check for errors
  CUDF_CUDA_TRY(cudaPeekAtLastError());
  int h_error = 0;
  CUDF_CUDA_TRY(cudaMemcpyAsync(&h_error, d_error.data(), sizeof(int),
    cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  if (fail_on_errors) {
    CUDF_EXPECTS(h_error == 0, "Malformed protobuf message or unsupported wire type");
  }

  // Build top-level struct column
  std::vector<std::unique_ptr<cudf::column>> top_children;
  for (int i = 0; i < num_fields; i++) {
    if (schema[i].parent_idx == -1) {
      top_children.push_back(sp_build_column_recursive(
        schema, schema_output_types, col_infos, schema_idx_to_info,
        col_data_ptrs, col_validity_ptrs, list_offsets_bufs, inner_offs_ptrs, inner_buf_sizes,
        col_sizes, count_totals, col_elem_bytes, i, num_fields, num_rows, stream, mr));
    }
  }

  auto result = cudf::make_structs_column(
    num_rows, std::move(top_children), 0, rmm::device_buffer{}, stream, mr);

  // Print timing results
  if (sp_timing_enabled) {
    cudaEventRecord(t6, stream.value());  // end column assembly
    cudaEventSynchronize(t6);

    float ms[7];
    cudaEventElapsedTime(&ms[0], t_start, t1);  // schema prep + device copy
    cudaEventElapsedTime(&ms[1], t1, t2);        // count kernel
    cudaEventElapsedTime(&ms[2], t2, t3);        // offsets + buffer alloc
    cudaEventElapsedTime(&ms[3], t3, t4);        // extract kernel
    cudaEventElapsedTime(&ms[4], t4, t5);        // inner offsets
    cudaEventElapsedTime(&ms[5], t5, t6);        // column assembly
    cudaEventElapsedTime(&ms[6], t_start, t6);   // total

    sp_call_count++;
    sp_phase_totals[0] += ms[0]; sp_phase_totals[1] += ms[1];
    sp_phase_totals[2] += ms[2]; sp_phase_totals[3] += ms[3];
    sp_phase_totals[4] += ms[4]; sp_phase_totals[5] += ms[5];
    sp_phase_totals[6] += ms[6];

    if (sp_call_count % 50 == 0) {
      fprintf(stderr,
        "[SP-TIMING] call#%d rows=%d fields=%d count_cols=%d out_cols=%d | "
        "THIS: prep=%.1f count=%.1f offsets=%.1f extract=%.1f inner=%.1f assembly=%.1f TOTAL=%.1f ms | "
        "CUMUL: prep=%.0f count=%.0f offsets=%.0f extract=%.0f inner=%.0f assembly=%.0f TOTAL=%.0f ms\n",
        sp_call_count, num_rows, num_fields, num_count_cols, num_output_cols,
        ms[0], ms[1], ms[2], ms[3], ms[4], ms[5], ms[6],
        sp_phase_totals[0], sp_phase_totals[1], sp_phase_totals[2],
        sp_phase_totals[3], sp_phase_totals[4], sp_phase_totals[5],
        sp_phase_totals[6]);
    }

    cudaEventDestroy(t_start); cudaEventDestroy(t1); cudaEventDestroy(t2);
    cudaEventDestroy(t3); cudaEventDestroy(t4); cudaEventDestroy(t5);
    cudaEventDestroy(t6); cudaEventDestroy(t7);
  }

  return result;
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

  // Try single-pass decoder (faster for complex nested schemas)
  // Can be disabled by setting PROTOBUF_NO_SINGLE_PASS=1
  {
    char const* no_sp = std::getenv("PROTOBUF_NO_SINGLE_PASS");
    bool use_single_pass = !(no_sp && std::string(no_sp) == "1");
    if (use_single_pass) {
      auto result = decode_nested_protobuf_single_pass(
        binary_input, schema, schema_output_types,
        default_ints, default_floats, default_bools, default_strings,
        fail_on_errors);
      CUDF_EXPECTS(result != nullptr,
        "Single-pass protobuf decoder failed: schema exceeds limits "
        "(SP_MAX_COUNTED=" + std::to_string(SP_MAX_COUNTED) +
        " or SP_MAX_OUTPUT_COLS=" + std::to_string(SP_MAX_OUTPUT_COLS) +
        ", actual num_count_cols or num_output_cols too large). "
        "Set PROTOBUF_NO_SINGLE_PASS=1 to use the old decoder.");
      return result;
    }
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

      // Get per-row counts for this repeated field entirely on GPU (performance fix!)
      rmm::device_uvector<int32_t> d_field_counts(num_rows, stream, mr);
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(num_rows),
                        d_field_counts.begin(),
                        extract_strided_count{d_repeated_info.data(), ri, num_repeated});
      
      int total_count = thrust::reduce(rmm::exec_policy(stream),
                                       d_field_counts.begin(), d_field_counts.end(), 0);

      // Still need host-side field_info for build_repeated_scalar_column
      std::vector<repeated_field_info> field_info(num_rows);
      for (int row = 0; row < num_rows; row++) {
        field_info[row] = h_repeated_info[row * num_repeated + ri];
      }

      if (total_count > 0) {
        // Build offsets for occurrence scanning on GPU (performance fix!)
        rmm::device_uvector<int32_t> d_occ_offsets(num_rows + 1, stream, mr);
        thrust::exclusive_scan(rmm::exec_policy(stream),
                               d_field_counts.begin(), d_field_counts.end(),
                               d_occ_offsets.begin(), 0);
        // Set last element
        CUDF_CUDA_TRY(cudaMemcpyAsync(d_occ_offsets.data() + num_rows, &total_count,
                                      sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));

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
          
          // Compute total_rep_count on GPU using thrust::reduce (performance fix!)
          // Extract counts from repeated_field_info on device
          rmm::device_uvector<int32_t> d_rep_counts(num_rows, stream, mr);
          thrust::transform(rmm::exec_policy(stream),
                            d_rep_info.begin(), d_rep_info.end(),
                            d_rep_counts.begin(),
                            [] __device__(repeated_field_info const& info) { return info.count; });
          
          int total_rep_count = thrust::reduce(rmm::exec_policy(stream),
                                               d_rep_counts.begin(), d_rep_counts.end(), 0);
          
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
            
            // Compute list offsets on GPU using exclusive_scan (performance fix!)
            rmm::device_uvector<int32_t> list_offs(num_rows + 1, stream, mr);
            thrust::exclusive_scan(rmm::exec_policy(stream),
                                   d_rep_counts.begin(), d_rep_counts.end(),
                                   list_offs.begin(), 0);
            // Set last element
            CUDF_CUDA_TRY(cudaMemcpyAsync(list_offs.data() + num_rows, &total_rep_count,
              sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));
            
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
              // Compute string offsets on GPU using thrust (performance fix!)
              // Extract lengths from occurrences on device
              rmm::device_uvector<int32_t> d_str_lengths(total_rep_count, stream, mr);
              thrust::transform(rmm::exec_policy(stream),
                                d_rep_occs.begin(), d_rep_occs.end(),
                                d_str_lengths.begin(),
                                [] __device__(repeated_occurrence const& occ) { return occ.length; });
              
              // Compute total chars and offsets
              int32_t total_chars = thrust::reduce(rmm::exec_policy(stream),
                                                   d_str_lengths.begin(), d_str_lengths.end(), 0);
              
              rmm::device_uvector<int32_t> str_offs(total_rep_count + 1, stream, mr);
              thrust::exclusive_scan(rmm::exec_policy(stream),
                                     d_str_lengths.begin(), d_str_lengths.end(),
                                     str_offs.begin(), 0);
              // Set last element
              CUDF_CUDA_TRY(cudaMemcpyAsync(str_offs.data() + total_rep_count, &total_chars,
                sizeof(int32_t), cudaMemcpyHostToDevice, stream.value()));
              
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

            // Get child struct locations for grandchild scanning using GPU kernel
            // IMPORTANT: Need to compute ABSOLUTE offsets (relative to row start)
            // d_child_locations contains offsets relative to parent message (Middle)
            // We need: child_offset_in_row = parent_offset_in_row + child_offset_in_parent
            // This is computed entirely on GPU to avoid D->H->D copy pattern (performance fix!)
            rmm::device_uvector<field_location> d_gc_parent(num_rows, stream, mr);
            compute_grandchild_parent_locations_kernel<<<blocks, threads, 0, stream.value()>>>(
              d_parent_locs.data(), d_child_locations.data(), ci, num_child_fields,
              d_gc_parent.data(), num_rows);

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
