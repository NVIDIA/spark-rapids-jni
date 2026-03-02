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

namespace spark_rapids_jni::protobuf_detail {

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
  int const* field_lookup,              // direct-mapped lookup table (nullable)
  int field_lookup_size,                // size of lookup table (0 if null)
  field_location* locations,            // [num_rows * num_fields] row-major
  int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::detail::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  for (int f = 0; f < num_fields; f++) {
    locations[row * num_fields + f] = {-1, 0};
  }

  if (in.nullable() && in.is_null(row)) { return; }

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  int32_t start     = in.offset_at(row) - base;
  int32_t end       = in.offset_at(row + 1) - base;

  if (!check_message_bounds(start, end, child.size(), error_flag)) { return; }

  uint8_t const* cur     = bytes + start;
  uint8_t const* msg_end = bytes + end;

  while (cur < msg_end) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) { return; }
    int fn = tag.field_number;
    int wt = tag.wire_type;

    int f = lookup_field(fn, field_lookup, field_lookup_size, field_descs, num_fields);
    if (f >= 0) {
      if (wt != field_descs[f].expected_wire_type) {
        set_error_once(error_flag, ERR_WIRE_TYPE);
        return;
      }

      // Record the location (relative to message start)
      int data_offset = static_cast<int>(cur - bytes - start);

      if (wt == WT_LEN) {
        // For length-delimited, record offset after length prefix and the data length
        uint64_t len;
        int len_bytes;
        if (!read_varint(cur, msg_end, len, len_bytes)) {
          set_error_once(error_flag, ERR_VARINT);
          return;
        }
        if (len > static_cast<uint64_t>(msg_end - cur - len_bytes) ||
            len > static_cast<uint64_t>(INT_MAX)) {
          set_error_once(error_flag, ERR_OVERFLOW);
          return;
        }
        // Record offset pointing to the actual data (after length prefix)
        locations[row * num_fields + f] = {data_offset + len_bytes, static_cast<int32_t>(len)};
      } else {
        // For fixed-size and varint fields, record offset and compute length
        int field_size = get_wire_type_size(wt, cur, msg_end);
        if (field_size < 0) {
          set_error_once(error_flag, ERR_FIELD_SIZE);
          return;
        }
        locations[row * num_fields + f] = {data_offset, field_size};
      }
    }

    // Skip to next field
    uint8_t const* next;
    if (!skip_field(cur, msg_end, wt, next)) {
      set_error_once(error_flag, ERR_SKIP);
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
 *
 * @note Time complexity: O(message_length * (num_repeated_fields + num_nested_fields)) per row.
 */
__global__ void count_repeated_fields_kernel(
  cudf::column_device_view const d_in,
  device_nested_field_descriptor const* schema,
  int num_fields,
  int depth_level,                     // Which depth level we're processing
  repeated_field_info* repeated_info,  // [num_rows * num_repeated_fields_at_this_depth]
  int num_repeated_fields,             // Number of repeated fields at this depth
  int const* repeated_field_indices,   // Indices into schema for repeated fields at this depth
  field_location*
    nested_locations,     // Locations of nested messages for next depth [num_rows * num_nested]
  int num_nested_fields,  // Number of nested message fields at this depth
  int const* nested_field_indices,  // Indices into schema for nested message fields
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

  if (in.nullable() && in.is_null(row)) { return; }

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  int32_t start     = in.offset_at(row) - base;
  int32_t end       = in.offset_at(row + 1) - base;
  if (!check_message_bounds(start, end, child.size(), error_flag)) { return; }

  uint8_t const* cur     = bytes + start;
  uint8_t const* msg_end = bytes + end;

  while (cur < msg_end) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) { return; }
    int fn = tag.field_number;
    int wt = tag.wire_type;

    // Check repeated fields at this depth
    for (int i = 0; i < num_repeated_fields; i++) {
      int schema_idx = repeated_field_indices[i];
      if (schema[schema_idx].field_number == fn && schema[schema_idx].depth == depth_level) {
        int expected_wt = schema[schema_idx].wire_type;

        // Handle both packed and unpacked encoding for repeated fields
        // Packed encoding uses wire type LEN (2) even for scalar types
        bool is_packed = (wt == WT_LEN && expected_wt != WT_LEN);

        if (!is_packed && wt != expected_wt) {
          set_error_once(error_flag, ERR_WIRE_TYPE);
          return;
        }

        if (is_packed) {
          // Packed encoding: read length, then count elements inside
          uint64_t packed_len;
          int len_bytes;
          if (!read_varint(cur, msg_end, packed_len, len_bytes)) {
            set_error_once(error_flag, ERR_VARINT);
            return;
          }

          // Count elements based on type
          uint8_t const* packed_start = cur + len_bytes;
          uint8_t const* packed_end   = packed_start + packed_len;
          if (packed_end > msg_end) {
            set_error_once(error_flag, ERR_OVERFLOW);
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
                set_error_once(error_flag, ERR_VARINT);
                return;
              }
              p += vbytes;
              count++;
            }
          } else if (expected_wt == WT_32BIT) {
            if ((packed_len % 4) != 0) {
              set_error_once(error_flag, ERR_FIXED_LEN);
              return;
            }
            count = static_cast<int>(packed_len / 4);
          } else if (expected_wt == WT_64BIT) {
            if ((packed_len % 8) != 0) {
              set_error_once(error_flag, ERR_FIXED_LEN);
              return;
            }
            count = static_cast<int>(packed_len / 8);
          }

          repeated_info[row * num_repeated_fields + i].count += count;
          repeated_info[row * num_repeated_fields + i].total_length +=
            static_cast<int32_t>(packed_len);
        } else {
          // Non-packed encoding: single element
          int32_t data_offset, data_length;
          if (!get_field_data_location(cur, msg_end, wt, data_offset, data_length)) {
            set_error_once(error_flag, ERR_FIELD_SIZE);
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
          set_error_once(error_flag, ERR_WIRE_TYPE);
          return;
        }

        uint64_t len;
        int len_bytes;
        if (!read_varint(cur, msg_end, len, len_bytes)) {
          set_error_once(error_flag, ERR_VARINT);
          return;
        }

        int32_t msg_offset = static_cast<int32_t>(cur - bytes - start) + len_bytes;
        nested_locations[row * num_nested_fields + i] = {msg_offset, static_cast<int32_t>(len)};
      }
    }

    // Skip to next field
    uint8_t const* next;
    if (!skip_field(cur, msg_end, wt, next)) {
      set_error_once(error_flag, ERR_SKIP);
      return;
    }
    cur = next;
  }
}

/**
 * Scan and record all occurrences of repeated fields.
 * Called after count_repeated_fields_kernel to fill in actual locations.
 *
 * @note Time complexity: O(message_length * num_repeated_fields) per row.
 */
__global__ void scan_repeated_field_occurrences_kernel(
  cudf::column_device_view const d_in,
  device_nested_field_descriptor const* schema,
  int schema_idx,                    // Which field in schema we're scanning
  int depth_level,
  int32_t const* output_offsets,     // Pre-computed offsets from prefix sum [num_rows + 1]
  repeated_occurrence* occurrences,  // Output: all occurrences [total_count]
  int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::detail::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  if (in.nullable() && in.is_null(row)) { return; }

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  int32_t start     = in.offset_at(row) - base;
  int32_t end       = in.offset_at(row + 1) - base;
  if (!check_message_bounds(start, end, child.size(), error_flag)) { return; }

  uint8_t const* cur     = bytes + start;
  uint8_t const* msg_end = bytes + end;

  int target_fn = schema[schema_idx].field_number;
  int target_wt = schema[schema_idx].wire_type;
  int write_idx = output_offsets[row];

  while (cur < msg_end) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) { return; }
    int fn = tag.field_number;
    int wt = tag.wire_type;

    if (fn == target_fn) {
      // Check for packed encoding: wire type LEN but expected non-LEN
      bool is_packed = (wt == WT_LEN && target_wt != WT_LEN);

      if (is_packed) {
        // Packed encoding: multiple elements in a length-delimited blob
        uint64_t packed_len;
        int len_bytes;
        if (!read_varint(cur, msg_end, packed_len, len_bytes)) {
          set_error_once(error_flag, ERR_VARINT);
          return;
        }

        uint8_t const* packed_start = cur + len_bytes;
        uint8_t const* packed_end   = packed_start + packed_len;
        if (packed_end > msg_end) {
          set_error_once(error_flag, ERR_OVERFLOW);
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
              set_error_once(error_flag, ERR_VARINT);
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
            int32_t elem_offset    = static_cast<int32_t>(p - bytes - start);
            occurrences[write_idx] = {static_cast<int32_t>(row), elem_offset, 4};
            write_idx++;
            p += 4;
          }
        } else if (target_wt == WT_64BIT) {
          // Fixed 64-bit: each element is 8 bytes
          uint8_t const* p = packed_start;
          while (p + 8 <= packed_end) {
            int32_t elem_offset    = static_cast<int32_t>(p - bytes - start);
            occurrences[write_idx] = {static_cast<int32_t>(row), elem_offset, 8};
            write_idx++;
            p += 8;
          }
        }
      } else if (wt == target_wt) {
        // Non-packed encoding: single element
        int32_t data_offset, data_length;
        if (!get_field_data_location(cur, msg_end, wt, data_offset, data_length)) {
          set_error_once(error_flag, ERR_FIELD_SIZE);
          return;
        }

        int32_t abs_offset     = static_cast<int32_t>(cur - bytes - start) + data_offset;
        occurrences[write_idx] = {static_cast<int32_t>(row), abs_offset, data_length};
        write_idx++;
      }
    }

    // Skip to next field
    uint8_t const* next;
    if (!skip_field(cur, msg_end, wt, next)) {
      set_error_once(error_flag, ERR_SKIP);
      return;
    }
    cur = next;
  }
}

// ============================================================================
// Nested message scanning kernels
// ============================================================================

/**
 * Scan nested message fields.
 * Each row represents a nested message at a specific parent location.
 * This kernel finds fields within the nested message bytes.
 */
__global__ void scan_nested_message_fields_kernel(uint8_t const* message_data,
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
  if (parent_loc.offset < 0) { return; }

  auto parent_row_start       = parent_row_offsets[row] - parent_base_offset;
  uint8_t const* nested_start = message_data + parent_row_start + parent_loc.offset;
  uint8_t const* nested_end   = nested_start + parent_loc.length;

  uint8_t const* cur = nested_start;

  while (cur < nested_end) {
    proto_tag tag;
    if (!decode_tag(cur, nested_end, tag, error_flag)) { return; }
    int fn = tag.field_number;
    int wt = tag.wire_type;

    for (int f = 0; f < num_fields; f++) {
      if (field_descs[f].field_number == fn) {
        if (wt != field_descs[f].expected_wire_type) {
          set_error_once(error_flag, ERR_WIRE_TYPE);
          return;
        }

        int data_offset = static_cast<int>(cur - nested_start);

        if (wt == WT_LEN) {
          uint64_t len;
          int len_bytes;
          if (!read_varint(cur, nested_end, len, len_bytes)) {
            set_error_once(error_flag, ERR_VARINT);
            return;
          }
          if (len > static_cast<uint64_t>(nested_end - cur - len_bytes) ||
              len > static_cast<uint64_t>(INT_MAX)) {
            set_error_once(error_flag, ERR_OVERFLOW);
            return;
          }
          output_locations[row * num_fields + f] = {data_offset + len_bytes,
                                                    static_cast<int32_t>(len)};
        } else {
          int field_size = get_wire_type_size(wt, cur, nested_end);
          if (field_size < 0) {
            set_error_once(error_flag, ERR_FIELD_SIZE);
            return;
          }
          output_locations[row * num_fields + f] = {data_offset, field_size};
        }
      }
    }

    uint8_t const* next;
    if (!skip_field(cur, nested_end, wt, next)) {
      set_error_once(error_flag, ERR_SKIP);
      return;
    }
    cur = next;
  }
}

/**
 * Scan for child fields within repeated message occurrences.
 * Each occurrence is a protobuf message, and we need to find child field locations within it.
 */
__global__ void scan_repeated_message_children_kernel(
  uint8_t const* message_data,
  int32_t const* msg_row_offsets,  // Row offset for each occurrence
  field_location const*
    msg_locs,  // Location of each message occurrence (offset within row, length)
  int num_occurrences,
  field_descriptor const* child_descs,
  int num_child_fields,
  field_location* child_locs,  // Output: [num_occurrences * num_child_fields]
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
  int32_t row_offset       = msg_row_offsets[occ_idx];
  uint8_t const* msg_start = message_data + row_offset + msg_loc.offset;
  uint8_t const* msg_end   = msg_start + msg_loc.length;

  uint8_t const* cur = msg_start;

  while (cur < msg_end) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) { return; }
    int fn = tag.field_number;
    int wt = tag.wire_type;

    // Check against child field descriptors
    for (int f = 0; f < num_child_fields; f++) {
      if (child_descs[f].field_number == fn) {
        bool is_packed = (wt == WT_LEN && child_descs[f].expected_wire_type != WT_LEN);
        if (!is_packed && wt != child_descs[f].expected_wire_type) {
          set_error_once(error_flag, ERR_WIRE_TYPE);
          return;
        }

        int data_offset = static_cast<int>(cur - msg_start);

        if (wt == WT_LEN) {
          uint64_t len;
          int len_bytes;
          if (!read_varint(cur, msg_end, len, len_bytes)) {
            set_error_once(error_flag, ERR_VARINT);
            return;
          }
          // Store offset (after length prefix) and length
          child_locs[occ_idx * num_child_fields + f] = {data_offset + len_bytes,
                                                        static_cast<int32_t>(len)};
        } else {
          // For varint/fixed types, store offset and estimated length
          int32_t data_length = 0;
          if (wt == WT_VARINT) {
            uint64_t dummy;
            int vbytes;
            if (read_varint(cur, msg_end, dummy, vbytes)) { data_length = vbytes; }
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
      set_error_once(error_flag, ERR_SKIP);
      return;
    }
    cur = next;
  }
}

/**
 * Count repeated field occurrences within nested messages.
 * Similar to count_repeated_fields_kernel but operates on nested message locations.

 */
__global__ void count_repeated_in_nested_kernel(uint8_t const* message_data,
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
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  // Initialize counts
  for (int ri = 0; ri < num_repeated; ri++) {
    repeated_info[row * num_repeated + ri] = {0, 0};
  }

  auto const& parent_loc = parent_locs[row];
  if (parent_loc.offset < 0) return;

  cudf::size_type row_off;
  row_off = row_offsets[row] - base_offset;

  uint8_t const* msg_start = message_data + row_off + parent_loc.offset;
  uint8_t const* msg_end   = msg_start + parent_loc.length;
  uint8_t const* cur       = msg_start;

  while (cur < msg_end) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) { return; }
    int fn = tag.field_number;
    int wt = tag.wire_type;

    // Check if this is one of our repeated fields
    for (int ri = 0; ri < num_repeated; ri++) {
      int schema_idx = repeated_indices[ri];
      if (schema[schema_idx].field_number == fn && schema[schema_idx].is_repeated) {
        int expected_wt = schema[schema_idx].wire_type;
        bool is_packed  = (wt == WT_LEN && expected_wt != WT_LEN);

        if (!is_packed && wt != expected_wt) {
          set_error_once(error_flag, ERR_WIRE_TYPE);
          return;
        }

        if (is_packed) {
          uint64_t packed_len;
          int len_bytes;
          if (!read_varint(cur, msg_end, packed_len, len_bytes)) {
            set_error_once(error_flag, ERR_VARINT);
            return;
          }
          uint8_t const* packed_start = cur + len_bytes;
          uint8_t const* packed_end   = packed_start + packed_len;
          if (packed_end > msg_end) {
            set_error_once(error_flag, ERR_OVERFLOW);
            return;
          }

          int count = 0;
          if (expected_wt == WT_VARINT) {
            uint8_t const* p = packed_start;
            while (p < packed_end) {
              uint64_t dummy;
              int vbytes;
              if (!read_varint(p, packed_end, dummy, vbytes)) {
                set_error_once(error_flag, ERR_VARINT);
                return;
              }
              p += vbytes;
              count++;
            }
          } else if (expected_wt == WT_32BIT) {
            if ((packed_len % 4) != 0) {
              set_error_once(error_flag, ERR_FIXED_LEN);
              return;
            }
            count = static_cast<int>(packed_len / 4);
          } else if (expected_wt == WT_64BIT) {
            if ((packed_len % 8) != 0) {
              set_error_once(error_flag, ERR_FIXED_LEN);
              return;
            }
            count = static_cast<int>(packed_len / 8);
          }
          repeated_info[row * num_repeated + ri].count += count;
          repeated_info[row * num_repeated + ri].total_length += static_cast<int32_t>(packed_len);
        } else {
          int32_t data_offset, data_len;
          if (!get_field_data_location(cur, msg_end, wt, data_offset, data_len)) {
            set_error_once(error_flag, ERR_FIELD_SIZE);
            return;
          }
          repeated_info[row * num_repeated + ri].count++;
          repeated_info[row * num_repeated + ri].total_length += data_len;
        }
      }
    }

    uint8_t const* next;
    if (!skip_field(cur, msg_end, wt, next)) {
      set_error_once(error_flag, ERR_SKIP);
      return;
    }
    cur = next;
  }
}

/**
 * Scan for repeated field occurrences within nested messages.
 */
__global__ void scan_repeated_in_nested_kernel(uint8_t const* message_data,
                                               cudf::size_type const* row_offsets,
                                               cudf::size_type base_offset,
                                               field_location const* parent_locs,
                                               int num_rows,
                                               device_nested_field_descriptor const* schema,
                                               int num_fields,
                                               int32_t const* occ_prefix_sums,
                                               int num_repeated,
                                               int const* repeated_indices,
                                               repeated_occurrence* occurrences,
                                               int* error_flag)
{
  auto row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  auto const& parent_loc = parent_locs[row];
  if (parent_loc.offset < 0) return;

  // Prefix sum gives the write start offset for this row.
  int occ_offset = occ_prefix_sums[row];

  cudf::size_type row_off = row_offsets[row] - base_offset;

  uint8_t const* msg_start = message_data + row_off + parent_loc.offset;
  uint8_t const* msg_end   = msg_start + parent_loc.length;
  uint8_t const* cur       = msg_start;

  int occ_idx = 0;

  while (cur < msg_end) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) { return; }
    int fn = tag.field_number;
    int wt = tag.wire_type;

    // Check if this is one of our repeated fields.
    for (int ri = 0; ri < num_repeated; ri++) {
      int schema_idx = repeated_indices[ri];
      if (schema[schema_idx].field_number == fn && schema[schema_idx].is_repeated) {
        int expected_wt = schema[schema_idx].wire_type;
        bool is_packed  = (wt == WT_LEN && expected_wt != WT_LEN);

        if (!is_packed && wt != expected_wt) {
          set_error_once(error_flag, ERR_WIRE_TYPE);
          return;
        }

        if (is_packed) {
          uint64_t packed_len;
          int len_bytes;
          if (!read_varint(cur, msg_end, packed_len, len_bytes)) {
            set_error_once(error_flag, ERR_VARINT);
            return;
          }
          uint8_t const* packed_start = cur + len_bytes;
          uint8_t const* packed_end   = packed_start + packed_len;
          if (packed_end > msg_end) {
            set_error_once(error_flag, ERR_OVERFLOW);
            return;
          }

          if (expected_wt == WT_VARINT) {
            uint8_t const* p = packed_start;
            while (p < packed_end) {
              int32_t elem_offset = static_cast<int32_t>(p - msg_start);
              uint64_t dummy;
              int vbytes;
              if (!read_varint(p, packed_end, dummy, vbytes)) {
                set_error_once(error_flag, ERR_VARINT);
                return;
              }
              occurrences[occ_offset + occ_idx] = {row, elem_offset, vbytes};
              occ_idx++;
              p += vbytes;
            }
          } else if (expected_wt == WT_32BIT) {
            if ((packed_len % 4) != 0) {
              set_error_once(error_flag, ERR_FIXED_LEN);
              return;
            }
            for (uint64_t i = 0; i < packed_len; i += 4) {
              occurrences[occ_offset + occ_idx] = {
                row, static_cast<int32_t>(packed_start - msg_start + i), 4};
              occ_idx++;
            }
          } else if (expected_wt == WT_64BIT) {
            if ((packed_len % 8) != 0) {
              set_error_once(error_flag, ERR_FIXED_LEN);
              return;
            }
            for (uint64_t i = 0; i < packed_len; i += 8) {
              occurrences[occ_offset + occ_idx] = {
                row, static_cast<int32_t>(packed_start - msg_start + i), 8};
              occ_idx++;
            }
          }
        } else {
          int32_t data_offset = static_cast<int32_t>(cur - msg_start);
          int32_t data_len    = 0;
          if (wt == WT_LEN) {
            uint64_t len;
            int len_bytes;
            if (!read_varint(cur, msg_end, len, len_bytes)) {
              set_error_once(error_flag, ERR_VARINT);
              return;
            }
            data_offset += len_bytes;
            data_len = static_cast<int32_t>(len);
          } else if (wt == WT_VARINT) {
            uint64_t dummy;
            int vbytes;
            if (read_varint(cur, msg_end, dummy, vbytes)) { data_len = vbytes; }
          } else if (wt == WT_32BIT) {
            data_len = 4;
          } else if (wt == WT_64BIT) {
            data_len = 8;
          }

          occurrences[occ_offset + occ_idx] = {row, data_offset, data_len};
          occ_idx++;
        }
      }
    }

    uint8_t const* next;
    if (!skip_field(cur, msg_end, wt, next)) {
      set_error_once(error_flag, ERR_SKIP);
      return;
    }
    cur = next;
  }
}

/**
 * Kernel to compute nested struct locations from child field locations.
 * Replaces host-side loop that was copying data D->H, processing, then H->D.
 * This is a critical performance optimization.
 */
__global__ void compute_nested_struct_locations_kernel(
  field_location const* child_locs,  // Child field locations from parent scan
  field_location const* msg_locs,    // Parent message locations
  int32_t const* msg_row_offsets,    // Parent message row offsets
  int child_idx,                     // Which child field is the nested struct
  int num_child_fields,              // Total number of child fields per occurrence
  field_location* nested_locs,       // Output: nested struct locations
  int32_t* nested_row_offsets,       // Output: nested struct row offsets
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
  field_location const* parent_locs,  // Parent locations (row count)
  field_location const* child_locs,   // Child locations (row * num_child_fields)
  int child_idx,                      // Which child field
  int num_child_fields,               // Total child fields per row
  field_location* gc_parent_abs,      // Output: absolute grandchild parent locations
  int num_rows)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  auto const& parent_loc = parent_locs[row];
  auto const& child_loc  = child_locs[row * num_child_fields + child_idx];

  if (parent_loc.offset >= 0 && child_loc.offset >= 0) {
    // Absolute offset = parent offset + child's relative offset
    gc_parent_abs[row].offset = parent_loc.offset + child_loc.offset;
    gc_parent_abs[row].length = child_loc.length;
  } else {
    gc_parent_abs[row] = {-1, 0};
  }
}

/**
 * Compute virtual parent row offsets and locations for repeated message occurrences
 * inside nested messages. Each occurrence becomes a virtual "row" so that
 * build_nested_struct_column can recursively process the children.
 */
__global__ void compute_virtual_parents_for_nested_repeated_kernel(
  repeated_occurrence const* occurrences,
  cudf::size_type const* row_list_offsets,  // original binary input list offsets
  field_location const* parent_locations,   // parent nested message locations
  cudf::size_type* virtual_row_offsets,     // output: [total_count]
  field_location* virtual_parent_locs,      // output: [total_count]
  int total_count)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_count) return;

  auto const& occ  = occurrences[idx];
  auto const& ploc = parent_locations[occ.row_idx];

  virtual_row_offsets[idx] = row_list_offsets[occ.row_idx];

  // Keep zero-length embedded messages as "present but empty".
  // Protobuf allows an embedded message with length=0, which maps to a non-null
  // struct with all-null children (not a null struct).
  if (ploc.offset >= 0) {
    virtual_parent_locs[idx] = {ploc.offset + occ.offset, occ.length};
  } else {
    virtual_parent_locs[idx] = {-1, 0};
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

  auto const& occ      = occurrences[idx];
  msg_row_offsets[idx] = static_cast<int32_t>(list_offsets[occ.row_idx] - base_offset);
  msg_locs[idx]        = {occ.offset, occ.length};
}

/**
 * Extract a single field's locations from a 2D strided array on the GPU.
 * Replaces a D2H + CPU loop + H2D pattern for nested message location extraction.
 */
__global__ void extract_strided_locations_kernel(field_location const* nested_locations,
                                                 int field_idx,
                                                 int num_fields,
                                                 field_location* parent_locs,
                                                 int num_rows)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  parent_locs[row] = nested_locations[row * num_fields + field_idx];
}

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
      set_error_once(error_flag, ERR_REQUIRED);
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
 *
 * @note Time complexity: O(log(num_valid_values)) per row.

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

/**
 * Compute output UTF-8 length for enum-as-string rows.
 * Invalid/missing values produce length 0 (null row/field semantics handled by valid[] and
 * row_has_invalid_enum).

 */
__global__ void compute_enum_string_lengths_kernel(
  int32_t const* values,             // [num_rows] enum numeric values
  bool const* valid,                 // [num_rows] field validity
  int32_t const* valid_enum_values,  // sorted enum numeric values
  int32_t const* enum_name_offsets,  // [num_valid_values + 1]
  int num_valid_values,
  int32_t* lengths,                  // [num_rows]
  int num_rows)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  if (!valid[row]) {
    lengths[row] = 0;
    return;
  }

  int32_t val = values[row];
  int left    = 0;
  int right   = num_valid_values - 1;
  while (left <= right) {
    int mid         = left + (right - left) / 2;
    int32_t mid_val = valid_enum_values[mid];
    if (mid_val == val) {
      lengths[row] = enum_name_offsets[mid + 1] - enum_name_offsets[mid];
      return;
    } else if (mid_val < val) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  // Should not happen when validate_enum_values_kernel has already run, but keep safe.
  lengths[row] = 0;
}

/**
 * Copy enum-as-string UTF-8 bytes into output chars buffer using precomputed row offsets.
 */
__global__ void copy_enum_string_chars_kernel(
  int32_t const* values,             // [num_rows] enum numeric values
  bool const* valid,                 // [num_rows] field validity
  int32_t const* valid_enum_values,  // sorted enum numeric values
  int32_t const* enum_name_offsets,  // [num_valid_values + 1]
  uint8_t const* enum_name_chars,    // concatenated enum UTF-8 names
  int num_valid_values,
  int32_t const* output_offsets,     // [num_rows + 1]
  char* out_chars,                   // [total_chars]
  int num_rows)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;
  if (!valid[row]) return;

  int32_t val = values[row];
  int left    = 0;
  int right   = num_valid_values - 1;
  while (left <= right) {
    int mid         = left + (right - left) / 2;
    int32_t mid_val = valid_enum_values[mid];
    if (mid_val == val) {
      int32_t src_begin = enum_name_offsets[mid];
      int32_t src_end   = enum_name_offsets[mid + 1];
      int32_t dst_begin = output_offsets[row];
      for (int32_t i = 0; i < (src_end - src_begin); ++i) {
        out_chars[dst_begin + i] = static_cast<char>(enum_name_chars[src_begin + i]);
      }
      return;
    } else if (mid_val < val) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
}

}  // namespace spark_rapids_jni::protobuf_detail
