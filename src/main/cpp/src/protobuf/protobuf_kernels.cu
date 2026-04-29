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

#include "protobuf/protobuf_kernels.cuh"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace spark_rapids_jni::protobuf::detail {

namespace {

// ============================================================================
// Pass 1: Scan all fields kernel - records (offset, length) for each field
// ============================================================================

CUDF_KERNEL void set_error_if_unset_kernel(int* error_flag, int error_code)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) { set_error_once(error_flag, error_code); }
}

/**
 * Fused scanning kernel: scans each message once and records the location
 * of all requested fields.
 *
 * For "last one wins" semantics (protobuf standard for repeated scalars),
 * we continue scanning even after finding a field.
 *
 * If a row hits a parse error that leaves the cursor in an unsafe state (for example, malformed
 * varint bytes or a schema-matching field with the wrong wire type), the scan aborts for that row
 * instead of guessing where the next field begins. In permissive mode the caller may also supply a
 * row-level invalidity buffer so the full struct row can be nulled to match Spark CPU semantics for
 * malformed messages.
 */
CUDF_KERNEL void scan_all_fields_kernel(
  cudf::column_device_view const d_in,
  field_descriptor const* field_descs,  // [num_fields]
  int num_fields,
  int const* field_lookup,    // direct-mapped lookup table (nullable)
  int field_lookup_size,      // size of lookup table (0 if null)
  field_location* locations,  // [num_rows * num_fields] row-major
  int* error_flag,
  bool* row_has_invalid_data)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::detail::lists_column_device_view in{d_in};
  if (row >= in.size()) { return; }

  auto mark_row_error = [&]() {
    if (row_has_invalid_data != nullptr) { row_has_invalid_data[row] = true; }
  };

  for (int f = 0; f < num_fields; f++) {
    locations[flat_index(
      static_cast<size_t>(row), static_cast<size_t>(num_fields), static_cast<size_t>(f))] = {-1, 0};
  }

  if (in.nullable() && in.is_null(row)) { return; }

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  int32_t start     = in.offset_at(row) - base;
  int32_t end       = in.offset_at(row + 1) - base;

  if (!check_message_bounds(start, end, child.size(), error_flag)) {
    mark_row_error();
    return;
  }

  uint8_t const* cur     = bytes + start;
  uint8_t const* msg_end = bytes + end;

  while (cur < msg_end) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) {
      mark_row_error();
      return;
    }
    int fn = tag.field_number;
    int wt = tag.wire_type;

    int f = lookup_field(fn, field_lookup, field_lookup_size, field_descs, num_fields);
    if (f >= 0) {
      if (wt != field_descs[f].expected_wire_type) {
        set_error_once(error_flag, ERR_WIRE_TYPE);
        mark_row_error();
        return;
      }

      // Record the location (relative to message start)
      int data_offset = static_cast<int>(cur - bytes - start);

      if (wt == wire_type_value(proto_wire_type::LEN)) {
        // For length-delimited, record offset after length prefix and the data length
        uint64_t len;
        int len_bytes;
        if (!read_varint(cur, msg_end, len, len_bytes)) {
          set_error_once(error_flag, ERR_VARINT);
          mark_row_error();
          return;
        }
        if (len > static_cast<uint64_t>(msg_end - cur - len_bytes) ||
            len > static_cast<uint64_t>(cuda::std::numeric_limits<int>::max())) {
          set_error_once(error_flag, ERR_OVERFLOW);
          mark_row_error();
          return;
        }
        // Record offset pointing to the actual data (after length prefix)
        int32_t data_location;
        if (!checked_add_int32(data_offset, len_bytes, data_location)) {
          set_error_once(error_flag, ERR_OVERFLOW);
          mark_row_error();
          return;
        }
        locations[flat_index(
          static_cast<size_t>(row), static_cast<size_t>(num_fields), static_cast<size_t>(f))] = {
          data_location, static_cast<int32_t>(len)};
      } else {
        // For fixed-size and varint fields, record offset and compute length
        int field_size = get_wire_type_size(wt, cur, msg_end);
        if (field_size < 0) {
          set_error_once(error_flag, ERR_FIELD_SIZE);
          mark_row_error();
          return;
        }
        locations[flat_index(
          static_cast<size_t>(row), static_cast<size_t>(num_fields), static_cast<size_t>(f))] = {
          data_offset, field_size};
      }
    }

    // Skip to next field
    uint8_t const* next;
    if (!skip_field(cur, msg_end, wt, next)) {
      set_error_once(error_flag, ERR_SKIP);
      mark_row_error();
      return;
    }
    cur = next;
  }
}

// ============================================================================
// Shared device functions for repeated field processing
// ============================================================================

/**
 * Count a single repeated field occurrence (packed or unpacked).
 * Updates info.count. Returns false on error (error_flag set), true on success.
 */
__device__ bool count_repeated_element(uint8_t const* cur,
                                       uint8_t const* msg_end,
                                       int wt,
                                       int expected_wt,
                                       repeated_field_info& info,
                                       int* error_flag)
{
  bool is_packed = (wt == wire_type_value(proto_wire_type::LEN) &&
                    expected_wt != wire_type_value(proto_wire_type::LEN));

  if (!is_packed && wt != expected_wt) {
    set_error_once(error_flag, ERR_WIRE_TYPE);
    return false;
  }

  if (is_packed) {
    uint64_t packed_len;
    int len_bytes;
    if (!read_varint(cur, msg_end, packed_len, len_bytes)) {
      set_error_once(error_flag, ERR_VARINT);
      return false;
    }
    uint8_t const* packed_start = cur + len_bytes;
    if (packed_len > static_cast<uint64_t>(msg_end - packed_start)) {
      set_error_once(error_flag, ERR_OVERFLOW);
      return false;
    }
    uint8_t const* packed_end = packed_start + packed_len;

    int count = 0;
    if (expected_wt == wire_type_value(proto_wire_type::VARINT)) {
      uint8_t const* p = packed_start;
      while (p < packed_end) {
        uint64_t dummy;
        int vbytes;
        if (!read_varint(p, packed_end, dummy, vbytes)) {
          set_error_once(error_flag, ERR_VARINT);
          return false;
        }
        p += vbytes;
        count++;
      }
    } else if (expected_wt == wire_type_value(proto_wire_type::I32BIT)) {
      if ((packed_len % 4) != 0) {
        set_error_once(error_flag, ERR_FIXED_LEN);
        return false;
      }
      count = static_cast<int>(packed_len / 4);
    } else if (expected_wt == wire_type_value(proto_wire_type::I64BIT)) {
      if ((packed_len % 8) != 0) {
        set_error_once(error_flag, ERR_FIXED_LEN);
        return false;
      }
      count = static_cast<int>(packed_len / 8);
    }

    info.count += count;
  } else {
    int32_t data_offset, data_length;
    if (!get_field_data_location(cur, msg_end, wt, data_offset, data_length)) {
      set_error_once(error_flag, ERR_FIELD_SIZE);
      return false;
    }
    info.count++;
  }
  return true;
}

/**
 * Record a single repeated field occurrence (packed or unpacked).
 * Writes to occurrences[write_idx] and advances write_idx.
 * Offsets are computed relative to msg_base.
 * Returns false on error (error_flag set), true on success.
 */
__device__ bool scan_repeated_element(uint8_t const* cur,
                                      uint8_t const* msg_end,
                                      uint8_t const* msg_base,
                                      int wt,
                                      int expected_wt,
                                      int32_t row,
                                      repeated_occurrence* occurrences,
                                      int& write_idx,
                                      int write_end,
                                      int* error_flag)
{
  bool is_packed = (wt == wire_type_value(proto_wire_type::LEN) &&
                    expected_wt != wire_type_value(proto_wire_type::LEN));

  if (!is_packed && wt != expected_wt) {
    set_error_once(error_flag, ERR_WIRE_TYPE);
    return false;
  }

  if (is_packed) {
    uint64_t packed_len;
    int len_bytes;
    if (!read_varint(cur, msg_end, packed_len, len_bytes)) {
      set_error_once(error_flag, ERR_VARINT);
      return false;
    }
    uint8_t const* packed_start = cur + len_bytes;
    if (packed_len > static_cast<uint64_t>(msg_end - packed_start)) {
      set_error_once(error_flag, ERR_OVERFLOW);
      return false;
    }
    uint8_t const* packed_end = packed_start + packed_len;

    if (expected_wt == wire_type_value(proto_wire_type::VARINT)) {
      uint8_t const* p = packed_start;
      while (p < packed_end) {
        int32_t elem_offset = static_cast<int32_t>(p - msg_base);
        uint64_t dummy;
        int vbytes;
        if (!read_varint(p, packed_end, dummy, vbytes)) {
          set_error_once(error_flag, ERR_VARINT);
          return false;
        }
        if (write_idx >= write_end) {
          set_error_once(error_flag, ERR_REPEATED_COUNT_MISMATCH);
          return false;
        }
        occurrences[write_idx] = {row, elem_offset, vbytes};
        write_idx++;
        p += vbytes;
      }
    } else if (expected_wt == wire_type_value(proto_wire_type::I32BIT)) {
      if ((packed_len % 4) != 0) {
        set_error_once(error_flag, ERR_FIXED_LEN);
        return false;
      }
      for (uint64_t i = 0; i < packed_len; i += 4) {
        if (write_idx >= write_end) {
          set_error_once(error_flag, ERR_REPEATED_COUNT_MISMATCH);
          return false;
        }
        occurrences[write_idx] = {row, static_cast<int32_t>(packed_start - msg_base + i), 4};
        write_idx++;
      }
    } else if (expected_wt == wire_type_value(proto_wire_type::I64BIT)) {
      if ((packed_len % 8) != 0) {
        set_error_once(error_flag, ERR_FIXED_LEN);
        return false;
      }
      for (uint64_t i = 0; i < packed_len; i += 8) {
        if (write_idx >= write_end) {
          set_error_once(error_flag, ERR_REPEATED_COUNT_MISMATCH);
          return false;
        }
        occurrences[write_idx] = {row, static_cast<int32_t>(packed_start - msg_base + i), 8};
        write_idx++;
      }
    }
  } else {
    int32_t data_offset, data_length;
    if (!get_field_data_location(cur, msg_end, wt, data_offset, data_length)) {
      set_error_once(error_flag, ERR_FIELD_SIZE);
      return false;
    }
    if (write_idx >= write_end) {
      set_error_once(error_flag, ERR_REPEATED_COUNT_MISMATCH);
      return false;
    }
    int32_t abs_offset     = static_cast<int32_t>(cur - msg_base) + data_offset;
    occurrences[write_idx] = {row, abs_offset, data_length};
    write_idx++;
  }
  return true;
}

// ============================================================================
// Pass 1b: Count repeated fields kernel
// ============================================================================

/**
 * Count occurrences of repeated fields in each row.
 * Also records locations of nested message fields for hierarchical processing.
 *
 * Optional lookup tables (fn_to_rep_idx, fn_to_nested_idx) provide O(1) field_number
 * to local index mapping. When nullptr, falls back to linear search.
 */
CUDF_KERNEL void count_repeated_fields_kernel(cudf::column_device_view const d_in,
                                              device_nested_field_descriptor const* schema,
                                              int num_fields,
                                              int depth_level,
                                              repeated_field_info* repeated_info,
                                              int num_repeated_fields,
                                              int const* repeated_field_indices,
                                              field_location* nested_locations,
                                              int num_nested_fields,
                                              int const* nested_field_indices,
                                              int* error_flag,
                                              int const* fn_to_rep_idx,
                                              int fn_to_rep_size,
                                              int const* fn_to_nested_idx,
                                              int fn_to_nested_size)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::detail::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  // Initialize repeated counts to 0
  for (int f = 0; f < num_repeated_fields; f++) {
    repeated_info[flat_index(
      static_cast<size_t>(row), static_cast<size_t>(num_repeated_fields), static_cast<size_t>(f))] =
      {0};
  }

  // Initialize nested locations to not found
  for (int f = 0; f < num_nested_fields; f++) {
    nested_locations[flat_index(
      static_cast<size_t>(row), static_cast<size_t>(num_nested_fields), static_cast<size_t>(f))] = {
      -1, 0};
  }

  if (in.nullable() && in.is_null(row)) return;

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  int32_t start     = in.offset_at(row) - base;
  int32_t end       = in.offset_at(row + 1) - base;
  if (!check_message_bounds(start, end, child.size(), error_flag)) return;

  uint8_t const* cur     = bytes + start;
  uint8_t const* msg_end = bytes + end;

  while (cur < msg_end) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) return;
    int fn = tag.field_number;
    int wt = tag.wire_type;

    if (fn_to_rep_idx != nullptr && fn > 0 && fn < fn_to_rep_size) {
      int i = fn_to_rep_idx[fn];
      if (i >= 0) {
        int schema_idx = repeated_field_indices[i];
        // Defensive: also verify field_number matches, mirroring the linear-scan fallback below.
        // A correct lookup table guarantees this, but failing closed on a buggy table is cheap
        // and avoids dispatching to the wrong field index silently.
        if (schema[schema_idx].field_number == fn && schema[schema_idx].depth == depth_level &&
            !count_repeated_element(
              cur,
              msg_end,
              wt,
              schema[schema_idx].wire_type,
              repeated_info[flat_index(static_cast<size_t>(row),
                                       static_cast<size_t>(num_repeated_fields),
                                       static_cast<size_t>(i))],
              error_flag)) {
          return;
        }
      }
    } else {
      for (int i = 0; i < num_repeated_fields; i++) {
        int schema_idx = repeated_field_indices[i];
        if (schema[schema_idx].field_number == fn && schema[schema_idx].depth == depth_level) {
          if (!count_repeated_element(
                cur,
                msg_end,
                wt,
                schema[schema_idx].wire_type,
                repeated_info[flat_index(static_cast<size_t>(row),
                                         static_cast<size_t>(num_repeated_fields),
                                         static_cast<size_t>(i))],
                error_flag)) {
            return;
          }
          break;  // Each tag dispatches to at most one repeated index (mirrors lookup-table path)
        }
      }
    }

    // Check nested message fields at this depth
    auto handle_nested = [&](int i) {
      if (wt != wire_type_value(proto_wire_type::LEN)) {
        set_error_once(error_flag, ERR_WIRE_TYPE);
        return false;
      }
      uint64_t len;
      int len_bytes;
      if (!read_varint(cur, msg_end, len, len_bytes)) {
        set_error_once(error_flag, ERR_VARINT);
        return false;
      }
      if (len > static_cast<uint64_t>(msg_end - cur - len_bytes) ||
          len > static_cast<uint64_t>(cuda::std::numeric_limits<int>::max())) {
        set_error_once(error_flag, ERR_OVERFLOW);
        return false;
      }
      auto const rel_offset64 = static_cast<int64_t>(cur - bytes - start);
      if (rel_offset64 < cuda::std::numeric_limits<int32_t>::min() ||
          rel_offset64 > cuda::std::numeric_limits<int32_t>::max()) {
        set_error_once(error_flag, ERR_OVERFLOW);
        return false;
      }
      int32_t msg_offset;
      if (!checked_add_int32(static_cast<int32_t>(rel_offset64), len_bytes, msg_offset)) {
        set_error_once(error_flag, ERR_OVERFLOW);
        return false;
      }
      nested_locations[flat_index(
        static_cast<size_t>(row), static_cast<size_t>(num_nested_fields), static_cast<size_t>(i))] =
        {msg_offset, static_cast<int32_t>(len)};
      return true;
    };

    if (fn_to_nested_idx != nullptr && fn > 0 && fn < fn_to_nested_size) {
      int i = fn_to_nested_idx[fn];
      if (i >= 0) {
        int schema_idx = nested_field_indices[i];
        // Defensive: verify field_number matches the lookup, mirroring the linear-scan fallback.
        if (schema[schema_idx].field_number == fn && schema[schema_idx].depth == depth_level) {
          if (!handle_nested(i)) return;
        }
      }
    } else {
      for (int i = 0; i < num_nested_fields; i++) {
        int schema_idx = nested_field_indices[i];
        if (schema[schema_idx].field_number == fn && schema[schema_idx].depth == depth_level) {
          if (!handle_nested(i)) return;
          break;  // Each tag dispatches to at most one nested index (mirrors lookup-table path)
        }
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
 * Combined occurrence scan: scans each message ONCE and writes occurrences for ALL
 * repeated fields simultaneously, scanning each message only once.
 */
CUDF_KERNEL void scan_all_repeated_occurrences_kernel(cudf::column_device_view const d_in,
                                                      repeated_field_scan_desc const* scan_descs,
                                                      int num_scan_fields,
                                                      int* error_flag,
                                                      int const* fn_to_desc_idx,
                                                      int fn_to_desc_size)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  cudf::detail::lists_column_device_view in{d_in};
  if (row >= in.size()) return;

  if (in.nullable() && in.is_null(row)) return;

  auto const base   = in.offset_at(0);
  auto const child  = in.get_sliced_child();
  auto const* bytes = reinterpret_cast<uint8_t const*>(child.data<int8_t>());
  int32_t start     = in.offset_at(row) - base;
  int32_t end       = in.offset_at(row + 1) - base;
  if (!check_message_bounds(start, end, child.size(), error_flag)) return;

  uint8_t const* cur     = bytes + start;
  uint8_t const* msg_end = bytes + end;

  // The host validates `num_repeated <= MAX_REPEATED_FIELDS_PER_KERNEL` in
  // `validate_decode_context`, so this device-side check is a defense-in-depth assert. Keeping
  // the cap small keeps `write_idx` in registers / L1 instead of spilling into local memory.
  if (num_scan_fields > MAX_REPEATED_FIELDS_PER_KERNEL) {
    set_error_once(error_flag, ERR_SCHEMA_TOO_LARGE);
    return;
  }
  int write_idx[MAX_REPEATED_FIELDS_PER_KERNEL];
  for (int f = 0; f < num_scan_fields; f++) {
    write_idx[f] = scan_descs[f].row_offsets[row];
  }

  while (cur < msg_end) {
    proto_tag tag;
    if (!decode_tag(cur, msg_end, tag, error_flag)) return;
    int fn = tag.field_number;
    int wt = tag.wire_type;

    auto try_scan = [&](int f) -> bool {
      int target_wt  = scan_descs[f].wire_type;
      bool is_packed = (wt == wire_type_value(proto_wire_type::LEN) &&
                        target_wt != wire_type_value(proto_wire_type::LEN));
      if (is_packed || wt == target_wt) {
        return scan_repeated_element(cur,
                                     msg_end,
                                     bytes + start,
                                     wt,
                                     target_wt,
                                     static_cast<int32_t>(row),
                                     scan_descs[f].occurrences,
                                     write_idx[f],
                                     scan_descs[f].row_offsets[row + 1],
                                     error_flag);
      }
      set_error_once(error_flag, ERR_WIRE_TYPE);
      return false;
    };

    if (fn_to_desc_idx != nullptr && fn > 0 && fn < fn_to_desc_size) {
      int f = fn_to_desc_idx[fn];
      // Defensive sanity check: a correct lookup table guarantees scan_descs[f].field_number == fn,
      // but verifying matches the linear-scan fallback's behavior so a buggy table cannot dispatch
      // to the wrong descriptor silently.
      if (f >= 0 && f < num_scan_fields && scan_descs[f].field_number == fn) {
        if (!try_scan(f)) return;
      }
    } else {
      for (int f = 0; f < num_scan_fields; f++) {
        if (scan_descs[f].field_number == fn) {
          if (!try_scan(f)) return;
          break;  // Each tag dispatches to at most one scan descriptor
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

  for (int f = 0; f < num_scan_fields; f++) {
    if (write_idx[f] != scan_descs[f].row_offsets[row + 1]) {
      set_error_once(error_flag, ERR_REPEATED_COUNT_MISMATCH);
      return;
    }
  }
}

// ============================================================================
// Kernel to check required fields after scan pass
// ============================================================================

/**
 * Check if any required fields are missing (offset < 0) and set error flag.
 * This is called after the scan pass to validate required field constraints.
 */
CUDF_KERNEL void check_required_fields_kernel(
  field_location const* locations,  // [num_rows * num_fields]
  uint8_t const* is_required,       // [num_fields] (1 = required, 0 = optional)
  int num_fields,
  int num_rows,
  cudf::bitmask_type const* input_null_mask,  // optional top-level input null mask
  cudf::size_type input_offset,               // bit offset for sliced top-level input
  field_location const* parent_locs,          // [num_rows] optional parent presence for nested rows
  bool* row_force_null,            // [top_level_num_rows] optional permissive row nulling
  int32_t const* top_row_indices,  // [num_rows] optional nested-row -> top-row mapping
  int* error_flag)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;
  if (input_null_mask != nullptr && !cudf::bit_is_set(input_null_mask, row + input_offset)) {
    return;
  }
  if (parent_locs != nullptr && parent_locs[row].offset < 0) return;

  for (int f = 0; f < num_fields; f++) {
    if (is_required[f] != 0 && locations[flat_index(static_cast<size_t>(row),
                                                    static_cast<size_t>(num_fields),
                                                    static_cast<size_t>(f))]
                                   .offset < 0) {
      if (row_force_null != nullptr) {
        auto const top_row =
          top_row_indices != nullptr ? top_row_indices[row] : static_cast<int32_t>(row);
        row_force_null[top_row] = true;
      }
      // Required field is missing - set error flag
      set_error_once(error_flag, ERR_REQUIRED);
      return;  // No need to check other fields for this row
    }
  }
}

/**
 * Binary search a sorted enum-value array. Returns the matched index or -1 if not found.
 * Shared between the validate / lengths / chars enum-as-string kernels.
 */
__device__ inline int enum_binary_search(int32_t const* valid_enum_values,
                                         int num_valid_values,
                                         int32_t val)
{
  int left  = 0;
  int right = num_valid_values - 1;
  while (left <= right) {
    int mid         = left + (right - left) / 2;
    int32_t mid_val = valid_enum_values[mid];
    if (mid_val == val) {
      return mid;
    } else if (mid_val < val) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
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
CUDF_KERNEL void validate_enum_values_kernel(
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

  if (enum_binary_search(valid_enum_values, num_valid_values, values[row]) < 0) {
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
CUDF_KERNEL void compute_enum_string_lengths_kernel(
  int32_t const* values,             // [num_rows] enum numeric values
  bool const* valid,                 // [num_rows] field validity
  int32_t const* valid_enum_values,  // sorted enum numeric values
  int32_t const* enum_name_offsets,  // [num_valid_values + 1]
  int num_valid_values,
  int32_t* lengths,  // [num_rows]
  int num_rows)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;

  if (!valid[row]) {
    lengths[row] = 0;
    return;
  }

  int idx = enum_binary_search(valid_enum_values, num_valid_values, values[row]);
  // Should not happen when validate_enum_values_kernel has already run, but keep safe.
  lengths[row] = idx >= 0 ? (enum_name_offsets[idx + 1] - enum_name_offsets[idx]) : 0;
}

/**
 * Copy enum-as-string UTF-8 bytes into output chars buffer using precomputed row offsets.
 */
CUDF_KERNEL void copy_enum_string_chars_kernel(
  int32_t const* values,             // [num_rows] enum numeric values
  bool const* valid,                 // [num_rows] field validity
  int32_t const* valid_enum_values,  // sorted enum numeric values
  int32_t const* enum_name_offsets,  // [num_valid_values + 1]
  uint8_t const* enum_name_chars,    // concatenated enum UTF-8 names
  int num_valid_values,
  int32_t const* output_offsets,  // [num_rows + 1]
  char* out_chars,                // [total_chars]
  int num_rows)
{
  auto row = static_cast<cudf::size_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= num_rows) return;
  if (!valid[row]) return;

  int idx = enum_binary_search(valid_enum_values, num_valid_values, values[row]);
  if (idx < 0) return;
  int32_t src_begin = enum_name_offsets[idx];
  int32_t src_end   = enum_name_offsets[idx + 1];
  int32_t dst_begin = output_offsets[row];
  memcpy(
    out_chars + dst_begin, enum_name_chars + src_begin, static_cast<size_t>(src_end - src_begin));
}

}  // anonymous namespace

// ============================================================================
// Host wrapper functions — callable from other translation units
// ============================================================================

void set_error_once_async(int* error_flag, int error_code, rmm::cuda_stream_view stream)
{
  set_error_if_unset_kernel<<<1, 1, 0, stream.value()>>>(error_flag, error_code);
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

void launch_scan_all_fields(cudf::column_device_view const& d_in,
                            field_descriptor const* field_descs,
                            int num_fields,
                            int const* field_lookup,
                            int field_lookup_size,
                            field_location* locations,
                            int* error_flag,
                            bool* row_has_invalid_data,
                            int num_rows,
                            rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(d_in,
                                                                           field_descs,
                                                                           num_fields,
                                                                           field_lookup,
                                                                           field_lookup_size,
                                                                           locations,
                                                                           error_flag,
                                                                           row_has_invalid_data);
}

void launch_count_repeated_fields(cudf::column_device_view const& d_in,
                                  device_nested_field_descriptor const* schema,
                                  int num_fields,
                                  int depth_level,
                                  repeated_field_info* repeated_info,
                                  int num_repeated_fields,
                                  int const* repeated_field_indices,
                                  field_location* nested_locations,
                                  int num_nested_fields,
                                  int const* nested_field_indices,
                                  int* error_flag,
                                  int const* fn_to_rep_idx,
                                  int fn_to_rep_size,
                                  int const* fn_to_nested_idx,
                                  int fn_to_nested_size,
                                  int num_rows,
                                  rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  count_repeated_fields_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    d_in,
    schema,
    num_fields,
    depth_level,
    repeated_info,
    num_repeated_fields,
    repeated_field_indices,
    nested_locations,
    num_nested_fields,
    nested_field_indices,
    error_flag,
    fn_to_rep_idx,
    fn_to_rep_size,
    fn_to_nested_idx,
    fn_to_nested_size);
}

void launch_scan_all_repeated_occurrences(cudf::column_device_view const& d_in,
                                          repeated_field_scan_desc const* scan_descs,
                                          int num_scan_fields,
                                          int* error_flag,
                                          int const* fn_to_desc_idx,
                                          int fn_to_desc_size,
                                          int num_rows,
                                          rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  scan_all_repeated_occurrences_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    d_in, scan_descs, num_scan_fields, error_flag, fn_to_desc_idx, fn_to_desc_size);
}

void launch_validate_enum_values(int32_t const* values,
                                 bool* valid,
                                 bool* row_has_invalid_enum,
                                 int32_t const* valid_enum_values,
                                 int num_valid_values,
                                 int num_rows,
                                 rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  validate_enum_values_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    values, valid, row_has_invalid_enum, valid_enum_values, num_valid_values, num_rows);
}

void launch_compute_enum_string_lengths(int32_t const* values,
                                        bool const* valid,
                                        int32_t const* valid_enum_values,
                                        int32_t const* enum_name_offsets,
                                        int num_valid_values,
                                        int32_t* lengths,
                                        int num_rows,
                                        rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  compute_enum_string_lengths_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    values, valid, valid_enum_values, enum_name_offsets, num_valid_values, lengths, num_rows);
}

void launch_copy_enum_string_chars(int32_t const* values,
                                   bool const* valid,
                                   int32_t const* valid_enum_values,
                                   int32_t const* enum_name_offsets,
                                   uint8_t const* enum_name_chars,
                                   int num_valid_values,
                                   int32_t const* output_offsets,
                                   char* out_chars,
                                   int num_rows,
                                   rmm::cuda_stream_view stream)
{
  if (num_rows == 0) return;
  auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  copy_enum_string_chars_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(values,
                                                                                  valid,
                                                                                  valid_enum_values,
                                                                                  enum_name_offsets,
                                                                                  enum_name_chars,
                                                                                  num_valid_values,
                                                                                  output_offsets,
                                                                                  out_chars,
                                                                                  num_rows);
}

void maybe_check_required_fields(field_location const* locations,
                                 std::vector<int> const& field_indices,
                                 std::vector<nested_field_descriptor> const& schema,
                                 int num_rows,
                                 cudf::bitmask_type const* input_null_mask,
                                 cudf::size_type input_offset,
                                 field_location const* parent_locs,
                                 bool* row_force_null,
                                 int32_t const* top_row_indices,
                                 int* error_flag,
                                 rmm::cuda_stream_view stream)
{
  if (num_rows == 0 || field_indices.empty()) { return; }

  bool has_required = false;
  auto h_is_required =
    cudf::detail::make_pinned_vector_async<uint8_t>(field_indices.size(), stream);
  for (size_t i = 0; i < field_indices.size(); ++i) {
    h_is_required[i] = schema[field_indices[i]].is_required ? 1 : 0;
    has_required |= (h_is_required[i] != 0);
  }
  if (!has_required) { return; }

  auto d_is_required = cudf::detail::make_device_uvector_async(
    h_is_required, stream, cudf::get_current_device_resource_ref());

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

void propagate_invalid_enum_flags_to_rows(rmm::device_uvector<bool> const& item_invalid,
                                          rmm::device_uvector<bool>& row_invalid,
                                          int num_items,
                                          int32_t const* top_row_indices,
                                          bool propagate_to_rows,
                                          rmm::cuda_stream_view stream)
{
  if (num_items == 0 || row_invalid.size() == 0 || !propagate_to_rows) return;

  if (top_row_indices == nullptr) {
    CUDF_EXPECTS(static_cast<size_t>(num_items) <= row_invalid.size(),
                 "enum invalid-row propagation exceeded row buffer");
    thrust::transform(rmm::exec_policy_nosync(stream),
                      row_invalid.begin(),
                      row_invalid.begin() + num_items,
                      item_invalid.begin(),
                      row_invalid.begin(),
                      [] __device__(bool row_is_invalid, bool item_is_invalid) {
                        return row_is_invalid || item_is_invalid;
                      });
    return;
  }

  // Multiple items may share the same `top_row_indices[idx]` (e.g. several occurrences of a
  // packed repeated enum within one row), so concurrent threads can race on the same byte.
  // Although every racing write stores the same value (`true`), non-atomic concurrent writes
  // to the same address are UB under the CUDA memory model. Use atomic_ref like set_error_once.
  thrust::for_each(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(num_items),
    [item_invalid = item_invalid.data(),
     top_row_indices,
     row_invalid = row_invalid.data()] __device__(int idx) {
      if (item_invalid[idx]) {
        cuda::atomic_ref<bool, cuda::thread_scope_device> ref(row_invalid[top_row_indices[idx]]);
        ref.store(true, cuda::memory_order_relaxed);
      }
    });
}

void validate_enum_and_propagate_rows(rmm::device_uvector<int32_t> const& values,
                                      rmm::device_uvector<bool>& valid,
                                      cudf::detail::host_vector<int32_t> const& valid_enums,
                                      rmm::device_uvector<bool>& row_invalid,
                                      int num_items,
                                      int32_t const* top_row_indices,
                                      bool propagate_to_rows,
                                      rmm::cuda_stream_view stream)
{
  if (num_items == 0 || valid_enums.empty()) return;

  auto const blocks  = static_cast<int>((num_items + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
  auto d_valid_enums = cudf::detail::make_device_uvector_async(
    valid_enums, stream, cudf::get_current_device_resource_ref());

  rmm::device_uvector<bool> item_invalid(
    num_items, stream, cudf::get_current_device_resource_ref());
  thrust::fill(rmm::exec_policy_nosync(stream), item_invalid.begin(), item_invalid.end(), false);
  validate_enum_values_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream.value()>>>(
    values.data(),
    valid.data(),
    item_invalid.data(),
    d_valid_enums.data(),
    static_cast<int>(valid_enums.size()),
    num_items);

  propagate_invalid_enum_flags_to_rows(
    item_invalid, row_invalid, num_items, top_row_indices, propagate_to_rows, stream);
}

}  // namespace spark_rapids_jni::protobuf::detail
