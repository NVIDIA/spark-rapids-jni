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

#include "protobuf.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <array>
#include <map>
#include <type_traits>

namespace spark_rapids_jni::protobuf_detail {

// Wire type constants (protobuf encoding spec)
constexpr int WT_VARINT = 0;
constexpr int WT_64BIT  = 1;
constexpr int WT_LEN    = 2;
constexpr int WT_SGROUP = 3;
constexpr int WT_EGROUP = 4;
constexpr int WT_32BIT  = 5;

// Protobuf varint encoding uses at most 10 bytes to represent a 64-bit value.
constexpr int MAX_VARINT_BYTES = 10;

// CUDA kernel launch configuration.
constexpr int THREADS_PER_BLOCK = 256;

// Error codes for kernel error reporting.
constexpr int ERR_BOUNDS       = 1;
constexpr int ERR_VARINT       = 2;
constexpr int ERR_FIELD_NUMBER = 3;
constexpr int ERR_WIRE_TYPE    = 4;
constexpr int ERR_OVERFLOW     = 5;
constexpr int ERR_FIELD_SIZE   = 6;
constexpr int ERR_SKIP         = 7;
constexpr int ERR_FIXED_LEN    = 8;
constexpr int ERR_REQUIRED     = 9;

// Maximum supported nesting depth for recursive struct decoding.
constexpr int MAX_NESTED_STRUCT_DECODE_DEPTH = 10;

// Threshold for using a direct-mapped lookup table for field_number -> field_index.
// Field numbers above this threshold fall back to linear search.
constexpr int FIELD_LOOKUP_TABLE_MAX = 4096;

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
  int32_t count;         // Number of occurrences in this row
  int32_t total_length;  // Total bytes for all occurrences (for varlen fields)
};

/**
 * Location of a single occurrence of a repeated field.
 */
struct repeated_occurrence {
  int32_t row_idx;  // Which row this occurrence belongs to
  int32_t offset;   // Offset within the message
  int32_t length;   // Length of the field data
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

  device_nested_field_descriptor() = default;

  explicit device_nested_field_descriptor(spark_rapids_jni::nested_field_descriptor const& src)
    : field_number(src.field_number),
      parent_idx(src.parent_idx),
      depth(src.depth),
      wire_type(src.wire_type),
      output_type_id(static_cast<int>(src.output_type)),
      encoding(src.encoding),
      is_repeated(src.is_repeated),
      is_required(src.is_required),
      has_default_value(src.has_default_value)
  {
  }
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
  // Protobuf varint uses 7 bits per byte with MSB as continuation flag.
  // A 64-bit value requires at most ceil(64/7) = 10 bytes.
  while (cur < end && bytes < MAX_VARINT_BYTES) {
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

__device__ inline void set_error_once(int* error_flag, int error_code)
{
  atomicCAS(error_flag, 0, error_code);
}

__device__ inline int get_wire_type_size(int wt, uint8_t const* cur, uint8_t const* end)
{
  switch (wt) {
    case WT_VARINT: {
      // Need to scan to find the end of varint
      int count = 0;
      while (cur < end && count < MAX_VARINT_BYTES) {
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
      if (len > static_cast<uint64_t>(end - cur - n) || len > static_cast<uint64_t>(INT_MAX - n))
        return -1;
      return n + static_cast<int>(len);
    }
    case WT_SGROUP: {
      auto const* start = cur;
      // Recursively skip until the matching end-group tag.
      while (cur < end) {
        uint64_t key;
        int key_bytes;
        if (!read_varint(cur, end, key, key_bytes)) return -1;
        cur += key_bytes;

        int inner_wt = static_cast<int>(key & 0x7);
        if (inner_wt == WT_EGROUP) { return static_cast<int>(cur - start); }

        int inner_size = get_wire_type_size(inner_wt, cur, end);
        if (inner_size < 0 || cur + inner_size > end) return -1;
        cur += inner_size;
      }
      return -1;
    }
    case WT_EGROUP: return 0;
    default: return -1;
  }
}

__device__ inline bool skip_field(uint8_t const* cur,
                                  uint8_t const* end,
                                  int wt,
                                  uint8_t const*& out_cur)
{
  // End-group is handled by the parent group parser.
  if (wt == WT_EGROUP) {
    out_cur = cur;
    return true;
  }

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
__device__ inline bool get_field_data_location(
  uint8_t const* cur, uint8_t const* end, int wt, int32_t& data_offset, int32_t& data_length)
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

__device__ inline bool check_message_bounds(int32_t start,
                                            int32_t end_pos,
                                            cudf::size_type total_size,
                                            int* error_flag)
{
  if (start < 0 || end_pos < start || end_pos > total_size) {
    set_error_once(error_flag, ERR_BOUNDS);
    return false;
  }
  return true;
}

struct proto_tag {
  int field_number;
  int wire_type;
};

__device__ inline bool decode_tag(uint8_t const*& cur,
                                  uint8_t const* end,
                                  proto_tag& tag,
                                  int* error_flag)
{
  uint64_t key;
  int key_bytes;
  if (!read_varint(cur, end, key, key_bytes)) {
    set_error_once(error_flag, ERR_VARINT);
    return false;
  }

  cur += key_bytes;
  tag.field_number = static_cast<int>(key >> 3);
  tag.wire_type    = static_cast<int>(key & 0x7);
  if (tag.field_number == 0) {
    set_error_once(error_flag, ERR_FIELD_NUMBER);
    return false;
  }
  return true;
}

/**
 * Load a little-endian value from unaligned memory.
 * Reads bytes individually to avoid unaligned-access issues on GPU.
 */
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
// Field number lookup table helpers
// ============================================================================

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

/**
 * O(1) lookup of field_number -> field_index using a direct-mapped table.
 * Falls back to linear search when the table is empty (field numbers too large).
 */
__device__ inline int lookup_field(int field_number,
                                   int const* lookup_table,
                                   int lookup_table_size,
                                   field_descriptor const* field_descs,
                                   int num_fields)
{
  if (lookup_table != nullptr && field_number > 0 && field_number < lookup_table_size) {
    return lookup_table[field_number];
  }
  for (int f = 0; f < num_fields; f++) {
    if (field_descs[f].field_number == field_number) return f;
  }
  return -1;
}

// ============================================================================
// Pass 2: Extract data kernels
// ============================================================================

// ============================================================================
// Data Extraction Location Providers
// ============================================================================

struct TopLevelLocationProvider {
  cudf::size_type const* offsets;
  cudf::size_type base_offset;
  field_location const* locations;
  int field_idx;
  int num_fields;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto loc = locations[thread_idx * num_fields + field_idx];
    if (loc.offset >= 0) { data_offset = offsets[thread_idx] - base_offset + loc.offset; }
    return loc;
  }
};

struct RepeatedLocationProvider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  repeated_occurrence const* occurrences;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto occ    = occurrences[thread_idx];
    data_offset = row_offsets[occ.row_idx] - base_offset + occ.offset;
    return {occ.offset, occ.length};
  }
};

struct NestedLocationProvider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* parent_locations;
  field_location const* child_locations;
  int field_idx;
  int num_fields;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto ploc = parent_locations[thread_idx];
    auto cloc = child_locations[thread_idx * num_fields + field_idx];
    if (ploc.offset >= 0 && cloc.offset >= 0) {
      data_offset = row_offsets[thread_idx] - base_offset + ploc.offset + cloc.offset;
    } else {
      cloc.offset = -1;
    }
    return cloc;
  }
};

struct NestedRepeatedLocationProvider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* parent_locations;
  repeated_occurrence const* occurrences;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto occ    = occurrences[thread_idx];
    auto ploc   = parent_locations[occ.row_idx];
    data_offset = row_offsets[occ.row_idx] - base_offset + ploc.offset + occ.offset;
    return {occ.offset, occ.length};
  }
};

struct RepeatedMsgChildLocationProvider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* msg_locations;
  field_location const* child_locations;
  int field_idx;
  int num_fields;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto mloc = msg_locations[thread_idx];
    auto cloc = child_locations[thread_idx * num_fields + field_idx];
    if (mloc.offset >= 0 && cloc.offset >= 0) {
      data_offset = row_offsets[thread_idx] - base_offset + mloc.offset + cloc.offset;
    } else {
      cloc.offset = -1;
    }
    return cloc;
  }
};

template <typename OutputType, bool ZigZag = false, typename LocationProvider>
__global__ void extract_varint_kernel(uint8_t const* message_data,
                                      LocationProvider loc_provider,
                                      int total_items,
                                      OutputType* out,
                                      bool* valid,
                                      int* error_flag,
                                      bool has_default      = false,
                                      int64_t default_value = 0)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) return;

  int32_t data_offset = 0;
  auto loc            = loc_provider.get(idx, data_offset);

  if (loc.offset < 0) {
    if (has_default) {
      out[idx] = static_cast<OutputType>(default_value);
      if (valid) valid[idx] = true;
    } else {
      if (valid) valid[idx] = false;
    }
    return;
  }

  uint8_t const* cur     = message_data + data_offset;
  uint8_t const* cur_end = cur + loc.length;

  uint64_t v;
  int n;
  if (!read_varint(cur, cur_end, v, n)) {
    set_error_once(error_flag, ERR_VARINT);
    if (valid) valid[idx] = false;
    return;
  }

  if constexpr (ZigZag) { v = (v >> 1) ^ (-(v & 1)); }
  out[idx] = static_cast<OutputType>(v);
  if (valid) valid[idx] = true;
}

template <typename OutputType, int WT, typename LocationProvider>
__global__ void extract_fixed_kernel(uint8_t const* message_data,
                                     LocationProvider loc_provider,
                                     int total_items,
                                     OutputType* out,
                                     bool* valid,
                                     int* error_flag,
                                     bool has_default         = false,
                                     OutputType default_value = OutputType{})
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) return;

  int32_t data_offset = 0;
  auto loc            = loc_provider.get(idx, data_offset);

  if (loc.offset < 0) {
    if (has_default) {
      out[idx] = default_value;
      if (valid) valid[idx] = true;
    } else {
      if (valid) valid[idx] = false;
    }
    return;
  }

  uint8_t const* cur = message_data + data_offset;
  OutputType value;

  if constexpr (WT == WT_32BIT) {
    if (loc.length < 4) {
      set_error_once(error_flag, ERR_FIXED_LEN);
      if (valid) valid[idx] = false;
      return;
    }
    uint32_t raw = load_le<uint32_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  } else {
    if (loc.length < 8) {
      set_error_once(error_flag, ERR_FIXED_LEN);
      if (valid) valid[idx] = false;
      return;
    }
    uint64_t raw = load_le<uint64_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  }

  out[idx] = value;
  if (valid) valid[idx] = true;
}

template <typename LocationProvider>
__global__ void extract_lengths_kernel(LocationProvider loc_provider,
                                       int total_items,
                                       int32_t* out_lengths,
                                       bool has_default       = false,
                                       int32_t default_length = 0)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) return;

  int32_t data_offset = 0;
  auto loc            = loc_provider.get(idx, data_offset);

  if (loc.offset >= 0) {
    out_lengths[idx] = loc.length;
  } else if (has_default) {
    out_lengths[idx] = default_length;
  } else {
    out_lengths[idx] = 0;
  }
}
template <typename LocationProvider>
__global__ void copy_varlen_data_kernel(uint8_t const* message_data,
                                        LocationProvider loc_provider,
                                        int total_items,
                                        cudf::size_type const* output_offsets,
                                        char* output_chars,
                                        int* error_flag,
                                        bool has_default             = false,
                                        uint8_t const* default_chars = nullptr,
                                        int default_len              = 0)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) return;

  int32_t data_offset = 0;
  auto loc            = loc_provider.get(idx, data_offset);

  auto out_start = output_offsets[idx];

  if (loc.offset < 0) {
    if (has_default && default_len > 0) {
      memcpy(output_chars + out_start, default_chars, default_len);
    }
    return;
  }

  uint8_t const* src = message_data + data_offset;
  memcpy(output_chars + out_start, src, loc.length);
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
  if (enable_zigzag && encoding == spark_rapids_jni::ENC_ZIGZAG) {
    extract_varint_kernel<T, true, LocationProvider>
      <<<blocks, threads, 0, stream.value()>>>(message_data,
                                               loc_provider,
                                               num_rows,
                                               out_ptr,
                                               valid_ptr,
                                               error_ptr,
                                               has_default,
                                               default_value);
  } else if (encoding == spark_rapids_jni::ENC_FIXED) {
    if constexpr (sizeof(T) == 4) {
      extract_fixed_kernel<T, WT_32BIT, LocationProvider>
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
      extract_fixed_kernel<T, WT_64BIT, LocationProvider>
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

  __device__ int32_t operator()(int row) const { return info[row * num_fields + field_idx].count; }
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


// ============================================================================
// Forward declarations of non-template __global__ kernels
// ============================================================================

__global__ void scan_all_fields_kernel(
  cudf::column_device_view const d_in, field_descriptor const* field_descs, int num_fields,
  int const* field_lookup, int field_lookup_size, field_location* locations, int* error_flag);

__global__ void count_repeated_fields_kernel(
  cudf::column_device_view const d_in, device_nested_field_descriptor const* schema,
  int num_fields, int depth_level, repeated_field_info* repeated_info, int num_repeated_fields,
  int const* repeated_field_indices, field_location* nested_locations, int num_nested_fields,
  int const* nested_field_indices, int* error_flag);

__global__ void scan_repeated_field_occurrences_kernel(
  cudf::column_device_view const d_in, device_nested_field_descriptor const* schema,
  int schema_idx, int depth_level, int32_t const* output_offsets,
  repeated_occurrence* occurrences, int* error_flag);

__global__ void scan_nested_message_fields_kernel(
  uint8_t const* message_data, cudf::size_type const* parent_row_offsets,
  cudf::size_type parent_base_offset, field_location const* parent_locations,
  int num_parent_rows, field_descriptor const* field_descs, int num_fields,
  field_location* output_locations, int* error_flag);

__global__ void scan_repeated_message_children_kernel(
  uint8_t const* message_data, int32_t const* msg_row_offsets, field_location const* msg_locs,
  int num_occurrences, field_descriptor const* child_descs, int num_child_fields,
  field_location* child_locs, int* error_flag);

__global__ void count_repeated_in_nested_kernel(
  uint8_t const* message_data, cudf::size_type const* row_offsets, cudf::size_type base_offset,
  field_location const* parent_locs, int num_rows, device_nested_field_descriptor const* schema,
  int num_fields, repeated_field_info* repeated_info, int num_repeated,
  int const* repeated_indices, int* error_flag);

__global__ void scan_repeated_in_nested_kernel(
  uint8_t const* message_data, cudf::size_type const* row_offsets, cudf::size_type base_offset,
  field_location const* parent_locs, int num_rows, device_nested_field_descriptor const* schema,
  int num_fields, int32_t const* occ_prefix_sums, int num_repeated,
  int const* repeated_indices, repeated_occurrence* occurrences, int* error_flag);

__global__ void compute_nested_struct_locations_kernel(
  field_location const* child_locs, field_location const* msg_locs,
  int32_t const* msg_row_offsets, int child_idx, int num_child_fields,
  field_location* nested_locs, int32_t* nested_row_offsets, int total_count);

__global__ void compute_grandchild_parent_locations_kernel(
  field_location const* parent_locs, field_location const* child_locs,
  int child_idx, int num_child_fields, field_location* gc_parent_abs, int num_rows);

__global__ void compute_virtual_parents_for_nested_repeated_kernel(
  repeated_occurrence const* occurrences, cudf::size_type const* row_list_offsets,
  field_location const* parent_locations, cudf::size_type* virtual_row_offsets,
  field_location* virtual_parent_locs, int total_count);

__global__ void compute_msg_locations_from_occurrences_kernel(
  repeated_occurrence const* occurrences, cudf::size_type const* list_offsets,
  cudf::size_type base_offset, field_location* msg_locs, int32_t* msg_row_offsets,
  int total_count);

__global__ void extract_strided_locations_kernel(
  field_location const* nested_locations, int field_idx, int num_fields,
  field_location* parent_locs, int num_rows);

__global__ void check_required_fields_kernel(
  field_location const* locations, uint8_t const* is_required, int num_fields,
  int num_rows, int* error_flag);

__global__ void validate_enum_values_kernel(
  int32_t const* values, bool* valid, bool* row_has_invalid_enum,
  int32_t const* valid_enum_values, int num_valid_values, int num_rows);

__global__ void compute_enum_string_lengths_kernel(
  int32_t const* values, bool const* valid, int32_t const* valid_enum_values,
  int32_t const* enum_name_offsets, int num_valid_values, int32_t* lengths, int num_rows);

__global__ void copy_enum_string_chars_kernel(
  int32_t const* values, bool const* valid, int32_t const* valid_enum_values,
  int32_t const* enum_name_offsets, uint8_t const* enum_name_chars, int num_valid_values,
  int32_t const* output_offsets, char* out_chars, int num_rows);

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
  rmm::device_uvector<bool>& d_row_has_invalid_enum,
  int num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

// Complex builder forward declarations
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
  rmm::device_async_resource_ref mr);

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
  auto const blocks  = (num_rows + threads - 1) / threads;
  extract_lengths_kernel<LengthProvider><<<blocks, threads, 0, stream.value()>>>(
    length_provider, num_rows, lengths.data(), has_default, def_len);

  auto [offsets_col, total_size] = cudf::strings::detail::make_offsets_child_column(
    lengths.begin(), lengths.end(), stream, mr);

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

  rmm::device_uvector<bool> valid((num_rows > 0 ? num_rows : 1), stream, mr);
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
    return cudf::make_lists_column(num_rows,
                                   std::move(offsets_col),
                                   std::move(bytes_child),
                                   null_count,
                                   std::move(mask),
                                   stream,
                                   mr);
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
  rmm::device_uvector<bool>& d_row_has_invalid_enum,
  rmm::device_uvector<int>& d_error,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
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
      rmm::device_uvector<int32_t> out(num_items, stream, mr);
      rmm::device_uvector<bool> valid((num_items > 0 ? num_items : 1), stream, mr);
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
          rmm::device_uvector<int32_t> d_valid_enums(valid_enums.size(), stream, mr);
          CUDF_CUDA_TRY(cudaMemcpyAsync(d_valid_enums.data(),
                                        valid_enums.data(),
                                        valid_enums.size() * sizeof(int32_t),
                                        cudaMemcpyHostToDevice,
                                        stream.value()));
          validate_enum_values_kernel<<<blocks, threads_per_block, 0, stream.value()>>>(
            out.data(),
            valid.data(),
            d_row_has_invalid_enum.data(),
            d_valid_enums.data(),
            static_cast<int>(valid_enums.size()),
            num_items);
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
          extract_fixed_kernel<float, WT_32BIT, LocationProvider>
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
          extract_fixed_kernel<double, WT_64BIT, LocationProvider>
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

  rmm::device_uvector<T> values(total_count, stream, mr);
  rmm::device_uvector<int> d_error(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_error.data(), 0, sizeof(int), stream.value()));

  auto const threads = THREADS_PER_BLOCK;
  auto const blocks  = (total_count + threads - 1) / threads;

  int encoding = field_desc.encoding;
  bool zigzag  = (encoding == spark_rapids_jni::ENC_ZIGZAG);

  // For float/double types, always use fixed kernel (they use wire type 32BIT/64BIT)
  // For integer types, use fixed kernel only if encoding is ENC_FIXED
  constexpr bool is_floating_point = std::is_same_v<T, float> || std::is_same_v<T, double>;
  bool use_fixed_kernel            = is_floating_point || (encoding == spark_rapids_jni::ENC_FIXED);

  RepeatedLocationProvider loc_provider{list_offsets, base_offset, d_occurrences.data()};
  if (use_fixed_kernel) {
    if constexpr (sizeof(T) == 4) {
      extract_fixed_kernel<T, WT_32BIT, RepeatedLocationProvider>
        <<<blocks, threads, 0, stream.value()>>>(
          message_data, loc_provider, total_count, values.data(), nullptr, d_error.data());
    } else {
      extract_fixed_kernel<T, WT_64BIT, RepeatedLocationProvider>
        <<<blocks, threads, 0, stream.value()>>>(
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
                                   std::move(null_mask),
                                   stream,
                                   mr);
  }

  return cudf::make_lists_column(
    num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{}, stream, mr);
}

}  // namespace spark_rapids_jni::protobuf_detail
