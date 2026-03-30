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

#include "protobuf/protobuf_device_helpers.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>

#include <cuda/std/type_traits>

namespace spark_rapids_jni::protobuf::detail {

// ============================================================================
// Pass 2: Extract data kernels
// ============================================================================

// ============================================================================
// Data Extraction Location Providers
// ============================================================================

struct top_level_location_provider {
  cudf::size_type const* offsets;
  cudf::size_type base_offset;
  field_location const* locations;
  int field_idx;
  int num_fields;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto loc = locations[flat_index(static_cast<size_t>(thread_idx),
                                    static_cast<size_t>(num_fields),
                                    static_cast<size_t>(field_idx))];
    if (loc.offset >= 0) { data_offset = offsets[thread_idx] - base_offset + loc.offset; }
    return loc;
  }
};

struct repeated_location_provider {
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

struct nested_location_provider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* parent_locations;
  field_location const* child_locations;
  int field_idx;
  int num_fields;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto ploc = parent_locations[thread_idx];
    auto cloc = child_locations[flat_index(static_cast<size_t>(thread_idx),
                                           static_cast<size_t>(num_fields),
                                           static_cast<size_t>(field_idx))];
    if (ploc.offset >= 0 && cloc.offset >= 0) {
      data_offset = row_offsets[thread_idx] - base_offset + ploc.offset + cloc.offset;
    } else {
      cloc.offset = -1;
    }
    return cloc;
  }
};

struct nested_repeated_location_provider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* parent_locations;
  repeated_occurrence const* occurrences;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto occ  = occurrences[thread_idx];
    auto ploc = parent_locations[occ.row_idx];
    if (ploc.offset >= 0) {
      data_offset = row_offsets[occ.row_idx] - base_offset + ploc.offset + occ.offset;
      return {occ.offset, occ.length};
    }
    data_offset = 0;
    return {-1, 0};
  }
};

struct repeated_msg_child_location_provider {
  cudf::size_type const* row_offsets;
  cudf::size_type base_offset;
  field_location const* msg_locations;
  field_location const* child_locations;
  int field_idx;
  int num_fields;

  __device__ inline field_location get(int thread_idx, int32_t& data_offset) const
  {
    auto mloc = msg_locations[thread_idx];
    auto cloc = child_locations[flat_index(static_cast<size_t>(thread_idx),
                                           static_cast<size_t>(num_fields),
                                           static_cast<size_t>(field_idx))];
    if (mloc.offset >= 0 && cloc.offset >= 0) {
      data_offset = row_offsets[thread_idx] - base_offset + mloc.offset + cloc.offset;
    } else {
      cloc.offset = -1;
    }
    return cloc;
  }
};

template <typename OutputType, bool ZigZag = false, typename LocationProvider>
CUDF_KERNEL void extract_varint_kernel(uint8_t const* message_data,
                                       LocationProvider loc_provider,
                                       int total_items,
                                       OutputType* out,
                                       bool* valid,
                                       int* error_flag,
                                       bool has_default      = false,
                                       int64_t default_value = 0)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) { return; }

  int32_t data_offset = 0;
  auto loc            = loc_provider.get(idx, data_offset);

  // For BOOL8 (uint8_t), protobuf spec says any non-zero varint is true.
  // A raw static_cast<uint8_t> would silently truncate values >= 256 to 0.
  auto const write_value = [](OutputType* dst, uint64_t val) {
    if constexpr (cuda::std::is_same_v<OutputType, uint8_t>) {
      *dst = static_cast<uint8_t>(val != 0 ? 1 : 0);
    } else {
      *dst = static_cast<OutputType>(val);
    }
  };

  if (loc.offset < 0) {
    if (has_default) {
      write_value(&out[idx], static_cast<uint64_t>(default_value));
      if (valid) { valid[idx] = true; }
    } else {
      if (valid) { valid[idx] = false; }
    }
    return;
  }

  uint8_t const* cur     = message_data + data_offset;
  uint8_t const* cur_end = cur + loc.length;

  uint64_t v;
  int n;
  if (!read_varint(cur, cur_end, v, n)) {
    set_error_once(error_flag, ERR_VARINT);
    if (valid) { valid[idx] = false; }
    return;
  }

  if constexpr (ZigZag) { v = (v >> 1) ^ (-(v & 1)); }
  write_value(&out[idx], v);
  if (valid) { valid[idx] = true; }
}

template <typename OutputType, int WT, typename LocationProvider>
CUDF_KERNEL void extract_fixed_kernel(uint8_t const* message_data,
                                      LocationProvider loc_provider,
                                      int total_items,
                                      OutputType* out,
                                      bool* valid,
                                      int* error_flag,
                                      bool has_default         = false,
                                      OutputType default_value = OutputType{})
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) { return; }

  int32_t data_offset = 0;
  auto loc            = loc_provider.get(idx, data_offset);

  if (loc.offset < 0) {
    if (has_default) {
      out[idx] = default_value;
      if (valid) { valid[idx] = true; }
    } else {
      if (valid) { valid[idx] = false; }
    }
    return;
  }

  uint8_t const* cur = message_data + data_offset;
  OutputType value;

  if constexpr (WT == wire_type_value(proto_wire_type::I32BIT)) {
    if (loc.length < 4) {
      set_error_once(error_flag, ERR_FIXED_LEN);
      if (valid) { valid[idx] = false; }
      return;
    }
    uint32_t raw = load_le<uint32_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  } else {
    if (loc.length < 8) {
      set_error_once(error_flag, ERR_FIXED_LEN);
      if (valid) { valid[idx] = false; }
      return;
    }
    uint64_t raw = load_le<uint64_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  }

  out[idx] = value;
  if (valid) { valid[idx] = true; }
}

// ============================================================================
// Batched scalar extraction — one 2D kernel for N fields of the same type
// ============================================================================

struct batched_scalar_desc {
  int loc_field_idx;  // index into the locations array (column within d_locations)
  void* output;       // pre-allocated output buffer (T*)
  bool* valid;        // pre-allocated validity buffer
  bool has_default;
  int64_t default_int;
  double default_float;
};

template <typename OutputType, bool ZigZag = false>
CUDF_KERNEL void extract_varint_batched_kernel(uint8_t const* message_data,
                                               cudf::size_type const* row_offsets,
                                               cudf::size_type base_offset,
                                               field_location const* locations,
                                               int num_loc_fields,
                                               batched_scalar_desc const* descs,
                                               int num_descs,
                                               int num_rows,
                                               int* error_flag)
{
  int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int fi  = static_cast<int>(blockIdx.y);
  if (row >= num_rows || fi >= num_descs) { return; }

  auto const& desc = descs[fi];
  auto loc         = locations[row * num_loc_fields + desc.loc_field_idx];
  auto* out        = static_cast<OutputType*>(desc.output);

  auto const write_value = [](OutputType* dst, uint64_t val) {
    if constexpr (cuda::std::is_same_v<OutputType, uint8_t>) {
      *dst = static_cast<uint8_t>(val != 0 ? 1 : 0);
    } else {
      *dst = static_cast<OutputType>(val);
    }
  };

  if (loc.offset < 0) {
    if (desc.has_default) {
      write_value(&out[row], static_cast<uint64_t>(desc.default_int));
      desc.valid[row] = true;
    } else {
      desc.valid[row] = false;
    }
    return;
  }

  int32_t data_offset = row_offsets[row] - base_offset + loc.offset;
  uint8_t const* cur  = message_data + data_offset;
  uint8_t const* end  = cur + loc.length;

  uint64_t v;
  int n;
  if (!read_varint(cur, end, v, n)) {
    set_error_once(error_flag, ERR_VARINT);
    desc.valid[row] = false;
    return;
  }
  if constexpr (ZigZag) { v = (v >> 1) ^ (-(v & 1)); }
  write_value(&out[row], v);
  desc.valid[row] = true;
}

template <typename OutputType, int WT>
CUDF_KERNEL void extract_fixed_batched_kernel(uint8_t const* message_data,
                                              cudf::size_type const* row_offsets,
                                              cudf::size_type base_offset,
                                              field_location const* locations,
                                              int num_loc_fields,
                                              batched_scalar_desc const* descs,
                                              int num_descs,
                                              int num_rows,
                                              int* error_flag)
{
  int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int fi  = static_cast<int>(blockIdx.y);
  if (row >= num_rows || fi >= num_descs) { return; }

  auto const& desc = descs[fi];
  auto loc         = locations[row * num_loc_fields + desc.loc_field_idx];
  auto* out        = static_cast<OutputType*>(desc.output);

  if (loc.offset < 0) {
    if (desc.has_default) {
      if constexpr (cuda::std::is_integral_v<OutputType>) {
        out[row] = static_cast<OutputType>(desc.default_int);
      } else {
        out[row] = static_cast<OutputType>(desc.default_float);
      }
      desc.valid[row] = true;
    } else {
      desc.valid[row] = false;
    }
    return;
  }

  int32_t data_offset = row_offsets[row] - base_offset + loc.offset;
  uint8_t const* cur  = message_data + data_offset;
  OutputType value;

  if constexpr (WT == wire_type_value(proto_wire_type::I32BIT)) {
    if (loc.length < 4) {
      set_error_once(error_flag, ERR_FIXED_LEN);
      desc.valid[row] = false;
      return;
    }
    uint32_t raw = load_le<uint32_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  } else {
    if (loc.length < 8) {
      set_error_once(error_flag, ERR_FIXED_LEN);
      desc.valid[row] = false;
      return;
    }
    uint64_t raw = load_le<uint64_t>(cur);
    memcpy(&value, &raw, sizeof(value));
  }
  out[row]        = value;
  desc.valid[row] = true;
}

// ============================================================================

template <typename LocationProvider>
CUDF_KERNEL void extract_lengths_kernel(LocationProvider loc_provider,
                                        int total_items,
                                        int32_t* out_lengths,
                                        bool has_default       = false,
                                        int32_t default_length = 0)
{
  auto idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total_items) { return; }

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
CUDF_KERNEL void copy_varlen_data_kernel(uint8_t const* message_data,
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
  if (idx >= total_items) { return; }

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

// ============================================================================
}  // namespace spark_rapids_jni::protobuf::detail
