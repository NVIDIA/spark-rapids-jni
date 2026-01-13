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

#include "sequence_file.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>

namespace spark_rapids_jni {

namespace {

// ============================================================================
// Constants and Utilities
// ============================================================================

constexpr int BLOCK_SIZE         = 256;
constexpr int MAX_SYNC_POSITIONS = 1024 * 1024;  // Max sync markers we track

/**
 * @brief Read a big-endian int32 from a byte pointer.
 */
__device__ __forceinline__ int32_t read_int32_be(uint8_t const* ptr)
{
  return (static_cast<int32_t>(ptr[0]) << 24) | (static_cast<int32_t>(ptr[1]) << 16) |
         (static_cast<int32_t>(ptr[2]) << 8) | static_cast<int32_t>(ptr[3]);
}

/**
 * @brief Check if 16 bytes at the given position match the sync marker.
 */
__device__ __forceinline__ bool matches_sync_marker(uint8_t const* data,
                                                    size_t pos,
                                                    size_t data_size,
                                                    uint8_t const* sync_marker)
{
  if (pos + SYNC_MARKER_SIZE > data_size) { return false; }

  // Compare 16 bytes
  for (int i = 0; i < SYNC_MARKER_SIZE; ++i) {
    if (data[pos + i] != sync_marker[i]) { return false; }
  }
  return true;
}

// ============================================================================
// Kernel 1: Find Sync Marker Positions
// ============================================================================

/**
 * @brief Scan data for sync marker positions.
 *
 * A sync marker in the record stream is indicated by:
 * - int32 = -1 (0xFFFFFFFF in big-endian)
 * - followed by 16 bytes matching the sync marker
 *
 * This kernel finds all such positions and records them.
 */
__global__ void find_sync_markers_kernel(uint8_t const* data,
                                         size_t data_size,
                                         uint8_t const* sync_marker,
                                         int32_t* sync_positions,
                                         int32_t* sync_count,
                                         int32_t max_syncs)
{
  int64_t const tid       = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const stride    = gridDim.x * blockDim.x;
  int64_t const num_items = static_cast<int64_t>(data_size) - 20;  // Need room for -1 + sync

  for (int64_t pos = tid; pos < num_items; pos += stride) {
    // Check for sync marker indicator (-1 as big-endian int32)
    int32_t indicator = read_int32_be(data + pos);
    if (indicator == SYNC_MARKER_INDICATOR) {
      // Check if next 16 bytes match sync marker
      if (matches_sync_marker(data, pos + 4, data_size, sync_marker)) {
        // Found a sync marker! Record the position of the -1 indicator
        int32_t idx = atomicAdd(sync_count, 1);
        if (idx < max_syncs) { sync_positions[idx] = static_cast<int32_t>(pos); }
      }
    }
  }
}

// ============================================================================
// Kernel 2: Parse Record Boundaries (Sequential within chunks)
// ============================================================================

/**
 * @brief Record metadata: offset and lengths.
 */
struct record_info {
  int32_t offset;     ///< Start of record data (after length fields)
  int32_t key_len;    ///< Key length in bytes
  int32_t value_len;  ///< Value length in bytes
};

/**
 * @brief Parse records sequentially from a starting position to an end position.
 *
 * This function is called by multiple threads, each handling a chunk between sync markers.
 * Records are parsed sequentially because record lengths are variable.
 *
 * @param data Input data buffer
 * @param start Start position in data (after sync marker if any)
 * @param end End position (next sync marker or end of data)
 * @param sync_marker The 16-byte sync marker to skip
 * @param records Output array for record info
 * @param record_count Atomic counter for total records found
 * @param max_records Maximum records to store
 */
__device__ void parse_records_in_chunk(uint8_t const* data,
                                       size_t data_size,
                                       int32_t start,
                                       int32_t end,
                                       uint8_t const* sync_marker,
                                       record_info* records,
                                       int32_t* record_count,
                                       int32_t max_records)
{
  int32_t pos = start;

  while (pos + 8 <= end) {  // Need at least 8 bytes for lengths
    int32_t record_len = read_int32_be(data + pos);

    // Check for sync marker indicator
    if (record_len == SYNC_MARKER_INDICATOR) {
      // Skip: -1 (4 bytes) + sync marker (16 bytes) = 20 bytes
      pos += 4 + SYNC_MARKER_SIZE;
      continue;
    }

    // Validate record length
    if (record_len < 0) {
      // Invalid record length (negative but not -1)
      break;
    }

    int32_t key_len = read_int32_be(data + pos + 4);
    if (key_len < 0 || key_len > record_len) {
      // Invalid key length
      break;
    }

    int32_t value_len = record_len - key_len;

    // Check if we have enough data for this record
    int32_t record_data_start = pos + 8;  // After the two length fields
    int32_t record_data_end   = record_data_start + record_len;
    if (record_data_end > end) {
      // Record extends past our chunk boundary
      break;
    }

    // Record this entry
    int32_t idx = atomicAdd(record_count, 1);
    if (idx < max_records) {
      records[idx].offset    = record_data_start;
      records[idx].key_len   = key_len;
      records[idx].value_len = value_len;
    }

    // Move to next record
    pos = record_data_end;
  }
}

/**
 * @brief Kernel to parse records between sync markers.
 *
 * Each thread processes one chunk (region between adjacent sync markers).
 */
__global__ void parse_records_kernel(uint8_t const* data,
                                     size_t data_size,
                                     int32_t const* sync_positions,
                                     int32_t sync_count,
                                     uint8_t const* sync_marker,
                                     record_info* records,
                                     int32_t* record_count,
                                     int32_t max_records)
{
  int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;

  // We have (sync_count + 1) chunks:
  // Chunk 0: [0, sync_positions[0])
  // Chunk i: [sync_positions[i-1] + 20, sync_positions[i])  for i in [1, sync_count)
  // Chunk sync_count: [sync_positions[sync_count-1] + 20, data_size)
  int32_t const num_chunks = sync_count + 1;

  if (tid >= num_chunks) { return; }

  int32_t start, end;

  if (tid == 0) {
    start = 0;
    end   = (sync_count > 0) ? sync_positions[0] : static_cast<int32_t>(data_size);
  } else if (tid == sync_count) {
    // Last chunk: from last sync marker to end of data
    start = sync_positions[sync_count - 1] + 4 + SYNC_MARKER_SIZE;
    end   = static_cast<int32_t>(data_size);
  } else {
    // Middle chunk: between two sync markers
    start = sync_positions[tid - 1] + 4 + SYNC_MARKER_SIZE;
    end   = sync_positions[tid];
  }

  if (start < end) {
    parse_records_in_chunk(
      data, data_size, start, end, sync_marker, records, record_count, max_records);
  }
}

/**
 * @brief Single-threaded kernel for small files (no sync markers).
 */
__global__ void parse_records_single_chunk_kernel(uint8_t const* data,
                                                  size_t data_size,
                                                  uint8_t const* sync_marker,
                                                  record_info* records,
                                                  int32_t* record_count,
                                                  int32_t max_records)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    parse_records_in_chunk(data,
                           data_size,
                           0,
                           static_cast<int32_t>(data_size),
                           sync_marker,
                           records,
                           record_count,
                           max_records);
  }
}

// ============================================================================
// Kernel 3: Extract Key/Value Data
// ============================================================================

/**
 * @brief Copy key/value data to output buffers.
 *
 * Each thread handles one record.
 */
__global__ void extract_data_kernel(uint8_t const* data,
                                    record_info const* records,
                                    int32_t num_records,
                                    int32_t const* key_offsets,
                                    int32_t const* value_offsets,
                                    uint8_t* key_data,
                                    uint8_t* value_data,
                                    bool wants_key,
                                    bool wants_value)
{
  int32_t const tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t const stride = gridDim.x * blockDim.x;

  for (int32_t i = tid; i < num_records; i += stride) {
    record_info const& rec = records[i];
    int32_t src_pos        = rec.offset;

    if (wants_key && key_data != nullptr) {
      int32_t dst_pos = key_offsets[i];
      for (int32_t j = 0; j < rec.key_len; ++j) {
        key_data[dst_pos + j] = data[src_pos + j];
      }
    }

    if (wants_value && value_data != nullptr) {
      int32_t dst_pos     = value_offsets[i];
      int32_t value_start = src_pos + rec.key_len;
      for (int32_t j = 0; j < rec.value_len; ++j) {
        value_data[dst_pos + j] = data[value_start + j];
      }
    }
  }
}

// ============================================================================
// Host Functions
// ============================================================================

/**
 * @brief Build a LIST<UINT8> column from data and offsets.
 */
std::unique_ptr<cudf::column> build_list_column(rmm::device_uvector<uint8_t>& data_buffer,
                                                rmm::device_uvector<int32_t>& offsets_buffer,
                                                cudf::size_type num_rows,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  // IMPORTANT: Capture sizes BEFORE calling release() to avoid undefined behavior
  // due to unspecified argument evaluation order in C++
  auto const child_size   = static_cast<cudf::size_type>(data_buffer.size());
  auto const offsets_size = static_cast<cudf::size_type>(offsets_buffer.size());

  // Release buffers BEFORE creating columns
  auto child_data   = data_buffer.release();
  auto offsets_data = offsets_buffer.release();

  // Create child column (UINT8)
  auto child_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                                     child_size,
                                                     std::move(child_data),
                                                     rmm::device_buffer{},
                                                     0);

  // Create offsets column (INT32)
  auto offsets_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                       offsets_size,
                                                       std::move(offsets_data),
                                                       rmm::device_buffer{},
                                                       0);

  // Create LIST column
  return cudf::make_lists_column(num_rows,
                                 std::move(offsets_column),
                                 std::move(child_column),
                                 0,
                                 rmm::device_buffer{},
                                 stream,
                                 mr);
}

}  // namespace

// ============================================================================
// Public API Implementation
// ============================================================================

sequence_file_result parse_sequence_file(uint8_t const* data,
                                         size_t data_size,
                                         std::vector<uint8_t> const& sync_marker,
                                         bool wants_key,
                                         bool wants_value,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Validate inputs
  CUDF_EXPECTS(sync_marker.size() == SYNC_MARKER_SIZE, "Sync marker must be exactly 16 bytes");

  sequence_file_result result;
  result.num_rows = 0;

  if (data_size == 0 || (!wants_key && !wants_value)) {
    // Return empty columns
    if (wants_key) {
      rmm::device_uvector<uint8_t> empty_data(0, stream, mr);
      rmm::device_uvector<int32_t> empty_offsets(1, stream, mr);
      cudaMemsetAsync(empty_offsets.data(), 0, sizeof(int32_t), stream.value());
      result.key_column = build_list_column(empty_data, empty_offsets, 0, stream, mr);
    }
    if (wants_value) {
      rmm::device_uvector<uint8_t> empty_data(0, stream, mr);
      rmm::device_uvector<int32_t> empty_offsets(1, stream, mr);
      cudaMemsetAsync(empty_offsets.data(), 0, sizeof(int32_t), stream.value());
      result.value_column = build_list_column(empty_data, empty_offsets, 0, stream, mr);
    }
    return result;
  }

  // Copy sync marker to device
  rmm::device_uvector<uint8_t> d_sync_marker(SYNC_MARKER_SIZE, stream, mr);
  cudaMemcpyAsync(d_sync_marker.data(),
                  sync_marker.data(),
                  SYNC_MARKER_SIZE,
                  cudaMemcpyHostToDevice,
                  stream.value());

  // ========================================================================
  // Step 1: Find sync marker positions
  // ========================================================================

  rmm::device_uvector<int32_t> sync_positions(MAX_SYNC_POSITIONS, stream, mr);
  rmm::device_scalar<int32_t> sync_count(0, stream, mr);

  if (data_size > 20) {  // Need at least room for one sync marker entry
    int num_blocks = std::min(static_cast<int>((data_size + BLOCK_SIZE - 1) / BLOCK_SIZE), 65535);
    find_sync_markers_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(data,
                                                                            data_size,
                                                                            d_sync_marker.data(),
                                                                            sync_positions.data(),
                                                                            sync_count.data(),
                                                                            MAX_SYNC_POSITIONS);
  }

  int32_t h_sync_count = sync_count.value(stream);

  // Sort sync positions (they were found out of order due to parallel scanning)
  if (h_sync_count > 1) {
    thrust::sort(
      rmm::exec_policy(stream), sync_positions.begin(), sync_positions.begin() + h_sync_count);
  }

  // ========================================================================
  // Step 2: Parse record boundaries
  // ========================================================================

  // Estimate max records (conservative: assume average record size of 100 bytes)
  int32_t max_records = std::max(static_cast<int32_t>(data_size / 8), 1024);
  rmm::device_uvector<record_info> records(max_records, stream, mr);
  rmm::device_scalar<int32_t> record_count(0, stream, mr);

  if (h_sync_count == 0) {
    // No sync markers - parse as single chunk
    parse_records_single_chunk_kernel<<<1, 1, 0, stream.value()>>>(
      data, data_size, d_sync_marker.data(), records.data(), record_count.data(), max_records);
  } else {
    // Parse chunks in parallel
    int32_t num_chunks = h_sync_count + 1;
    int num_blocks     = (num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    parse_records_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(data,
                                                                        data_size,
                                                                        sync_positions.data(),
                                                                        h_sync_count,
                                                                        d_sync_marker.data(),
                                                                        records.data(),
                                                                        record_count.data(),
                                                                        max_records);
  }

  int32_t num_records = record_count.value(stream);
  result.num_rows     = num_records;

  if (num_records == 0) {
    // Return empty columns
    if (wants_key) {
      rmm::device_uvector<uint8_t> empty_data(0, stream, mr);
      rmm::device_uvector<int32_t> empty_offsets(1, stream, mr);
      cudaMemsetAsync(empty_offsets.data(), 0, sizeof(int32_t), stream.value());
      result.key_column = build_list_column(empty_data, empty_offsets, 0, stream, mr);
    }
    if (wants_value) {
      rmm::device_uvector<uint8_t> empty_data(0, stream, mr);
      rmm::device_uvector<int32_t> empty_offsets(1, stream, mr);
      cudaMemsetAsync(empty_offsets.data(), 0, sizeof(int32_t), stream.value());
      result.value_column = build_list_column(empty_data, empty_offsets, 0, stream, mr);
    }
    return result;
  }

  // ========================================================================
  // Step 3: Compute output offsets (exclusive scan of lengths)
  // ========================================================================

  rmm::device_uvector<int32_t> key_lengths(num_records, stream, mr);
  rmm::device_uvector<int32_t> value_lengths(num_records, stream, mr);

  // Extract lengths from record_info
  thrust::transform(rmm::exec_policy(stream),
                    records.begin(),
                    records.begin() + num_records,
                    key_lengths.begin(),
                    [] __device__(record_info const& r) { return r.key_len; });

  thrust::transform(rmm::exec_policy(stream),
                    records.begin(),
                    records.begin() + num_records,
                    value_lengths.begin(),
                    [] __device__(record_info const& r) { return r.value_len; });

  // Compute offsets via exclusive scan
  rmm::device_uvector<int32_t> key_offsets(num_records + 1, stream, mr);
  rmm::device_uvector<int32_t> value_offsets(num_records + 1, stream, mr);

  thrust::exclusive_scan(
    rmm::exec_policy(stream), key_lengths.begin(), key_lengths.end(), key_offsets.begin(), 0);
  thrust::exclusive_scan(
    rmm::exec_policy(stream), value_lengths.begin(), value_lengths.end(), value_offsets.begin(), 0);

  // Set the final offset (total size)
  int32_t total_key_bytes   = 0;
  int32_t total_value_bytes = 0;

  // Get last offset + last length = total size
  {
    int32_t h_last_key_offset, h_last_key_len;
    int32_t h_last_value_offset, h_last_value_len;

    cudaMemcpyAsync(&h_last_key_offset,
                    key_offsets.data() + num_records - 1,
                    sizeof(int32_t),
                    cudaMemcpyDeviceToHost,
                    stream.value());
    cudaMemcpyAsync(&h_last_key_len,
                    key_lengths.data() + num_records - 1,
                    sizeof(int32_t),
                    cudaMemcpyDeviceToHost,
                    stream.value());
    cudaMemcpyAsync(&h_last_value_offset,
                    value_offsets.data() + num_records - 1,
                    sizeof(int32_t),
                    cudaMemcpyDeviceToHost,
                    stream.value());
    cudaMemcpyAsync(&h_last_value_len,
                    value_lengths.data() + num_records - 1,
                    sizeof(int32_t),
                    cudaMemcpyDeviceToHost,
                    stream.value());
    stream.synchronize();

    total_key_bytes   = h_last_key_offset + h_last_key_len;
    total_value_bytes = h_last_value_offset + h_last_value_len;

    cudaMemcpyAsync(key_offsets.data() + num_records,
                    &total_key_bytes,
                    sizeof(int32_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
    cudaMemcpyAsync(value_offsets.data() + num_records,
                    &total_value_bytes,
                    sizeof(int32_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
  }

  // ========================================================================
  // Step 4: Extract data to output buffers
  // ========================================================================

  rmm::device_uvector<uint8_t> key_data(wants_key ? total_key_bytes : 0, stream, mr);
  rmm::device_uvector<uint8_t> value_data(wants_value ? total_value_bytes : 0, stream, mr);

  int num_blocks = (num_records + BLOCK_SIZE - 1) / BLOCK_SIZE;
  extract_data_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
    data,
    records.data(),
    num_records,
    key_offsets.data(),
    value_offsets.data(),
    wants_key ? key_data.data() : nullptr,
    wants_value ? value_data.data() : nullptr,
    wants_key,
    wants_value);

  // Synchronize to ensure all data is written before building columns
  stream.synchronize();

  // ========================================================================
  // Step 5: Build output columns
  // ========================================================================

  if (wants_key) {
    result.key_column = build_list_column(key_data, key_offsets, num_records, stream, mr);
  }

  if (wants_value) {
    result.value_column = build_list_column(value_data, value_offsets, num_records, stream, mr);
  }

  return result;
}

cudf::size_type count_records(uint8_t const* data,
                              size_t data_size,
                              std::vector<uint8_t> const& sync_marker,
                              rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(sync_marker.size() == SYNC_MARKER_SIZE, "Sync marker must be exactly 16 bytes");

  if (data_size == 0) { return 0; }

  auto mr = rmm::mr::get_current_device_resource_ref();

  // Copy sync marker to device
  rmm::device_uvector<uint8_t> d_sync_marker(SYNC_MARKER_SIZE, stream, mr);
  cudaMemcpyAsync(d_sync_marker.data(),
                  sync_marker.data(),
                  SYNC_MARKER_SIZE,
                  cudaMemcpyHostToDevice,
                  stream.value());

  // Find sync markers
  rmm::device_uvector<int32_t> sync_positions(MAX_SYNC_POSITIONS, stream, mr);
  rmm::device_scalar<int32_t> sync_count(0, stream, mr);

  if (data_size > 20) {
    int num_blocks = std::min(static_cast<int>((data_size + BLOCK_SIZE - 1) / BLOCK_SIZE), 65535);
    find_sync_markers_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(data,
                                                                            data_size,
                                                                            d_sync_marker.data(),
                                                                            sync_positions.data(),
                                                                            sync_count.data(),
                                                                            MAX_SYNC_POSITIONS);
  }

  int32_t h_sync_count = sync_count.value(stream);

  if (h_sync_count > 1) {
    thrust::sort(
      rmm::exec_policy(stream), sync_positions.begin(), sync_positions.begin() + h_sync_count);
  }

  // Count records
  int32_t max_records = std::max(static_cast<int32_t>(data_size / 8), 1024);
  rmm::device_uvector<record_info> records(max_records, stream, mr);
  rmm::device_scalar<int32_t> record_count(0, stream, mr);

  if (h_sync_count == 0) {
    parse_records_single_chunk_kernel<<<1, 1, 0, stream.value()>>>(
      data, data_size, d_sync_marker.data(), records.data(), record_count.data(), max_records);
  } else {
    int32_t num_chunks = h_sync_count + 1;
    int num_blocks     = (num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    parse_records_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(data,
                                                                        data_size,
                                                                        sync_positions.data(),
                                                                        h_sync_count,
                                                                        d_sync_marker.data(),
                                                                        records.data(),
                                                                        record_count.data(),
                                                                        max_records);
  }

  return record_count.value(stream);
}

}  // namespace spark_rapids_jni
