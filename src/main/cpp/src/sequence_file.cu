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

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <iostream>

namespace spark_rapids_jni {

namespace {

// ============================================================================
// Constants and Utilities
// ============================================================================

constexpr int BLOCK_SIZE         = 256;
constexpr int MAX_SYNC_POSITIONS = 1024 * 1024;  // Max sync markers we track

// Minimum average chunk size to use the block-per-chunk kernel (bytes)
// For smaller chunks, the overhead of kernel launch per chunk is not worth it
constexpr int MIN_CHUNK_SIZE_FOR_BLOCK_KERNEL = 16 * 1024;  // 16KB

// Threshold for using warp-cooperative copy (bytes)
// Records larger than this will be copied by multiple threads in a warp
constexpr int WARP_COPY_THRESHOLD = 256;

// Warp size
constexpr int WARP_SIZE = 32;
constexpr int SYNC_VALIDATE_RECORDS = 3;

int32_t get_env_int(char const* name, int32_t default_value, int32_t min_value, int32_t max_value)
{
  char const* value = std::getenv(name);
  if (value == nullptr || *value == '\0') { return default_value; }
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') { return default_value; }
  if (parsed < min_value) { return min_value; }
  if (parsed > max_value) { return max_value; }
  return static_cast<int32_t>(parsed);
}

// Debug error codes
constexpr int32_t ERR_INVALID_RECORD_LEN = 1;
constexpr int32_t ERR_INVALID_KEY_LEN    = 2;
constexpr int32_t ERR_RECORD_PAST_CHUNK  = 3;

__device__ __forceinline__ int32_t read_int32_be(uint8_t const* ptr);

__device__ __forceinline__ void set_error(int32_t* error_flag, int32_t code)
{
  if (error_flag != nullptr) { atomicCAS(error_flag, 0, code); }
}

__device__ __forceinline__ void set_error_info(int32_t* error_flag,
                                               int32_t* error_file_idx,
                                               int32_t* error_pos,
                                               int32_t code,
                                               int32_t file_idx,
                                               int32_t pos)
{
  if (error_flag != nullptr) {
    int32_t prev = atomicCAS(error_flag, 0, code);
    if (prev == 0) {
      if (error_file_idx != nullptr) { *error_file_idx = file_idx; }
      if (error_pos != nullptr) { *error_pos = pos; }
    }
  }
}

/**
 * @brief Validate N consecutive records starting at pos.
 */
__device__ __forceinline__ bool validate_records(uint8_t const* data,
                                                 int64_t pos,
                                                 int64_t end,
                                                 int32_t num_records)
{
  for (int32_t i = 0; i < num_records; ++i) {
    if (pos + 8 > end) { return false; }
    int32_t record_len = read_int32_be(data + pos);
    if (record_len == SYNC_MARKER_INDICATOR || record_len < 0) { return false; }
    int32_t key_len = read_int32_be(data + pos + 4);
    if (key_len < 0 || key_len > record_len) { return false; }
    int64_t record_end = pos + 8LL + record_len;
    if (record_end > end) { return false; }
    pos = record_end;
  }
  return true;
}

// ============================================================================
// Vectorized Memory Copy Utilities
// ============================================================================

/**
 * @brief Fast memory copy that handles unaligned addresses safely.
 *
 * Uses byte-level access to avoid misaligned address errors on GPU.
 * The compiler will optimize this for aligned cases.
 *
 * @param dst Destination pointer
 * @param src Source pointer (read-only)
 * @param len Number of bytes to copy
 */
__device__ __forceinline__ void fast_copy(uint8_t* __restrict__ dst,
                                          uint8_t const* __restrict__ src,
                                          int32_t len)
{
  // Use standard memcpy - CUDA compiler optimizes this well
  // and handles alignment correctly
  memcpy(dst, src, len);
}

/**
 * @brief Warp-cooperative memory copy for large buffers.
 *
 * All threads in a warp cooperate to copy data in parallel.
 * Each thread handles a strided portion of the data using byte-level access
 * to avoid misaligned address errors.
 *
 * @param dst Destination pointer
 * @param src Source pointer
 * @param len Number of bytes to copy
 * @param lane Thread's lane ID within the warp (0-31)
 */
__device__ __forceinline__ void warp_cooperative_copy(uint8_t* __restrict__ dst,
                                                      uint8_t const* __restrict__ src,
                                                      int32_t len,
                                                      int lane)
{
  if (len <= 0) return;

  // Each thread copies bytes with stride of WARP_SIZE for coalesced access
  for (int32_t i = lane; i < len; i += WARP_SIZE) {
    dst[i] = src[i];
  }
}

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
                                         int32_t max_syncs,
                                         int32_t validate_records_count)
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
        // Validate N consecutive records to avoid false-positive sync markers
        int64_t next_pos = static_cast<int64_t>(pos) + 4 + SYNC_MARKER_SIZE;
        int64_t end = static_cast<int64_t>(data_size);
        if (!validate_records(data, next_pos, end, validate_records_count)) { continue; }
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
                                       int32_t max_records,
                                       int32_t* error_flag)
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
      set_error(error_flag, ERR_INVALID_RECORD_LEN);
      break;
    }

    int32_t key_len = read_int32_be(data + pos + 4);
    if (key_len < 0 || key_len > record_len) {
      set_error(error_flag, ERR_INVALID_KEY_LEN);
      break;
    }

    int32_t value_len = record_len - key_len;

    // Check if we have enough data for this record
    int32_t record_data_start = pos + 8;  // After the two length fields
    int64_t record_data_end64 = static_cast<int64_t>(record_data_start) + record_len;
    if (record_data_end64 > static_cast<int64_t>(data_size)) {
      set_error(error_flag, ERR_RECORD_PAST_CHUNK);
      break;
    }
    if (record_data_end64 > static_cast<int64_t>(end)) { break; }
    if (record_data_end64 > static_cast<int64_t>(INT32_MAX)) {
      set_error(error_flag, ERR_RECORD_PAST_CHUNK);
      break;
    }
    int32_t record_data_end = static_cast<int32_t>(record_data_end64);

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
 * @brief Count records sequentially from a starting position to an end position.
 */
__device__ void count_records_in_chunk(uint8_t const* data,
                                       size_t data_size,
                                       int32_t start,
                                       int32_t end,
                                       uint8_t const* sync_marker,
                                       int32_t* record_count)
{
  (void)sync_marker;
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
    if (record_len < 0) { break; }

    int32_t key_len = read_int32_be(data + pos + 4);
    if (key_len < 0 || key_len > record_len) { break; }

    int32_t record_data_start = pos + 8;  // After the two length fields
    int32_t record_data_end   = record_data_start + record_len;
    if (record_data_end > end) { break; }

    atomicAdd(record_count, 1);
    pos = record_data_end;
  }
}

// ============================================================================
// Optimized Chunk Parsing Kernel
// ============================================================================

/**
 * @brief Kernel for parsing chunks with optimized memory access.
 *
 * This kernel is launched with one block per chunk. Thread 0 does the actual
 * sequential parsing (required by SequenceFile format), but we use optimized
 * memory access patterns and minimize synchronization.
 *
 * Future optimization: implement speculative parallel parsing where multiple
 * threads scan for potential record boundaries in parallel, then validate
 * which ones are on the correct chain.
 *
 * @param data Input data buffer
 * @param data_size Total data size
 * @param chunk_starts Start positions of each chunk
 * @param chunk_ends End positions of each chunk
 * @param num_chunks Number of chunks to process
 * @param records Output array for all record info
 * @param record_count Global atomic counter for records
 * @param max_records Maximum records to store
 */
__global__ void parse_records_parallel_kernel(uint8_t const* data,
                                              size_t data_size,
                                              int32_t const* chunk_starts,
                                              int32_t const* chunk_ends,
                                              int32_t num_chunks,
                                              record_info* records,
                                              int32_t* record_count,
                                              int32_t max_records,
                                              int32_t* error_flag)
{
  // Each block handles one chunk, thread 0 does the work
  int32_t const chunk_idx = blockIdx.x;
  if (chunk_idx >= num_chunks) return;
  if (threadIdx.x != 0) return;  // Only thread 0 parses

  int32_t const chunk_start = chunk_starts[chunk_idx];
  int32_t const chunk_end   = chunk_ends[chunk_idx];

  // Sequential parsing with optimized inner loop
  int32_t pos = chunk_start;

  while (pos + 8 <= chunk_end) {
    // Use __ldg for cached reads (read-only data)
    int32_t record_len = read_int32_be(data + pos);

    // Check for sync marker indicator
    if (record_len == SYNC_MARKER_INDICATOR) {
      pos += 4 + SYNC_MARKER_SIZE;  // Skip -1 and sync marker
      continue;
    }

    // Validate record length
    if (record_len < 0) {
      set_error(error_flag, ERR_INVALID_RECORD_LEN);
      break;
    }

    int32_t key_len = read_int32_be(data + pos + 4);
    if (key_len < 0 || key_len > record_len) {
      set_error(error_flag, ERR_INVALID_KEY_LEN);
      break;
    }

    int32_t value_len = record_len - key_len;
    int64_t record_data_end64 = static_cast<int64_t>(pos) + 8LL + record_len;

    if (record_data_end64 > static_cast<int64_t>(data_size)) {
      set_error(error_flag, ERR_RECORD_PAST_CHUNK);
      break;
    }
    if (record_data_end64 > static_cast<int64_t>(chunk_end)) { break; }
    if (record_data_end64 > static_cast<int64_t>(INT32_MAX)) {
      set_error(error_flag, ERR_RECORD_PAST_CHUNK);
      break;
    }

    // Store record info
    int32_t idx = atomicAdd(record_count, 1);
    if (idx < max_records) {
      records[idx].offset    = pos + 8;
      records[idx].key_len   = key_len;
      records[idx].value_len = value_len;
    }

    pos = static_cast<int32_t>(record_data_end64);
  }
}

/**
 * @brief Kernel for counting records per chunk (one block per chunk).
 */
__global__ void count_records_parallel_kernel(uint8_t const* data,
                                              size_t data_size,
                                              int32_t const* chunk_starts,
                                              int32_t const* chunk_ends,
                                              int32_t num_chunks,
                                              int32_t* record_count)
{
  int32_t const chunk_idx = blockIdx.x;
  if (chunk_idx >= num_chunks) return;
  if (threadIdx.x != 0) return;  // Only thread 0 counts

  int32_t const chunk_start = chunk_starts[chunk_idx];
  int32_t const chunk_end   = chunk_ends[chunk_idx];

  count_records_in_chunk(data, data_size, chunk_start, chunk_end, nullptr, record_count);
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
                                     int32_t max_records,
                                     int32_t* error_flag)
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
      data, data_size, start, end, sync_marker, records, record_count, max_records, error_flag);
  }
}

/**
 * @brief Kernel to count records between sync markers.
 */
__global__ void count_records_kernel(uint8_t const* data,
                                     size_t data_size,
                                     int32_t const* sync_positions,
                                     int32_t sync_count,
                                     uint8_t const* sync_marker,
                                     int32_t* record_count)
{
  int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;

  int32_t const num_chunks = sync_count + 1;
  if (tid >= num_chunks) { return; }

  int32_t start, end;
  if (tid == 0) {
    start = 0;
    end   = (sync_count > 0) ? sync_positions[0] : static_cast<int32_t>(data_size);
  } else if (tid == sync_count) {
    start = sync_positions[sync_count - 1] + 4 + SYNC_MARKER_SIZE;
    end   = static_cast<int32_t>(data_size);
  } else {
    start = sync_positions[tid - 1] + 4 + SYNC_MARKER_SIZE;
    end   = sync_positions[tid];
  }

  if (start < end) {
    count_records_in_chunk(data, data_size, start, end, sync_marker, record_count);
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
                                                  int32_t max_records,
                                                  int32_t* error_flag)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    parse_records_in_chunk(data,
                           data_size,
                           0,
                           static_cast<int32_t>(data_size),
                           sync_marker,
                           records,
                           record_count,
                           max_records,
                           error_flag);
  }
}

/**
 * @brief Single-threaded kernel for counting records in small files.
 */
__global__ void count_records_single_chunk_kernel(uint8_t const* data,
                                                  size_t data_size,
                                                  uint8_t const* sync_marker,
                                                  int32_t* record_count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    count_records_in_chunk(
      data, data_size, 0, static_cast<int32_t>(data_size), sync_marker, record_count);
  }
}

// ============================================================================
// Kernel 3: Extract Key/Value Data
// ============================================================================

/**
 * @brief Copy key/value data to output buffers with warp-cooperative copy.
 *
 * Each warp handles one record. Large records use warp-wide copy for better throughput.
 */
__global__ void extract_data_kernel(uint8_t const* __restrict__ data,
                                    record_info const* __restrict__ records,
                                    int32_t num_records,
                                    int32_t const* __restrict__ key_offsets,
                                    int32_t const* __restrict__ value_offsets,
                                    uint8_t* __restrict__ key_data,
                                    uint8_t* __restrict__ value_data,
                                    bool wants_key,
                                    bool wants_value)
{
  int32_t const warp_id   = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int32_t const lane      = threadIdx.x & (WARP_SIZE - 1);
  int32_t const num_warps = (gridDim.x * blockDim.x) / WARP_SIZE;

  for (int32_t i = warp_id; i < num_records; i += num_warps) {
    record_info const& rec = records[i];
    int32_t const total_len = rec.key_len + rec.value_len;
    bool const use_warp_copy = (total_len >= WARP_COPY_THRESHOLD);

    if (use_warp_copy) {
      if (wants_key && key_data != nullptr && rec.key_len > 0) {
        warp_cooperative_copy(key_data + key_offsets[i], data + rec.offset, rec.key_len, lane);
      }
      if (wants_value && value_data != nullptr && rec.value_len > 0) {
        warp_cooperative_copy(value_data + value_offsets[i],
                              data + rec.offset + rec.key_len,
                              rec.value_len,
                              lane);
      }
    } else {
      if (lane == 0) {
        if (wants_key && key_data != nullptr && rec.key_len > 0) {
          fast_copy(key_data + key_offsets[i], data + rec.offset, rec.key_len);
        }
        if (wants_value && value_data != nullptr && rec.value_len > 0) {
          fast_copy(value_data + value_offsets[i],
                    data + rec.offset + rec.key_len,
                    rec.value_len);
        }
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
  bool const debug = (std::getenv("SEQFILE_GPU_DEBUG") != nullptr);
  int32_t const sync_validate_records =
    get_env_int("SEQFILE_GPU_SYNC_VALIDATE", SYNC_VALIDATE_RECORDS, 1, 64);
  rmm::device_scalar<int32_t> error_flag(0, stream, mr);
  rmm::device_scalar<int32_t> error_file_idx(-1, stream, mr);
  rmm::device_scalar<int32_t> error_pos(-1, stream, mr);
  int32_t* error_flag_ptr = debug ? error_flag.data() : nullptr;
  int32_t* error_file_idx_ptr = debug ? error_file_idx.data() : nullptr;
  int32_t* error_pos_ptr = debug ? error_pos.data() : nullptr;

  // ========================================================================
  // Step 1: Find sync marker positions
  // ========================================================================
  cudf::scoped_range sync_range{"seqfile::find_syncs"};

  rmm::device_uvector<int32_t> sync_positions(MAX_SYNC_POSITIONS, stream, mr);
  rmm::device_scalar<int32_t> sync_count(0, stream, mr);

  if (data_size > 20) {  // Need at least room for one sync marker entry
    int num_blocks = std::min(static_cast<int>((data_size + BLOCK_SIZE - 1) / BLOCK_SIZE), 65535);
    find_sync_markers_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(data,
                                                                            data_size,
                                                                            d_sync_marker.data(),
                                                                            sync_positions.data(),
                                                                            sync_count.data(),
                                                                            MAX_SYNC_POSITIONS,
                                                                            sync_validate_records);
  }

  int32_t h_sync_count = sync_count.value(stream);
  if (debug && h_sync_count > MAX_SYNC_POSITIONS) {
    std::cerr << "SequenceFile GPU parser sync_count overflow (single-file): "
              << "sync_count=" << h_sync_count
              << ", max_syncs=" << MAX_SYNC_POSITIONS
              << ", data_size=" << data_size << std::endl;
  }
  if (h_sync_count > MAX_SYNC_POSITIONS) { h_sync_count = MAX_SYNC_POSITIONS; }

  // Sort sync positions (they were found out of order due to parallel scanning)
  if (h_sync_count > 1) {
    thrust::sort(
      rmm::exec_policy(stream), sync_positions.begin(), sync_positions.begin() + h_sync_count);
  }

  // ========================================================================
  // Step 2: Count record boundaries first (avoid over-allocation)
  // ========================================================================
  cudf::scoped_range count_range{"seqfile::count_records"};

  rmm::device_scalar<int32_t> record_count(0, stream, mr);
  int32_t const num_chunks = h_sync_count + 1;

  int64_t const avg_chunk_size   = static_cast<int64_t>(data_size) / num_chunks;
  bool const use_block_per_chunk = avg_chunk_size >= MIN_CHUNK_SIZE_FOR_BLOCK_KERNEL;

  if (use_block_per_chunk && num_chunks > 0) {
    rmm::device_uvector<int32_t> chunk_starts(num_chunks, stream, mr);
    rmm::device_uvector<int32_t> chunk_ends(num_chunks, stream, mr);

    thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<int32_t>(0),
      thrust::make_counting_iterator<int32_t>(num_chunks),
      chunk_starts.begin(),
      [sync_positions = sync_positions.data(), h_sync_count] __device__(int32_t i) -> int32_t {
        if (i == 0) return 0;
        return sync_positions[i - 1] + 4 + static_cast<int32_t>(SYNC_MARKER_SIZE);
      });

    thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<int32_t>(0),
      thrust::make_counting_iterator<int32_t>(num_chunks),
      chunk_ends.begin(),
      [sync_positions = sync_positions.data(),
       h_sync_count,
       data_size = static_cast<int32_t>(data_size)] __device__(int32_t i) -> int32_t {
        if (i == h_sync_count) return data_size;
        return sync_positions[i];
      });

    count_records_parallel_kernel<<<num_chunks, 32, 0, stream.value()>>>(
      data, data_size, chunk_starts.data(), chunk_ends.data(), num_chunks, record_count.data());
  } else if (h_sync_count == 0) {
    count_records_single_chunk_kernel<<<1, 1, 0, stream.value()>>>(
      data, data_size, d_sync_marker.data(), record_count.data());
  } else {
    int num_blocks = (num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_records_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(data,
                                                                        data_size,
                                                                        sync_positions.data(),
                                                                        h_sync_count,
                                                                        d_sync_marker.data(),
                                                                        record_count.data());
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
  // Step 3: Parse record boundaries with exact allocation
  // ========================================================================
  cudf::scoped_range parse_range{"seqfile::parse_records"};

  rmm::device_uvector<record_info> records(num_records, stream, mr);
  record_count = rmm::device_scalar<int32_t>(0, stream, mr);

  if (use_block_per_chunk && num_chunks > 0) {
    rmm::device_uvector<int32_t> chunk_starts(num_chunks, stream, mr);
    rmm::device_uvector<int32_t> chunk_ends(num_chunks, stream, mr);

    thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<int32_t>(0),
      thrust::make_counting_iterator<int32_t>(num_chunks),
      chunk_starts.begin(),
      [sync_positions = sync_positions.data(), h_sync_count] __device__(int32_t i) -> int32_t {
        if (i == 0) return 0;
        return sync_positions[i - 1] + 4 + static_cast<int32_t>(SYNC_MARKER_SIZE);
      });

    thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<int32_t>(0),
      thrust::make_counting_iterator<int32_t>(num_chunks),
      chunk_ends.begin(),
      [sync_positions = sync_positions.data(),
       h_sync_count,
       data_size = static_cast<int32_t>(data_size)] __device__(int32_t i) -> int32_t {
        if (i == h_sync_count) return data_size;
        return sync_positions[i];
      });

    parse_records_parallel_kernel<<<num_chunks, 32, 0, stream.value()>>>(
      data,
      data_size,
      chunk_starts.data(),
      chunk_ends.data(),
      num_chunks,
      records.data(),
      record_count.data(),
      num_records,
      error_flag_ptr);
  } else if (h_sync_count == 0) {
    parse_records_single_chunk_kernel<<<1, 1, 0, stream.value()>>>(
      data,
      data_size,
      d_sync_marker.data(),
      records.data(),
      record_count.data(),
      num_records,
      error_flag_ptr);
  } else {
    int num_blocks = (num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE;
    parse_records_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(data,
                                                                        data_size,
                                                                        sync_positions.data(),
                                                                        h_sync_count,
                                                                        d_sync_marker.data(),
                                                                        records.data(),
                                                                        record_count.data(),
                                                                        num_records,
                                                                        error_flag_ptr);
  }
  if (debug) {
    int32_t err = error_flag.value(stream);
    if (err != 0) {
      CUDF_FAIL("SequenceFile GPU parser error in parse_records_in_chunk");
    }
  }

  // Sort records by offset to ensure correct order
  // (parallel parsing may produce records out of order)
  if (num_records > 1) {
    thrust::sort(rmm::exec_policy(stream),
                 records.begin(),
                 records.begin() + num_records,
                 [] __device__(record_info const& a, record_info const& b) {
                   return a.offset < b.offset;
                 });
  }

  // ========================================================================
  // Step 3: Compute output offsets using optimized inclusive scan
  // ========================================================================
  cudf::scoped_range offsets_range{"seqfile::compute_offsets"};
  // Strategy: Use inclusive_scan starting at offset[1], so offset[n] directly
  // contains the total size. This eliminates multiple D2H/H2D copies.

  rmm::device_uvector<int32_t> key_offsets(num_records + 1, stream, mr);
  rmm::device_uvector<int32_t> value_offsets(num_records + 1, stream, mr);

  // Set offsets[0] = 0 for both key and value
  cudaMemsetAsync(key_offsets.data(), 0, sizeof(int32_t), stream.value());
  cudaMemsetAsync(value_offsets.data(), 0, sizeof(int32_t), stream.value());

  // Transform records to lengths and inclusive_scan directly into offsets[1..n]
  // This computes: offsets[i+1] = sum(lengths[0..i]), so offsets[n] = total
  thrust::transform(rmm::exec_policy(stream),
                    records.begin(),
                    records.begin() + num_records,
                    key_offsets.begin() + 1,
                    [] __device__(record_info const& r) { return r.key_len; });
  thrust::inclusive_scan(
    rmm::exec_policy(stream), key_offsets.begin() + 1, key_offsets.end(), key_offsets.begin() + 1);

  thrust::transform(rmm::exec_policy(stream),
                    records.begin(),
                    records.begin() + num_records,
                    value_offsets.begin() + 1,
                    [] __device__(record_info const& r) { return r.value_len; });
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         value_offsets.begin() + 1,
                         value_offsets.end(),
                         value_offsets.begin() + 1);

  // Get total sizes with a single combined D2H copy (only 2 values instead of 4)
  int32_t total_key_bytes   = 0;
  int32_t total_value_bytes = 0;
  cudaMemcpyAsync(&total_key_bytes,
                  key_offsets.data() + num_records,
                  sizeof(int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  cudaMemcpyAsync(&total_value_bytes,
                  value_offsets.data() + num_records,
                  sizeof(int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  stream.synchronize();
  if (debug) {
    int64_t total_bytes = static_cast<int64_t>(total_key_bytes) + total_value_bytes;
    if (total_key_bytes < 0 || total_value_bytes < 0 || total_bytes > static_cast<int64_t>(data_size)) {
      CUDF_FAIL("SequenceFile GPU parser produced invalid offsets");
    }
  }

  // ========================================================================
  // Step 4: Extract data to output buffers
  // ========================================================================
  cudf::scoped_range extract_range{"seqfile::extract_data"};

  rmm::device_uvector<uint8_t> key_data(wants_key ? total_key_bytes : 0, stream, mr);
  rmm::device_uvector<uint8_t> value_data(wants_value ? total_value_bytes : 0, stream, mr);

  int32_t const warps_per_block = BLOCK_SIZE / WARP_SIZE;
  int num_blocks = (num_records + warps_per_block - 1) / warps_per_block;
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
  cudf::scoped_range build_range{"seqfile::build_columns"};

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
  int32_t const sync_validate_records =
    get_env_int("SEQFILE_GPU_SYNC_VALIDATE", SYNC_VALIDATE_RECORDS, 1, 64);

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
                                                                            MAX_SYNC_POSITIONS,
                                                                            sync_validate_records);
  }

  int32_t h_sync_count = sync_count.value(stream);
  if (h_sync_count > MAX_SYNC_POSITIONS) { h_sync_count = MAX_SYNC_POSITIONS; }

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
      data,
      data_size,
      d_sync_marker.data(),
      records.data(),
      record_count.data(),
      max_records,
      nullptr);
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
                                                                        max_records,
                                                                        nullptr);
  }

  return record_count.value(stream);
}

// ============================================================================
// Multi-file Parsing Implementation
// ============================================================================

namespace {

/**
 * @brief Extended record info that includes file index for multi-file parsing.
 */
struct multi_file_record_info {
  int32_t offset;      ///< Offset in combined buffer where record data starts
  int32_t key_len;     ///< Key length in bytes
  int32_t value_len;   ///< Value length in bytes
  int32_t file_idx;    ///< Index of the file this record belongs to
};

/**
 * @brief Chunk info for multi-file parsing.
 */
struct multi_file_chunk {
  int32_t start;       ///< Start offset in combined buffer
  int32_t end;         ///< End offset in combined buffer
  int32_t file_idx;    ///< File this chunk belongs to
};

/**
 * @brief Device-side file descriptor (copy of host file_descriptor).
 */
struct device_file_desc {
  int64_t data_offset;
  int64_t data_size;
  uint8_t sync_marker[SYNC_MARKER_SIZE];
};

/**
 * @brief Build chunk list from per-file sync positions.
 *
 * Reads from per-file sync arrays (already sorted within each file).
 */
__global__ void build_chunks_from_syncs_kernel(multi_file_chunk* chunks,
                                               device_file_desc const* file_descs,
                                               int32_t num_files,
                                               int32_t const* file_chunk_offsets,
                                               int32_t const* per_file_sync_positions,
                                               int32_t const* per_file_sync_counts,
                                               int32_t max_syncs_per_file)
{
  int32_t const file_idx = static_cast<int32_t>(blockIdx.x);
  if (file_idx >= num_files) { return; }
  if (threadIdx.x != 0) { return; }

  device_file_desc const& fd = file_descs[file_idx];
  if (fd.data_size <= 0) { return; }

  int32_t sync_count = per_file_sync_counts[file_idx];
  if (sync_count > max_syncs_per_file) sync_count = max_syncs_per_file;

  int32_t chunk_write_idx = file_chunk_offsets[file_idx];
  int32_t chunk_end_idx   = file_chunk_offsets[file_idx + 1];

  int32_t chunk_start = static_cast<int32_t>(fd.data_offset);
  int32_t file_end    = static_cast<int32_t>(fd.data_offset + fd.data_size);

  // Read from per-file sync array (already sorted)
  int32_t const* syncs = per_file_sync_positions + file_idx * max_syncs_per_file;

  for (int32_t i = 0; i < sync_count; ++i) {
    int32_t sync_pos = syncs[i];
    // Guard against duplicates/out-of-order markers
    if (sync_pos <= chunk_start) { continue; }
    if (sync_pos >= file_end) { continue; }

    chunks[chunk_write_idx++] = {chunk_start, sync_pos, file_idx};
    chunk_start = sync_pos + 4 + static_cast<int32_t>(SYNC_MARKER_SIZE);
  }

  if (chunk_start < file_end) {
    chunks[chunk_write_idx++] = {chunk_start, file_end, file_idx};
  }

  // Fill any remaining chunk slots with empty ranges to avoid uninitialized reads.
  while (chunk_write_idx < chunk_end_idx) {
    chunks[chunk_write_idx++] = {file_end, file_end, file_idx};
  }
}

/**
 * @brief Find sync markers across multiple files, writing to per-file arrays.
 *
 * Each file has its own sync position array to avoid global sorting.
 * per_file_sync_positions is a 2D array flattened as [file_idx * max_syncs_per_file + local_idx]
 */
__global__ void find_sync_markers_multi_file_kernel(
    uint8_t const* combined_data,
    device_file_desc const* file_descs,
    int32_t num_files,
    int32_t* per_file_sync_positions,  // Output: [num_files * max_syncs_per_file]
    int32_t* per_file_sync_counts,     // Output: count per file [num_files]
    int32_t max_syncs_per_file,
    int32_t validate_records_count)
{
  int32_t const file_idx = static_cast<int32_t>(blockIdx.y);
  if (file_idx >= num_files) { return; }

  device_file_desc const& fd = file_descs[file_idx];
  int64_t const file_start   = fd.data_offset;
  int64_t const file_end     = fd.data_offset + fd.data_size;

  // Scan this file's data for sync markers
  int64_t const tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t const stride   = gridDim.x * blockDim.x;
  int64_t const scan_end = file_end - 20;  // Need room for -1 + sync

  for (int64_t pos = file_start + tid; pos < scan_end; pos += stride) {
    int32_t indicator = read_int32_be(combined_data + pos);
    if (indicator == SYNC_MARKER_INDICATOR) {
      // Check if next 16 bytes match this file's sync marker
      bool matches = true;
      for (int i = 0; i < SYNC_MARKER_SIZE && matches; ++i) {
        if (combined_data[pos + 4 + i] != fd.sync_marker[i]) {
          matches = false;
        }
      }
      if (matches) {
        // Validate N consecutive records to avoid false-positive sync markers
        int64_t next_pos = static_cast<int64_t>(pos) + 4 + SYNC_MARKER_SIZE;
        if (!validate_records(combined_data, next_pos, file_end, validate_records_count)) {
          continue;
        }
        // Write to per-file array (no global coordination needed)
        int32_t local_idx = atomicAdd(&per_file_sync_counts[file_idx], 1);
        if (local_idx < max_syncs_per_file) {
          per_file_sync_positions[file_idx * max_syncs_per_file + local_idx] =
            static_cast<int32_t>(pos);
        }
      }
    }
  }
}

/**
 * @brief Sort sync positions within each file using simple insertion sort.
 *
 * Each block handles one file. For typical sync counts (< 1000 per file),
 * insertion sort is efficient and has low overhead.
 */
__global__ void sort_per_file_syncs_kernel(
    int32_t* per_file_sync_positions,
    int32_t const* per_file_sync_counts,
    int32_t max_syncs_per_file,
    int32_t num_files)
{
  int32_t const file_idx = blockIdx.x;
  if (file_idx >= num_files) return;
  if (threadIdx.x != 0) return;  // Only thread 0 does the sort

  int32_t count = per_file_sync_counts[file_idx];
  if (count <= 1) return;

  // Clamp count to max
  if (count > max_syncs_per_file) count = max_syncs_per_file;

  int32_t* syncs = per_file_sync_positions + file_idx * max_syncs_per_file;

  // Simple insertion sort (efficient for small arrays, no extra memory needed)
  for (int32_t i = 1; i < count; i++) {
    int32_t key = syncs[i];
    int32_t j = i - 1;
    while (j >= 0 && syncs[j] > key) {
      syncs[j + 1] = syncs[j];
      j--;
    }
    syncs[j + 1] = key;
  }
}

/**
 * @brief Parse records from a chunk in multi-file mode.
 *
 * Supports two modes:
 * 1. Counting mode (records=nullptr): Just counts records per chunk
 * 2. Writing mode (records!=nullptr): Writes records at deterministic positions
 *
 * In writing mode, chunk_base_offset provides the pre-computed starting index
 * for this chunk, and chunk_local_count is used for local counting within the chunk.
 */
__device__ void parse_records_in_chunk_multi_file(
    uint8_t const* data,
    int32_t start,
    int32_t end,
    int32_t file_idx,
    uint8_t const* sync_marker,
    multi_file_record_info* records,
    int32_t* chunk_record_count,     // Per-chunk count (for counting pass)
    int32_t chunk_base_offset,       // Pre-computed base offset (for writing pass)
    int32_t max_records,
    int32_t* error_flag,
    int32_t* error_file_idx,
    int32_t* error_pos,
    int64_t* record_bytes_sum)
{
  int32_t pos = start;
  int32_t local_count = 0;  // Local count within this chunk

  while (pos + 8 <= end) {
    int32_t record_len = read_int32_be(data + pos);

    // Check for sync marker indicator
    if (record_len == SYNC_MARKER_INDICATOR) {
      pos += 4 + static_cast<int32_t>(SYNC_MARKER_SIZE);
      continue;
    }

    if (record_len < 0) {
      set_error_info(error_flag, error_file_idx, error_pos,
        ERR_INVALID_RECORD_LEN, file_idx, pos);
      break;
    }

    int32_t key_len = read_int32_be(data + pos + 4);
    if (key_len < 0 || key_len > record_len) {
      set_error_info(error_flag, error_file_idx, error_pos,
        ERR_INVALID_KEY_LEN, file_idx, pos);
      break;
    }

    int32_t value_len = record_len - key_len;
    int64_t record_data_end64 = static_cast<int64_t>(pos) + 8LL + record_len;

    if (record_data_end64 > static_cast<int64_t>(end)) { break; }
    if (record_data_end64 > static_cast<int64_t>(INT32_MAX)) {
      set_error_info(error_flag, error_file_idx, error_pos,
        ERR_RECORD_PAST_CHUNK, file_idx, static_cast<int32_t>(record_data_end64));
      break;
    }

    if (record_bytes_sum != nullptr) {
      atomicAdd(reinterpret_cast<unsigned long long*>(record_bytes_sum),
                static_cast<unsigned long long>(record_len));
    }

    // Store record at deterministic position (writing mode)
    if (records != nullptr) {
      int32_t idx = chunk_base_offset + local_count;
      if (idx < max_records) {
        records[idx].offset    = pos + 8;
        records[idx].key_len   = key_len;
        records[idx].value_len = value_len;
        records[idx].file_idx  = file_idx;
      }
    }

    local_count++;
    pos = static_cast<int32_t>(record_data_end64);
  }

  // Store the count for this chunk (counting mode)
  if (chunk_record_count != nullptr) {
    *chunk_record_count = local_count;
  }
}

/**
 * @brief Kernel to parse records from multiple files using chunk information.
 *
 * Supports two modes:
 * 1. Counting mode: chunk_record_counts != nullptr, chunk_record_offsets == nullptr
 *    - Each chunk writes its count to chunk_record_counts[chunk_id]
 * 2. Writing mode: chunk_record_counts == nullptr, chunk_record_offsets != nullptr
 *    - Each chunk writes records starting at chunk_record_offsets[chunk_id]
 */
__global__ void parse_records_multi_file_kernel(
    uint8_t const* combined_data,
    multi_file_chunk const* chunks,
    int32_t num_chunks,
    device_file_desc const* file_descs,
    multi_file_record_info* records,
    int32_t* chunk_record_counts,    // For counting pass: output per-chunk counts
    int32_t const* chunk_record_offsets,  // For writing pass: input per-chunk offsets
    int32_t max_records,
    int32_t* error_flag,
    int32_t* error_file_idx,
    int32_t* error_pos,
    int64_t* record_bytes_sum)
{
  int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_chunks) return;

  multi_file_chunk const& chunk = chunks[tid];
  device_file_desc const& fd    = file_descs[chunk.file_idx];

  // Determine mode and get base offset
  int32_t base_offset = (chunk_record_offsets != nullptr) ? chunk_record_offsets[tid] : 0;
  int32_t* count_ptr = (chunk_record_counts != nullptr) ? &chunk_record_counts[tid] : nullptr;

  parse_records_in_chunk_multi_file(combined_data,
                                    chunk.start,
                                    chunk.end,
                                    chunk.file_idx,
                                    fd.sync_marker,
                                    records,
                                    count_ptr,
                                    base_offset,
                                    max_records,
                                    error_flag,
                                    error_file_idx,
                                    error_pos,
                                    record_bytes_sum);
}

/**
 * @brief Count records per file from the records array.
 *
 * This kernel is used after parsing to compute per-file record counts,
 * since we no longer track them during parsing (we track per-chunk counts instead).
 */
__global__ void count_records_per_file_kernel(
    multi_file_record_info const* records,
    int32_t num_records,
    int32_t* file_record_counts,
    int32_t num_files)
{
  int32_t const tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_records) return;

  int32_t file_idx = records[tid].file_idx;
  if (file_idx >= 0 && file_idx < num_files) {
    atomicAdd(&file_record_counts[file_idx], 1);
  }
}

/**
 * @brief Extract data kernel for multi-file records with warp-cooperative copy.
 *
 * This kernel uses a hybrid approach:
 * - Small records (< WARP_COPY_THRESHOLD): Single thread uses fast_copy
 * - Large records (>= WARP_COPY_THRESHOLD): Entire warp cooperates to copy
 *
 * Each warp processes one record at a time. For large records (like 4KB values),
 * all 32 threads in the warp work together to copy the data in parallel,
 * achieving better memory bandwidth utilization.
 */
__global__ void extract_data_multi_file_kernel(
    uint8_t const* __restrict__ data,
    multi_file_record_info const* __restrict__ records,
    int32_t num_records,
    int32_t const* __restrict__ key_offsets,
    int32_t const* __restrict__ value_offsets,
    uint8_t* __restrict__ key_data,
    uint8_t* __restrict__ value_data,
    bool wants_key,
    bool wants_value)
{
  // Each warp processes one record
  int32_t const warp_id     = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int32_t const lane        = threadIdx.x & (WARP_SIZE - 1);
  int32_t const num_warps   = (gridDim.x * blockDim.x) / WARP_SIZE;

  for (int32_t i = warp_id; i < num_records; i += num_warps) {
    multi_file_record_info const& rec = records[i];
    int32_t const total_len = rec.key_len + rec.value_len;

    // Decide strategy based on record size
    bool const use_warp_copy = (total_len >= WARP_COPY_THRESHOLD);

    if (use_warp_copy) {
      // Large record: all threads in warp cooperate
      if (wants_key && key_data != nullptr && rec.key_len > 0) {
        warp_cooperative_copy(key_data + key_offsets[i], data + rec.offset, rec.key_len, lane);
      }
      if (wants_value && value_data != nullptr && rec.value_len > 0) {
        warp_cooperative_copy(value_data + value_offsets[i],
                              data + rec.offset + rec.key_len,
                              rec.value_len,
                              lane);
      }
    } else {
      // Small record: only lane 0 does the copy
      if (lane == 0) {
        if (wants_key && key_data != nullptr && rec.key_len > 0) {
          fast_copy(key_data + key_offsets[i], data + rec.offset, rec.key_len);
        }
        if (wants_value && value_data != nullptr && rec.value_len > 0) {
          fast_copy(value_data + value_offsets[i],
                    data + rec.offset + rec.key_len,
                    rec.value_len);
        }
      }
    }
  }
}

}  // anonymous namespace

multi_file_result parse_multiple_sequence_files(
    uint8_t const* combined_data,
    std::vector<file_descriptor> const& file_descs,
    bool wants_key,
    bool wants_value,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  multi_file_result result;
  result.total_rows = 0;

  int32_t const num_files = static_cast<int32_t>(file_descs.size());
  if (num_files == 0) {
    return result;
  }

  // Calculate total data size and max file size for kernel sizing
  int64_t total_data_size = 0;
  int64_t max_file_size   = 0;
  for (auto const& fd : file_descs) {
    total_data_size += fd.data_size;
    max_file_size = std::max(max_file_size, fd.data_size);
  }

  if (total_data_size == 0 || (!wants_key && !wants_value)) {
    result.file_row_counts.resize(num_files, 0);
    return result;
  }

  // Copy file descriptors to device
  rmm::device_uvector<device_file_desc> d_file_descs(num_files, stream, mr);
  std::vector<device_file_desc> h_file_descs(num_files);
  for (int32_t i = 0; i < num_files; ++i) {
    h_file_descs[i].data_offset = file_descs[i].data_offset;
    h_file_descs[i].data_size   = file_descs[i].data_size;
    memcpy(h_file_descs[i].sync_marker, file_descs[i].sync_marker, SYNC_MARKER_SIZE);
  }
  cudaMemcpyAsync(d_file_descs.data(),
                  h_file_descs.data(),
                  num_files * sizeof(device_file_desc),
                  cudaMemcpyHostToDevice,
                  stream.value());

  // Initialize per-file record counts
  rmm::device_uvector<int32_t> d_file_record_counts(num_files, stream, mr);
  cudaMemsetAsync(d_file_record_counts.data(), 0, num_files * sizeof(int32_t), stream.value());
  bool const debug = (std::getenv("SEQFILE_GPU_DEBUG") != nullptr);
  int32_t const sync_validate_records =
    get_env_int("SEQFILE_GPU_SYNC_VALIDATE", SYNC_VALIDATE_RECORDS, 1, 64);
  rmm::device_scalar<int32_t> error_flag(0, stream, mr);
  rmm::device_scalar<int32_t> error_file_idx(-1, stream, mr);
  rmm::device_scalar<int32_t> error_pos(-1, stream, mr);
  rmm::device_scalar<int64_t> record_bytes_sum(0, stream, mr);
  int32_t* error_flag_ptr = debug ? error_flag.data() : nullptr;
  int32_t* error_file_idx_ptr = debug ? error_file_idx.data() : nullptr;
  int32_t* error_pos_ptr = debug ? error_pos.data() : nullptr;
  int64_t* record_bytes_sum_ptr = debug ? record_bytes_sum.data() : nullptr;

  // ========================================================================
  // Step 1: Find sync markers across all files
  // ========================================================================
  cudf::scoped_range sync_range{"seqfile::multi_find_syncs"};

  // Estimate max syncs per file: file_size / MIN_SYNC_INTERVAL
  // Hadoop writes sync markers roughly every 2KB, but we use a conservative estimate
  constexpr int64_t MIN_SYNC_INTERVAL = 1024;  // ~1KB minimum between syncs
  int32_t max_syncs_per_file =
    static_cast<int32_t>(std::min((max_file_size / MIN_SYNC_INTERVAL) + 1,
                                  static_cast<int64_t>(MAX_SYNC_POSITIONS / num_files)));
  // Ensure at least some space per file
  max_syncs_per_file = std::max(max_syncs_per_file, 128);

  // Allocate per-file sync storage (2D array flattened)
  rmm::device_uvector<int32_t> per_file_sync_positions(
    static_cast<size_t>(num_files) * max_syncs_per_file, stream, mr);
  rmm::device_uvector<int32_t> per_file_sync_counts(num_files, stream, mr);
  cudaMemsetAsync(per_file_sync_counts.data(), 0, num_files * sizeof(int32_t), stream.value());

  if (total_data_size > 20 && max_file_size > 20) {
    int num_blocks =
      std::min(static_cast<int>((max_file_size + BLOCK_SIZE - 1) / BLOCK_SIZE), 65535);
    dim3 grid(num_blocks, num_files, 1);
    find_sync_markers_multi_file_kernel<<<grid, BLOCK_SIZE, 0, stream.value()>>>(
      combined_data,
      d_file_descs.data(),
      num_files,
      per_file_sync_positions.data(),
      per_file_sync_counts.data(),
      max_syncs_per_file,
      sync_validate_records);
  }

  // Sort syncs within each file in parallel (no global sort needed!)
  sort_per_file_syncs_kernel<<<num_files, 1, 0, stream.value()>>>(
    per_file_sync_positions.data(),
    per_file_sync_counts.data(),
    max_syncs_per_file,
    num_files);

  // ========================================================================
  // Step 2: Build chunk list from per-file syncs (no global sorting!)
  // ========================================================================
  cudf::scoped_range chunk_range{"seqfile::multi_build_chunks"};

  // file_sync_counts already computed by find_sync_markers_multi_file_kernel
  // Compute file_chunk_counts: each file has (sync_count + 1) chunks
  rmm::device_uvector<int32_t> file_chunk_counts(num_files, stream, mr);
  thrust::transform(rmm::exec_policy(stream),
                    per_file_sync_counts.begin(),
                    per_file_sync_counts.end(),
                    file_chunk_counts.begin(),
                    [] __device__(int32_t sync_count) {
                      return sync_count + 1;  // syncs divide file into (sync_count + 1) chunks
                    });

  // Build file_chunk_offsets (exclusive scan)
  rmm::device_uvector<int32_t> file_chunk_offsets(num_files + 1, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         file_chunk_counts.begin(),
                         file_chunk_counts.end(),
                         file_chunk_offsets.begin());

  // Get total chunks
  int32_t num_chunks = thrust::reduce(rmm::exec_policy(stream),
                                      file_chunk_counts.begin(),
                                      file_chunk_counts.end(),
                                      0,
                                      thrust::plus<int32_t>());

  // Set file_chunk_offsets[num_files] = num_chunks
  cudaMemcpyAsync(file_chunk_offsets.data() + num_files,
                  &num_chunks,
                  sizeof(int32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());

  if (num_chunks == 0) {
    result.file_row_counts.resize(num_files, 0);
    return result;
  }

  // Allocate chunks on device and build them from per-file syncs
  rmm::device_uvector<multi_file_chunk> d_chunks(num_chunks, stream, mr);
  build_chunks_from_syncs_kernel<<<num_files, 1, 0, stream.value()>>>(
    d_chunks.data(),
    d_file_descs.data(),
    num_files,
    file_chunk_offsets.data(),
    per_file_sync_positions.data(),
    per_file_sync_counts.data(),
    max_syncs_per_file);

  // ========================================================================
  // Step 3: Count records per chunk (avoid over-allocation and enable sorting-free parsing)
  // ========================================================================
  cudf::scoped_range count_range{"seqfile::multi_count_records"};

  // Allocate per-chunk record counts
  rmm::device_uvector<int32_t> chunk_record_counts(num_chunks, stream, mr);
  cudaMemsetAsync(chunk_record_counts.data(), 0, num_chunks * sizeof(int32_t), stream.value());

  int num_blocks = (num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE;
  parse_records_multi_file_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
    combined_data,
    d_chunks.data(),
    num_chunks,
    d_file_descs.data(),
    nullptr,              // records = nullptr (counting mode)
    chunk_record_counts.data(),  // per-chunk counts output
    nullptr,              // chunk_record_offsets = nullptr (counting mode)
    0,
    error_flag_ptr,
    error_file_idx_ptr,
    error_pos_ptr,
    record_bytes_sum_ptr);

  if (debug) {
    int32_t err = error_flag.value(stream);
    if (err != 0) {
      int32_t fidx = error_file_idx.value(stream);
      int32_t epos = error_pos.value(stream);
      std::cerr << "SequenceFile GPU parser error in multi-file count pass (code="
                << err << ", file=" << fidx << ", pos=" << epos << ")"
                << std::endl;
      CUDF_FAIL("SequenceFile GPU parser error in multi-file count pass");
    }
    cudaMemsetAsync(error_flag.data(), 0, sizeof(int32_t), stream.value());
    cudaMemsetAsync(error_file_idx.data(), 0xFF, sizeof(int32_t), stream.value());
    cudaMemsetAsync(error_pos.data(), 0xFF, sizeof(int32_t), stream.value());
    cudaMemsetAsync(record_bytes_sum.data(), 0, sizeof(int64_t), stream.value());
  }

  // Compute chunk_record_offsets using exclusive_scan
  rmm::device_uvector<int32_t> chunk_record_offsets(num_chunks + 1, stream, mr);
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         chunk_record_counts.begin(),
                         chunk_record_counts.end(),
                         chunk_record_offsets.begin());

  // Compute total record count
  int32_t num_records = thrust::reduce(rmm::exec_policy(stream),
                                       chunk_record_counts.begin(),
                                       chunk_record_counts.end(),
                                       0,
                                       thrust::plus<int32_t>());
  result.total_rows = num_records;

  if (num_records == 0) {
    stream.synchronize();
    result.file_row_counts.resize(num_files, 0);
    return result;
  }

  // ========================================================================
  // Step 4: Parse records with deterministic positions (no sorting needed!)
  // ========================================================================
  cudf::scoped_range parse_range{"seqfile::multi_parse_records"};

  rmm::device_uvector<multi_file_record_info> records(num_records, stream, mr);

  parse_records_multi_file_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
    combined_data,
    d_chunks.data(),
    num_chunks,
    d_file_descs.data(),
    records.data(),           // records output
    nullptr,                  // chunk_record_counts = nullptr (writing mode)
    chunk_record_offsets.data(),  // per-chunk offsets for deterministic writing
    num_records,
    error_flag_ptr,
    error_file_idx_ptr,
    error_pos_ptr,
    record_bytes_sum_ptr);

  if (debug) {
    int32_t err = error_flag.value(stream);
    if (err != 0) {
      int32_t fidx = error_file_idx.value(stream);
      int32_t epos = error_pos.value(stream);
      std::cerr << "SequenceFile GPU parser error in multi-file parse_records_in_chunk (code="
                << err << ", file=" << fidx << ", pos=" << epos << ")"
                << std::endl;
      CUDF_FAIL("SequenceFile GPU parser error in multi-file parse_records_in_chunk");
    }
    int64_t record_bytes = record_bytes_sum.value(stream);
    std::cerr << "SequenceFile GPU parser record_bytes_sum=" << record_bytes
              << " total_data_size=" << total_data_size << std::endl;
  }

  // Records are already in order (deterministic positions based on chunk offsets)
  // No sorting needed!

  // Compute per-file record counts (since we tracked per-chunk, not per-file)
  cudaMemsetAsync(d_file_record_counts.data(), 0, num_files * sizeof(int32_t), stream.value());
  int count_blocks = (num_records + BLOCK_SIZE - 1) / BLOCK_SIZE;
  count_records_per_file_kernel<<<count_blocks, BLOCK_SIZE, 0, stream.value()>>>(
    records.data(),
    num_records,
    d_file_record_counts.data(),
    num_files);

  // Copy per-file record counts to host
  result.file_row_counts.resize(num_files);
  cudaMemcpyAsync(result.file_row_counts.data(),
                  d_file_record_counts.data(),
                  num_files * sizeof(int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  stream.synchronize();

  // ========================================================================
  // Step 5: Compute output offsets
  // ========================================================================
  cudf::scoped_range offsets_range{"seqfile::multi_compute_offsets"};

  rmm::device_uvector<int32_t> key_offsets(num_records + 1, stream, mr);
  rmm::device_uvector<int32_t> value_offsets(num_records + 1, stream, mr);

  cudaMemsetAsync(key_offsets.data(), 0, sizeof(int32_t), stream.value());
  cudaMemsetAsync(value_offsets.data(), 0, sizeof(int32_t), stream.value());

  thrust::transform(rmm::exec_policy(stream),
                    records.begin(),
                    records.begin() + num_records,
                    key_offsets.begin() + 1,
                    [] __device__(multi_file_record_info const& r) { return r.key_len; });
  thrust::inclusive_scan(
      rmm::exec_policy(stream), key_offsets.begin() + 1, key_offsets.end(), key_offsets.begin() + 1);

  thrust::transform(rmm::exec_policy(stream),
                    records.begin(),
                    records.begin() + num_records,
                    value_offsets.begin() + 1,
                    [] __device__(multi_file_record_info const& r) { return r.value_len; });
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         value_offsets.begin() + 1,
                         value_offsets.end(),
                         value_offsets.begin() + 1);

  int32_t total_key_bytes   = 0;
  int32_t total_value_bytes = 0;
  cudaMemcpyAsync(&total_key_bytes,
                  key_offsets.data() + num_records,
                  sizeof(int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  cudaMemcpyAsync(&total_value_bytes,
                  value_offsets.data() + num_records,
                  sizeof(int32_t),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  stream.synchronize();
  if (debug) {
    int64_t total_bytes = static_cast<int64_t>(total_key_bytes) + total_value_bytes;
    if (total_key_bytes < 0 || total_value_bytes < 0 ||
        total_bytes > static_cast<int64_t>(total_data_size)) {
      int64_t record_bytes = record_bytes_sum.value(stream);
      std::cerr << "SequenceFile GPU parser invalid offsets (multi-file): "
                << "total_key_bytes=" << total_key_bytes
                << ", total_value_bytes=" << total_value_bytes
                << ", total_data_size=" << total_data_size
                << ", num_records=" << num_records
                << ", record_bytes_sum=" << record_bytes
                << ", num_chunks=" << num_chunks
                << std::endl;
      CUDF_FAIL("SequenceFile GPU parser produced invalid offsets (multi-file)");
    }
  }

  // ========================================================================
  // Step 6: Extract data
  // ========================================================================
  cudf::scoped_range extract_range{"seqfile::multi_extract_data"};

  rmm::device_uvector<uint8_t> key_data(wants_key ? total_key_bytes : 0, stream, mr);
  rmm::device_uvector<uint8_t> value_data(wants_value ? total_value_bytes : 0, stream, mr);

  // Each warp (32 threads) processes one record.
  // With BLOCK_SIZE=256, we have 8 warps per block.
  int32_t const warps_per_block = BLOCK_SIZE / WARP_SIZE;
  num_blocks = (num_records + warps_per_block - 1) / warps_per_block;
  extract_data_multi_file_kernel<<<num_blocks, BLOCK_SIZE, 0, stream.value()>>>(
      combined_data,
      records.data(),
      num_records,
      key_offsets.data(),
      value_offsets.data(),
      wants_key ? key_data.data() : nullptr,
      wants_value ? value_data.data() : nullptr,
      wants_key,
      wants_value);

  stream.synchronize();

  // ========================================================================
  // Step 7: Build output columns
  // ========================================================================
  cudf::scoped_range build_range{"seqfile::multi_build_columns"};

  if (wants_key) {
    result.key_column = build_list_column(key_data, key_offsets, num_records, stream, mr);
  }

  if (wants_value) {
    result.value_column = build_list_column(value_data, value_offsets, num_records, stream, mr);
  }

  return result;
}

}  // namespace spark_rapids_jni
