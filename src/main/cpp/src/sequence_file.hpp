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

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace spark_rapids_jni {

/**
 * @brief Size of the sync marker in Hadoop SequenceFile format.
 */
constexpr size_t SYNC_MARKER_SIZE = 16;

/**
 * @brief Magic number indicating a sync marker in the record stream.
 *
 * When parsing records, if the "record length" field equals -1 (0xFFFFFFFF),
 * the next 16 bytes are a sync marker, not a record.
 */
constexpr int32_t SYNC_MARKER_INDICATOR = -1;

/**
 * @brief Result of parsing a SequenceFile.
 *
 * Contains the key and/or value columns as LIST<UINT8> (equivalent to Spark BinaryType).
 */
struct sequence_file_result {
  std::unique_ptr<cudf::column> key_column;    ///< Key column (LIST<UINT8>), may be null
  std::unique_ptr<cudf::column> value_column;  ///< Value column (LIST<UINT8>), may be null
  cudf::size_type num_rows;                    ///< Number of records parsed
};

/**
 * @brief Parse uncompressed SequenceFile data on the GPU.
 *
 * This function parses the record portion of a Hadoop SequenceFile (version 6, uncompressed)
 * and returns key/value data as cuDF LIST<UINT8> columns.
 *
 * The parsing is performed entirely on the GPU using multiple CUDA kernels:
 * 1. Scan for sync marker positions to identify chunk boundaries
 * 2. Parse record boundaries within each chunk
 * 3. Extract key/value data to contiguous output buffers
 *
 * @param data Pointer to device memory containing SequenceFile record data (excluding header).
 * @param data_size Size of the data in bytes.
 * @param sync_marker The 16-byte sync marker from the file header.
 * @param wants_key If true, extract and return the key column.
 * @param wants_value If true, extract and return the value column.
 * @param stream CUDA stream to use for operations.
 * @param mr Device memory resource for allocations.
 * @return A sequence_file_result containing the requested columns.
 *
 * @throws cudf::logic_error if sync_marker size is not 16 bytes.
 * @throws cudf::logic_error if data contains malformed records.
 */
sequence_file_result parse_sequence_file(
  uint8_t const* data,
  size_t data_size,
  std::vector<uint8_t> const& sync_marker,
  bool wants_key,
  bool wants_value,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/**
 * @brief Count the number of records in SequenceFile data.
 *
 * This is a lightweight operation that only scans for record boundaries
 * without extracting the actual data.
 *
 * @param data Pointer to device memory containing SequenceFile record data.
 * @param data_size Size of the data in bytes.
 * @param sync_marker The 16-byte sync marker from the file header.
 * @param stream CUDA stream to use for operations.
 * @return The number of records in the data.
 */
cudf::size_type count_records(uint8_t const* data,
                              size_t data_size,
                              std::vector<uint8_t> const& sync_marker,
                              rmm::cuda_stream_view stream = cudf::get_default_stream());

}  // namespace spark_rapids_jni
