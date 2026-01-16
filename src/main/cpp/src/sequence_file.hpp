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
 * @brief Descriptor for a single file in a multi-file batch.
 *
 * Used when parsing multiple SequenceFiles in a single GPU kernel launch.
 */
struct file_descriptor {
  int64_t data_offset;                ///< Offset of this file's data in the combined buffer
  int64_t data_size;                  ///< Size of this file's data in bytes
  uint8_t sync_marker[SYNC_MARKER_SIZE];  ///< This file's 16-byte sync marker
};

/**
 * @brief Result of parsing multiple SequenceFiles.
 *
 * Contains combined key/value columns plus per-file record counts for partition value assignment.
 */
struct multi_file_result {
  std::unique_ptr<cudf::column> key_column;    ///< Combined key column (LIST<UINT8>)
  std::unique_ptr<cudf::column> value_column;  ///< Combined value column (LIST<UINT8>)
  std::vector<int32_t> file_row_counts;        ///< Number of records from each file
  cudf::size_type total_rows;                  ///< Total number of records across all files
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

/**
 * @brief Parse multiple SequenceFiles in a single GPU operation.
 *
 * This function parses multiple SequenceFiles that have been concatenated into
 * a single buffer, using their individual sync markers. This enables higher
 * GPU parallelism by processing chunks from all files simultaneously.
 *
 * @param combined_data Pointer to device memory containing concatenated file data.
 * @param file_descs Vector of file descriptors (offset, size, sync_marker for each file).
 * @param wants_key If true, extract and return the key column.
 * @param wants_value If true, extract and return the value column.
 * @param stream CUDA stream to use for operations.
 * @param mr Device memory resource for allocations.
 * @return A multi_file_result containing combined columns and per-file record counts.
 */
multi_file_result parse_multiple_sequence_files(
  uint8_t const* combined_data,
  std::vector<file_descriptor> const& file_descs,
  bool wants_key,
  bool wants_value,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
