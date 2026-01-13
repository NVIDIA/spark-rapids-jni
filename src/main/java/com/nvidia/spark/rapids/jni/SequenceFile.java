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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.NativeDepsLoader;

/**
 * JNI interface for GPU-accelerated parsing of uncompressed Hadoop SequenceFiles.
 *
 * <p>This class provides native methods to parse SequenceFile records directly on the GPU,
 * bypassing the CPU-based Hadoop SequenceFile.Reader for improved performance.</p>
 *
 * <h2>SequenceFile Format (Uncompressed, Version 6)</h2>
 * <pre>
 * Header:
 *   - Magic: "SEQ" (3 bytes) + Version (1 byte, typically 6)
 *   - KeyClassName: UTF-8 string (2-byte length prefix)
 *   - ValueClassName: UTF-8 string (2-byte length prefix)
 *   - isCompressed: boolean (1 byte) = 0x00 (uncompressed)
 *   - isBlockCompressed: boolean (1 byte) = 0x00
 *   - Metadata: key-value pairs
 *   - Sync Marker: 16 bytes (random, used for split alignment)
 *
 * Records:
 *   [Optional] Sync Marker (-1 as int32 + 16 bytes sync)
 *
 *   Record:
 *     - recordLength: int32 (4 bytes, big-endian) = keyLen + valueLen
 *     - keyLength: int32 (4 bytes, big-endian)
 *     - keyBytes: byte[keyLength]
 *     - valueBytes: byte[recordLength - keyLength]
 *
 *   Sync markers are inserted approximately every 2000 records.
 * </pre>
 *
 * <h2>Usage</h2>
 * <ol>
 *   <li>Read the file into a DeviceMemoryBuffer (including header)</li>
 *   <li>Parse the header on CPU to extract the 16-byte sync marker and header size</li>
 *   <li>Call {@link #parseSequenceFile} with the data buffer (excluding header), sync marker,
 *       and flags indicating which columns are needed</li>
 *   <li>The method returns an array of ColumnVectors (key and/or value as LIST&lt;UINT8&gt;)</li>
 * </ol>
 */
public class SequenceFile {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Parse uncompressed SequenceFile data on the GPU and return key/value columns.
   *
   * <p>The input data buffer should contain only the record data portion of the SequenceFile,
   * excluding the header. The header should be parsed on the CPU to extract the sync marker.</p>
   *
   * @param data Device memory buffer containing SequenceFile record data (excluding header).
   *             The buffer is not modified and remains owned by the caller.
   * @param dataSize The actual size of valid data in the buffer (may be less than buffer capacity).
   * @param syncMarker The 16-byte sync marker extracted from the file header.
   *                   Must be exactly 16 bytes.
   * @param wantsKey If true, include the key column in the output.
   * @param wantsValue If true, include the value column in the output.
   * @return An array of ColumnVectors. The array length depends on wantsKey and wantsValue:
   *         <ul>
   *           <li>If both true: [keyColumn, valueColumn]</li>
   *           <li>If only wantsKey: [keyColumn]</li>
   *           <li>If only wantsValue: [valueColumn]</li>
   *           <li>If neither: empty array (but why would you call this?)</li>
   *         </ul>
   *         Each column is of type LIST&lt;UINT8&gt; (equivalent to Spark BinaryType).
   *         The caller is responsible for closing the returned columns.
   * @throws IllegalArgumentException if syncMarker is not exactly 16 bytes,
   *         or if data is null, or if dataSize is negative.
   * @throws RuntimeException if parsing fails due to malformed data.
   */
  public static ColumnVector[] parseSequenceFile(
      DeviceMemoryBuffer data,
      long dataSize,
      byte[] syncMarker,
      boolean wantsKey,
      boolean wantsValue) {
    if (data == null) {
      throw new IllegalArgumentException("data buffer cannot be null");
    }
    if (dataSize < 0) {
      throw new IllegalArgumentException("dataSize cannot be negative: " + dataSize);
    }
    if (syncMarker == null || syncMarker.length != 16) {
      throw new IllegalArgumentException(
          "syncMarker must be exactly 16 bytes, got: " +
          (syncMarker == null ? "null" : syncMarker.length + " bytes"));
    }
    if (!wantsKey && !wantsValue) {
      return new ColumnVector[0];
    }

    long[] columnHandles = parseSequenceFileNative(
        data.getAddress(),
        dataSize,
        syncMarker,
        wantsKey,
        wantsValue);

    ColumnVector[] result = new ColumnVector[columnHandles.length];
    for (int i = 0; i < columnHandles.length; i++) {
      result[i] = new ColumnVector(columnHandles[i]);
    }
    return result;
  }

  /**
   * Get the number of records in SequenceFile data without fully parsing.
   *
   * <p>This is a lightweight operation that only counts records by scanning
   * for record boundaries and sync markers.</p>
   *
   * @param data Device memory buffer containing SequenceFile record data (excluding header).
   * @param dataSize The actual size of valid data in the buffer.
   * @param syncMarker The 16-byte sync marker extracted from the file header.
   * @return The number of records in the data.
   */
  public static long countRecords(
      DeviceMemoryBuffer data,
      long dataSize,
      byte[] syncMarker) {
    if (data == null) {
      throw new IllegalArgumentException("data buffer cannot be null");
    }
    if (dataSize < 0) {
      throw new IllegalArgumentException("dataSize cannot be negative: " + dataSize);
    }
    if (syncMarker == null || syncMarker.length != 16) {
      throw new IllegalArgumentException(
          "syncMarker must be exactly 16 bytes, got: " +
          (syncMarker == null ? "null" : syncMarker.length + " bytes"));
    }
    if (dataSize == 0) {
      return 0;
    }

    return countRecordsNative(data.getAddress(), dataSize, syncMarker);
  }

  // Native method declarations

  /**
   * Native implementation of SequenceFile parsing.
   *
   * @param dataAddress Device memory address of the data buffer.
   * @param dataSize Size of the data in bytes.
   * @param syncMarker 16-byte sync marker.
   * @param wantsKey Whether to extract keys.
   * @param wantsValue Whether to extract values.
   * @return Array of native column handles (long pointers).
   */
  private static native long[] parseSequenceFileNative(
      long dataAddress,
      long dataSize,
      byte[] syncMarker,
      boolean wantsKey,
      boolean wantsValue);

  /**
   * Native implementation of record counting.
   *
   * @param dataAddress Device memory address of the data buffer.
   * @param dataSize Size of the data in bytes.
   * @param syncMarker 16-byte sync marker.
   * @return Number of records.
   */
  private static native long countRecordsNative(
      long dataAddress,
      long dataSize,
      byte[] syncMarker);
}
