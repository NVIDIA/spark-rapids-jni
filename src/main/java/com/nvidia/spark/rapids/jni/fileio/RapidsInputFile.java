/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.jni.fileio;

import ai.rapids.cudf.HostMemoryBuffer;

import java.io.IOException;
import java.util.List;
import java.util.OptionalLong;

/**
 * Represents an input file that can be read from.
 * <br/>
 * The implementation of this interface should be thread-safe.
 */
public interface RapidsInputFile {
  /**
   * Get the path of this input file.
   * @return the file path string
   */
  default String path() {
    throw new UnsupportedOperationException("path is not supported");
  }

  /**
   * Get the length of the file in bytes.
   * @return the length of the file in bytes
   * @throws IOException if an I/O error occurs while getting the length
   */
  long getLength() throws IOException;

  /**
   * Get the last modification time of the file in milliseconds since epoch.
   * @return an OptionalLong containing the last modification time, or empty if not available
   * @throws IOException if an I/O error occurs while getting the modification time
   */
  default OptionalLong getLastModificationTime() throws IOException {
    return OptionalLong.empty();
  }

  /**
   * Reads data from the input file into the provided output buffer using vectored read.
   *
   * <p>The output buffer will not be closed by this method. It is the caller's responsibility
   * to close it.</p>
   *
   * @param output the buffer to read data into
   * @param copyRanges a list of copy ranges specifying the input offsets, lengths, and output
   *                   offsets
   * @throws IOException if an I/O error occurs during reading
   */
  default void readVectored(HostMemoryBuffer output, List<CopyRange> copyRanges)
      throws IOException {
    throw new UnsupportedOperationException(
        "readVectored is not supported for " + getClass().getName());
  }

  /**
   * Reads the last {@code length} bytes of the input file into the provided output buffer.
   *
   * <p>The output buffer will not be closed by this method. It is the caller's responsibility
   * to close it.</p>
   *
   * @param length the number of bytes to read from the tail
   * @param output the buffer to read data into
   * @throws IOException if an I/O error occurs during reading
   */
  default void readTail(long length, HostMemoryBuffer output) throws IOException {
    throw new UnsupportedOperationException(
        "readTail is not supported for " + getClass().getName());
  }

  /**
   * Open the file for reading.
   * @return a {@link SeekableInputStream } to read from the file
   * @throws IOException if an I/O error occurs while opening the file
   */
  SeekableInputStream open() throws IOException;

  /**
   * Describes a range of bytes to copy from the input file into an output buffer.
   */
  final class CopyRange {
    private final long inputOffset;
    private final long length;
    private final long outputOffset;

    public CopyRange(long inputOffset, long length, long outputOffset) {
      this.inputOffset = inputOffset;
      this.length = length;
      this.outputOffset = outputOffset;
    }

    public long getInputOffset() {
      return inputOffset;
    }

    public long getLength() {
      return length;
    }

    public long getOutputOffset() {
      return outputOffset;
    }
  }
}