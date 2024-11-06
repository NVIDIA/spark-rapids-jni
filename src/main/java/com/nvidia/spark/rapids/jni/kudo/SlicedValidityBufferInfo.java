/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.jni.kudo;

/**
 * A simple utility class to hold information about serializing/deserializing sliced validity buffer.
 */
class SlicedValidityBufferInfo {
  private final long bufferOffset;
  private final long bufferLength;
  /// The bit offset within the buffer where the slice starts
  private final long beginBit;
  private final long endBit; // Exclusive

  SlicedValidityBufferInfo(long bufferOffset, long bufferLength, long beginBit, long endBit) {
    this.bufferOffset = bufferOffset;
    this.bufferLength = bufferLength;
    this.beginBit = beginBit;
    this.endBit = endBit;
  }

  @Override
  public String toString() {
    return "SlicedValidityBufferInfo{" + "bufferOffset=" + bufferOffset + ", bufferLength=" + bufferLength +
        ", beginBit=" + beginBit + ", endBit=" + endBit + '}';
  }

  public long getBufferOffset() {
    return bufferOffset;
  }

  public long getBufferLength() {
    return bufferLength;
  }

  public long getBeginBit() {
    return beginBit;
  }

  public long getEndBit() {
    return endBit;
  }

  static SlicedValidityBufferInfo calc(long rowOffset, long numRows) {
    if (rowOffset < 0) {
      throw new IllegalArgumentException("rowOffset must be >= 0, but was " + rowOffset);
    }
    if (numRows < 0) {
      throw new IllegalArgumentException("numRows must be >= 0, but was " + numRows);
    }
    long bufferOffset = rowOffset / 8;
    long beginBit = rowOffset % 8;
    long bufferLength = 0;
    if (numRows > 0) {
      bufferLength = (rowOffset + numRows - 1) / 8 - bufferOffset + 1;
    }
    long endBit = beginBit + numRows;
    return new SlicedValidityBufferInfo(bufferOffset, bufferLength, beginBit, endBit);
  }
}
