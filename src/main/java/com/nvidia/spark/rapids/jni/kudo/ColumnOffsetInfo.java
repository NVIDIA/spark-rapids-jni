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

import ai.rapids.cudf.DeviceMemoryBufferView;

import static com.nvidia.spark.rapids.jni.Preconditions.ensureNonNegative;

/**
 * This class is used to store the offsets of the buffer of a column in the serialized data.
 */
class ColumnOffsetInfo {
  static final long INVALID_OFFSET = -1L;
  private final long validity;
  private final long validityBufferLen;
  private final long offset;
  private final long offsetBufferLen;
  private final long data;
  private final long dataBufferLen;

  public ColumnOffsetInfo(long validity, long validityBufferLen, long offset, long offsetBufferLen, long data,
                          long dataBufferLen) {
    ensureNonNegative(validityBufferLen, "validityBuffeLen");
    ensureNonNegative(offsetBufferLen, "offsetBufferLen");
    ensureNonNegative(dataBufferLen, "dataBufferLen");
    this.validity = validity;
    this.validityBufferLen = validityBufferLen;
    this.offset = offset;
    this.offsetBufferLen = offsetBufferLen;
    this.data = data;
    this.dataBufferLen = dataBufferLen;
  }

  /**
   * Get the validity buffer offset.
   * @return {@value #INVALID_OFFSET} if the validity buffer is not present, otherwise the offset.
   */
  long getValidity() {
    return validity;
  }

  /**
   * Get a view of the validity buffer from underlying buffer.
   * @param baseAddress the base address of underlying buffer.
   * @return null if the validity buffer is not present, otherwise a view of the buffer.
   */
  DeviceMemoryBufferView getValidityBuffer(long baseAddress) {
    if (validity == INVALID_OFFSET) {
      return null;
    }
    return new DeviceMemoryBufferView(validity + baseAddress, validityBufferLen);
  }

  /**
   * Get the offset buffer offset.
   * @return {@value #INVALID_OFFSET} if the offset buffer is not present, otherwise the offset.
   */
  long getOffset() {
    return offset;
  }

  /**
   * Get a view of the offset buffer from underlying buffer.
   * @param baseAddress the base address of underlying buffer.
   * @return null if the offset buffer is not present, otherwise a view of the buffer.
   */
  DeviceMemoryBufferView getOffsetBuffer(long baseAddress) {
    if (offset == INVALID_OFFSET) {
      return null;
    }
    return new DeviceMemoryBufferView(offset + baseAddress, offsetBufferLen);
  }

  /**
   * Get the data buffer offset.
   * @return {@value #INVALID_OFFSET} if the data buffer is not present, otherwise the offset.
   */
  long getData() {
    return data;
  }

  /**
   * Get a view of the data buffer from underlying buffer.
   * @param baseAddress the base address of underlying buffer.
   * @return null if the data buffer is not present, otherwise a view of the buffer.
   */
  DeviceMemoryBufferView getDataBuffer(long baseAddress) {
    if (data == INVALID_OFFSET) {
      return null;
    }
    return new DeviceMemoryBufferView(data + baseAddress, dataBufferLen);
  }

  long getDataBufferLen() {
    return dataBufferLen;
  }

  @Override
  public String toString() {
    return "ColumnOffsets{" +
        "validity=" + validity +
        ", offset=" + offset +
        ", data=" + data +
        ", dataLen=" + dataBufferLen +
        '}';
  }
}
