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

import java.util.Optional;
import java.util.OptionalLong;

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;

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
    ensure(dataBufferLen >= 0, () -> "dataLen must be non-negative, but was " + dataBufferLen);
    this.validity = validity;
    this.validityBufferLen = validityBufferLen;
    this.offset = offset;
    this.offsetBufferLen = offsetBufferLen;
    this.data = data;
    this.dataBufferLen = dataBufferLen;
  }

  public OptionalLong getValidity() {
    return (validity == INVALID_OFFSET) ? OptionalLong.empty() : OptionalLong.of(validity);
  }

  public Optional<DeviceMemoryBufferView> getValidityBuffer(long baseAddress) {
    if (validity == INVALID_OFFSET) {
      return Optional.empty();
    }
    return Optional.of(new DeviceMemoryBufferView(validity + baseAddress, validityBufferLen));
  }

  public OptionalLong getOffset() {
    return (offset == INVALID_OFFSET) ? OptionalLong.empty() : OptionalLong.of(offset);
  }

  public Optional<DeviceMemoryBufferView> getOffsetBuffer(long baseAddress) {
    if (offset == INVALID_OFFSET) {
      return Optional.empty();
    }
    return Optional.of(new DeviceMemoryBufferView(offset + baseAddress, offsetBufferLen));
  }

  public OptionalLong getData() {
    return (data == INVALID_OFFSET) ? OptionalLong.empty() : OptionalLong.of(data);
  }

  public Optional<DeviceMemoryBufferView> getDataBuffer(long baseAddress) {
    if (data == INVALID_OFFSET) {
      return Optional.empty();
    }
    return Optional.of(new DeviceMemoryBufferView(data + baseAddress, dataBufferLen));
  }

  public long getDataBufferLen() {
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
