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

import ai.rapids.cudf.*;

import java.util.Optional;

class ColumnViewInfo {
  private final DType dtype;
  private final ColumnOffsetInfo offsetInfo;
  private final long nullCount;
  private final long rowCount;

  public ColumnViewInfo(DType dtype, ColumnOffsetInfo offsetInfo,
                        long nullCount, long rowCount) {
    this.dtype = dtype;
    this.offsetInfo = offsetInfo;
    this.nullCount = nullCount;
    this.rowCount = rowCount;
  }

  ColumnView buildColumnView(DeviceMemoryBuffer buffer, ColumnView[] childrenView) {
    long baseAddress = buffer.getAddress();

    if (dtype.isNestedType()) {
      return new ColumnView(dtype, rowCount, Optional.of(nullCount),
          offsetInfo.getValidityBuffer(baseAddress).orElse(null),
          offsetInfo.getOffsetBuffer(baseAddress).orElse(null),
          childrenView);
    } else {
      return new ColumnView(dtype, rowCount, Optional.of(nullCount),
          offsetInfo.getDataBuffer(baseAddress).orElse(null),
          offsetInfo.getValidityBuffer(baseAddress).orElse(null),
          offsetInfo.getOffsetBuffer(baseAddress).orElse(null));
    }
  }

  @Override
  public String toString() {
    return "ColumnViewInfo{" +
        "dtype=" + dtype +
        ", offsetInfo=" + offsetInfo +
        ", nullCount=" + nullCount +
        ", rowCount=" + rowCount +
        '}';
  }
}
