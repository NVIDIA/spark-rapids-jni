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

import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.DeviceMemoryBuffer;

import static java.lang.Math.toIntExact;

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

    public long buildColumnView(DeviceMemoryBuffer buffer, long[] childrenView) {
        long bufferAddress = buffer.getAddress();

        long dataAddress = 0;
        if (offsetInfo.getData().isPresent()) {
            dataAddress = buffer.getAddress() + offsetInfo.getData().getAsLong();
        }

        long validityAddress = 0;
        if (offsetInfo.getValidity().isPresent()) {
            validityAddress = offsetInfo.getValidity().getAsLong() + bufferAddress;
        }

        long offsetsAddress = 0;
        if (offsetInfo.getOffset().isPresent()) {
            offsetsAddress = offsetInfo.getOffset().getAsLong() + bufferAddress;
        }

        return ColumnView.makeCudfColumnView(
                dtype.getTypeId().getNativeId(), dtype.getScale(),
                dataAddress, offsetInfo.getDataLen(),
                offsetsAddress, validityAddress,
                toIntExact(nullCount), toIntExact(rowCount),
                childrenView);
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
