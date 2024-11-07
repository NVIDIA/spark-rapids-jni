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

import java.util.OptionalLong;

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static com.nvidia.spark.rapids.jni.Preconditions.ensureNonNegative;

/**
 * This class is used to store the offsets of the buffer of a column in the serialized data.
 */
class ColumnOffsetInfo {
    private static final long INVALID_OFFSET = -1L;
    private final long validity;
    private final long offset;
    private final long data;
    private final long dataLen;

    public ColumnOffsetInfo(long validity, long offset, long data, long dataLen) {
        ensure(dataLen >= 0, () -> "dataLen must be non-negative, but was " + dataLen);
        this.validity = validity;
        this.offset = offset;
        this.data = data;
        this.dataLen = dataLen;
    }

    public OptionalLong getValidity() {
        return (validity == INVALID_OFFSET) ? OptionalLong.empty() : OptionalLong.of(validity);
    }

    public OptionalLong getOffset() {
        return (offset == INVALID_OFFSET) ? OptionalLong.empty() : OptionalLong.of(offset);
    }

    public OptionalLong getData() {
        return (data == INVALID_OFFSET) ? OptionalLong.empty() : OptionalLong.of(data);
    }

    public long getDataLen() {
        return dataLen;
    }

    @Override
    public String toString() {
        return "ColumnOffsets{" +
                "validity=" + validity +
                ", offset=" + offset +
                ", data=" + data +
                ", dataLen=" + dataLen +
                '}';
    }
}
