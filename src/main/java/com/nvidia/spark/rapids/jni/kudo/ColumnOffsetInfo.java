package com.nvidia.spark.rapids.jni.kudo;

import java.util.OptionalLong;

/**
 * This class is used to store the offsets of the buffer of a column in the serialized data.
 */
public class ColumnOffsetInfo {
    private static final long INVALID_OFFSET = -1L;
    private final long validity;
    private final long offset;
    private final long data;
    private final long dataLen;

    public ColumnOffsetInfo(long validity, long offset, long data, long dataLen) {
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
