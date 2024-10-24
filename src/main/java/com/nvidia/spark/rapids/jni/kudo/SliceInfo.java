package com.nvidia.spark.rapids.jni.kudo;

public class SliceInfo {
    final long offset;
    final long rowCount;
    private final SlicedValidityBufferInfo validityBufferInfo;

    SliceInfo(long offset, long rowCount) {
        this.offset = offset;
        this.rowCount = rowCount;
        this.validityBufferInfo = SlicedValidityBufferInfo.calc(offset, rowCount);
    }

    public SlicedValidityBufferInfo getValidityBufferInfo() {
        return validityBufferInfo;
    }

    public long getOffset() {
        return offset;
    }

    public long getRowCount() {
        return rowCount;
    }

    @Override
    public String toString() {
        return "SliceInfo{" +
                "offset=" + offset +
                ", rowCount=" + rowCount +
                ", validityBufferInfo=" + validityBufferInfo +
                '}';
    }
}
