package com.nvidia.spark.rapids.jni.kudo;

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
        return "SlicedValidityBufferInfo{" + "bufferOffset=" + bufferOffset + ", bufferLength=" + bufferLength + ", beginBit=" + beginBit + ", endBit=" + endBit + '}';
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
