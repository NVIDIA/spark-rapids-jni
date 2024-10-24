package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.DType;
import ai.rapids.cudf.DeviceMemoryBuffer;

import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.safeLongToInt;


public class ColumnViewInfo {
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

        return RefUtils.makeCudfColumnView(
                dtype.getTypeId().getNativeId(), dtype.getScale(),
                dataAddress, offsetInfo.getDataLen(),
                offsetsAddress, validityAddress,
                safeLongToInt(nullCount), safeLongToInt(rowCount),
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
