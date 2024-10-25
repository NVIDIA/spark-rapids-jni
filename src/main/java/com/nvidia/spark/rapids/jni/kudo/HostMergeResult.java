package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.*;
import com.nvidia.spark.rapids.jni.TableUtils;
import com.nvidia.spark.rapids.jni.schema.Visitors;

import java.util.List;

public class HostMergeResult implements AutoCloseable {
    private final List<ColumnViewInfo> columnOffsets;
    private final HostMemoryBuffer hostBuf;

    public HostMergeResult(HostMemoryBuffer hostBuf, List<ColumnViewInfo> columnOffsets) {
        this.columnOffsets = columnOffsets;
        this.hostBuf = hostBuf;
    }

    @Override
    public void close() throws Exception {
        if (hostBuf != null) {
            hostBuf.close();
        }
    }

    public ContiguousTable toContiguousTable(Schema schema) {
        return TableUtils.closeIfException(DeviceMemoryBuffer.allocate(hostBuf.getLength()),
            deviceMemBuf -> {
            if (hostBuf.getLength() > 0) {
                deviceMemBuf.copyFromHostBuffer(hostBuf);
            }

            TableBuilder builder = new TableBuilder(columnOffsets, deviceMemBuf);
            Table t = Visitors.visitSchema(schema, builder);

            return RefUtils.makeContiguousTable(t, deviceMemBuf);
        });
    }

    @Override
    public String toString() {
        return "HostMergeResult{" +
                "columnOffsets=" + columnOffsets +
                ", hostBuf length =" + hostBuf.getLength() +
                '}';
    }
}
