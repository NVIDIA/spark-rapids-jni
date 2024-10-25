package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.HostMemoryBuffer;

public class SerializedTable implements AutoCloseable {
    private final SerializedTableHeader header;
    private final HostMemoryBuffer buffer;

    SerializedTable(SerializedTableHeader header, HostMemoryBuffer buffer) {
        this.header = header;
        this.buffer = buffer;
    }

    public SerializedTableHeader getHeader() {
        return header;
    }

    public HostMemoryBuffer getBuffer() {
        return buffer;
    }

    @Override
    public String toString() {
        return "SerializedTable{" +
                "header=" + header +
                ", buffer=" + buffer +
                '}';
    }

    @Override
    public void close() throws Exception {
        if (buffer != null) {
            buffer.close();
        }
    }
}
