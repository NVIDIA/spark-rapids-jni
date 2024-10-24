package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.HostMemoryBuffer;

import java.io.IOException;

/**
 * Visible for testing
 */
abstract class DataWriter {

    public abstract void writeByte(byte b) throws IOException;

    public abstract void writeShort(short s) throws IOException;

    public abstract void writeInt(int i) throws IOException;

    public abstract void writeIntNativeOrder(int i) throws IOException;

    public abstract void writeLong(long val) throws IOException;

    /**
     * Copy data from src starting at srcOffset and going for len bytes.
     *
     * @param src       where to copy from.
     * @param srcOffset offset to start at.
     * @param len       amount to copy.
     */
    public abstract void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len) throws IOException;

    public void flush() throws IOException {
        // NOOP by default
    }

    public abstract void write(byte[] arr, int offset, int length) throws IOException;
}
