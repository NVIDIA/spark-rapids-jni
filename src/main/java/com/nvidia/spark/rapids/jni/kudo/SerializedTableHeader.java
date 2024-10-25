package com.nvidia.spark.rapids.jni.kudo;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Optional;

/**
 * Holds the metadata about a serialized table. If this is being read from a stream
 * isInitialized will return true if the metadata was read correctly from the stream.
 * It will return false if an EOF was encountered at the beginning indicating that
 * there was no data to be read.
 */
public final class SerializedTableHeader {
    /**
     * Magic number "KUDO" in ASCII.
     */
    private static final int SER_FORMAT_MAGIC_NUMBER = 0x4B55444F;
    private static final short VERSION_NUMBER = 0x0001;

    // Useful for reducing calculations in writing.
    private long offset;
    private long numRows;
    private long validityBufferLen;
    private long offsetBufferLen;
    private long totalDataLen;
    // This is used to indicate the validity buffer for the columns.
    // 1 means that this column has validity data, 0 means it does not.
    private byte[] hasValidityBuffer;

    private boolean initialized = false;


    public SerializedTableHeader(DataInputStream din) throws IOException {
        readFrom(din);
    }

    SerializedTableHeader(long offset, long numRows, long validityBufferLen, long offsetBufferLen, long totalDataLen, byte[] hasValidityBuffer) {
        this.offset = offset;
        this.numRows = numRows;
        this.validityBufferLen = validityBufferLen;
        this.offsetBufferLen = offsetBufferLen;
        this.totalDataLen = totalDataLen;
        this.hasValidityBuffer = hasValidityBuffer;

        this.initialized = true;
    }

    /**
     * Returns the size of a buffer needed to read data into the stream.
     */
    public long getTotalDataLen() {
        return totalDataLen;
    }

    /**
     * Returns the number of rows stored in this table.
     */
    public long getNumRows() {
        return numRows;
    }

    public long getOffset() {
        return offset;
    }

    /**
     * Returns true if the metadata for this table was read, else false indicating an EOF was
     * encountered.
     */
    public boolean wasInitialized() {
        return initialized;
    }

    public boolean hasValidityBuffer(int columnIndex) {
        return hasValidityBuffer[columnIndex] != 0;
    }

    public long getSerializedSize() {
        return 4 + 2 + 8 + 8 + 8 + 8 + 8 + 4 + hasValidityBuffer.length;
    }

    public int getNumColumns() {
        return Optional.ofNullable(hasValidityBuffer).map(arr -> arr.length).orElse(0);
    }

    public long getValidityBufferLen() {
        return validityBufferLen;
    }

    public long getOffsetBufferLen() {
        return offsetBufferLen;
    }

    public boolean isInitialized() {
        return initialized;
    }

    private void readFrom(DataInputStream din) throws IOException {
        try {
            int num = din.readInt();
            if (num != SER_FORMAT_MAGIC_NUMBER) {
                throw new IllegalStateException("THIS DOES NOT LOOK LIKE CUDF SERIALIZED DATA. " + "Expected magic number " + SER_FORMAT_MAGIC_NUMBER + " Found " + num);
            }
        } catch (EOFException e) {
            // If we get an EOF at the very beginning don't treat it as an error because we may
            // have finished reading everything...
            return;
        }
        short version = din.readShort();
        if (version != VERSION_NUMBER) {
            throw new IllegalStateException("READING THE WRONG SERIALIZATION FORMAT VERSION FOUND " + version + " EXPECTED " + VERSION_NUMBER);
        }

        offset = din.readLong();
        numRows = din.readLong();

        validityBufferLen = din.readLong();
        offsetBufferLen = din.readLong();
        totalDataLen = din.readLong();
        int validityBufferLength = din.readInt();
        hasValidityBuffer = new byte[validityBufferLength];
        din.readFully(hasValidityBuffer);

        initialized = true;
    }

    public void writeTo(DataWriter dout) throws IOException {
        // Now write out the data
        dout.writeInt(SER_FORMAT_MAGIC_NUMBER);
        dout.writeShort(VERSION_NUMBER);

        dout.writeLong(offset);
        dout.writeLong(numRows);
        dout.writeLong(validityBufferLen);
        dout.writeLong(offsetBufferLen);
        dout.writeLong(totalDataLen);
        dout.writeInt(hasValidityBuffer.length);
        dout.write(hasValidityBuffer, 0, hasValidityBuffer.length);
    }

    @Override
    public String toString() {
        return "SerializedTableHeader{" +
                "offset=" + offset +
                ", numRows=" + numRows +
                ", validityBufferLen=" + validityBufferLen +
                ", offsetBufferLen=" + offsetBufferLen +
                ", totalDataLen=" + totalDataLen +
                ", hasValidityBuffer=" + Arrays.toString(hasValidityBuffer) +
                ", initialized=" + initialized +
                '}';
    }
}
