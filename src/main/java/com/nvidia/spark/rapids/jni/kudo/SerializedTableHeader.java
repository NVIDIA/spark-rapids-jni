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

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.util.Arrays;

/**
 * Holds the metadata about a serialized table. If this is being read from a stream
 * isInitialized will return true if the metadata was read correctly from the stream.
 * It will return false if an EOF was encountered at the beginning indicating that
 * there was no data to be read.
 */
public final class SerializedTableHeader {
    /**
     * Magic number "KUD0" in ASCII.
     */
    private static final int SER_FORMAT_MAGIC_NUMBER = 0x4B554400;

    // The offset in the original table where row starts. For example, if we want to serialize rows [3, 9) of the
    // original table, offset would be 3, and numRows would be 6.
    private long offset;
    private long numRows;
    private long validityBufferLen;
    private long offsetBufferLen;
    private long totalDataLen;
    private int numColumns;
    // A bit set to indicate if a column has a validity buffer or not. Each column is represented by a single bit.
    private byte[] hasValidityBuffer;

    private boolean initialized = false;


    public SerializedTableHeader(DataInputStream din) throws IOException {
        readFrom(din);
    }

    SerializedTableHeader(long offset, long numRows, long validityBufferLen, long offsetBufferLen,
        long totalDataLen, int numColumns, byte[] hasValidityBuffer) {
        this.offset = offset;
        this.numRows = numRows;
        this.validityBufferLen = validityBufferLen;
        this.offsetBufferLen = offsetBufferLen;
        this.totalDataLen = totalDataLen;
        this.numColumns = numColumns;
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
        int pos = columnIndex / 8;
        int bit = columnIndex % 8;
        return (hasValidityBuffer[pos] & (1 << bit)) != 0;
    }

    public int getSerializedSize() {
        return 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + hasValidityBuffer.length;
    }

    public int getNumColumns() {
        return numColumns;
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

        offset = din.readInt();
        numRows = din.readInt();

        validityBufferLen = din.readInt();
        offsetBufferLen = din.readInt();
        totalDataLen = din.readInt();
        numColumns = din.readInt();
        int validityBufferLength = din.readInt();
        hasValidityBuffer = new byte[validityBufferLength];
        din.readFully(hasValidityBuffer);

        initialized = true;
    }

    public void writeTo(DataWriter dout) throws IOException {
        // Now write out the data
        dout.writeInt(SER_FORMAT_MAGIC_NUMBER);

        dout.writeInt((int)offset);
        dout.writeInt((int)numRows);
        dout.writeInt((int)validityBufferLen);
        dout.writeInt((int)offsetBufferLen);
        dout.writeInt((int)totalDataLen);
        dout.writeInt(numColumns);
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
            ", numColumns=" + numColumns +
            ", hasValidityBuffer=" + Arrays.toString(hasValidityBuffer) +
            ", initialized=" + initialized +
            '}';
    }
}
