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

import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.safeLongToInt;

/**
 * Holds the metadata about a serialized table. If this is being read from a stream
 * isInitialized will return true if the metadata was read correctly from the stream.
 * It will return false if an EOF was encountered at the beginning indicating that
 * there was no data to be read.
 */
public final class KudoTableHeader {
  /**
   * Magic number "KUD0" in ASCII.
   */
  private static final int SER_FORMAT_MAGIC_NUMBER = 0x4B554400;

  // The offset in the original table where row starts. For example, if we want to serialize rows [3, 9) of the
  // original table, offset would be 3, and numRows would be 6.
  private int offset;
  private int numRows;
  private int validityBufferLen;
  private int offsetBufferLen;
  private int totalDataLen;
  private int numColumns;
  // A bit set to indicate if a column has a validity buffer or not. Each column is represented by a single bit.
  private byte[] hasValidityBuffer;

  private boolean initialized = false;


  public KudoTableHeader(DataInputStream din) throws IOException {
    readFrom(din);
  }

  KudoTableHeader(long offset, long numRows, long validityBufferLen, long offsetBufferLen,
                  long totalDataLen, int numColumns, byte[] hasValidityBuffer) {
    this.offset = safeLongToInt(offset);
    this.numRows = safeLongToInt(numRows);
    this.validityBufferLen = safeLongToInt(validityBufferLen);
    this.offsetBufferLen = safeLongToInt(offsetBufferLen);
    this.totalDataLen = safeLongToInt(totalDataLen);
    this.numColumns = safeLongToInt(numColumns);
    this.hasValidityBuffer = hasValidityBuffer;

    this.initialized = true;
  }

  /**
   * Returns the size of a buffer needed to read data into the stream.
   */
  public int getTotalDataLen() {
    return totalDataLen;
  }

  /**
   * Returns the number of rows stored in this table.
   */
  public int getNumRows() {
    return numRows;
  }

  public int getOffset() {
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

  /**
   * Get the size of the serialized header.
   *
   * <p>
   * It consists of the following fields:
   * <ol>
   *   <li>Magic Number</li>
   *   <li>Row Offset</li>
   *   <li>Number of rows</li>
   *   <li>Validity buffer length</li>
   *   <li>Offset buffer length</li>
   *   <li>Total data length</li>
   *   <li>Number of columns</li>
   *   <li>hasValidityBuffer</li>
   * </ol>
   *
   * For more details of each field, please refer to {@link KudoSerializer}.
   * <p/>
   *
   * @return the size of the serialized header.
   */
  public int getSerializedSize() {
    return 7 * Integer.BYTES + hasValidityBuffer.length;
  }

  public int getNumColumns() {
    return numColumns;
  }

  public int getValidityBufferLen() {
    return validityBufferLen;
  }

  public int getOffsetBufferLen() {
    return offsetBufferLen;
  }

  public boolean isInitialized() {
    return initialized;
  }

  private void readFrom(DataInputStream din) throws IOException {
    try {
      int num = din.readInt();
      if (num != SER_FORMAT_MAGIC_NUMBER) {
        throw new IllegalStateException("Kudo format error, expected magic number " + SER_FORMAT_MAGIC_NUMBER +
            " found " + num);
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
    int validityBufferLength = lengthOfHasValidityBuffer(numColumns);
    hasValidityBuffer = new byte[validityBufferLength];
    din.readFully(hasValidityBuffer);

    initialized = true;
  }

  public void writeTo(DataWriter dout) throws IOException {
    // Now write out the data
    dout.writeInt(SER_FORMAT_MAGIC_NUMBER);

    dout.writeInt(offset);
    dout.writeInt(numRows);
    dout.writeInt(validityBufferLen);
    dout.writeInt(offsetBufferLen);
    dout.writeInt(totalDataLen);
    dout.writeInt(numColumns);
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

  private static int lengthOfHasValidityBuffer(int numColumns) {
    return (numColumns + 1 + 7) / 8;
  }
}
