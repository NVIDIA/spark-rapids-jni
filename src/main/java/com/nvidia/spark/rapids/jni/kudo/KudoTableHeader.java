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
import java.util.Optional;

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static com.nvidia.spark.rapids.jni.Preconditions.ensureNonNegative;
import static java.util.Objects.requireNonNull;

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
  private static final int SER_FORMAT_MAGIC_NUMBER = 0x4B554430;

  // The offset in the original table where row starts. For example, if we want to serialize rows [3, 9) of the
  // original table, offset would be 3, and numRows would be 6.
  private final int offset;
  private final int numRows;
  private final int validityBufferLen;
  private final int offsetBufferLen;
  private final int totalDataLen;
  // A bit set to indicate if a column has a validity buffer or not. Each column is represented by a single bit.
  private final byte[] hasValidityBuffer;

  /**
   * Reads the table header from the given input stream.
   *
   * @param din input stream
   * @return the table header. If an EOFException is encountered at the beginning, returns empty result.
   * @throws IOException if an I/O error occurs
   */
  public static Optional<KudoTableHeader> readFrom(DataInputStream din, int numColumns) throws IOException {
    int num;
    try {
      System.err.println("READ HEADER...");
      num = din.readInt();
      System.err.println("GOT MAGIC NUMBER " + num + " vs " + SER_FORMAT_MAGIC_NUMBER);
      if (num != SER_FORMAT_MAGIC_NUMBER) {
        throw new IllegalStateException("Kudo format error, expected magic number " + SER_FORMAT_MAGIC_NUMBER +
            " found " + num);
      }
    } catch (EOFException e) {
      // If we get an EOF at the very beginning don't treat it as an error because we may
      // have finished reading everything...
      return Optional.empty();
    }

    int offset = din.readInt();
    System.err.println("GOT OFFSET " + offset);
    int numRows = din.readInt();
    System.err.println("GOT NUM ROWS " + numRows);

    int validityBufferLen = din.readInt();
    System.err.println("GOT VALID LEN " + validityBufferLen);
    int offsetBufferLen = din.readInt();
    System.err.println("GOT OFFSET LEN " + offsetBufferLen);
    int totalDataLen = din.readInt();
    System.err.println("GOT TOTAL LEN " + totalDataLen);
    int validityBufferLength = lengthOfHasValidityBuffer(numColumns);
    System.err.println("VALID BUFFER LEN CALC FOR NUM COLS " + validityBufferLength + " FROM " + numColumns);
    byte[] hasValidityBuffer = new byte[validityBufferLength];
    din.readFully(hasValidityBuffer);

    KudoTableHeader header = new KudoTableHeader(offset, numRows, validityBufferLen, offsetBufferLen, totalDataLen,
        numColumns, hasValidityBuffer);
    int amountPadded = header.getSerializedSize() - header.getNonPaddedSerializedSize();

    din.skipBytes(amountPadded);

    return Optional.of(header);
  }

  KudoTableHeader(int offset, int numRows, int validityBufferLen, int offsetBufferLen,
                  int totalDataLen, int numColumns, byte[] hasValidityBuffer) {
    this.offset = ensureNonNegative(offset, "offset");
    this.numRows = ensureNonNegative(numRows, "numRows");
    this.validityBufferLen = ensureNonNegative(validityBufferLen, "validityBufferLen");
    this.offsetBufferLen = ensureNonNegative(offsetBufferLen, "offsetBufferLen");
    this.totalDataLen = ensureNonNegative(totalDataLen, "totalDataLen");
    ensureNonNegative(numColumns, "numColumns");

    requireNonNull(hasValidityBuffer, "hasValidityBuffer cannot be null");
    ensure(hasValidityBuffer.length == lengthOfHasValidityBuffer(numColumns),
        () -> numColumns + " columns expects hasValidityBuffer with length " + lengthOfHasValidityBuffer(numColumns) +
            ", but found " + hasValidityBuffer.length);
    this.hasValidityBuffer = hasValidityBuffer;
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
   *   <li>hasValidityBuffer</li>
   * </ol>
   * <p>
   * For more details of each field, please refer to {@link KudoSerializer}.
   * <p/>
   *
   * @return the size of the serialized header.
   */
  public int getSerializedSize() {
    return (int)KudoSerializer.padForHostAlignment(getNonPaddedSerializedSize());
  }

  int getNonPaddedSerializedSize() {
    return (6 * Integer.BYTES) + hasValidityBuffer.length;
  }

  public int getValidityBufferLen() {
    return validityBufferLen;
  }

  public int getOffsetBufferLen() {
    return offsetBufferLen;
  }

  public int writeTo(DataWriter dout) throws IOException {
    int streamIndex = 0;
    // Now write out the data
    System.err.println("MAGIC NUMBER AT " + streamIndex);
    dout.writeInt(SER_FORMAT_MAGIC_NUMBER);
    streamIndex += 4;

    dout.writeInt(offset);
    System.err.println("OFFSET AT " + streamIndex);
    streamIndex += 4;

    dout.writeInt(numRows);
    System.err.println("NUM ROWS AT " + streamIndex);
    streamIndex += 4;

    dout.writeInt(validityBufferLen);
    System.err.println("VALIDITY BUF LEN AT " + streamIndex);
    streamIndex += 4;

    dout.writeInt(offsetBufferLen);
    System.err.println("OFFSET BUFF LEN AT " + streamIndex);
    streamIndex += 4;

    dout.writeInt(totalDataLen);
    System.err.println("TOTAL DATA LEN AT " + streamIndex);
    streamIndex += 4;

    dout.write(hasValidityBuffer, 0, hasValidityBuffer.length);
    System.err.println("VALIDITY AT " + streamIndex + " TO " + (streamIndex + hasValidityBuffer.length));
    streamIndex += hasValidityBuffer.length;

    streamIndex = (int) KudoSerializer.padForHostAlignment(dout, streamIndex);
    System.err.println("HEADER PADDED TO " + streamIndex);
    return streamIndex;
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
        '}';
  }

  private static int lengthOfHasValidityBuffer(int numColumns) {
    return (numColumns + 7) / 8;
  }
}
