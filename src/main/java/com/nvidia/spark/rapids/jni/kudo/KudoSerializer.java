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

import ai.rapids.cudf.*;
import com.nvidia.spark.rapids.jni.schema.Visitors;

import java.io.*;
import java.util.Arrays;
import java.util.stream.IntStream;

import static com.nvidia.spark.rapids.jni.Preconditions.ensure;
import static java.util.Objects.requireNonNull;

/**
 * This class is used to serialize/deserialize a table using the Kudo format.
 *
 * <h1>Background</h1>
 *
 * The Kudo format is a binary format that is optimized for serializing/deserializing a table partition during Spark
 * shuffle. The optimizations are based on two key observations:
 *
 * <ol>
 *     <li>The binary format doesn't need to be self descriptive, since shuffle runtime could provide information such
 *     as schema, which helped us to reduce header size a lot.
 *     </li>
 *     <li>In most cases we need to concat several small tables into a larger table during shuffle read time, since
 *     gpu's vectorized execution engine typically requires larger batch size, which makes write time concatenation
 *     meaningless. This relaxed the requirement of calculating exact validity buffer and offset buffer at write time,
 *     which makes write almost a memory copy process, without sacrificing read performance much.
 *     </li>
 * </ol>
 *
 * <h1>Format</h1>
 *
 * Similar to {@link JCudfSerialization}, it still consists of two parts: header and body.
 *
 * <h2>Header</h2>
 *
 * Header consists of following fields:
 *
 * <table>
 *     <tr>
 *         <th>Field Name</th>
 *         <th>Size</th>
 *         <th>Comments</th>
 *     </tr>
 *     <tr>
 *         <td>Magic Number</td>
 *         <td>4</td>
 *         <td>ASCII codes for "KUD0"</td>
 *     </tr>
 *     <tr>
 *         <td>Offset</td>
 *         <td>4</td>
 *         <td>Row offset in original table, in big endian format</td>
 *     </tr>
 *     <tr>
 *         <td>Number of rows</td>
 *         <td>4</td>
 *         <td>Number of rows, in big endian format</td>
 *     </tr>
 *     <tr>
 *         <td>Length of validity buffer</td>
 *         <td>4</td>
 *         <td>Length of validity buffer, in big endian format</td>
 *     </tr>
 *     <tr>
 *         <td>Length of offset buffer</td>
 *         <td>4</td>
 *         <td>Length of offset buffer, in big endian format</td>
 *     </tr>
 *     <tr>
 *         <td>Length of total body</td>
 *         <td>4</td>
 *         <td>Length of total body, in big endian format</td>
 *     </tr>
 *     <tr>
 *         <td>Number of columns</td>
 *         <td>4</td>
 *         <td>Number of columns in flattened schema, in big endian format. For details of <q>flattened schema</q>,
 *         see {@link com.nvidia.spark.rapids.jni.schema.SchemaVisitor}
 *         </td>
 *     </tr>
 *     <tr>
 *         <td>hasValidityBuffer</td>
 *         <td>(number of columns + 7) / 8</td>
 *         <td>A bit set to indicate whether a column has validity buffer. To test if column
 *         <code>col<sub>i<sub></code> has validity buffer, use the following code:
 *         <br/>
 *         <code>
 *           int pos = col<sub>i</sub> / 8; <br/>
 *           int bit = col<sub>i</sub> % 8; <br/>
 *           return (hasValidityBuffer[pos] & (1 << bit)) != 0;
 *         </code>
 *         </td>
 *     </tr>
 * </table>
 *
 * <h2>Body</h2>
 *
 * The body consists of three part:
 * <ol>
 *     <li>Validity buffers for every column with validity in depth-first ordering of schema columns. Each buffer of
 *     each column is 4 bytes padded.
 *     </li>
 *     <li>Offset buffers for every column with offsets in depth-first ordering of schema columns. Each buffer of each
 *     column is 4 bytes padded.</li>
 *     <li>Data buffers for every column with data in depth-first ordering of schema columns. Each buffer of each
 *     column is 4 bytes padded.</li>
 * </ol>
 *
 * <h1>Serialization</h1>
 *
 * The serialization process writes the header first, then writes the body. There are two optimizations when writing
 * validity buffer and offset buffer:
 *
 * <ol>
 *     <li>For validity buffer, it only copies buffers without calculating an exact validity buffer. For example, when
 *     we want to serialize rows [3, 9) of the original table, instead of calculating the exact validity buffer, it
 *     just copies first two bytes of the validity buffer. At read time, the deserializer will know that the true
 *     validity buffer starts from the fourth bit, since we have recorded the row offset in the header.
 *     </li>
 *     <li>For offset buffer, it only copies buffers without calculating an exact offset buffer. For example, when we want
 *  *  to serialize rows [3, 9) of the original table, instead of calculating the exact offset values by subtracting
 *  *  first value, it just copies the offset buffer values of rows [3, 9).
 *  *  </li>
 *  </ol>
 */
public class KudoSerializer {

  private static final byte[] PADDING = new byte[64];
  private static final BufferType[] ALL_BUFFER_TYPES = new BufferType[] {BufferType.VALIDITY, BufferType.OFFSET,
      BufferType.DATA};

  static {
    Arrays.fill(PADDING, (byte) 0);
  }

  private final int flattenedColumnCount;

  public KudoSerializer(Schema schema) {
    requireNonNull(schema, "schema is null");
    this.flattenedColumnCount = schema.getFlattenedColumnNames().length;
  }

  /**
   * Write partition of a table to a stream.
   * <br/>
   *
   * The caller should ensure that table's schema matches the schema used to create this serializer, otherwise behavior
   * is undefined.
   *
   * @param table table to write
   * @param out output stream
   * @param rowOffset row offset in original table
   * @param numRows number of rows to write
   * @return number of bytes written
   */
  public long writeToStream(Table table, OutputStream out, int rowOffset, int numRows) {
    HostColumnVector[] columns = null;
    try {
      columns = IntStream.range(0, table.getNumberOfColumns())
          .mapToObj(table::getColumn)
          .map(c -> c.copyToHostAsync(Cuda.DEFAULT_STREAM))
          .toArray(HostColumnVector[]::new);

      Cuda.DEFAULT_STREAM.sync();
      return writeToStream(columns, out, rowOffset, numRows);
    } finally {
      if (columns != null) {
        for (HostColumnVector column : columns) {
          column.close();
        }
      }
    }
  }

  /**
   * Write partition of an array of {@link HostColumnVector} to an output stream.
   * <br/>
   *
   * The caller should ensure that table's schema matches the schema used to create this serializer, otherwise behavior
   * is undefined.
   *
   * @param columns columns to write
   * @param out output stream
   * @param rowOffset row offset in original column vector.
   * @param numRows number of rows to write
   * @return number of bytes written
   */
  public long writeToStream(HostColumnVector[] columns, OutputStream out, int rowOffset, int numRows) {
    ensure(numRows > 0, () -> "numRows must be > 0, but was " + numRows);
    ensure(columns.length > 0, () -> "columns must not be empty, for row count only records " +
            "please call writeRowCountToStream");

    try {
      return writeSliced(columns, writerFrom(out), rowOffset, numRows);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Write a row count only record to an output stream.
   * @param out output stream
   * @param numRows number of rows to write
   * @return number of bytes written
   */
  public static long writeRowCountToStream(OutputStream out, int numRows) {
    if (numRows <= 0) {
      throw new IllegalArgumentException("Number of rows must be > 0, but was " + numRows);
    }
    try {
      DataWriter writer = writerFrom(out);
      KudoTableHeader header = new KudoTableHeader(0, numRows, 0, 0, 0
          , 0, new byte[0]);
      header.writeTo(writer);
      writer.flush();
      return header.getSerializedSize();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private long writeSliced(HostColumnVector[] columns, DataWriter out, int rowOffset, int numRows) throws Exception {
    KudoTableHeaderCalc headerCalc = new KudoTableHeaderCalc(rowOffset, numRows, flattenedColumnCount);
    Visitors.visitColumns(columns, headerCalc);
    KudoTableHeader header = headerCalc.getHeader();
    header.writeTo(out);

    long bytesWritten = 0;
    for (BufferType bufferType : ALL_BUFFER_TYPES) {
      SlicedBufferSerializer serializer = new SlicedBufferSerializer(rowOffset, numRows, bufferType, out);
      Visitors.visitColumns(columns, serializer);
      bytesWritten += serializer.getTotalDataLen();
    }

    if (bytesWritten != header.getTotalDataLen()) {
      throw new IllegalStateException("Header total data length: " + header.getTotalDataLen() +
          " does not match actual written data length: " + bytesWritten +
          ", rowOffset: " + rowOffset + " numRows: " + numRows);
    }

    out.flush();

    return header.getSerializedSize() + bytesWritten;
  }

  private static DataWriter writerFrom(OutputStream out) {
    if (!(out instanceof DataOutputStream)) {
      out = new DataOutputStream(new BufferedOutputStream(out));
    }
    return new DataOutputStreamWriter((DataOutputStream) out);
  }


  static long padForHostAlignment(long orig) {
    return ((orig + 3) / 4) * 4;
  }

  static long padForHostAlignment(DataWriter out, long bytes) throws IOException {
    final long paddedBytes = padForHostAlignment(bytes);
    if (paddedBytes > bytes) {
      out.write(PADDING, 0, (int) (paddedBytes - bytes));
    }
    return paddedBytes;
  }

  static long padFor64byteAlignment(long orig) {
    return ((orig + 63) / 64) * 64;
  }

  static int safeLongToNonNegativeInt(long value) {
    ensure(value >= 0, () -> "Expected non negative value, but was " + value);
    ensure(value <= Integer.MAX_VALUE, () -> "Value is too large to fit in an int");
    return (int) value;
  }
}