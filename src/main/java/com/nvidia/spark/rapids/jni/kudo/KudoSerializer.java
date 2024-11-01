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

/**
 * This class is used to serialize/deserialize a table using the Kudo format.
 *
 * <h1>Background</h1>
 *
 * The Kudo format is a binary format that is optimized for serializing/deserializing a table during spark shuffle. The
 * optimizations are based on two key observations:
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
 * Similar to {@link JCudfSerialization}, it still consists of two pars: header and body.
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
 *     </tr>
 *     <tr>
 *         <td>Offset</td>
 *         <td>4</td>
 *         <td>Offset in original table</td>
 *     </tr>
 *     <tr>
 *         <td>Number of rows</td>
 *         <td>4</td>
 *     </tr>
 *     <tr>
 *         <td>Length of validity buffer</td>
 *         <td>4</td>
 *     </tr>
 *     <tr>
 *         <td>Length of offset buffer</td>
 *         <td>4</td>
 *     </tr>
 *     <tr>
 *         <td>Length of total body</td>
 *         <td>4</td>
 *     </tr>
 *     <tr>
 *         <td>Number of columns</td>
 *         <td>4</td>
 *     </tr>
 *     <tr>
 *         <td>Length of hasValidityBuffer</td>
 *         <td>4</td>
 *         <td>Length of hasValidityBuffer bitset</td>
 *     </tr>
 *     <tr>
 *         <td>hasValidityBuffer</td>
 *         <td>(number of columns + 1 + 7) / 8</td>
 *         <td>A bit set to indicate whether a column has validity buffer.</td>
 *     </tr>
 * </table>
 *
 * <h2>Body</h2>
 *
 * The body consists of three part:
 * <ol>
 *     <li>Validity buffer of all columns if it has</li>
 *     <li>Offset buffer of all columns if it has</li>
 *     <li>Data buffer of all columns</li>
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
 *     just copies first two bytes of the validity buffer.
 *     </li>
 *     <li>For offset buffer, it only copies buffers without calculating an exact offset buffer. For example, when we want
 *  *  to serialize rows [3, 9) of the original table, instead of calculating the exact offset values by subtracting
 *  *  first value, it just copies the offset buffer values of rows [3, 9).
 *  *  </li>
 *  </ol>
 */
public class KudoSerializer {

  private static final byte[] PADDING = new byte[64];

  static {
    Arrays.fill(PADDING, (byte) 0);
  }

  public String version() {
    return "MultiTableSerializer-v7";
  }

  public long writeToStream(Table table, OutputStream out, long rowOffset, long numRows) {

    HostColumnVector[] columns = null;
    try {
      columns = IntStream.range(0, table.getNumberOfColumns())
          .mapToObj(table::getColumn)
          .map(ColumnView::copyToHost)
          .toArray(HostColumnVector[]::new);
      return writeToStream(columns, out, rowOffset, numRows);
    } finally {
      if (columns != null) {
        for (HostColumnVector column : columns) {
          column.close();
        }
      }
    }
  }

  public long writeToStream(HostColumnVector[] columns, OutputStream out, long rowOffset, long numRows) {
    if (numRows < 0) {
      throw new IllegalArgumentException("numRows must be >= 0");
    }

    if (numRows == 0 || columns.length == 0) {
      return 0;
    }

    try {
      return writeSliced(columns, writerFrom(out), rowOffset, numRows);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public long writeRowsToStream(OutputStream out, long numRows) {
    if (numRows <= 0) {
      throw new IllegalArgumentException("Number of rows must be > 0, but was " + numRows);
    }
    try {
      DataWriter writer = writerFrom(out);
      SerializedTableHeader header = new SerializedTableHeader(0, safeLongToInt(numRows), 0, 0, 0
          , 0, new byte[0]);
      header.writeTo(writer);
      writer.flush();
      return header.getSerializedSize();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private static long writeSliced(HostColumnVector[] columns, DataWriter out, long rowOffset, long numRows) throws Exception {
    SerializedTableHeaderCalc headerCalc = new SerializedTableHeaderCalc(rowOffset, numRows, columns.length);
    Visitors.visitColumns(columns, headerCalc);
    SerializedTableHeader header = headerCalc.getHeader();
    header.writeTo(out);

    long bytesWritten = 0;
    for (BufferType bufferType : Arrays.asList(BufferType.VALIDITY, BufferType.OFFSET, BufferType.DATA)) {
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

  private static DataInputStream readerFrom(InputStream in) {
    if (!(in instanceof DataInputStream)) {
      in = new DataInputStream(in);
    }
    return new DataInputStream(in);
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

  static long padFor64byteAlignment(DataWriter out, long bytes) throws IOException {
    final long paddedBytes = padFor64byteAlignment(bytes);
    if (paddedBytes > bytes) {
      out.write(PADDING, 0, (int) (paddedBytes - bytes));
    }
    return paddedBytes;
  }

  static int safeLongToInt(long value) {
//        if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE) {
//            throw new ArithmeticException("Overflow: long value is too large to fit in an int");
//        }
    return (int) value;
  }

}