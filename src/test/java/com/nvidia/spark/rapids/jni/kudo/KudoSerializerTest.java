/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import com.nvidia.spark.rapids.jni.Arms;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.function.Supplier;

import static java.lang.Math.toIntExact;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;
import static org.junit.jupiter.api.Assertions.*;

public class KudoSerializerTest {
  @Test
  public void testSerializeAndDeserializeTable() {
    try(Table expected = buildTestTable()) {
      int rowCount = toIntExact(expected.getRowCount());
      for (int sliceSize = rowCount; sliceSize >= 1; sliceSize--) {
        List<TableSlice> tableSlices = new ArrayList<>();
        for (int startRow = 0; startRow < rowCount; startRow += sliceSize) {
          tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount - startRow), expected));
        }

        checkMergeTable(expected, tableSlices);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void testRowCountOnly() throws Exception {
    OpenByteArrayOutputStream out = new OpenByteArrayOutputStream();
    long bytesWritten = KudoSerializer.writeRowCountToStream(out, 5);
    assertEquals(28, bytesWritten);

    ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());
    KudoTableHeader header = KudoTableHeader.readFrom(new DataInputStream(in)).get();

    assertEquals(0, header.getNumColumns());
    assertEquals(0, header.getOffset());
    assertEquals(5, header.getNumRows());
    assertEquals(0, header.getValidityBufferLen());
    assertEquals(0, header.getOffsetBufferLen());
    assertEquals(0, header.getTotalDataLen());
  }

  @Test
  public void testSerializeAndDeserializeEmptyStructTable() {
    try(Table expected = buildEmptyStructTable()) {
      int rowCount = toIntExact(expected.getRowCount());
      for (int sliceSize = rowCount; sliceSize >= 1; sliceSize--) {
        List<TableSlice> tableSlices = new ArrayList<>();
        for (int startRow = 0; startRow < rowCount; startRow += sliceSize) {
          tableSlices.add(new TableSlice(startRow, Math.min(sliceSize, rowCount - startRow), expected));
        }

        checkMergeTable(expected, tableSlices);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void testWriteSimple() throws Exception {
    KudoSerializer serializer = new KudoSerializer(buildSimpleTestSchema());

    try (Table t = buildSimpleTable()) {
      OpenByteArrayOutputStream out = new OpenByteArrayOutputStream();
      long bytesWritten = serializer.writeToStreamWithMetrics(t, out, 0, 4).getWrittenBytes();

      assertEquals(172, bytesWritten);

      ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());

      KudoTableHeader header = KudoTableHeader.readFrom(new DataInputStream(in)).get();
      assertEquals(7, header.getNumColumns());
      assertEquals(0, header.getOffset());
      assertEquals(4, header.getNumRows());
      assertEquals(7, header.getValidityBufferLen());
      assertEquals(40, header.getOffsetBufferLen());
      assertEquals(143, header.getTotalDataLen());

      // First integer column has no validity buffer
      assertFalse(header.hasValidityBuffer(0));
      for (int i = 1; i < 7; i++) {
        assertTrue(header.hasValidityBuffer(i));
      }
    }
  }

  @Test
  public void testMergeTableWithDifferentValidity() {
    Arms.withResource(new ArrayList<Table>(), tables -> {
      Table table1 = new Table.TestBuilder()
          .column(-83182L, 5822L, 3389L, 7384L, 7297L)
          .column(-2.06, -2.14, 8.04, 1.16, -1.0)
          .build();
      tables.add(table1);

      Table table2 = new Table.TestBuilder()
          .column(-47L, null, -83L, -166L, -220L, 470L, 619L, 803L, 661L)
          .column(-6.08, 1.6, 1.78, -8.01, 1.22, 1.43, 2.13, -1.65, null)
          .build();
      tables.add(table2);

      Table table3 = new Table.TestBuilder()
          .column(8722L, 8733L)
          .column(2.51, 0.0)
          .build();
      tables.add(table3);


      Table expected = new Table.TestBuilder()
          .column(7384L, 7297L, 803L, 661L, 8733L)
          .column(1.16, -1.0, -1.65, null, 0.0)
          .build();
      tables.add(expected);

      checkMergeTable(expected, asList(
          new TableSlice(3, 2, table1),
          new TableSlice(7, 2, table2),
          new TableSlice(1, 1, table3)));
      return null;
    });
  }

  @Test
  public void testMergeString() {
      Arms.withResource(new ArrayList<Table>(), tables -> {
                  Table table1 = new Table.TestBuilder()
                          .column("A", "B", "C", "D", null, "TESTING", "1", "2", "3", "4",
                                  "5", "6", "7", null, "9", "10", "11", "12", "13", null, "15")
                          .build();
                  tables.add(table1);

                  Table table2 = new Table.TestBuilder()
                          .column("A", "A", "C", "C", "E", "TESTING", "1", "2", "3", "4", "5",
                                  "6", "7", "", "9", "10", "11", "12", "13", "", "15")
                          .build();
                  tables.add(table2);

                  Table expected = new Table.TestBuilder()
                          .column("C", "D", null, "TESTING", "1", "2", "3", "4",
                                  "5", "6", "7", null, "9", "C", "E", "TESTING", "1", "2")
                          .build();
                  tables.add(expected);

                  checkMergeTable(expected, asList(
                          new TableSlice(2, 13, table1),
                          new TableSlice(3, 5, table2)));

                  return null;
              }
      );
  }

  @Test
  public void testMergeList() {
    Arms.withResource(new ArrayList<Table>(), tables -> {
      Table table1 = new Table.TestBuilder()
          .column(-881L, 482L, 660L, 896L, -129L, -108L, -428L, 0L, 617L, 782L)
          .column(integers(665), integers(-267), integers(398), integers(-314),
              integers(-370), integers(181), integers(665, 544), integers(222), integers(-587),
              integers(544))
          .build();
      tables.add(table1);

      Table table2 = new Table.TestBuilder()
          .column(-881L, 482L, 660L, 896L, 122L, 241L, 281L, 680L, 783L, null)
          .column(integers(-370), integers(398), integers(-587, 398), integers(-314),
              integers(307), integers(-397, -633), integers(-314, 307), integers(-633), integers(-397),
              integers(181, -919, -175))
          .build();
      tables.add(table2);

      Table expected = new Table.TestBuilder()
          .column(896L, -129L, -108L, -428L, 0L, 617L, 782L, 482L, 660L, 896L, 122L, 241L,
              281L, 680L, 783L, null)
          .column(integers(-314), integers(-370), integers(181), integers(665, 544), integers(222),
              integers(-587), integers(544), integers(398), integers(-587, 398), integers(-314),
              integers(307), integers(-397, -633), integers(-314, 307), integers(-633), integers(-397),
              integers(181, -919, -175))
          .build();
      tables.add(expected);

      checkMergeTable(expected, asList(
          new TableSlice(3, 7, table1),
          new TableSlice(1, 9, table2)));

      return null;
    });
  }

  @Test
  public void testMergeComplexStructList() {
    Arms.withResource(new ArrayList<Table>(), tables -> {
      HostColumnVector.ListType listMapType = new HostColumnVector.ListType(true,
              new HostColumnVector.ListType(true,
                      new HostColumnVector.StructType(true,
                              new HostColumnVector.BasicType(false, DType.STRING),
                              new HostColumnVector.BasicType(true, DType.STRING))));

      Table table = new Table.TestBuilder()
              .column(listMapType, asList(asList(struct("k1", "v1"), struct("k2", "v2")),
                              singletonList(struct("k3", "v3"))),
                      null,
                      singletonList(asList(struct("k14", "v14"), struct("k15", "v15"))),
                      null,
                      asList(null, null, null),
                      asList(singletonList(struct("k22", null)), singletonList(struct("k23", null))),
                      null, null,
                      null)
              .build();
      tables.add(table);

      checkMergeTable(table, asList(
              new TableSlice(0, 3, table),
              new TableSlice(3, 3, table),
              new TableSlice(6, 3, table))
      );
      return null;
    });
  }


  @Test
  public void testSerializeValidity() {
    Arms.withResource(new ArrayList<Table>(), tables -> {
      List<Integer> col1 = new ArrayList<>(512);
      col1.add(null);
      col1.add(null);
      col1.addAll(IntStream.range(2, 512).boxed().collect(Collectors.toList()));

      Table table1 = new Table.TestBuilder()
          .column(col1.toArray(new Integer[0]))
          .build();
      tables.add(table1);

      Table table2 = new Table.TestBuilder()
          .column(509, 510, 511)
          .build();
      tables.add(table2);

      checkMergeTable(table2, asList(new TableSlice(509, 3, table1)));
      return null;
    });
  }

  @Test
  public void testByteArrayOutputStreamWriter() throws Exception {
    ByteArrayOutputStream bout = new ByteArrayOutputStream(32);
    DataWriter writer = new ByteArrayOutputStreamWriter(bout);

    writer.writeInt(0x12345678);

    byte[] testByteArr1 = new byte[2097];
    ThreadLocalRandom.current().nextBytes(testByteArr1);
    writer.write(testByteArr1, 0, testByteArr1.length);

    byte[] testByteArr2 = new byte[7896];
    ThreadLocalRandom.current().nextBytes(testByteArr2);
    try(HostMemoryBuffer buffer = HostMemoryBuffer.allocate(testByteArr2.length)) {
      buffer.setBytes(0, testByteArr2, 0, testByteArr2.length);
      writer.copyDataFrom(buffer, 0, testByteArr2.length);
    }

    byte[] expected = new byte[4 + testByteArr1.length + testByteArr2.length];
    expected[0] = 0x12;
    expected[1] = 0x34;
    expected[2] = 0x56;
    expected[3] = 0x78;
    System.arraycopy(testByteArr1, 0, expected, 4, testByteArr1.length);
    System.arraycopy(testByteArr2, 0, expected, 4 + testByteArr1.length,
        testByteArr2.length);

    assertArrayEquals(expected, bout.toByteArray());
  }


  private static Schema buildSimpleTestSchema() {
    Schema.Builder builder = Schema.builder();

    builder.addColumn(DType.INT32, "a");
    builder.addColumn(DType.STRING, "b");
    Schema.Builder listBuilder = builder.addColumn(DType.LIST, "c");
    listBuilder.addColumn(DType.INT32, "c1");

    Schema.Builder structBuilder = builder.addColumn(DType.STRUCT, "d");
    structBuilder.addColumn(DType.INT8, "d1");
    structBuilder.addColumn(DType.INT64, "d2");

    return builder.build();
  }

  static Table buildSimpleTable() {
    HostColumnVector.StructType st = new HostColumnVector.StructType(
        true,
        new HostColumnVector.BasicType(true, DType.INT8),
        new HostColumnVector.BasicType(true, DType.INT64)
    );
    return new Table.TestBuilder()
        .column(1, 2, 3, 4)
        .column("1", "12", null, "45")
        .column(new Integer[]{1, null, 3}, new Integer[]{4, 5, 6}, null, new Integer[]{7, 8, 9})
        .column(st, new HostColumnVector.StructData((byte) 1, 11L),
            new HostColumnVector.StructData((byte) 2, null), null,
            new HostColumnVector.StructData((byte) 3, 33L))
        .build();
  }

  static Table buildEmptyStructTable() {
    HostColumnVector.StructType st = new HostColumnVector.StructType(true);
    return new Table.TestBuilder()
        .column(st,
            struct(), null, null, struct(), null, null, struct(), struct(),
            null, struct(), struct(), null, struct(), struct(), null, null,
            struct(), null, null, struct(), null, null, struct(), struct(),
            null, struct(), struct(), null, struct(), struct(), null, null,
            struct(), struct(), null, struct(), null, null, struct(), struct(),
            null, struct(), struct(), null, struct(), null, null, null,
            struct(), null, null, struct(), null, struct(), struct(), null,
            null, struct(), struct(), null, struct(), struct(), null, null,
            struct(), null, null, struct(), null, null, struct(), struct(),
            null, struct(), struct(), null, struct(), struct(), null, null,
            struct(), null, null, null, null, null, struct(), struct(),
            null, struct(), struct(), null, struct(), struct(), null, null,
            struct())
        .build();
  }

  static Table buildTestTable() {
    HostColumnVector.ListType listMapType = new HostColumnVector.ListType(true,
        new HostColumnVector.ListType(true,
            new HostColumnVector.StructType(true,
                new HostColumnVector.BasicType(false, DType.STRING),
                new HostColumnVector.BasicType(true, DType.STRING))));
    HostColumnVector.ListType mapStructType = new HostColumnVector.ListType(true,
        new HostColumnVector.StructType(true,
            new HostColumnVector.BasicType(false, DType.STRING),
            new HostColumnVector.BasicType(false, DType.STRING)));
    HostColumnVector.StructType structType = new HostColumnVector.StructType(true,
        new HostColumnVector.BasicType(true, DType.INT32),
        new HostColumnVector.BasicType(false, DType.FLOAT32));
    HostColumnVector.ListType listDateType = new HostColumnVector.ListType(true,
        new HostColumnVector.StructType(false,
            new HostColumnVector.BasicType(false, DType.INT32),
            new HostColumnVector.BasicType(true, DType.INT32)));

    return new Table.TestBuilder()
        .column(100, 202, 3003, 40004, 5, -60, 1, null, 3, null, 5, null, 7, null, 9, null, 11, null, 13, null, 15)
        .column(true, true, false, false, true, null, true, true, null, false, false, null, true,
            true, null, false, false, null, true, true, null)
        .column((byte)1, (byte)2, null, (byte)4, (byte)5,(byte)6,(byte)1,(byte)2,(byte)3, null,(byte)5, (byte)6,
            (byte) 7, null,(byte) 9,(byte) 10,(byte) 11, null,(byte) 13,(byte) 14,(byte) 15)
        .column((short)6, (short)5, (short)4, null, (short)2, (short)1,
            (short)1, (short)2, (short)3, null, (short)5, (short)6, (short)7, null, (short)9,
            (short)10, null, (short)12, (short)13, (short)14, null)
        .column(1L, null, 1001L, 50L, -2000L, null, 1L, 2L, 3L, 4L, null, 6L,
            7L, 8L, 9L, null, 11L, 12L, 13L, 14L, null)
        .column(10.1f, 20f, -1f, 3.1415f, -60f, null, 1f, 2f, 3f, 4f, 5f, null, 7f, 8f, 9f, 10f, 11f, null, 13f, 14f, 15f)
        .column(10.1f, 20f, -2f, 3.1415f, -60f, -50f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f)
        .column(10.1, 20.0, 33.1, 3.1415, -60.5, null, 1d, 2.0, 3.0, 4.0, 5.0,
            6.0, null, 8.0, 9.0, 10.0, 11.0, 12.0, null, 14.0, 15.0)
        .column((Float)null, null, null, null, null, null, null, null, null, null,
            null, null, null, null, null, null, null, null, null, null, null)
        .timestampDayColumn(99, 100, 101, 102, 103, 104, 1, 2, 3, 4, 5, 6, 7, null, 9, 10, 11, 12, 13, null, 15)
        .timestampMillisecondsColumn(9L, 1006L, 101L, 5092L, null, 88L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, null, 10L, 11L, 12L, 13L, 14L, 15L)
        .timestampSecondsColumn(1L, null, 3L, 4L, 5L, 6L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, 15L)
        .decimal32Column(-3, 100, 202, 3003, 40004, 5, -60, 1, null, 3,
            null, 5, null, 7, null, 9, null, 11, null, 13, null, 15)
        .decimal64Column(-8, 1L, null, 1001L, 50L, -2000L, null, 1L, 2L, 3L,
            4L, null, 6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, null)
        .decimal128Column(-2, RoundingMode.UNNECESSARY, new BigInteger("1"), null, new BigInteger("1001"),
            new BigInteger("50"), new BigInteger("-2000"), null, new BigInteger("1"), new BigInteger("2"),
            new BigInteger("3"), new BigInteger("4"), null, new BigInteger("6"), new BigInteger("7"),
            new BigInteger("8"), new BigInteger("9"), null, new BigInteger("11"), new BigInteger("12"),
            new BigInteger("13"), new BigInteger("14"), null)
        .column("A", "B", "C", "D", null, "TESTING", "1", "2", "3", "4",
            "5", "6", "7", null, "9", "10", "11", "12", "13", null, "15")
        .column("A", "A", "C", "C", "E", "TESTING", "1", "2", "3", "4", "5",
            "6", "7", "", "9", "10", "11", "12", "13", "", "15")
        .column("", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            "", "", "", "", "", "")
        .column("", null, "", "", null, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "")
        .column((String)null, null, null, null, null, null, null, null, null, null,
            null, null, null, null, null, null, null, null, null, null, null)
        .column(mapStructType, structs(struct("1", "2")), structs(struct("3", "4")), null, null,
            structs(struct("key", "value"), struct("a", "b")), null, null,
            structs(struct("3", "4"), struct("1", "2")), structs(),
            structs(null, struct("foo", "bar")),
            structs(null, null, null), null, null, null, null, null, null, null, null, null,
            structs(struct("the", "end")))
        .column(structType, struct(1, 1f), null, struct(2, 3f),
            null, struct(8, 7f), struct(0, 0f), null,
            null, struct(-1, -1f), struct(-100, -100f),
            struct(Integer.MAX_VALUE, Float.MAX_VALUE), null,
            null, null,
            null, null,
            null, null,
            null, null,
            struct(Integer.MIN_VALUE, Float.MIN_VALUE))
        .column(integers(1, 2), null, integers(3, 4, null, 5, null), null, null, integers(6, 7, 8),
            integers(null, null, null), integers(1, 2, 3), integers(4, 5, 6), integers(7, 8, 9),
            integers(10, 11, 12), integers((Integer)null), integers(14, null),
            integers(14, 15, null, 16, 17, 18), integers(19, 20, 21), integers(22, 23, 24),
            integers(25, 26, 27), integers(28, 29, 30), integers(31, 32, 33), null,
            integers(37, 38, 39))
        .column(integers(), integers(), integers(), integers(), integers(), integers(), integers(),
            integers(), integers(), integers(), integers(), integers(), integers(), integers(),
            integers(), integers(), integers(), integers(), integers(), integers(), integers())
        .column(integers(null, null), integers(null, null, null, null), integers(),
            integers(null, null, null), integers(), integers(null, null, null, null, null),
            integers((Integer)null), integers(null, null, null), integers(null, null),
            integers(null, null, null, null), integers(null, null, null, null, null), integers(),
            integers(null, null, null, null), integers(null, null, null), integers(null, null),
            integers(null, null, null), integers(null, null), integers((Integer)null),
            integers((Integer)null), integers(null, null),
            integers(null, null, null, null, null))
        .column((Integer)null, null, null, null, null, null, null, null, null, null,
            null, null, null, null, null, null, null, null, null, null, null)
        .column(strings("1", "2", "3"), strings("4"), strings("5"), strings("6, 7"),
            strings("", "9", null), strings("11"), strings(""), strings(null, null),
            strings("15", null), null, null, strings("18", "19", "20"), null, strings("22"),
            strings("23", ""), null, null, null, null, strings(), strings("the end"))
        .column(strings(), strings(), strings(), strings(), strings(), strings(), strings(),
            strings(), strings(), strings(), strings(), strings(), strings(), strings(), strings(),
            strings(), strings(), strings(), strings(), strings(), strings())
        .column(strings(null, null), strings(null, null, null, null), strings(),
            strings(null, null, null), strings(), strings(null, null, null, null, null),
            strings((String)null), strings(null, null, null), strings(null, null),
            strings(null, null, null, null), strings(null, null, null, null, null), strings(),
            strings(null, null, null, null), strings(null, null, null), strings(null, null),
            strings(null, null, null), strings(null, null), strings((String)null),
            strings((String)null), strings(null, null),
            strings(null, null, null, null, null))
        .column((String)null, null, null, null, null, null, null, null, null, null,
            null, null, null, null, null, null, null, null, null, null, null)
        .column(listMapType, asList(asList(struct("k1", "v1"), struct("k2", "v2")),
                singletonList(struct("k3", "v3"))),
            asList(asList(struct("k4", "v4"), struct("k5", "v5"),
                struct("k6", "v6")), singletonList(struct("k7", "v7"))),
            null, null, null, asList(asList(struct("k8", "v8"), struct("k9", "v9")),
                asList(struct("k10", "v10"), struct("k11", "v11"), struct("k12", "v12"),
                    struct("k13", "v13"))),
            singletonList(asList(struct("k14", "v14"), struct("k15", "v15"))), null, null, null, null,
            asList(asList(struct("k16", "v16"), struct("k17", "v17")),
                singletonList(struct("k18", "v18"))),
            asList(asList(struct("k19", "v19"), struct("k20", "v20")),
                singletonList(struct("k21", "v21"))),
            asList(singletonList(struct("k22", "v22")), singletonList(struct("k23", "v23"))),
            asList(null, null, null),
            asList(singletonList(struct("k22", null)), singletonList(struct("k23", null))),
            null, null, null, null, null)
        .column(listDateType, asList(struct(-210, 293), struct(-719, 205), struct(-509, 183),
                struct(174, 122), struct(647, 683)),
            asList(struct(311, 992), struct(-169, 482), struct(166, 525)),
            asList(struct(156, 197), struct(926, 134), struct(747, 312), struct(293, 801)),
            asList(struct(647, null), struct(293, 387)), emptyList(),
            null, emptyList(), null,
            asList(struct(-210, 293), struct(-719, 205), struct(-509, 183), struct(174, 122),
                struct(647, 683)),
            asList(struct(311, 992), struct(-169, 482), struct(166, 525)),
            asList(struct(156, 197), struct(926, 134), struct(747, 312), struct(293, 801)),
            asList(struct(647, null), struct(293, 387)), emptyList(), null,
            emptyList(), null,
            singletonList(struct(778, 765)), asList(struct(7, 87), struct(8, 96)),
            asList(struct(9, 56), struct(10, 532), struct(11, 456)), null, emptyList())
        .build();
  }

  private static void checkMergeTable(Table expected, List<TableSlice> tableSlices) {
    try {
      KudoSerializer serializer = new KudoSerializer(schemaOf(expected));

      OpenByteArrayOutputStream bout = new OpenByteArrayOutputStream();
      for (TableSlice slice : tableSlices) {
        serializer.writeToStreamWithMetrics(slice.getBaseTable(), bout, slice.getStartRow(), slice.getNumRows());
      }
      bout.flush();

      ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
      Arms.withResource(new ArrayList<KudoTable>(tableSlices.size()), kudoTables -> {
        try {
          for (int i = 0; i < tableSlices.size(); i++) {
            kudoTables.add(KudoTable.from(bin).get());
          }

          long rows = kudoTables.stream().mapToLong(t -> t.getHeader().getNumRows()).sum();
          assertEquals(expected.getRowCount(), toIntExact(rows));

          try (Table merged = serializer.mergeToTable(kudoTables.toArray(new KudoTable[0]))) {
            assertEquals(expected.getRowCount(), merged.getRowCount());

            AssertUtils.assertTablesAreEqual(expected, merged);
          }
        } catch (Exception e) {
          throw new RuntimeException(e);
        }

        return null;
      });
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private static Integer[] integers(Integer... values) {
    return values;
  }

  public static HostColumnVector.StructData struct(Object... values) {
    return new HostColumnVector.StructData(values);
  }

  public static List<HostColumnVector.StructData> structs(HostColumnVector.StructData... values) {
    return asList(values);
  }

  public static String[] strings(String... values) {
    return values;
  }

  public static Schema schemaOf(Table t) {
    Schema.Builder builder = Schema.builder();

    for (int i = 0; i < t.getNumberOfColumns(); i++) {
      ColumnVector cv = t.getColumn(i);
      addToSchema(cv, "col_" + i + "_", builder);
    }

    return builder.build();
  }

  private static void addToSchema(ColumnView cv, String namePrefix, Schema.Builder builder) {
    toSchemaInner(cv, 0, namePrefix, builder);
  }

  private static int toSchemaInner(ColumnView cv, int idx, String namePrefix,
                                   Schema.Builder builder) {
    String name = namePrefix + idx;

    Schema.Builder thisBuilder = builder.addColumn(cv.getType(), name);
    int lastIdx = idx;
    for (int i = 0; i < cv.getNumChildren(); i++) {
      lastIdx = toSchemaInner(cv.getChildColumnView(i), lastIdx + 1, namePrefix,
          thisBuilder);
    }

    return lastIdx;
  }

  private static class TableSlice {
    private final int startRow;
    private final int numRows;
    private final Table baseTable;

    private TableSlice(int startRow, int numRows, Table baseTable) {
      this.startRow = startRow;
      this.numRows = numRows;
      this.baseTable = baseTable;
    }

    public int getStartRow() {
      return startRow;
    }

    public int getNumRows() {
      return numRows;
    }

    public Table getBaseTable() {
      return baseTable;
    }
  }

  @Test
  public void testMergeWithDumpPath() {
    File tempFile = null;
    try {
      //Create a temporary file for dumping
      tempFile = File.createTempFile("kudo_dump_test", ".bin");
      tempFile.deleteOnExit();

      String dumpPath = tempFile.getAbsolutePath();
      
      Table table1 = new Table.TestBuilder()
          .column(1, 2, 3, 4)
          .column("a", "b", "c", "d")
          .build();
      
      Table table2 = new Table.TestBuilder()
          .column(5, 6, 7, 8)
          .column("e", "f", "g", "h")
          .build();
      
      // Create KudoSerializer with table1's schema
      KudoSerializer serializer = new KudoSerializer(schemaOf(table1));
      
      // Serialize both tables
      ByteArrayOutputStream bout = new ByteArrayOutputStream();
      serializer.writeToStreamWithMetrics(table1, bout, 0, (int)table1.getRowCount());
      
      // Serialize table2 using same serializer - this will create an inconsistent state
      serializer.writeToStreamWithMetrics(table2, bout, 0, (int)table2.getRowCount());
      bout.flush();
      
      ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
      KudoTable[] kudoTables = new KudoTable[2];
      
      // Read the KudoTables from the stream
      kudoTables[0] = KudoTable.from(bin).get();
      kudoTables[1] = KudoTable.from(bin).get();
      
      // merge the two tables and dump the result to the temp file
      Supplier<OutputStream> outputStreamSupplier = () -> {
        try {
          return new FileOutputStream(dumpPath);
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      };
      MergeOptions options = new MergeOptions(DumpOption.Always, outputStreamSupplier, dumpPath);
      serializer.mergeOnHost(kudoTables, options);
      
      // Verify dump file exists and has content
      assertTrue(tempFile.exists(), "Dump file should exist");
      assertTrue(tempFile.length() > 0, "Dump file should not be empty");
      
      // Basic check that file contains schema info
      byte[] fileContent = java.nio.file.Files.readAllBytes(tempFile.toPath());
      String contentStart = new String(fileContent, 0, Math.min(100, fileContent.length));
      assertTrue(contentStart.contains("col_0_0") || contentStart.contains("Schema"), 
          "Dump file should contain schema information");
      
    } catch (Exception e) {
      fail("Test failed with exception: " + e.getMessage());
    } finally {
      // Cleanup
      if (tempFile != null && tempFile.exists()) {
        tempFile.delete();
      }
    }
  }
}
