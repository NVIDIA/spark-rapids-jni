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

import static java.lang.Math.toIntExact;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Schema;
import ai.rapids.cudf.Table;
import com.nvidia.spark.rapids.jni.Arms;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

public class KudoSerializerTest {
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

  private static Table buildSimpleTable() {
    HostColumnVector.StructType st = new HostColumnVector.StructType(
        true,
        new HostColumnVector.BasicType(true, DType.INT8),
        new HostColumnVector.BasicType(true, DType.INT64)
    );
    return new Table.TestBuilder()
        .column(1, 2, 3, 4)
        .column("1", "12", null, "45")
        .column(new Integer[] {1, null, 3}, new Integer[] {4, 5, 6}, null, new Integer[] {7, 8, 9})
        .column(st, new HostColumnVector.StructData((byte) 1, 11L),
            new HostColumnVector.StructData((byte) 2, null), null,
            new HostColumnVector.StructData((byte) 3, 33L))
        .build();
  }

  private static Table buildTestTable() {
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
        .column(100, 202, 3003, 40004, 5, -60, 1, null, 3, null, 5, null, 7, null, 9, null, 11,
            null, 13, null, 15)
        .column(true, true, false, false, true, null, true, true, null, false, false, null, true,
            true, null, false, false, null, true, true, null)
        .column((byte) 1, (byte) 2, null, (byte) 4, (byte) 5, (byte) 6, (byte) 1, (byte) 2,
            (byte) 3, null, (byte) 5, (byte) 6,
            (byte) 7, null, (byte) 9, (byte) 10, (byte) 11, null, (byte) 13, (byte) 14, (byte) 15)
        .column((short) 6, (short) 5, (short) 4, null, (short) 2, (short) 1,
            (short) 1, (short) 2, (short) 3, null, (short) 5, (short) 6, (short) 7, null, (short) 9,
            (short) 10, null, (short) 12, (short) 13, (short) 14, null)
        .column(1L, null, 1001L, 50L, -2000L, null, 1L, 2L, 3L, 4L, null, 6L,
            7L, 8L, 9L, null, 11L, 12L, 13L, 14L, null)
        .column(10.1f, 20f, -1f, 3.1415f, -60f, null, 1f, 2f, 3f, 4f, 5f, null, 7f, 8f, 9f, 10f,
            11f, null, 13f, 14f, 15f)
        .column(10.1f, 20f, -2f, 3.1415f, -60f, -50f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f,
            12f, 13f, 14f, 15f)
        .column(10.1, 20.0, 33.1, 3.1415, -60.5, null, 1d, 2.0, 3.0, 4.0, 5.0,
            6.0, null, 8.0, 9.0, 10.0, 11.0, 12.0, null, 14.0, 15.0)
        .column((Float) null, null, null, null, null, null, null, null, null, null,
            null, null, null, null, null, null, null, null, null, null, null)
        .timestampDayColumn(99, 100, 101, 102, 103, 104, 1, 2, 3, 4, 5, 6, 7, null, 9, 10, 11, 12,
            13, null, 15)
        .timestampMillisecondsColumn(9L, 1006L, 101L, 5092L, null, 88L, 1L, 2L, 3L, 4L, 5L, 6L, 7L,
            8L, null, 10L, 11L, 12L, 13L, 14L, 15L)
        .timestampSecondsColumn(1L, null, 3L, 4L, 5L, 6L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L, null,
            11L, 12L, 13L, 14L, 15L)
        .decimal32Column(-3, 100, 202, 3003, 40004, 5, -60, 1, null, 3,
            null, 5, null, 7, null, 9, null, 11, null, 13, null, 15)
        .decimal64Column(-8, 1L, null, 1001L, 50L, -2000L, null, 1L, 2L, 3L,
            4L, null, 6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, null)
        .column("A", "B", "C", "D", null, "TESTING", "1", "2", "3", "4",
            "5", "6", "7", null, "9", "10", "11", "12", "13", null, "15")
        .column("A", "A", "C", "C", "E", "TESTING", "1", "2", "3", "4", "5",
            "6", "7", "", "9", "10", "11", "12", "13", "", "15")
        .column("", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            "", "", "", "", "", "")
        .column("", null, "", "", null, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            "")
        .column((String) null, null, null, null, null, null, null, null, null, null,
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
            integers(10, 11, 12), integers((Integer) null), integers(14, null),
            integers(14, 15, null, 16, 17, 18), integers(19, 20, 21), integers(22, 23, 24),
            integers(25, 26, 27), integers(28, 29, 30), integers(31, 32, 33), null,
            integers(37, 38, 39))
        .column(integers(), integers(), integers(), integers(), integers(), integers(), integers(),
            integers(), integers(), integers(), integers(), integers(), integers(), integers(),
            integers(), integers(), integers(), integers(), integers(), integers(), integers())
        .column(integers(null, null), integers(null, null, null, null), integers(),
            integers(null, null, null), integers(), integers(null, null, null, null, null),
            integers((Integer) null), integers(null, null, null), integers(null, null),
            integers(null, null, null, null), integers(null, null, null, null, null), integers(),
            integers(null, null, null, null), integers(null, null, null), integers(null, null),
            integers(null, null, null), integers(null, null), integers((Integer) null),
            integers((Integer) null), integers(null, null),
            integers(null, null, null, null, null))
        .column((Integer) null, null, null, null, null, null, null, null, null, null,
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
            strings((String) null), strings(null, null, null), strings(null, null),
            strings(null, null, null, null), strings(null, null, null, null, null), strings(),
            strings(null, null, null, null), strings(null, null, null), strings(null, null),
            strings(null, null, null), strings(null, null), strings((String) null),
            strings((String) null), strings(null, null),
            strings(null, null, null, null, null))
        .column((String) null, null, null, null, null, null, null, null, null, null,
            null, null, null, null, null, null, null, null, null, null, null)
        .column(listMapType, asList(asList(struct("k1", "v1"), struct("k2", "v2")),
                singletonList(struct("k3", "v3"))),
            asList(asList(struct("k4", "v4"), struct("k5", "v5"),
                struct("k6", "v6")), singletonList(struct("k7", "v7"))),
            null, null, null, asList(asList(struct("k8", "v8"), struct("k9", "v9")),
                asList(struct("k10", "v10"), struct("k11", "v11"), struct("k12", "v12"),
                    struct("k13", "v13"))),
            singletonList(asList(struct("k14", "v14"), struct("k15", "v15"))), null, null, null,
            null,
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

      ByteArrayOutputStream bout = new ByteArrayOutputStream();
      for (TableSlice slice : tableSlices) {
        serializer.writeToStream(slice.getBaseTable(), bout, slice.getStartRow(),
            slice.getNumRows());
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

          try (Table merged = serializer.mergeToTable(kudoTables).getLeft()) {
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

  private static HostColumnVector.StructData struct(Object... values) {
    return new HostColumnVector.StructData(values);
  }

  private static List<HostColumnVector.StructData> structs(HostColumnVector.StructData... values) {
    return asList(values);
  }

  private static String[] strings(String... values) {
    return values;
  }

  private static Schema schemaOf(Table t) {
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

  @Test
  public void testSerializeAndDeserializeTable() {
    try (Table expected = buildTestTable()) {
      int rowCount = toIntExact(expected.getRowCount());
      for (int sliceSize = 1; sliceSize <= rowCount; sliceSize++) {
        List<TableSlice> tableSlices = new ArrayList<>();
        for (int startRow = 0; startRow < rowCount; startRow += sliceSize) {
          tableSlices.add(
              new TableSlice(startRow, Math.min(sliceSize, rowCount - startRow), expected));
        }

        checkMergeTable(expected, tableSlices);
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void testRowCountOnly() throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
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
  public void testWriteSimple() throws Exception {
    KudoSerializer serializer = new KudoSerializer(buildSimpleTestSchema());

    try (Table t = buildSimpleTable()) {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      long bytesWritten = serializer.writeToStream(t, out, 0, 4);
      assertEquals(189, bytesWritten);

      ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());

      KudoTableHeader header = KudoTableHeader.readFrom(new DataInputStream(in)).get();
      assertEquals(7, header.getNumColumns());
      assertEquals(0, header.getOffset());
      assertEquals(4, header.getNumRows());
      assertEquals(24, header.getValidityBufferLen());
      assertEquals(40, header.getOffsetBufferLen());
      assertEquals(160, header.getTotalDataLen());

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
              integers(307), integers(-397, -633), integers(-314, 307), integers(-633),
              integers(-397),
              integers(181, -919, -175))
          .build();
      tables.add(table2);

      Table expected = new Table.TestBuilder()
          .column(896L, -129L, -108L, -428L, 0L, 617L, 782L, 482L, 660L, 896L, 122L, 241L,
              281L, 680L, 783L, null)
          .column(integers(-314), integers(-370), integers(181), integers(665, 544), integers(222),
              integers(-587), integers(544), integers(398), integers(-587, 398), integers(-314),
              integers(307), integers(-397, -633), integers(-314, 307), integers(-633),
              integers(-397),
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

      checkMergeTable(table2, singletonList(new TableSlice(509, 3, table1)));
      return null;
    });
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
}
