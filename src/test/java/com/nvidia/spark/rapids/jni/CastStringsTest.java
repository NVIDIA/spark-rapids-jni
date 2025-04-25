/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.jni;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;

import org.junit.jupiter.api.Test;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;

public class CastStringsTest {
  @Test
  void castToIntegerTest() {
    Table.TestBuilder tb = new Table.TestBuilder();
    tb.column(3l, 9l, 4l, 2l, 20l, null, null, 1l);
    tb.column(5, 1, 0, 2, 7, null, null, 1);
    tb.column(new Byte[]{2, 3, 4, 5, 9, null, null, 1});
    try (Table expected = tb.build()) {
      Table.TestBuilder tb2 = new Table.TestBuilder();
      tb2.column(" 3", "9", "4", "2", "20.5", null, "7.6asd", "\u0000 \u001f1\u0014");
      tb2.column("5", "1  ", "0", "2", "7.1", null, "asdf", "\u0000 \u001f1\u0014");
      tb2.column("2", "3", " 4 ", "5", " 9.2 ", null, "7.8.3", "\u0000 \u001f1\u0014");

      List<ColumnVector> result = new ArrayList<>();
      try (Table origTable = tb2.build()) {
        for (int i = 0; i < origTable.getNumberOfColumns(); i++) {
          ColumnVector string_col = origTable.getColumn(i);
          result.add(CastStrings.toInteger(string_col, false,
              expected.getColumn(i).getType()));
        }
        try (Table result_tbl = new Table(
            result.toArray(new ColumnVector[result.size()]))) {
          AssertUtils.assertTablesAreEqual(expected, result_tbl);
        }
      } finally {
        result.forEach(ColumnVector::close);
      }
    }
  }

  @Test
  void castToIntegerNoStripTest() {
    Table.TestBuilder tb = new Table.TestBuilder();
    tb.column(null, 9l, 4l, 2l, 20l, null, null);
    tb.column(5, null, 0, 2, 7, null, null);
    tb.column(new Byte[]{2, 3, null, 5, null, null, null});
    try (Table expected = tb.build()) {
      Table.TestBuilder tb2 = new Table.TestBuilder();
      tb2.column(" 3", "9", "4", "2", "20.5", null, "7.6asd");
      tb2.column("5", "1 ", "0", "2", "7.1", null, "asdf");
      tb2.column("2", "3", " 4 ", "5.6", " 9.2 ", null, "7.8.3");

      List<ColumnVector> result = new ArrayList<>();
      try (Table origTable = tb2.build()) {
        for (int i = 0; i < origTable.getNumberOfColumns(); i++) {
          ColumnVector string_col = origTable.getColumn(i);
          result.add(CastStrings.toInteger(string_col, false, false,
              expected.getColumn(i).getType()));
        }
        try (Table result_tbl = new Table(
            result.toArray(new ColumnVector[result.size()]))) {
          AssertUtils.assertTablesAreEqual(expected, result_tbl);
        }
      } finally {
        result.forEach(ColumnVector::close);
      }
    }
  }

  @Test
  void castToIntegerAnsiTest() {
    Table.TestBuilder tb = new Table.TestBuilder();
    tb.column(3l, 9l, 4l, 2l, 20l);
    tb.column(5, 1, 0, 2, 7);
    tb.column(new Byte[]{2, 3, 4, 5, 9});
    try (Table expected = tb.build()) {
      Table.TestBuilder tb2 = new Table.TestBuilder();
      tb2.column("3", "9", "4", "2", "20");
      tb2.column("5", "1", "0", "2", "7");
      tb2.column("2", "3", "4", "5", "9");

      List<ColumnVector> result = new ArrayList<>();
      try (Table origTable = tb2.build()) {
        for (int i = 0; i < origTable.getNumberOfColumns(); i++) {
          ColumnVector string_col = origTable.getColumn(i);
          result.add(CastStrings.toInteger(string_col, true,
              expected.getColumn(i).getType()));
        }
        try (Table result_tbl = new Table(
            result.toArray(new ColumnVector[result.size()]))) {
          AssertUtils.assertTablesAreEqual(expected, result_tbl);
        }
      } finally {
        result.forEach(ColumnVector::close);
      }
      Table.TestBuilder fail = new Table.TestBuilder();
      fail.column("asdf", "9.0.2", "- 4e", "b2", "20-fe");

      try (Table failTable = fail.build();
           ColumnVector cv =
               CastStrings.toInteger(failTable.getColumn(0), true,
                   expected.getColumn(0).getType());) {
        fail("Should have thrown");
      } catch (CastException e) {
        assertEquals("asdf", e.getStringWithError());
        assertEquals(0, e.getRowWithError());
      }
    }
  }

  @Test
  void castToFloatsTrimTest() {
    Table.TestBuilder tb = new Table.TestBuilder();
    tb.column(1.1f, 1.2f, 1.3f, 1.4f, 1.5f, null, null);
    tb.column(1.1d, 1.2d, 1.3d, 1.4d, 1.5d, null, null);
    try (Table expected = tb.build()) {
      Table.TestBuilder tb2 = new Table.TestBuilder();
      tb2.column("1.1\u0000", "1.2\u0014", "1.3\u001f", 
          "\u0000\u00001.4\u0000", "1.5\u0000\u0020\u0000", "1.6\u009f", "1.7\u0021");
      tb2.column("1.1\u0000", "1.2\u0014", "1.3\u001f", 
          "\u0000\u00001.4\u0000", "1.5\u0000\u0020\u0000", "1.6\u009f", "1.7\u0021");

      List<ColumnVector> result = new ArrayList<>();
      try (Table origTable = tb2.build()) {
        for (int i = 0; i < origTable.getNumberOfColumns(); i++) {
          ColumnVector string_col = origTable.getColumn(i);
          result.add(CastStrings.toFloat(string_col, false, 
              expected.getColumn(i).getType()));
        }
        try (Table result_tbl = new Table(
            result.toArray(new ColumnVector[result.size()]))) {
          AssertUtils.assertTablesAreEqual(expected, result_tbl);
        }
      } finally {
        result.forEach(ColumnVector::close);
      }
    }
  }

  @Test
  void castToFloatNanTest(){
    Table.TestBuilder tb2 = new Table.TestBuilder();
    tb2.column("nan", "nan ", " nan ", "NAN", "nAn ", " NAn ", "Nan 0", "+naN", "-nAn");

    Table.TestBuilder tb = new Table.TestBuilder();
    tb.column(Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN, null, null, null);

    try (Table expected = tb.build()) {
      List<ColumnVector> result = new ArrayList<>();
      try (Table origTable = tb2.build()) {
        for (int i = 0; i < origTable.getNumberOfColumns(); i++) {
          ColumnVector string_col = origTable.getColumn(i);
          result.add(CastStrings.toFloat(string_col, false, expected.getColumn(i).getType()));
        }
        try (Table result_tbl = new Table(result.toArray(new ColumnVector[result.size()]))) {
          AssertUtils.assertTablesAreEqual(expected, result_tbl);
        }
      } finally {
        result.forEach(ColumnVector::close);
      }
    }
  }

  @Test
  void castToFloatsInfTest(){
    // The test data: Table.TestBuilder object with a column containing the string "inf"
    Table.TestBuilder tb2 = new Table.TestBuilder();
    tb2.column("INFINITY ", "inf", "+inf ", " -INF  ", "INFINITY AND BEYOND", "INF");

    Table.TestBuilder tb = new Table.TestBuilder();
    tb.column(Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, null, Float.POSITIVE_INFINITY);

    try (Table expected = tb.build()) {
      List<ColumnVector> result = new ArrayList<>();
      try (Table origTable = tb2.build()) {
        for (int i = 0; i < origTable.getNumberOfColumns(); i++) {
          ColumnVector string_col = origTable.getColumn(i);
          result.add(CastStrings.toFloat(string_col, false, expected.getColumn(i).getType()));
        }
        System.out.println(result);
        try (Table result_tbl = new Table(result.toArray(new ColumnVector[result.size()]))) {
          AssertUtils.assertTablesAreEqual(expected, result_tbl);
        }
      } finally {
        result.forEach(ColumnVector::close);
      }
    }
  }

  @Test
  void castToDecimalTest() {
    Table.TestBuilder tb = new Table.TestBuilder();
    tb.decimal32Column(0,3, 9, 4, 2, 21, null, null, 1);
    tb.decimal64Column(0, 5l, 1l, 0l, 2l, 7l, null, null, 1l);
    tb.decimal32Column(-1, 20, 30, 40, 51, 92, null, null, 10);
    try (Table expected = tb.build()) {
      int[] desiredPrecision = new int[]{2, 10, 3};
      int[] desiredScale = new int[]{0, 0, -1};

      Table.TestBuilder tb2 = new Table.TestBuilder();
      tb2.column(" 3", "9", "4", "2", "20.5", null, "7.6asd", "\u0000 \u001f1\u0014");
      tb2.column("5", "1 ", "0", "2", "7.1", null, "asdf", "\u0000 \u001f1\u0014");
      tb2.column("2", "3", " 4 ", "5.07", "9.23", null, "7.8.3", "\u0000 \u001f1\u0014");

      List<ColumnVector> result = new ArrayList<>();
      try (Table origTable = tb2.build()) {
        for (int i = 0; i < origTable.getNumberOfColumns(); i++) {
          ColumnVector string_col = origTable.getColumn(i);
          result.add(CastStrings.toDecimal(string_col, false,
              desiredPrecision[i], desiredScale[i]));
        }
        try (Table result_tbl = new Table(
            result.toArray(new ColumnVector[result.size()]))) {
          AssertUtils.assertTablesAreEqual(expected, result_tbl);
        }
      } finally {
        result.forEach(ColumnVector::close);
      }
    }
  }

  @Test
  void castToDecimalNoStripTest() {
    Table.TestBuilder tb = new Table.TestBuilder();
    tb.decimal32Column(0, null, 9, 4, 2, 21, null, null);
    tb.decimal64Column(0, 5l, null, 0l, 2l, 7l, null, null);
    tb.decimal32Column(-1, 20, 30, null, 51, 92, null, null);
    try (Table expected = tb.build()) {
      int[] desiredPrecision = new int[]{2, 10, 3};
      int[] desiredScale = new int[]{0, 0, -1};

      Table.TestBuilder tb2 = new Table.TestBuilder();
      tb2.column(" 3", "9", "4", "2", "20.5", null, "7.6asd");
      tb2.column("5", "1 ", "0", "2", "7.1", null, "asdf");
      tb2.column("2", "3", " 4 ", "5.07", "9.23", null, "7.8.3");

      List<ColumnVector> result = new ArrayList<>();
      try (Table origTable = tb2.build()) {
        for (int i = 0; i < origTable.getNumberOfColumns(); i++) {
          ColumnVector string_col = origTable.getColumn(i);
          result.add(CastStrings.toDecimal(string_col, false, false,
              desiredPrecision[i], desiredScale[i]));
        }
        try (Table result_tbl = new Table(
            result.toArray(new ColumnVector[result.size()]))) {
          AssertUtils.assertTablesAreEqual(expected, result_tbl);
        }
      } finally {
        result.forEach(ColumnVector::close);
      }
    }
  }

  @Test
  void castFromLongToBinaryStringTest() {
    try (ColumnVector v0 = ColumnVector.fromBoxedLongs(null, 0L, 1L, 10L, -1L, Long.MAX_VALUE, Long.MIN_VALUE);
         ColumnVector result = CastStrings.fromLongToBinary(v0);
         ColumnVector expected = ColumnVector.fromStrings(null, "0", "1", "1010",
         "1111111111111111111111111111111111111111111111111111111111111111",
         "111111111111111111111111111111111111111111111111111111111111111",
         "1000000000000000000000000000000000000000000000000000000000000000")) {
          AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }

  private void convTestInternal(Table input, Table expected, int fromBase) {
    try(
      ColumnVector intCol = CastStrings.toIntegersWithBase(input.getColumn(0), fromBase, false,
        DType.UINT64);
      ColumnVector decStrCol = CastStrings.fromIntegersWithBase(intCol, 10);
      ColumnVector hexStrCol = CastStrings.fromIntegersWithBase(intCol, 16);
    ) {
      AssertUtils.assertColumnsAreEqual(expected.getColumn(0), decStrCol, "decStrCol");
      AssertUtils.assertColumnsAreEqual(expected.getColumn(1), hexStrCol, "hexStrCol");
    }
  }

  @Test
  void  baseDec2HexTestNoNulls() {
    try (
      Table input = new Table.TestBuilder().column(
        "510",
        "00510",
        "00-510"
      ).build();

      Table expected = new Table.TestBuilder().column(
        "510",
        "510",
        "0"
      ).column(
        "1FE",
        "1FE",
        "0"
      ).build()
    )
    {
      convTestInternal(input, expected, 10);
    }
  }

  @Test
  void  baseDec2HexTestMixed() {
    try (
      Table input = new Table.TestBuilder().column(
        null,
        " ",
        "junk-510junk510",
        "--510",
        "   -510junk510",
        "  510junk510",
        "510",
        "00510",
        "00-510"
      ).build();

      Table expected = new Table.TestBuilder().column(
        null,
        null,
        "0",
        "0",
        "18446744073709551106",
        "510",
        "510",
        "510",
        "0"
      ).column(
        null,
        null,
        "0",
        "0",
        "FFFFFFFFFFFFFE02",
        "1FE",
        "1FE",
        "1FE",
        "0"
      ).build()
    )
    {
      convTestInternal(input, expected, 10);
    }
  }

  @Test
  void baseHex2DecTest() {
    try(
      Table input = new Table.TestBuilder().column(
        null,
        "junk",
        "0",
        "f",
        "junk-5Ajunk5A",
        "--5A",
        "   -5Ajunk5A",
        "  5Ajunk5A",
        "5a",
        "05a",
        "005a",
        "00-5a",
        "NzGGImWNRh"
      ).build();

      Table expected = new Table.TestBuilder().column(
        null,
        "0",
        "0",
        "15",
        "0",
        "0",
        "18446744073709551526",
        "90",
        "90",
        "90",
        "90",
        "0",
        "0"
      ).column(
        null,
        "0",
        "0",
        "F",
        "0",
        "0",
        "FFFFFFFFFFFFFFA6",
        "5A",
        "5A",
        "5A",
        "5A",
        "0",
        "0"
      ).build();
    )
    {
      convTestInternal(input, expected, 16);
    }
  }

  /**
   * Mock timezone info for testing.
   */
  private ColumnVector getTimezoneInfoMock() {
    HostColumnVector.DataType type = new HostColumnVector.StructType(false,
        new HostColumnVector.BasicType(false, DType.STRING),
        new HostColumnVector.BasicType(false, DType.INT32),
        new HostColumnVector.BasicType(false, DType.BOOL8));
    ArrayList<HostColumnVector.StructData> data = new ArrayList<>();
    // Only test 3 timezones: CTT, JST, PST
    data.add(new HostColumnVector.StructData("CTT", 0, false));
    data.add(new HostColumnVector.StructData("JST", 1, false));
    data.add(new HostColumnVector.StructData("PST", 2, true));
    return HostColumnVector.fromStructs(type, data).copyToDevice();
  }

  @Test
  void toTimestampIntermediateResultTest() {
    List<List<Object>> list = new ArrayList<>();
    // Row is: input, return type, UTC ts, just time, tz type, offset, is DST, tz
    // index
    // Intermediate result:
    // - Return type: 0 success, 1 invalid, 2 unsupported
    // - UTC timestamp
    // - Just time: 0 no, 1 yes
    // - TZ type: 0 not specified 1 fixed, 2 other, 3 invalid
    // - TZ offset: record offset in seconds when tz type is fixed
    // - TZ is DST: 0 no, 1 yes
    // - TZ index: 0 CTT, 1 JST, 2 PST
    list.add(Arrays.asList("2023-11-05T03:04:55 +00:00", 0, 1699153495000000L, 0, 1, 0, 0, -1)); // row 0
    list.add(Arrays.asList("2023-11-05 03:04:55 +01:02", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +1:02", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 -01:2", 0, 1699153495000000L, 0, 1, -(3600 * 1 + 60 * 2), 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +1:2", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +10:59", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 +10:59:03", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +105903", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +1059", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +10", 0, 1699153495000000L, 0, 1, 3600 * 10, 0, -1));

    list.add(Arrays.asList("2023-11-05T03:04:55 UT+00:00", 0, 1699153495000000L, 0, 1, 0, 0, -1)); // rwo 10
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+01:02", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+1:02", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+01:2", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+1:2", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+10:59", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT-10:59:03", 0, 1699153495000000L, 0, 1, -(3600 * 10 + 60 * 59 + 3), 0,
        -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 UT+105903", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+1059", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+10", 0, 1699153495000000L, 0, 1, 3600 * 10, 0, -1));

    list.add(Arrays.asList("2023-11-05T03:04:55 UTC+00:00", 0, 1699153495000000L, 0, 1, 0, 0, -1)); // row 20
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+01:02", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+1:02", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+01:2", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+1:2", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+10:59", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 UTC+10:59:03", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 UTC+105903", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+1059", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC-10", 0, 1699153495000000L, 0, 1, -(3600 * 10), 0, -1));

    list.add(Arrays.asList("2023-11-05T03:04:55 GMT+00:00", 0, 1699153495000000L, 0, 1, 0, 0, -1)); // row 30
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+01:02", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+1:02", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT-01:2", 0, 1699153495000000L, 0, 1, -(3600 * 1 + 60 * 2), 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+1:2", 0, 1699153495000000L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+10:59", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 GMT+10:59:03", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 GMT+105903", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+1059", 0, 1699153495000000L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+10", 0, 1699153495000000L, 0, 1, 3600 * 10, 0, -1));

    list.add(Arrays.asList("2023-11-05T03:04:55.123456789 PST", 0, 1699153495123456L, 0, 2, 0, 1, 2)); // row 40
    list.add(Arrays.asList("2023-11-05 03:04:55.123456 PST", 0, 1699153495123456L, 0, 2, 0, 1, 2));
    list.add(Arrays.asList("2023-11-05T03:04:55 CTT", 0, 1699153495000000L, 0, 2, 0, 0, 0));
    list.add(Arrays.asList("2023-11-05 03:04:55 JST", 0, 1699153495000000L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-11-05 03:04:55 PST", 0, 1699153495000000L, 0, 2, 0, 1, 2));

    // use default timezone: index is 1
    list.add(Arrays.asList("2023-11-05 03:04:55", 0, 1699153495000000L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-11-05", 0, 1699142400000000L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-11", 0, 1698796800000000L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023", 0, 1672531200000000L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("T23:17:50.201567 CTT", 0, 83870201567L, 1, 2, 0, 0, 0));
    list.add(Arrays.asList("T23:17:50.201567 JST", 0, 83870201567L, 1, 2, 0, 0, 1)); // row 50
    list.add(Arrays.asList("T23:17:50.201567 PST", 0, 83870201567L, 1, 2, 0, 1, 2));
    list.add(Arrays.asList("T23:17:50 CTT", 0, 83870000000L, 1, 2, 0, 0, 0));
    list.add(Arrays.asList("T23:17:50 JST", 0, 83870000000L, 1, 2, 0, 0, 1));
    list.add(Arrays.asList("T23:17:50 PST", 0, 83870000000L, 1, 2, 0, 1, 2));
    list.add(Arrays.asList("T23:17:50", 0, 83870000000L, 1, 2, 0, 0, 1));
    list.add(Arrays.asList("T23:17:50", 0, 83870000000L, 1, 2, 0, 0, 1));
    list.add(Arrays.asList("T23:17:50", 0, 83870000000L, 1, 2, 0, 0, 1));

    list.add(Arrays.asList("12345", 0, 327403382400000000L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-1-1", 0, 1672531200000000L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-1-01", 0, 1672531200000000L, 0, 2, 0, 0, 1)); // row 60
    list.add(Arrays.asList("2023-01-1", 0, 1672531200000000L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-01-01", 0, 1672531200000000L, 0, 2, 0, 0, 1));

    // test leap year
    list.add(Arrays.asList("2028-02-29", 0, 1835395200000000L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-01-01 00:00:00Z", 0, 1672531200000000L, 0, 1, 0, 0, -1));
    list.add(Arrays.asList("2023-01-01 00:00:00 Z", 0, 1672531200000000L, 0, 1, 0, 0, -1));
    // special case GMT0
    list.add(Arrays.asList("2023-01-01 00:00:00 GMT0", 0, 1672531200000000L, 0, 1, 0, 0, -1));
    // test trim
    list.add(Arrays.asList(" \r\n\tT23:17:50 \r\n\t", 0, 83870000000L, 1, 2, 0, 0, 1));

    list.add(Arrays.asList("T00", 0, 0L, 1, 2, 0, 0, 1));
    list.add(Arrays.asList("T1:2", 0, 3720000000L, 1, 2, 0, 0, 1));
    list.add(Arrays.asList("T01:2", 0, 3720000000L, 1, 2, 0, 0, 1)); // row 70
    list.add(Arrays.asList("T1:02", 0, 3720000000L, 1, 2, 0, 0, 1));
    list.add(Arrays.asList("T01:02", 0, 3720000000L, 1, 2, 0, 0, 1));
    list.add(Arrays.asList("T01:02:03", 0, 3723000000L, 1, 2, 0, 0, 1));
    list.add(Arrays.asList("T1:2:3", 0, 3723000000L, 1, 2, 0, 0, 1));
    list.add(Arrays.asList("T01:02:03 UTC", 0, 3723000000L, 1, 1, 0, 0, -1));

    // ################################################
    // ############ Below is invalid cases ############
    // ################################################
    // empty
    list.add(Arrays.asList("", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("  ", 1, 0L, 0, 0, 0, 0, -1));

    // test non leap year and day is 29
    list.add(Arrays.asList(" -2025-2-29 ", 1, 0L, 0, 0, 0, 0, -1));
    // non-existence tz, result is invalid, but keeps tz type is other
    list.add(Arrays.asList(" 2023-11-05 03:04:55 non-existence-tz ", 1, 1699153495000000L, 0, 2, 0, 0, -1));

    // invalid month
    list.add(Arrays.asList("-2025-13-1", 1, 0L, 0, 0, 0, 0, -1)); // row 80
    list.add(Arrays.asList("-2025-99-1", 1, 0L, 0, 0, 0, 0, -1));

    // invalid day
    list.add(Arrays.asList("-2025-01-32", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("-2025-01-99", 1, 0L, 0, 0, 0, 0, -1));

    // invalid hour
    list.add(Arrays.asList("2000-01-01 24:00:00", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 99:00:00", 1, 0L, 0, 0, 0, 0, -1));

    // invalid minute
    list.add(Arrays.asList("2000-01-01 00:60:00", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:61:00", 1, 0L, 0, 0, 0, 0, -1));

    // invalid second
    list.add(Arrays.asList("2000-01-01 00:00:60", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:61", 1, 0L, 0, 0, 0, 0, -1));

    // invalid date
    list.add(Arrays.asList("x2025", 1, 0L, 0, 0, 0, 0, -1)); // row 90
    list.add(Arrays.asList("12", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("123", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("1234567", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-123", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-12x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-01-", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-01-x", 1, 0L, 0, 0, 0, 0, -1)); // row number 100
    list.add(Arrays.asList("2200-01-11x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-01-113", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-03-25T", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-03-25 x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-03-25Tx", 1, 0L, 0, 0, 0, 0, -1));

    // invalid time
    list.add(Arrays.asList("Tx", 1, 0L, 1, 0, 0, 0, -1));
    list.add(Arrays.asList("T00x", 1, 0L, 1, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:", 1, 0L, 1, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:x", 1, 0L, 1, 0, 0, 0, -1));
    list.add(Arrays.asList("T000:00", 1, 0L, 1, 0, 0, 0, -1)); // row 110
    list.add(Arrays.asList("T00:022", 1, 0L, 1, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:02:", 1, 0L, 1, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:02:x", 1, 0L, 1, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:02:003", 1, 0L, 1, 0, 0, 0, -1));
    list.add(Arrays.asList("T123", 1, 0L, 1, 0, 0, 0, -1));
    list.add(Arrays.asList("T12345", 1, 0L, 1, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:02:003", 1, 0L, 1, 0, 0, 0, -1));

    // invalid TZs
    list.add(Arrays.asList("2000-01-01 00:00:00 +", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 -X", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +07:", 1, 0L, 0, 3, 0, 0, -1)); // row 120
    list.add(Arrays.asList("2000-01-01 00:00:00 +09:x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +15:07x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +15:07:", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +15:07:x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +15:07:1", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +15:07:12x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +01x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +0102x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +010203x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +111", 1, 0L, 0, 3, 0, 0, -1)); // row 130
    list.add(Arrays.asList("2000-01-01 00:00:00 +11111", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +1x", 1, 0L, 0, 3, 0, 0, -1));
    // exceeds the max value 18 hours
    list.add(Arrays.asList("2000-01-01 00:00:00 +180001", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 -180001", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 -08:1:08", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 -8:1:08", 1, 0L, 0, 3, 0, 0, -1));
    // U is invalid tz
    list.add(Arrays.asList("2000-01-01 00:00:00 U", 1, 0L, 0, 3, 0, 0, -1));
    // Result is invalid, although ts is not zero
    list.add(Arrays.asList("2023-11-05 03:04:55 Ux", 1, 1699153495000000L, 0, 2, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTx", 1, 1699153495000000L, 0, 2, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTCx", 1, 1699153495000000L, 0, 2, 0, 0, -1)); // row 140
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT-08:1:08", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT-8:1:08", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 Gx", 1, 1699153495000000L, 0, 2, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMTx", 1, 1699153495000000L, 0, 2, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT-08:1:08", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT-8:1:08", 1, 0L, 0, 3, 0, 0, -1));

    // invalid year: e.g. abs(year) > 30,000
    list.add(Arrays.asList("300001-01-01", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("-300001-01-01", 1, 0L, 0, 0, 0, 0, -1)); // row 150

    List<String> input = new ArrayList<>(list.size());
    List<Byte> expected_return_type = new ArrayList<>(list.size());
    List<Long> expected_utc_ts = new ArrayList<>(list.size());
    List<Byte> expected_just_time = new ArrayList<>(list.size());
    List<Byte> expected_tz_type = new ArrayList<>(list.size());
    List<Integer> expected_tz_offset = new ArrayList<>(list.size());
    List<Byte> expected_tz_is_dst = new ArrayList<>(list.size());
    List<Integer> expected_tz_index = new ArrayList<>(list.size());

    for (List<Object> row : list) {
      input.add(row.get(0).toString());
      expected_return_type.add(Byte.parseByte(row.get(1).toString()));
      expected_utc_ts.add(Long.parseLong(row.get(2).toString()));
      expected_just_time.add(Byte.parseByte(row.get(3).toString()));
      expected_tz_type.add(Byte.parseByte(row.get(4).toString()));
      expected_tz_offset.add(Integer.parseInt(row.get(5).toString()));
      expected_tz_is_dst.add(Byte.parseByte(row.get(6).toString()));
      expected_tz_index.add(Integer.parseInt(row.get(7).toString()));
    }

    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector tzInfo = getTimezoneInfoMock(); // mock timezone info
        ColumnVector result = CastStrings.parseTimestampStrings(inputCv, 1, tzInfo);
        ColumnVector expectedReturnType = ColumnVector
            .fromBoxedUnsignedBytes(expected_return_type.toArray(new Byte[0]));
        ColumnVector expectedUtcTs = ColumnVector.fromBoxedLongs(expected_utc_ts.toArray(new Long[0]));
        ColumnVector expectedJustTime = ColumnVector.fromBoxedUnsignedBytes(expected_just_time.toArray(new Byte[0]));
        ColumnVector expectedTzType = ColumnVector.fromBoxedUnsignedBytes(expected_tz_type.toArray(new Byte[0]));
        ColumnVector expectedTzOffset = ColumnVector.fromBoxedInts(expected_tz_offset.toArray(new Integer[0]));
        ColumnVector expectedTzIsDst = ColumnVector.fromBoxedUnsignedBytes(expected_tz_is_dst.toArray(new Byte[0]));
        ColumnVector expectedTzIndex = ColumnVector.fromBoxedInts(expected_tz_index.toArray(new Integer[0]))) {
      AssertUtils.assertColumnsAreEqual(expectedReturnType, result.getChildColumnView(0));
      AssertUtils.assertColumnsAreEqual(expectedUtcTs, result.getChildColumnView(1));
      AssertUtils.assertColumnsAreEqual(expectedJustTime, result.getChildColumnView(2));
      AssertUtils.assertColumnsAreEqual(expectedTzType, result.getChildColumnView(3));
      AssertUtils.assertColumnsAreEqual(expectedTzOffset, result.getChildColumnView(4));
      AssertUtils.assertColumnsAreEqual(expectedTzIsDst, result.getChildColumnView(5));
      AssertUtils.assertColumnsAreEqual(expectedTzIndex, result.getChildColumnView(6));
    }
  }
}
