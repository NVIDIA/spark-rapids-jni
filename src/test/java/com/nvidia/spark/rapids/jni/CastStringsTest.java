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
import org.junit.jupiter.api.Assertions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.time.Instant;
import java.time.ZoneId;
import java.time.LocalDate;
import java.time.LocalDateTime;

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

  private static final Version SPARK_320_VERSION =
      new Version(SparkPlatformType.VANILLA_SPARK, 3, 2, 0);
  private static final Version SPARK_330_VERSION =
      new Version(SparkPlatformType.VANILLA_SPARK, 3, 3, 0);

  @Test
  void castStringToTimestampFirstPhaseJustTimeTest() {
    GpuTimeZoneDB.cacheDatabase(2200);
    long defaultEpochDay = 1;
    long secondsOfEpochDay = defaultEpochDay * 24 * 60 * 60;
    List<List<Object>> list = new ArrayList<>();
    // Row is: input, return type, UTC seconds, UTC microseconds, tz type, offset, is DST, tz
    // index
    // Intermediate result:
    // - Return type: 0 success, 1 invalid, 2 unsupported
    // - UTC timestamp: seconds
    // - UTC timestamp: microseconds
    // - TZ type: 0 not specified 1 fixed, 2 other, 3 invalid
    // - TZ offset: record offset in seconds when tz type is fixed
    // - TZ is DST: 0 no, 1 yes
    // - TZ index: 0 CTT, 1 JST, 2 PST

    // valid time
    list.add(Arrays.asList("T00", 0, 0L + secondsOfEpochDay, 0, 2, 0, 0, 1));
    list.add(Arrays.asList(" T1:2", 0, 3720L + secondsOfEpochDay, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("T01:2", 0, 3720L + secondsOfEpochDay, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("T1:02", 0, 3720L + secondsOfEpochDay, 0, 2, 0, 0, 1));
    list.add(Arrays.asList(" T01:02", 0, 3720L + secondsOfEpochDay, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("T01:02:03", 0, 3723L + secondsOfEpochDay, 0, 2, 0, 0, 1));
    list.add(Arrays.asList(" T1:2:3", 0, 3723L + secondsOfEpochDay, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("T01:02:03", 0, 3723L + secondsOfEpochDay, 0, 2, 0, 0, 1));
    list.add(Arrays.asList(" \r\n\tT23:17:50 \r\n\t", 0, 83870L + secondsOfEpochDay, 0, 2, 0, 0, 1));

    // valid time: begin with not T
    list.add(Arrays.asList("01:2", 0, 3720L + secondsOfEpochDay, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("1:2", 0, 3720L + secondsOfEpochDay, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("01:2:3", 0, 3723L + secondsOfEpochDay, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("1:2:3", 0, 3723L + secondsOfEpochDay, 0, 2, 0, 0, 1));

    // invalid time
    list.add(Arrays.asList("Tx", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("T00x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("T000:00", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:022", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:02:", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:02:x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:02:003", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("T123", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("T12345", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("T00:02:003", 1, 0L, 0, 0, 0, 0, -1));

    // invalid time: time leading by sign
    list.add(Arrays.asList("+00:00:00", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("-00:00:00", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("+T00:00:00", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("-T00:00:00", 1, 0L, 0, 0, 0, 0, -1));

    List<String> input = new ArrayList<>(list.size());
    List<Byte> expected_return_type = new ArrayList<>(list.size());
    List<Long> expected_utc_seconds = new ArrayList<>(list.size());
    List<Integer> expected_utc_microseconds = new ArrayList<>(list.size());
    List<Byte> expected_tz_type = new ArrayList<>(list.size());
    List<Integer> expected_tz_offset = new ArrayList<>(list.size());
    List<Byte> expected_tz_is_dst = new ArrayList<>(list.size());
    List<Integer> expected_tz_index = new ArrayList<>(list.size());

    for (List<Object> row : list) {
      input.add(row.get(0).toString());
      expected_return_type.add(Byte.parseByte(row.get(1).toString()));
      expected_utc_seconds.add(Long.parseLong(row.get(2).toString()));
      expected_utc_microseconds.add(Integer.parseInt(row.get(3).toString()));
      expected_tz_type.add(Byte.parseByte(row.get(4).toString()));
      expected_tz_offset.add(Integer.parseInt(row.get(5).toString()));
      expected_tz_is_dst.add(Byte.parseByte(row.get(6).toString()));
      expected_tz_index.add(Integer.parseInt(row.get(7).toString()));
    }

    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector tzInfo = getTimezoneInfoMock(); // mock timezone info
        Table transitions = GpuTimeZoneDB.getTransitions();
        ColumnVector result = CastStrings.parseTimestampStrings(
            inputCv,
            1,
            /* is default tz DST */ false,
            defaultEpochDay,
            tzInfo,
            transitions,
            SPARK_330_VERSION);
        ColumnVector expectedReturnType = ColumnVector
            .fromBoxedUnsignedBytes(expected_return_type.toArray(new Byte[0]));
        ColumnVector expectedUtcTs = ColumnVector.fromBoxedLongs(expected_utc_seconds.toArray(new Long[0]));
        ColumnVector expectedUtcTsMicro = ColumnVector.fromBoxedInts(expected_utc_microseconds.toArray(new Integer[0]));
        ColumnVector expectedTzType = ColumnVector.fromBoxedUnsignedBytes(expected_tz_type.toArray(new Byte[0]));
        ColumnVector expectedTzOffset = ColumnVector.fromBoxedInts(expected_tz_offset.toArray(new Integer[0]));
        ColumnVector expectedTzIsDst = ColumnVector.fromBoxedUnsignedBytes(expected_tz_is_dst.toArray(new Byte[0]));
        ColumnVector expectedTzIndex = ColumnVector.fromBoxedInts(expected_tz_index.toArray(new Integer[0]))) {
      AssertUtils.assertColumnsAreEqual(expectedReturnType, result.getChildColumnView(0));
      AssertUtils.assertColumnsAreEqual(expectedUtcTs, result.getChildColumnView(1));
      AssertUtils.assertColumnsAreEqual(expectedUtcTsMicro, result.getChildColumnView(2));
      AssertUtils.assertColumnsAreEqual(expectedTzType, result.getChildColumnView(3));
      AssertUtils.assertColumnsAreEqual(expectedTzOffset, result.getChildColumnView(4));
      AssertUtils.assertColumnsAreEqual(expectedTzIsDst, result.getChildColumnView(5));
      AssertUtils.assertColumnsAreEqual(expectedTzIndex, result.getChildColumnView(6));
    }
  }

  /**
   * Test just time, specify fixed timezone.
   * Note: this test case is sensitive to the execution time,
   * because the current date at the execution time in the specified timezone may
   * be today,
   * tomorrow or yesterday compared to the UTC date.
   */
  @Test
  void castStringToTimestampJustTimeWithFixedTzTest() {
    GpuTimeZoneDB.cacheDatabase(2200);
    GpuTimeZoneDB.verifyDatabaseCached();

    ZoneId eightHours = ZoneId.of("+08:00");
    Instant now = Instant.now();
    long expectedSeconds1 = now.getEpochSecond() - now.getEpochSecond() % (24 * 60 * 60);
    long expectedTs1 = expectedSeconds1 * 1000000L;

    LocalDateTime ldt = now.atZone(eightHours).toLocalDateTime();
    // 5 minutes later, assume the execution time is within 5 minutes
    LocalDateTime fiveMinutesLater = now.plusSeconds(5 * 60).atZone(eightHours).toLocalDateTime();
    if (ldt.getDayOfMonth() != fiveMinutesLater.getDayOfMonth()) {
      // 5 minutes later is another day, skip this test case
      // the expected ts is from Java code, the actual ts is from Kernel code
      // both are based on the current date, so the current dates may be different
      return;
    }

    // calcute the expected UTC seconds
    LocalDateTime ldtAtMidNight = LocalDateTime.of(ldt.getYear(), ldt.getMonth(),
        ldt.getDayOfMonth(), 0, 0, 0);
    Instant ins = Instant.from(ldtAtMidNight.atZone(eightHours));
    long expectedTs2 = ins.getEpochSecond() * 1000000L + ins.getNano() / 1000L;

    List<List<Object>> list = new ArrayList<>();
    List<String> input = new ArrayList<>(list.size());
    List<Long> expectedTS = new ArrayList<>(list.size());

    list.add(Arrays.asList("T00:00:00", expectedTs1, true));
    list.add(Arrays.asList("T00:00:00 +08:00", expectedTs2, true));
    for (List<Object> row : list) {
      input.add(row.get(0).toString());
      if ((Boolean) row.get(2)) {
        expectedTS.add(Long.parseLong(row.get(1).toString()));
      } else {
        expectedTS.add(null);
      }
    }
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ false, SPARK_330_VERSION);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            expectedTS.toArray(new Long[0]))) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test just time, specify fixed timezone.
   * Note: this test case is sensitive to the execution time,
   * because the current date in the specified timezone may be today, tomorrow or
   * yesterday compared
   * to the UTC date.
   */
  @Test
  void castStringToTimestampJustTimeWithOtherTzTest() {
    GpuTimeZoneDB.cacheDatabase(2200);
    GpuTimeZoneDB.verifyDatabaseCached();

    ZoneId pstTZ = ZoneId.of("America/Los_Angeles");
    Instant now = Instant.now();
    long expectedSeconds1 = now.getEpochSecond() - now.getEpochSecond() % (24 * 60 * 60);
    long expectedTs1 = expectedSeconds1 * 1000000L;

    LocalDateTime ldt = now.atZone(pstTZ).toLocalDateTime();
    // 5 minutes later, assume the execution time is within 5 minutes
    LocalDateTime fiveMinutesLater = now.plusSeconds(5 * 60).atZone(pstTZ).toLocalDateTime();
    if (ldt.getDayOfMonth() != fiveMinutesLater.getDayOfMonth()) {
      // 5 minutes later is another day, skip this test case
      // the expected ts is from Java code, the actual ts is from Kernel code
      // both are based on the current date, so the current dates may be different
      return;
    }

    // calcute the expected UTC seconds
    LocalDateTime ldtAtMidNight = LocalDateTime.of(ldt.getYear(), ldt.getMonth(),
        ldt.getDayOfMonth(), 0, 0, 0);
    Instant ins = Instant.from(ldtAtMidNight.atZone(pstTZ));
    long expectedTs2 = ins.getEpochSecond() * 1000000L + ins.getNano() / 1000L;

    List<List<Object>> list = new ArrayList<>();
    List<String> input = new ArrayList<>(list.size());
    List<Long> expectedTS = new ArrayList<>(list.size());

    list.add(Arrays.asList("T00:00:00", expectedTs1, true));
    list.add(Arrays.asList("T00:00:00 America/Los_Angeles", expectedTs2, true));
    for (List<Object> row : list) {
      input.add(row.get(0).toString());
      if ((Boolean) row.get(2)) {
        expectedTS.add(Long.parseLong(row.get(1).toString()));
      } else {
        expectedTS.add(null);
      }
    }
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ false, SPARK_330_VERSION);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            expectedTS.toArray(new Long[0]))) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void castStringToTimestampFirstPhaseTest() {
    GpuTimeZoneDB.cacheDatabase(2200);
    List<List<Object>> list = new ArrayList<>();
    // make a dummy epoch day, this test case does not test just time.
    long defaultEpochDay = 1;

    // Row is: input, return type, UTC seconds, UTC microseconds, tz type, offset, is DST, tz index
    // Intermediate result:
    // - Parse Result type: 0 Success, 1 invalid e.g. year is 7 digits 1234567
    // - seconds part of parsed UTC timestamp
    // - microseconds part of parsed UTC timestamp
    // - Timezone type: 0 unspecified, 1 fixed type, 2 other type, 3 invalid
    // - Timezone offset for fixed type, only applies to fixed type
    // - Timezone is DST, only applies to other type
    // - Timezone index to `GpuTimeZoneDB.transitions` table
    list.add(Arrays.asList("2023-11-05T03:04:55 +00:00", 0, 1699153495L, 0, 1, 0, 0, -1)); // row 0
    list.add(Arrays.asList("2023-11-05 03:04:55 +01:02", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +1:02", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 -01:2", 0, 1699153495L, 0, 1, -(3600 * 1 + 60 * 2), 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +1:2", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +10:59", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 +10:59:03", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +105903", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +1059", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 +10", 0, 1699153495L, 0, 1, 3600 * 10, 0, -1));

    list.add(Arrays.asList("2023-11-05T03:04:55 UT+00:00", 0, 1699153495L, 0, 1, 0, 0, -1)); // rwo 10
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+01:02", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+1:02", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+01:2", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+1:2", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+10:59", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT-10:59:03", 0, 1699153495L, 0, 1, -(3600 * 10 + 60 * 59 + 3), 0,
        -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 UT+105903", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+1059", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+10", 0, 1699153495L, 0, 1, 3600 * 10, 0, -1));

    list.add(Arrays.asList("2023-11-05T03:04:55 UTC+00:00", 0, 1699153495L, 0, 1, 0, 0, -1)); // row 20
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+01:02", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+1:02", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+01:2", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+1:2", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+10:59", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 UTC+10:59:03", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 UTC+105903", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC+1059", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTC-10", 0, 1699153495L, 0, 1, -(3600 * 10), 0, -1));

    list.add(Arrays.asList("2023-11-05T03:04:55 GMT+00:00", 0, 1699153495L, 0, 1, 0, 0, -1)); // row 30
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+01:02", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+1:02", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT-01:2", 0, 1699153495L, 0, 1, -(3600 * 1 + 60 * 2), 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+1:2", 0, 1699153495L, 0, 1, 3600 * 1 + 60 * 2, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+10:59", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 GMT+10:59:03", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(
        Arrays.asList("2023-11-05 03:04:55 GMT+105903", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59 + 3, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+1059", 0, 1699153495L, 0, 1, 3600 * 10 + 60 * 59, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+10", 0, 1699153495L, 0, 1, 3600 * 10, 0, -1));

    list.add(Arrays.asList("2023-11-05T03:04:55.123456789 PST", 0, 1699153495L, 123456L, 2, 0, 1, 2)); // row 40
    list.add(Arrays.asList("2023-11-05 03:04:55.123456 PST", 0, 1699153495L, 123456L, 2, 0, 1, 2));
    list.add(Arrays.asList("2023-11-05T03:04:55 CTT", 0, 1699153495L, 0, 2, 0, 0, 0));
    list.add(Arrays.asList("2023-11-05 03:04:55 JST", 0, 1699153495L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-11-05 03:04:55 PST", 0, 1699153495L, 0, 2, 0, 1, 2));

    // use default timezone: index is 1
    list.add(Arrays.asList("2023-11-05 03:04:55", 0, 1699153495L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-11-05", 0, 1699142400L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-11", 0, 1698796800L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023", 0, 1672531200L, 0, 2, 0, 0, 1));

    list.add(Arrays.asList("12345", 0, 327403382400L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-1-1", 0, 1672531200L, 0, 2, 0, 0, 1)); // row 50
    list.add(Arrays.asList("2023-1-01", 0, 1672531200L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-01-1", 0, 1672531200L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-01-01", 0, 1672531200L, 0, 2, 0, 0, 1));

    // test leap year
    list.add(Arrays.asList("2028-02-29", 0, 1835395200L, 0, 2, 0, 0, 1));
    list.add(Arrays.asList("2023-01-01 00:00:00Z", 0, 1672531200L, 0, 1, 0, 0, -1));
    list.add(Arrays.asList("2023-01-01 00:00:00 Z", 0, 1672531200L, 0, 1, 0, 0, -1));
    // special case GMT0
    list.add(Arrays.asList("2023-01-01 00:00:00 GMT0", 0, 1672531200L, 0, 1, 0, 0, -1));

    // ################################################
    // ############ Below is invalid cases ############
    // ################################################
    // empty
    list.add(Arrays.asList("", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("  ", 1, 0L, 0, 0, 0, 0, -1));

    // test non leap year and day is 29
    list.add(Arrays.asList(" -2025-2-29 ", 1, 0L, 0, 0, 0, 0, -1)); // row 60
    // non-existence tz, result is invalid, but keeps tz type is other
    list.add(Arrays.asList(" 2023-11-05 03:04:55 non-existence-tz ", 1, 1699153495L, 0, 2, 0, 0, -1));

    // invalid month
    list.add(Arrays.asList("-2025-13-1", 1, 0L, 0, 0, 0, 0, -1));
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
    list.add(Arrays.asList("2000-01-01 00:00:60", 1, 0L, 0, 0, 0, 0, -1)); // row 70
    list.add(Arrays.asList("2000-01-01 00:00:61", 1, 0L, 0, 0, 0, 0, -1));

    // invalid date
    list.add(Arrays.asList("x2025", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("12", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("123", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("1234567", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-123", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-12x", 1, 0L, 0, 0, 0, 0, -1)); // row 80
    list.add(Arrays.asList("2200-01-", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-01-x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-01-11x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-01-113", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-03-25T", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-03-25 x", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("2200-03-25Tx", 1, 0L, 0, 0, 0, 0, -1));

    // invalid TZs
    list.add(Arrays.asList("2000-01-01 00:00:00 +", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 -X", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +07:", 1, 0L, 0, 3, 0, 0, -1)); // row 90
    list.add(Arrays.asList("2000-01-01 00:00:00 +09:x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +15:07x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +15:07:", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +15:07:x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +15:07:1", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +15:07:12x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +01x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +0102x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +010203x", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2000-01-01 00:00:00 +111", 1, 0L, 0, 3, 0, 0, -1)); // row 100
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
    list.add(Arrays.asList("2023-11-05 03:04:55 Ux", 1, 1699153495L, 0, 2, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTx", 1, 1699153495L, 0, 2, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UTCx", 1, 1699153495L, 0, 2, 0, 0, -1)); // row 110
    list.add(Arrays.asList("2023-11-05 03:04:55 UT+", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT-08:1:08", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 UT-8:1:08", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 Gx", 1, 1699153495L, 0, 2, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMTx", 1, 1699153495L, 0, 2, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT+", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT-08:1:08", 1, 0L, 0, 3, 0, 0, -1));
    list.add(Arrays.asList("2023-11-05 03:04:55 GMT-8:1:08", 1, 0L, 0, 3, 0, 0, -1));

    // invalid year: e.g. abs(year) > 30,000
    list.add(Arrays.asList("300001-01-01", 1, 0L, 0, 0, 0, 0, -1));
    list.add(Arrays.asList("-300001-01-01", 1, 0L, 0, 0, 0, 0, -1)); // row 120

    // ################################################
    // ############ Below is valid case ############
    // ################################################
    // although input exceeds max Spark year, but the intermidiate can hold it.
    // only years not in range [-300000, 300000] are invalid
    list.add(Arrays.asList("296271-05-22", 0, 9287254713600L, 0, 2, 0, 0, 1));

    List<String> input = new ArrayList<>(list.size());
    List<Byte> expected_return_type = new ArrayList<>(list.size());
    List<Long> expected_utc_seconds = new ArrayList<>(list.size());
    List<Integer> expected_utc_microseconds = new ArrayList<>(list.size());
    List<Byte> expected_tz_type = new ArrayList<>(list.size());
    List<Integer> expected_tz_offset = new ArrayList<>(list.size());
    List<Byte> expected_tz_is_dst = new ArrayList<>(list.size());
    List<Integer> expected_tz_index = new ArrayList<>(list.size());

    for (List<Object> row : list) {
      input.add(row.get(0).toString());
      expected_return_type.add(Byte.parseByte(row.get(1).toString()));
      expected_utc_seconds.add(Long.parseLong(row.get(2).toString()));
      expected_utc_microseconds.add(Integer.parseInt(row.get(3).toString()));
      expected_tz_type.add(Byte.parseByte(row.get(4).toString()));
      expected_tz_offset.add(Integer.parseInt(row.get(5).toString()));
      expected_tz_is_dst.add(Byte.parseByte(row.get(6).toString()));
      expected_tz_index.add(Integer.parseInt(row.get(7).toString()));
    }

    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector tzInfo = getTimezoneInfoMock(); // mock timezone info
        Table transitions = GpuTimeZoneDB.getTransitions();
        ColumnVector result = CastStrings.parseTimestampStrings(
            inputCv,
            1,
            /* is default tz DST */ false,
            defaultEpochDay,
            tzInfo,
            transitions,
            SPARK_330_VERSION);
        ColumnVector expectedReturnType = ColumnVector
            .fromBoxedUnsignedBytes(expected_return_type.toArray(new Byte[0]));
        ColumnVector expectedUtcTs = ColumnVector.fromBoxedLongs(expected_utc_seconds.toArray(new Long[0]));
        ColumnVector expectedUtcTsMicro = ColumnVector.fromBoxedInts(expected_utc_microseconds.toArray(new Integer[0]));
        ColumnVector expectedTzType = ColumnVector.fromBoxedUnsignedBytes(expected_tz_type.toArray(new Byte[0]));
        ColumnVector expectedTzOffset = ColumnVector.fromBoxedInts(expected_tz_offset.toArray(new Integer[0]));
        ColumnVector expectedTzIsDst = ColumnVector.fromBoxedUnsignedBytes(expected_tz_is_dst.toArray(new Byte[0]));
        ColumnVector expectedTzIndex = ColumnVector.fromBoxedInts(expected_tz_index.toArray(new Integer[0]))) {
      AssertUtils.assertColumnsAreEqual(expectedReturnType, result.getChildColumnView(0));
      AssertUtils.assertColumnsAreEqual(expectedUtcTs, result.getChildColumnView(1));
      AssertUtils.assertColumnsAreEqual(expectedUtcTsMicro, result.getChildColumnView(2));
      AssertUtils.assertColumnsAreEqual(expectedTzType, result.getChildColumnView(3));
      AssertUtils.assertColumnsAreEqual(expectedTzOffset, result.getChildColumnView(4));
      AssertUtils.assertColumnsAreEqual(expectedTzIsDst, result.getChildColumnView(5));
      AssertUtils.assertColumnsAreEqual(expectedTzIndex, result.getChildColumnView(6));
    }
  }

  @Test
  void castStringToTimestampOnCpu() {
    GpuTimeZoneDB.cacheDatabase(2200);
    GpuTimeZoneDB.verifyDatabaseCached();

    Instant ins1 = Instant.parse("2023-11-05T03:04:55Z");
    Instant ins2 = Instant.parse("2023-11-01T03:04:55Z");
    Instant ins3 = Instant.parse("2500-01-01T00:00:00Z"); // exceeds the max year 2200
    long base_ts1 = ins1.getEpochSecond() * 1000000L + ins1.getNano() / 1000L;
    long base_ts2 = ins2.getEpochSecond() * 1000000L + ins2.getNano() / 1000L;
    long base_ts3 = ins3.getEpochSecond() * 1000000L + ins2.getNano() / 1000L;

    // offset +01:02
    long offset = -(3600L * 1 + 60L * 2) * 1000000L;
    long cttOffset = -8 * 3600L * 1000000L;
    long pstOffset1 = +(8 * 3600L * 1000000L);
    long pstOffset2 = +(7 * 3600L * 1000000L);

    List<List<Object>> list = new ArrayList<>();
    List<String> input = new ArrayList<>(list.size());
    List<Long> expectedTS = new ArrayList<>(list.size());

    // 1. test fallback to cpu, has large year and has DST
    // Row is: input, expected ts, is null mask
    // CTT = Asia/Shanghai
    // PST = America/Los_Angeles
    list.add(Arrays.asList("2500-01-01", base_ts3, true)); // this value contributes to fallback
    list.add(Arrays.asList("2023-11-05T03:04:55 +00:00", base_ts1, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 +01:02", base_ts1 + offset, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 CTT", base_ts1 + cttOffset, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 Asia/Shanghai", base_ts1 + cttOffset, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 PST", base_ts1 + pstOffset1, true)); // PST contributes to fallback
    list.add(Arrays.asList("2023-11-05 03:04:55 America/Los_Angeles", base_ts1 + pstOffset1, true));
    list.add(Arrays.asList("2023-11-01 03:04:55 PST", base_ts2 + pstOffset2, true));
    list.add(Arrays.asList("2023-11-01 03:04:55 America/Los_Angeles", base_ts2 + pstOffset2, true));
    list.add(Arrays.asList("invalid", 0, false)); // ansi mode will cause exception
    for (List<Object> row : list) {
      input.add(row.get(0).toString());
      if ((Boolean) row.get(2)) {
        expectedTS.add(Long.parseLong(row.get(1).toString()));
      } else {
        expectedTS.add(null);
      }
    }
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ false, SPARK_330_VERSION);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(expectedTS.toArray(new Long[0]))) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }

    // 2. test ansi mode true, has large year and has DST
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ true, SPARK_330_VERSION)) {
      Assertions.assertNull(actual);
    }

    // 3. test ansi mode true, but without invalid input
    list.clear();
    input.clear();
    expectedTS.clear();
    list.add(Arrays.asList("2023-11-05 03:04:55 PST", base_ts1 + pstOffset1, true)); // PST contributes to fallback
    list.add(Arrays.asList("2500-01-01", base_ts3, true)); // this value contributes to fallback
    list.add(Arrays.asList("2023-11-05T03:04:55 +00:00", base_ts1, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 Z", base_ts1, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 +01:02", base_ts1 + offset, true));
    input.clear();
    expectedTS.clear();
    for (List<Object> row : list) {
      input.add(row.get(0).toString());
      expectedTS.add(Long.parseLong(row.get(1).toString()));
    }
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ true, SPARK_330_VERSION);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(expectedTS.toArray(new Long[0]))) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }

    // 4. test just time
    input.clear();
    list.clear();
    list.add(Arrays.asList("T00:00:01 +00:00", base_ts1, true));
    list.add(Arrays.asList("2500-01-01", base_ts3, true)); // this value contributes to fallback
    list.add(Arrays.asList("2023-11-05 03:04:55 PST", base_ts1 + pstOffset1, true)); // PST contributes to fallback

    for (List<Object> row : list) {
      input.add(row.get(0).toString());
    }
    long days = LocalDate.now().toEpochDay();
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */false, SPARK_330_VERSION);
        HostColumnVector hcv = actual.copyToHost();) {
      // this test may happen at mid-night, so the date may be different    
      long expectedTs1 = (days * 24 * 3600 + 1) * 1000000L;
      long expectedTs2 = ((days + 1) * 24 * 3600 + 1) * 1000000L;
      long actualTs = hcv.getLong(0);
      Assertions.assertTrue(actualTs == expectedTs1 || actualTs == expectedTs2);
    }

    // 5. test overflow after convert timezone for other timezone
    input.clear();
    list.clear();
    list.add(Arrays.asList("+294247-01-10T04:00:54.775807 PST", base_ts1, false)); // Spark max year
    list.add(Arrays.asList("2500-01-01", base_ts3, true)); // this value contributes to fallback
    list.add(Arrays.asList("2023-11-05 03:04:55 PST", base_ts1 + pstOffset1, true)); // PST contributes to fallback
    for (List<Object> row : list) {
      input.add(row.get(0).toString());
    }
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */true, SPARK_330_VERSION)) {
      Assertions.assertNull(actual);
    }

    // 6 test overflow after convert timezone for fixed timezone
    input.clear();
    list.clear();
    list.add(Arrays.asList("+294247-01-10T04:00:54.775807 -00:00:01", base_ts1, false)); // Spark max year
    list.add(Arrays.asList("2500-01-01", base_ts3, true)); // this value contributes to fallback
    list.add(Arrays.asList("2023-11-05 03:04:55 PST", base_ts1 + pstOffset1, true)); // PST contributes to fallback
    for (List<Object> row : list) {
      input.add(row.get(0).toString());
    }
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */true, SPARK_330_VERSION)) {
      Assertions.assertNull(actual);
    }
  }

  @Test
  void castStringToTimestampOnGpu() {
    GpuTimeZoneDB.cacheDatabase(2200);
    GpuTimeZoneDB.verifyDatabaseCached();

    Instant ins1 = Instant.parse("2023-11-05T03:04:55Z");
    Instant ins2 = Instant.parse("2023-11-01T03:04:55Z");
    long base_ts1 = ins1.getEpochSecond() * 1000000L + ins1.getNano() / 1000L;
    long base_ts2 = ins2.getEpochSecond() * 1000000L + ins2.getNano() / 1000L;

    // offset +01:02
    long offset = -(3600L * 1 + 60L * 2) * 1000000L;
    long cttOffset = -8 * 3600L * 1000000L;
    long pstOffset1 = +(8 * 3600L * 1000000L);
    long pstOffset2 = +(7 * 3600L * 1000000L);

    List<List<Object>> list = new ArrayList<>();
    List<String> input = new ArrayList<>(list.size());
    List<Long> expectedTS = new ArrayList<>(list.size());

    // 1. test fallback to cpu, has large year and has DST
    // Row is: input, expected ts, is null
    // CTT = Asia/Shanghai
    // PST = America/Los_Angeles
    list.add(Arrays.asList("2023-11-05T03:04:55 +00:00", base_ts1, true));
    list.add(Arrays.asList("2023-11-05T03:04:55.101", base_ts1 + 101000L, true));
    list.add(Arrays.asList("2023-11-05T03:04:55.123456", base_ts1 + 123456L, true));
    list.add(Arrays.asList("2023-11-05T03:04:55.9909", base_ts1 + 990900L, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 +01:02", base_ts1 + offset, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 CTT", base_ts1 + cttOffset, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 Asia/Shanghai", base_ts1 + cttOffset, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 PST", base_ts1 + pstOffset1, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 America/Los_Angeles", base_ts1 + pstOffset1, true));
    list.add(Arrays.asList("2023-11-01 03:04:55 PST", base_ts2 + pstOffset2, true));
    list.add(Arrays.asList("2023-11-01 03:04:55 America/Los_Angeles", base_ts2 + pstOffset2, true));
    list.add(Arrays.asList("invalid", 0, false)); // ansi mode will cause exception
    for (List<Object> row : list) {
      input.add(row.get(0).toString());
      if ((Boolean) row.get(2)) {
        expectedTS.add(Long.parseLong(row.get(1).toString()));
      } else {
        expectedTS.add(null);
      }
    }
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ false, SPARK_330_VERSION);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            expectedTS.toArray(new Long[0]))) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }

    // 2. test ansi mode true
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ true, SPARK_330_VERSION)) {
      Assertions.assertNull(actual);
    }

    // 3. test ansi mode true, but without invalid input
    list.clear();
    input.clear();
    expectedTS.clear();
    list.add(Arrays.asList("2023-11-05 03:04:55 PST", base_ts1 + pstOffset1, true));
    list.add(Arrays.asList("2023-11-05T03:04:55 +00:00", base_ts1, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 Z", base_ts1, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 +01:02", base_ts1 + offset, true));
    input.clear();
    expectedTS.clear();
    for (List<Object> row : list) {
      input.add(row.get(0).toString());
      expectedTS.add(Long.parseLong(row.get(1).toString()));
    }
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ true, SPARK_330_VERSION);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            expectedTS.toArray(new Long[0]))) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }

    // 4. test just time
    input.clear();
    list.clear();
    list.add(Arrays.asList("T00:00:01 +00:00", base_ts1, true));
    list.add(Arrays.asList("2023-11-05 03:04:55 PST", base_ts1 + pstOffset1, true));

    for (List<Object> row : list) {
      input.add(row.get(0).toString());
    }
    long days = LocalDate.now().toEpochDay();
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ false, SPARK_330_VERSION);
        HostColumnVector hcv = actual.copyToHost();) {
      long expectedTs1 = (days * 24 * 3600 + 1) * 1000000L;
      long expectedTs2 = ((days + 1) * 24 * 3600 + 1) * 1000000L;
      long actualTs = hcv.getLong(0);
      Assertions.assertTrue(actualTs == expectedTs1 || actualTs == expectedTs2);
    }

    // 5. test overflow after convert timezone for other timezone
    input.clear();
    list.clear();
    // Spark max year
    list.add(Arrays.asList("+294247-01-10T04:00:54.775807 -00:00:01", base_ts1, false));
    for (List<Object> row : list) {
      input.add(row.get(0).toString());
    }
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ true, SPARK_330_VERSION)) {
      Assertions.assertNull(actual);
    }

    // 6 test overflow after convert timezone for fixed timezone
    input.clear();
    list.clear();
    // Spark max year
    list.add(Arrays.asList("-294247-01-10T04:00:54.224191 +00:00:01", base_ts1, false));
    for (List<Object> row : list) {
      input.add(row.get(0).toString());
    }
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ true, SPARK_330_VERSION)) {
      Assertions.assertNull(actual);
    }
  }

  @Test
  void castStringToDate() {
    int expectedDays = (int) LocalDate.of(2025, 1, 1).toEpochDay();
    int negExpectedDays = (int) LocalDate.of(-2025, 1, 1).toEpochDay();
    int expectedLargeDays = (int) LocalDate.of(1000000, 1, 1).toEpochDay();
    int negExpectedLargeDays = (int) LocalDate.of(-1000000, 1, 1).toEpochDay();
    try (ColumnVector inputCv = ColumnVector.fromStrings(
        null,
        "  2025",
        "2025-01 ",
        "2025-1  ",
        "2025-1-1",
        "2025-1-01",
        "2025-01-1",
        "2025-01-01",
        "2025-01-01T",
        "+2025-01-01Txxx",
        "-2025-01-01 xxx",
        "1000000-01-01", // valid large year
        "-1000000-01-01", // valid large year
        "10000001-01-01", // invalid large year
        "-10000001-01-01" // invalid large year
    );
        ColumnVector actual = CastStrings.toDate(inputCv, /* ansi */ false);
        ColumnVector expected = ColumnVector.timestampDaysFromBoxedInts(
            null,
            expectedDays,
            expectedDays,
            expectedDays,
            expectedDays,
            expectedDays,
            expectedDays,
            expectedDays,
            expectedDays,
            expectedDays,
            negExpectedDays,
            expectedLargeDays,
            negExpectedLargeDays,
            null,
            null)) {

      AssertUtils.assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void castStringToDateAnsi() {
    // 2025x is invalid
    try (ColumnVector inputCv = ColumnVector.fromStrings("2025", "2025x");
        ColumnVector actual = CastStrings.toDate(inputCv, /* ansi */true)) {
      Assertions.assertNull(actual);
    }
  }

  /**
   * Test cast string to timestamp with non-UTC default timezone.
   */
  @Test
  void castStringToTimestampUseNonUTCDefaultTimezone() {
    GpuTimeZoneDB.cacheDatabase(2200);
    GpuTimeZoneDB.verifyDatabaseCached();

    // 1. test fallback to cpu: has year > 2200 and has DST
    String ts1 = "6663-09-28T00:00:00";
    // calculated from Spark:
    // spark.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
    // spark.createDataFrame([('6663-09-28T00:00:00',)], 'a string')
    // .selectExpr("cast(cast(a as timestamp) as long) * 1000000L").show()
    long micors1 = 148120124400000000L;
    try (ColumnVector inputCv = ColumnVector.fromStrings(ts1);
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv,
            "America/Los_Angeles", // non-UTC default timezone
            /* ansi */ false,
            SPARK_330_VERSION);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(micors1)) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }

    // 2. test run on GPU: has no year > 2200 although has DST
    String ts2 = "2025-09-28T00:00:00";
    // calculated from Spark: refer to the above code
    long micors2 = 1759042800000000L;
    try (ColumnVector inputCv = ColumnVector.fromStrings(ts2);
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv,
            "America/Los_Angeles", // non-UTC default timezone
            /* ansi */ false,
            SPARK_330_VERSION);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(micors2)) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void castStringToTimestampOnGpuForSpark320() {
    GpuTimeZoneDB.cacheDatabase(2200);
    GpuTimeZoneDB.verifyDatabaseCached();

    Instant ins1 = Instant.parse("2023-11-05T03:04:55Z");
    long base_ts1 = ins1.getEpochSecond() * 1000000L + ins1.getNano() / 1000L;

    List<List<Object>> list = new ArrayList<>();
    List<String> input = new ArrayList<>(list.size());
    List<Long> expectedTS = new ArrayList<>(list.size());

    // minute must be 2 digits for Spark 320
    list.add(Arrays.asList("2023-11-05T03:04:55 +00:00:00", base_ts1, true));
    list.add(Arrays.asList("2023-11-05T03:04:55 +00:1", 0, false));
    list.add(Arrays.asList("2023-11-05T03:04:55 -00:1", 0, false));
    list.add(Arrays.asList("2023-11-05T03:04:55 UT+00:1", 0, false));
    list.add(Arrays.asList("2023-11-05T03:04:55 UT-00:1", 0, false));
    list.add(Arrays.asList("2023-11-05T03:04:55 GMT+00:1", 0, false));
    list.add(Arrays.asList("2023-11-05T03:04:55 GMT-00:1", 0, false));
    // minute must be in ragne: 0-59
    // offsets must be in range: -18:00 to +18:00
    list.add(Arrays.asList("2015-03-18T12:03:17-0:70", 0, false));
    list.add(Arrays.asList("2015-03-18T12:03:17-18:01", 0, false));
    list.add(Arrays.asList("2015-03-18T12:03:17+19:00", 0, false));
    list.add(Arrays.asList("2015-03-18T12:03:17+10:60", 0, false));

    for (List<Object> row : list) {
      input.add(row.get(0).toString());
      if ((Boolean) row.get(2)) {
        expectedTS.add(Long.parseLong(row.get(1).toString()));
      } else {
        expectedTS.add(null);
      }
    }
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        // test spark 320
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ false, SPARK_320_VERSION);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            expectedTS.toArray(new Long[0]))) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Spark400+ and DB14.3+: do not support pattern: spaces + Thh:mm:ss
   * Refer to https://github.com/NVIDIA/spark-rapids-jni/issues/3401
   */
  @Test
  void castStringToTimestampOnGpuForSpark400PlusDB14_3Plus() {
    GpuTimeZoneDB.cacheDatabase(2200);
    GpuTimeZoneDB.verifyDatabaseCached();

    List<List<Object>> list = new ArrayList<>();
    List<String> input = new ArrayList<>(list.size());
    List<Long> expectedTS = new ArrayList<>(list.size());

    // invalid value
    list.add(Arrays.asList("    T00:00:00", 0, false));

    for (List<Object> row : list) {
      input.add(row.get(0).toString());
      if ((Boolean) row.get(2)) {
        expectedTS.add(Long.parseLong(row.get(1).toString()));
      } else {
        expectedTS.add(null);
      }
    }
    // test spark 400+
    Version v400 = new Version(SparkPlatformType.VANILLA_SPARK, 4, 0, 0);
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        // test spark 320
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ false, v400);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            expectedTS.toArray(new Long[0]))) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }

    // test spark DB 14.3+
    Version vDB14_3 = new Version(SparkPlatformType.DATABRICKS, 14, 3, 0);
    try (ColumnVector inputCv = ColumnVector.fromStrings(input.toArray(new String[0]));
        // test spark 320
        ColumnVector actual = CastStrings.toTimestamp(
            inputCv, "Z", /* ansi */ false, vDB14_3);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            expectedTS.toArray(new Long[0]))) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }
  }
}
