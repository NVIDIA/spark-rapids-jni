/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

import java.time.*;
import java.util.ArrayList;
import java.util.List;
import java.util.AbstractMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Table;

public class CastStringsTest {
  @Test
  void castToIntegerTest() {
    Table.TestBuilder tb = new Table.TestBuilder();
    tb.column(3l, 9l, 4l, 2l, 20l, null, null);
    tb.column(5, 1, 0, 2, 7, null, null);
    tb.column(new Byte[]{2, 3, 4, 5, 9, null, null});
    try (Table expected = tb.build()) {
      Table.TestBuilder tb2 = new Table.TestBuilder();
      tb2.column(" 3", "9", "4", "2", "20.5", null, "7.6asd");
      tb2.column("5", "1  ", "0", "2", "7.1", null, "asdf");
      tb2.column("2", "3", " 4 ", "5", " 9.2 ", null, "7.8.3");

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
  void castToDecimalTest() {
    Table.TestBuilder tb = new Table.TestBuilder();
    tb.decimal32Column(0,3, 9, 4, 2, 21, null, null);
    tb.decimal64Column(0, 5l, 1l, 0l, 2l, 7l, null, null);
    tb.decimal32Column(-1, 20, 30, 40, 51, 92, null, null);
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

  @Test
  void toTimestampTestAnsiWithoutTz() {
    assertThrows(IllegalArgumentException.class, () -> {
      try (ColumnVector input = ColumnVector.fromStrings(" invalid_value ")) {
        // ansiEnabled is true
        CastStrings.toTimestampWithoutTimeZone(input, false, true);
      }
    });

    Instant instant = LocalDateTime.parse("2023-11-05T03:04:55").toInstant(ZoneOffset.UTC);
    long expectedResults = instant.getEpochSecond() * 1000000L;

    try (
        ColumnVector input = ColumnVector.fromStrings("2023-11-05 3:04:55");
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(expectedResults);
        ColumnVector actual = CastStrings.toTimestampWithoutTimeZone(input, false, true)) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void toTimestampTestWithTz() {
    List<Map.Entry<String, Long>> entries = new ArrayList<>();
    // Without timezone
    entries.add(new AbstractMap.SimpleEntry<>("  2000-01-29 ", 949104000000000L));
    // Timezone IDs
    entries.add(new AbstractMap.SimpleEntry<>("2023-11-05 3:4:55 America/Sao_Paulo", 1699164295000000L));
    entries.add(new AbstractMap.SimpleEntry<>("2023-11-5T03:04:55.1   Asia/Shanghai", 1699124695100000L));
    entries.add(new AbstractMap.SimpleEntry<>("2000-1-29 13:59:8 Iran", 949141748000000L));
    entries.add(new AbstractMap.SimpleEntry<>("1968-03-25T23:59:1.123Asia/Tokyo", -55846858877000L));
    entries.add(new AbstractMap.SimpleEntry<>("1968-03-25T23:59:1.123456Asia/Tokyo", -55846858876544L));
  
    // UTC-like timezones
    //  no adjustment
    entries.add(new AbstractMap.SimpleEntry<>("1970-9-9 2:33:44 Z", 21695624000000L));
    entries.add(new AbstractMap.SimpleEntry<>(" 1969-12-1 2:3:4.999Z", -2671015001000L));
    entries.add(new AbstractMap.SimpleEntry<>("1954-10-20 00:11:22 GMT  ", -479692118000000L));
    entries.add(new AbstractMap.SimpleEntry<>("1984-1-3 00:11:22UTC", 441936682000000L));
    //  hh
    entries.add(new AbstractMap.SimpleEntry<>("1998-11-05T20:00:1.12 UTC+18 ", 910231201120000L));
    entries.add(new AbstractMap.SimpleEntry<>("1998-11-05T20:00:1.12UTC+0", 910296001120000L));
    entries.add(new AbstractMap.SimpleEntry<>("1998-11-05T20:00:1.12UTC-00", 910296001120000L));
    entries.add(new AbstractMap.SimpleEntry<>(" 1998-11-05T20:00:1.12   GMT+09 ", 910263601120000L));
    entries.add(new AbstractMap.SimpleEntry<>("1998-11-05T20:00:1.12   GMT-1", 910299601120000L));
    entries.add(new AbstractMap.SimpleEntry<>("1998-11-05T20:00:1.12  UTC-6", 910317601120000L));
    entries.add(new AbstractMap.SimpleEntry<>("1998-11-05T20:00:1.12  UTC-18", 910360801120000L));
    entries.add(new AbstractMap.SimpleEntry<>("1998-11-05T20:00:1.12UTC-00", 910296001120000L));
    entries.add(new AbstractMap.SimpleEntry<>(" 1998-11-05T20:00:1.12   +09 ", 910263601120000L));
    entries.add(new AbstractMap.SimpleEntry<>("1998-11-05T20:00:1.12   -1", 910299601120000L));
    entries.add(new AbstractMap.SimpleEntry<>("1998-11-05T20:00:1.12 +18 ", 910231201120000L));
    entries.add(new AbstractMap.SimpleEntry<>("1998-11-05T20:00:1.12-00", 910296001120000L));
    //  hh:mm
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.999 UTC+1428", -2723095001000L));
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.999 GMT-1501", -2616955001000L));
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.999 GMT+1:22", -2675935001000L));
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.8888 GMT+8:2", -2699935111200L));
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.999 UTC+17:9", -2732755001000L));
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.999 UTC-09:11", -2637955001000L));
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.999 +1428  ", -2723095001000L));
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.999-1501  ", -2616955001000L));
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.999 +1:22 ", -2675935001000L));
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.8888 +8:2  ", -2699935111200L));
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.999+17:9", -2732755001000L));
    entries.add(new AbstractMap.SimpleEntry<>("1969-12-1 2:3:4.999    -09:11", -2637955001000L));
    //  hh:mm::ss
    entries.add(new AbstractMap.SimpleEntry<>("2019-10-20 22:33:44.1 GMT+112233", 1571569871100000L));
    entries.add(new AbstractMap.SimpleEntry<>("2019-10-20 22:33:44.1 UTC-100102", 1571646886100000L));
    entries.add(new AbstractMap.SimpleEntry<>("2019-10-20 22:33:44.1 UTC+11:22:33", 1571569871100000L));
    entries.add(new AbstractMap.SimpleEntry<>("2019-10-20 22:33:44.1 GMT-10:10:10", 1571647434100000L));
    entries.add(new AbstractMap.SimpleEntry<>("2019-10-20 22:33:44.1 GMT-8:08:01", 1571640105100000L));
    entries.add(new AbstractMap.SimpleEntry<>("2019-10-20 22:33:44.1 UTC+4:59:59", 1571592825100000L));
    entries.add(new AbstractMap.SimpleEntry<>("2019-10-20 00:1:20.3  +102030", 1571492450300000L));
    entries.add(new AbstractMap.SimpleEntry<>("2019-10-20 00:1:20.3   -020103", 1571536943300000L));
    entries.add(new AbstractMap.SimpleEntry<>("2019-10-20 22:33:44.1   -8:08:01  ", 1571640105100000L));
    entries.add(new AbstractMap.SimpleEntry<>("2019-10-20 22:33:44.1+4:59:59", 1571592825100000L));
    // short TZ ID: BST->Asia/Dhaka, CTT->Asia/Shanghai
    entries.add(new AbstractMap.SimpleEntry<>("2023-11-5T03:04:55.1 CTT", 1699124695100000L));
    entries.add(new AbstractMap.SimpleEntry<>("2023-11-5T03:04:55.1 BST", 1699124695100000L + 7200L * 1000000L)); // BST is 2 hours later than CTT

    int validDataSize = entries.size();

    // Invalid instances
    // Timezone without hh:mm:ss
    entries.add(new AbstractMap.SimpleEntry<>("2000-01-29 Iran", null));
    // Invalid Timezone ID
    entries.add(new AbstractMap.SimpleEntry<>("2000-01-29 10:20:30 Asia/London", null));
    // Invalid UTC-like timezone
    //  overflow
    entries.add(new AbstractMap.SimpleEntry<>("2000-01-29 10:20:30 +10:60", null));
    entries.add(new AbstractMap.SimpleEntry<>("2000-01-29 10:20:30 UTC-7:59:60", null));
    entries.add(new AbstractMap.SimpleEntry<>("2000-01-29 10:20:30 +19", null));
    entries.add(new AbstractMap.SimpleEntry<>("2000-01-29 10:20:30 UTC-23", null));
    entries.add(new AbstractMap.SimpleEntry<>("2000-01-29 10:20:30 GMT+1801", null));
    entries.add(new AbstractMap.SimpleEntry<>("2000-01-29 10:20:30 -180001", null));
    entries.add(new AbstractMap.SimpleEntry<>("2000-01-29 10:20:30 UTC+18:00:10", null));
    entries.add(new AbstractMap.SimpleEntry<>("2000-01-29 10:20:30 GMT-23:5", null));

    List<String> inputs = new ArrayList<>();
    List<Long> expects = new ArrayList<>();
    for (Map.Entry<String, Long> entry : entries) {
      inputs.add(entry.getKey());
      expects.add(entry.getValue());
    }

    // Throw unsupported exception for symbols because Europe/London contains DST rules
    assertThrows(ai.rapids.cudf.CudfException.class, () -> {
      try (ColumnVector input = ColumnVector.fromStrings("2000-01-29 1:2:3 Europe/London")) {
        CastStrings.toTimestamp(input, ZoneId.of("UTC"), false);
      }
    });

    // Throw unsupported exception for symbols of special dates
    // Note: Spark 31x supports "epoch", "now", "today", "yesterday", "tomorrow".
    // But Spark 32x to Spark 35x do not supports.
    // Currently JNI do not supports
    for (String date : new String[]{"epoch", "now", "today", "yesterday", "tomorrow"})
    assertThrows(IllegalArgumentException.class, () -> {
      try (ColumnVector input = ColumnVector.fromStrings(date)) {
        CastStrings.toTimestamp(input, ZoneId.of("UTC"), true);
      }
    });

    // non-ANSI mode
    try (
        ColumnVector input = ColumnVector.fromStrings(inputs.toArray(new String[0]));
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(expects.toArray(new Long[0]));
        ColumnVector actual = CastStrings.toTimestamp(input, ZoneId.of("UTC"), false)) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }

    // Should NOT throw exception because all inputs are valid
    String[] validInputs = inputs.stream().limit(validDataSize).toArray(String[]::new);
    Long[] validExpects = expects.stream().limit(validDataSize).toArray(Long[]::new);
    try (
        ColumnVector input = ColumnVector.fromStrings(validInputs);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(validExpects);
        ColumnVector actual = CastStrings.toTimestamp(input, ZoneId.of("UTC"), true)) {
      AssertUtils.assertColumnsAreEqual(expected, actual);
    }

    // Throw IllegalArgumentException for invalid timestamps under ANSI mode
    assertThrows(IllegalArgumentException.class, () -> {
      try (ColumnVector input = ColumnVector.fromStrings(inputs.toArray(new String[0]))) {
        CastStrings.toTimestamp(input, ZoneId.of("UTC"), true);
      }
    });

    // Throw IllegalArgumentException for non-exist-tz in ANSI mode
    assertThrows(IllegalArgumentException.class, () -> {
      try (ColumnVector input = ColumnVector.fromStrings("2000-01-29 1:2:3 non-exist-tz")) {
        CastStrings.toTimestamp(input, ZoneId.of("UTC"), true);
      }
    });

    // Return null for non-exist-tz in non-Ansi mode
    Long[] nullExpected = {null};
    try (
      ColumnVector input = ColumnVector.fromStrings("2000-01-29 1:2:3 non-exist-tz");
      ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(nullExpected);
      ColumnVector actual = CastStrings.toTimestamp(input, ZoneId.of("UTC"), false)) {
        AssertUtils.assertColumnsAreEqual(expected, actual);
    }

  }
}
