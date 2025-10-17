/*
* Copyright (c)  2023-2025, NVIDIA CORPORATION.
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

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import ai.rapids.cudf.ColumnVector;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;


public class TimeZoneTest {
  @BeforeAll
  static void cacheTimezoneDatabase() {
    GpuTimeZoneDB.cacheDatabase(2200);
  }
  
  @AfterAll
  static void cleanup() {
    GpuTimeZoneDB.shutdown();
  }
  
  @Test
  void databaseLoadedTest() {
    // Check for a few timezones
    GpuTimeZoneDB.verifyDatabaseCached();
    List transitions = GpuTimeZoneDB.getHostTransitions("UTC+8");
    assertNotNull(transitions);
    assertEquals(1, transitions.size());
    transitions = GpuTimeZoneDB.getHostTransitions("Asia/Shanghai");
    assertNotNull(transitions);
    ZoneId shanghai = ZoneId.of("Asia/Shanghai").normalized();
    assertEquals(shanghai.getRules().getTransitions().size() + 1, transitions.size());
    transitions = GpuTimeZoneDB.getHostTransitions("America/Los_Angeles");
    assertNotNull(transitions);
  }
  
  @Test
  void convertToUtcSecondsTest() {
    try (ColumnVector input = ColumnVector.timestampSecondsFromBoxedLongs(
          -1262260800L,
          -908838000L,
          -908840700L,
          -888800400L,
          -888799500L,
          -888796800L,
          0L,
          1699571634L,
          568036800L
        );
        ColumnVector expected = ColumnVector.timestampSecondsFromBoxedLongs(
          -1262289600L,
          -908870400L,
          -908869500L,
          -888832800L,
          -888831900L,
          -888825600L,
          -28800L,
          1699542834L,
          568008000L
        );
        ColumnVector actual = GpuTimeZoneDB.fromTimestampToUtcTimestamp(input,
          ZoneId.of("Asia/Shanghai"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void convertToUtcSecondsTestTzRules() {
    try (
        ColumnVector input = ColumnVector.timestampSecondsFromBoxedLongs(
            -1262289600L,
            -908866800L,
            -908869500L,
            -888829200L,
            -888828300L,
            -868822000L,
            -25825600L,
            -23125600L,
            -20122000L,
            -28800L,
            1699542834L,
            2299542834L,
            568008000L,
            1741482000L, // 2025-03-09 01:00:00, gap begin - 1 hour
            1741485599L, // 2025-03-09 01:59:59, gap begin - 1s
            1741485600L, // 2025-03-09 02:00:00, gap begin
            1741485601L, // 2025-03-09 02:00:01, gap begin + 1s
            1741489199L, // 2025-03-09 02:59:59, gap begin + 1 hour - 1s
            1741489200L, // 2025-03-09 02:00:00, gap end = gap begin + 1 hour
            1741489201L, // 2025-03-09 03:00:01, gap end + 1s
            1741492800L, // 2025-03-09 04:00:00, gap end + 1 hour
            1762045200L, // 2025-11-02 01:00:00, overlap begin - 1 hour
            1762048799L, // 2025-11-02 01:59:59, overlap begin - 1s
            1762048800L, // 2025-11-02 02:00:00, overlap begin
            1762048801L, // 2025-11-02 02:00:01, overlap begin + 1s
            1762052399L, // 2025-11-02 02:59:59, overlap begin + 1 hour - 1s
            1762052400L, // 2025-11-02 03:00:00, overlap end = overlap begin + 1 hour
            1762052401L, // 2025-11-02 03:00:01, overlap end + 1s
            1762056000L // 2025-11-02 04:00:00, overlap end + 1 hour
        );
        ColumnVector expected = ColumnVector.timestampSecondsFromBoxedLongs(
            -1262260800L,
            -908838000L,
            -908840700L,
            -888800400L,
            -888799500L,
            -868796800L,
            -25796800L,
            -23096800L,
            -20096800L,
            0L,
            1699571634L,
            2299571634L,
            568036800L,
            1741510800L, // 2025-03-09T09:00:00Z, diff -8 hours
            1741514399L, // 2025-03-09T09:59:59Z, diff -8 hours
            1741514400L, // 2025-03-09T10:00:00Z, diff -8 hours
            1741514401L, // 2025-03-09T10:00:01Z, diff -8 hours
            1741517999L, // 2025-03-09T10:59:59Z, diff -8 hours
            1741514400L, // 2025-03-09T10:00:00Z, diff -7 hours
            1741514401L, // 2025-03-09T10:00:01Z, diff -7 hours
            1741518000L, // 2025-03-09T11:00:00Z, diff -7 hours
            1762070400L, // 2025-11-02T08:00:00Z, diff -7 hours
            1762073999L, // 2025-11-02T08:59:59Z, diff -7 hours
            1762077600L, // 2025-11-02T10:00:00Z, diff -8 hours
            1762077601L, // 2025-11-02T10:00:01Z, diff -8 hours
            1762081199L, // 2025-11-02T10:59:59Z, diff -8 hours
            1762081200L, // 2025-11-02T11:00:00Z, diff -8 hours
            1762081201L, // 2025-11-02T11:00:01Z, diff -8 hours
            1762084800L // 2025-11-02T12:00:00Z, diff -8 hours
        );
        ColumnVector actual = GpuTimeZoneDB.fromTimestampToUtcTimestamp(input,
            ZoneId.of("US/Pacific"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  // has a year > 2200, still runs on GPU.
  @Test
  void convertToUtcSecondsTzRulesBigThan2200year() {
    try (ColumnVector input = ColumnVector.timestampSecondsFromBoxedLongs(
          -1262289600L,
          -908866800L,
          -908869500L,
          -888829200L,
          -888828300L,
          -868822000L,
          -25825600L,
          -23125600L,
          -20122000L,
          -28800L,
          1699542834L,
          2299542834L,
          568008000L,
          11684584557L, // 2340 year, 
          16820549757L
        );
        ColumnVector expected = ColumnVector.timestampSecondsFromBoxedLongs(
          -1262260800L,
          -908838000L,
          -908840700L,
          -888800400L,
          -888799500L,
          -868796800L,
          -25796800L,
          -23096800L,
          -20096800L,
          0L,
          1699571634L,
          2299571634L,
          568036800L,
          11684609757L,
          16820578557L
        );
        ColumnVector actual = GpuTimeZoneDB.fromTimestampToUtcTimestamp(input,
          ZoneId.of("US/Pacific"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void convertToUtcNullTest() {
    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(
          233843186247521117L
        );
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
          233843211447521117L
        );
        ColumnVector actual = GpuTimeZoneDB.fromTimestampToUtcTimestamp(input,
          ZoneId.of("US/Pacific"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void convertToUtcMicroSecondsTzRulesNulls() {
    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(
          null,
          null,
          null,
          null
        );
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
          null,
          null,
          null,
          null
        );
        ColumnVector actual = GpuTimeZoneDB.fromTimestampToUtcTimestamp(input,
          ZoneId.of("US/Pacific"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void convertToUtcMilliSecondsTestTzRules() {
    try (ColumnVector input = ColumnVector.timestampMilliSecondsFromBoxedLongs(
          -1262289600000L,
          -908866800000L,
          -908869500000L,
          -888829200000L,
          -888828300000L,
          -888825600000L,
          -28800000L,
          1699542834312L,
          568008000000L
        );
        ColumnVector expected = ColumnVector.timestampMilliSecondsFromBoxedLongs(
          -1262260800000L,
          -908838000000L,
          -908840700000L,
          -888800400000L,
          -888799500000L,
          -888796800000L,
          0L,
          1699571634312L,
          568036800000L
        );
        ColumnVector actual = GpuTimeZoneDB.fromTimestampToUtcTimestamp(input,
          ZoneId.of("America/Los_Angeles"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void convertToUtcMilliSecondsTest() {
    try (ColumnVector input = ColumnVector.timestampMilliSecondsFromBoxedLongs(
          -1262260800000L,
          -908838000000L,
          -908840700000L,
          -888800400000L,
          -888799500000L,
          -888796800000L,
          0L,
          1699571634312L,
          568036800000L
        );
        ColumnVector expected = ColumnVector.timestampMilliSecondsFromBoxedLongs(
          -1262289600000L,
          -908870400000L,
          -908869500000L,
          -888832800000L,
          -888831900000L,
          -888825600000L,
          -28800000L,
          1699542834312L,
          568008000000L
        );
        ColumnVector actual = GpuTimeZoneDB.fromTimestampToUtcTimestamp(input,
          ZoneId.of("Asia/Shanghai"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }
  
  @Test
  void convertToUtcMicroSecondsTest() {
    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(
          -1262260800000000L,
          -908838000000000L,
          -908840700000000L,
          -888800400000000L,
          -888799500000000L,
          -888796800000000L,
          0L,
          1699571634312000L,
          568036800000000L
        );
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
          -1262289600000000L,
          -908870400000000L,
          -908869500000000L,
          -888832800000000L,
          -888831900000000L,
          -888825600000000L,
          -28800000000L,
          1699542834312000L,
          568008000000000L
        );
        ColumnVector actual = GpuTimeZoneDB.fromTimestampToUtcTimestamp(input,
          ZoneId.of("Asia/Shanghai"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void convertFromUtcSecondsTest() {
    try (ColumnVector input = ColumnVector.timestampSecondsFromBoxedLongs(
          -1262289600L,
          -908870400L,
          -908869500L,
          -888832800L,
          -888831900L,
          -888825600L,
          0L,
          1699542834L,
          568008000L);
        ColumnVector expected = ColumnVector.timestampSecondsFromBoxedLongs(
          -1262260800L,
          -908838000L,
          -908837100L,
          -888800400L,
          -888799500L,
          -888796800L,
          28800L,
          1699571634L,
          568036800L);
        ColumnVector actual = GpuTimeZoneDB.fromUtcTimestampToTimestamp(input,
          ZoneId.of("Asia/Shanghai"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void convertFromUtcMilliSecondsTest() {
    try (ColumnVector input = ColumnVector.timestampMilliSecondsFromBoxedLongs(
          -1262289600000L,
          -908870400000L,
          -908869500000L,
          -888832800000L,
          -888831900000L,
          -888825600000L,
          0L,
          1699542834312L,
          568008000000L);
        ColumnVector expected = ColumnVector.timestampMilliSecondsFromBoxedLongs(
          -1262260800000L,
          -908838000000L,
          -908837100000L,
          -888800400000L,
          -888799500000L,
          -888796800000L,
          28800000L,
          1699571634312L,
          568036800000L);
        ColumnVector actual = GpuTimeZoneDB.fromUtcTimestampToTimestamp(input,
          ZoneId.of("Asia/Shanghai"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }
  
  @Test
  void convertFromUtcMicroSecondsTest() {
    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(
          -1262289600000000L,
          -908870400000000L,
          -908869500000000L,
          -888832800000000L,
          -888831900000000L,
          -888825600000000L,
          0L,
          1699542834312000L,
          568008000000000L,
          null);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
          -1262260800000000L,
          -908838000000000L,
          -908837100000000L,
          -888800400000000L,
          -888799500000000L,
          -888796800000000L,
          28800000000L,
          1699571634312000L,
          568036800000000L,
          null);
        ColumnVector actual = GpuTimeZoneDB.fromUtcTimestampToTimestamp(input,
          ZoneId.of("Asia/Shanghai"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void nonNonNormalizedTimezone() {
    GpuTimeZoneDB.verifyDatabaseCached();
    List transitions;

    transitions = GpuTimeZoneDB.getHostTransitions("Etc/GMT");
    assertNotNull(transitions);

    transitions = GpuTimeZoneDB.getHostTransitions("Z");
    assertNotNull(transitions);
  }

  /**
   * test `Australia/Sydney` and `America/Los_Angeles` timezones.
   * Australia/Sydney: the start rule is overlap, the end rule is gap.
   * America/Los_Angeles: the start rule is gap, the end rule is overlap.
   */
  @Test
  void convertToUtcSecondsCompareToJava() {
    GpuTimeZoneDB.verifyDatabaseCached();

    // test time range: (0001-01-01 00:00:00, 9999-12-31 23:59:59)
    long min = LocalDateTime.of(1, 1, 1, 0, 0, 0)
        .toEpochSecond(ZoneOffset.UTC);
    long max = LocalDateTime.of(9999, 12, 31, 23, 59, 59)
        .toEpochSecond(ZoneOffset.UTC);

    // use today as the random seed so we get different values each day
    Random rng = new Random(LocalDate.now().toEpochDay());
    for (String tz : Arrays.asList("America/Los_Angeles", "Australia/Sydney")) {
      ZoneId zid = ZoneId.of(tz);

      int num_rows = 10 * 1024;
      long[] seconds = new long[num_rows];
      for (int i = 0; i < seconds.length; ++i) {
        // range is years from 0001 to 9999
        seconds[i] = min + (long) (rng.nextDouble() * (max - min));
      }

      long[] expectedSeconds = new long[num_rows];
      for (int i = 0; i < expectedSeconds.length; ++i) {
        expectedSeconds[i] = Instant.ofEpochSecond(seconds[i]).atZone(ZoneOffset.UTC).toLocalDateTime().atZone(zid)
            .toInstant().getEpochSecond();
      }

      try (ColumnVector input = ColumnVector.timestampSecondsFromLongs(seconds);
          ColumnVector actual = GpuTimeZoneDB.fromTimestampToUtcTimestamp(input, zid);
          ColumnVector expected = ColumnVector.timestampSecondsFromLongs(expectedSeconds)) {

        assertColumnsAreEqual(expected, actual);
      }
    }
  }
}
