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

import java.time.ZoneId;
import java.util.List;

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
    // storing up to year 2200, years 2000-2200 have DST
    assert(transitions.size() > 400);
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
          568008000L
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
          568036800L
        );
        ColumnVector actual = GpuTimeZoneDB.fromTimestampToUtcTimestamp(input,
          ZoneId.of("US/Pacific"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void convertToUtcSecondsTzRulesCpuFallbackTest() {
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
          11684584557L,
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
  void convertToUtcMicroSecondsTzRulesNullFallbackTest() {
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
  
}
