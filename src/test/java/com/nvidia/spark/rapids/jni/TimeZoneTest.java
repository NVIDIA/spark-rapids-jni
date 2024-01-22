/*
* Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Scalar;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;


public class TimeZoneTest {
  @BeforeAll
  static void cacheTimezoneDatabase() {
    GpuTimeZoneDB.cacheDatabase();
  }
  
  @AfterAll
  static void cleanup() {
    GpuTimeZoneDB.shutdown();
  }
  
  @Test
  void databaseLoadedTest() {
    // Check for a few timezones
    GpuTimeZoneDB instance = GpuTimeZoneDB.getInstance();
    List transitions = instance.getHostFixedTransitions("UTC+8");
    assertNotNull(transitions);
    assertEquals(1, transitions.size());
    transitions = instance.getHostFixedTransitions("Asia/Shanghai");
    assertNotNull(transitions);
    ZoneId shanghai = ZoneId.of("Asia/Shanghai").normalized();
    assertEquals(shanghai.getRules().getTransitions().size() + 2, transitions.size());
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
          568008000000000L);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
          -1262260800000000L,
          -908838000000000L,
          -908837100000000L,
          -888800400000000L,
          -888799500000000L,
          -888796800000000L,
          28800000000L,
          1699571634312000L,
          568036800000000L);
        ColumnVector actual = GpuTimeZoneDB.fromUtcTimestampToTimestamp(input,
          ZoneId.of("Asia/Shanghai"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void timeAddCCTest() {
    // Some edge cases related to overlap transitions
    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(
      -57954592249912415L,
      -2177453143500000L,
      -43013395848980300L,
      -2177485200000000L,
      -2177481695679933L,
      -2177481944610644L,
      0L,
      -2177481944610644L,
      -2177481944610644L);
      ColumnVector duration = ColumnVector.durationMicroSecondsFromBoxedLongs(
        56087020233685111L,
        1000000L,
        173001810506226873L,
        1000000L,
        1000000L,
        1000000L,
        173001810506226873L,
        86399999999L,
        86400000000L
      );
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
          -1867571673227304L,
          -2177453142500000L,
          129988415000246573L,
          -2177485199000000L,
          -2177481694679933L,
          -2177481943610644L,
          173001810506226873L,
          -2177395544610645L,
          -2177395201610644L);
        ColumnVector actual = GpuTimeZoneDB.timeAdd(input, duration,
          ZoneId.of("Asia/Shanghai"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void timeAddCSTest() {
    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(
        -57954592249912415L,
        -2177453143500000L,
        -43013395848980300L,
        -2177485200000000L,
        -2177481695679933L,
        -2177481944610644L,
        0L);
        Scalar duration = Scalar.durationFromLong(DType.DURATION_MICROSECONDS, 1800000000L);
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
          -57954590449912415L,
          -2177451343500000L,
          -43013394048980300L,
          -2177483400000000L,
          -2177479895679933L,
          -2177481934610644L,
          1800000000L);
        ColumnVector actual = GpuTimeZoneDB.timeAdd(input, duration,
          ZoneId.of("Asia/Shanghai"))) {
      assertColumnsAreEqual(expected, actual);
    }
  }
  
}
