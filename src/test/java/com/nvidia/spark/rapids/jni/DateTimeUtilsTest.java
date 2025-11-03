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

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

import java.time.Instant;
import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.util.Random;

import org.junit.jupiter.api.Test;

import ai.rapids.cudf.ColumnVector;

public class DateTimeUtilsTest {
  @Test
  void rebaseDaysToJulianTest() {
    try (
        ColumnVector input = ColumnVector.timestampDaysFromBoxedInts(-719162, -354285, null,
            -141714, -141438, -141437,
            null, null,
            -141432, -141427, -31463, -31453, -1, 0, 18335);
        ColumnVector expected = ColumnVector.timestampDaysFromBoxedInts(-719164, -354280, null,
            -141704, -141428, -141427,
            null, null,
            -141427, -141427, -31463, -31453, -1, 0, 18335);
        ColumnVector result = DateTimeUtils.rebaseGregorianToJulian(input)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void rebaseDaysToGregorianTest() {
    try (
        ColumnVector input = ColumnVector.timestampDaysFromBoxedInts(-719164, -354280, null,
            -141704, -141428, -141427,
            null, null,
            -141427, -141427, -31463, -31453, -1, 0, 18335);
        ColumnVector expected = ColumnVector.timestampDaysFromBoxedInts(-719162, -354285, null,
            -141714, -141438, -141427,
            null, null,
            -141427, -141427, -31463, -31453, -1, 0, 18335);
        ColumnVector result = DateTimeUtils.rebaseJulianToGregorian(input)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void rebaseMicroToJulian() {
    try (
        ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(-62135593076345679L,
            -30610213078876544L,
            null,
            -12244061221876544L,
            -12220243200000000L,
            -12219639001448163L,
            -12219292799000001L,
            -45446999900L,
            1L,
            null,
            1584178381500000L);
        ColumnVector expected =
            ColumnVector.timestampMicroSecondsFromBoxedLongs(-62135765876345679L,
                -30609781078876544L,
                null,
                -12243197221876544L,
                -12219379200000000L,
                -12219207001448163L,
                -12219292799000001L,
                -45446999900L,
                1L,
                null,
                1584178381500000L);
        ColumnVector result = DateTimeUtils.rebaseGregorianToJulian(input)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void rebaseMicroToGregorian() {
    try (
        ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(-62135765876345679L,
            -30609781078876544L,
            null,
            -12243197221876544L,
            -12219379200000000L,
            -12219207001448163L,
            -12219292799000001L,
            -45446999900L,
            1L,
            null,
            1584178381500000L);
        ColumnVector expected =
            ColumnVector.timestampMicroSecondsFromBoxedLongs(-62135593076345679L,
                -30610213078876544L,
                null,
                -12244061221876544L,
                -12220243200000000L,
                -12219207001448163L,
                -12219292799000001L,
                -45446999900L,
                1L,
                null,
                1584178381500000L);
        ColumnVector result = DateTimeUtils.rebaseJulianToGregorian(input)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void truncateDateTest() {
    try (ColumnVector input = ColumnVector.timestampDaysFromBoxedInts(-31463, -31453, null, 0, 18335);
         ColumnVector format = ColumnVector.fromStrings("YEAR", "MONTH", "WEEK", "QUARTER", "YY");
         ColumnVector expected = ColumnVector.timestampDaysFromBoxedInts(-31776, -31472, null, 0, 18262);
         ColumnVector result = DateTimeUtils.truncate(input, format)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void truncateTimestampTest() {
    try (
        ColumnVector input = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            -12219292799000001L,
            -45446999900L,
            1L,
            null,
            1584178381500000L);
        ColumnVector format = ColumnVector.fromStrings("YEAR", "HOUR", "WEEK", "QUARTER", "SECOND");
        ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(
            -12244089600000000L,
            -46800000000L,
            -259200000000L,
            null,
            1584178381000000L);
        ColumnVector result = DateTimeUtils.truncate(input, format)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  // =========== CPU date utils for validation, block begin ===========
  // copied from Iceberg org.apache.iceberg.util.DateTimeUtil
  private static final OffsetDateTime EPOCH = Instant.ofEpochSecond(0).atOffset(ZoneOffset.UTC);
  private static final LocalDate EPOCH_DAY = EPOCH.toLocalDate();
  private static final long MICROS_PER_SECOND = 1_000_000L;

  private static int daysToYearsOnCpu(int days) {
    return convertDaysOnCpu(days, ChronoUnit.YEARS);
  }

  private static int daysToMonthsOnCpu(int days) {
    return convertDaysOnCpu(days, ChronoUnit.MONTHS);
  }

  private static int convertDaysOnCpu(int days, ChronoUnit granularity) {
    if (days >= 0) {
      LocalDate date = EPOCH_DAY.plusDays(days);
      return (int) granularity.between(EPOCH_DAY, date);
    } else {
      // add 1 day to the value to account for the case where there is exactly 1 unit
      // between the
      // date and epoch because the result will always be decremented.
      LocalDate date = EPOCH_DAY.plusDays(days + 1);
      return (int) granularity.between(EPOCH_DAY, date) - 1;
    }
  }

  private static int microsToYearsOnCpu(long micros) {
    return convertMicros(micros, ChronoUnit.YEARS);
  }

  private static int microsToMonthsOnCpu(long micros) {
    return convertMicros(micros, ChronoUnit.MONTHS);
  }

  private static int microsToDaysOnCpu(long micros) {
    return convertMicros(micros, ChronoUnit.DAYS);
  }

  private static int microsToHoursOnCpu(long micros) {
    return convertMicros(micros, ChronoUnit.HOURS);
  }

  private static int convertMicros(long micros, ChronoUnit granularity) {
    if (micros >= 0) {
      long epochSecond = Math.floorDiv(micros, MICROS_PER_SECOND);
      long nanoAdjustment = Math.floorMod(micros, MICROS_PER_SECOND) * 1000;
      return (int) granularity.between(EPOCH, toOffsetDateTime(epochSecond, nanoAdjustment));
    } else {
      // add 1 micro to the value to account for the case where there is exactly 1
      // unit between
      // the timestamp and epoch because the result will always be decremented.
      long epochSecond = Math.floorDiv(micros, MICROS_PER_SECOND);
      long nanoAdjustment = Math.floorMod(micros + 1, MICROS_PER_SECOND) * 1000;
      return (int) granularity.between(EPOCH, toOffsetDateTime(epochSecond, nanoAdjustment)) - 1;
    }
  }

  private static OffsetDateTime toOffsetDateTime(long epochSecond, long nanoAdjustment) {
    return Instant.ofEpochSecond(epochSecond, nanoAdjustment).atOffset(ZoneOffset.UTC);
  }

  // =========== CPU date utils for validation: block end ===========

  @Test
  void computeYearDiffTest() {
    // basic test
    try (
        ColumnVector input = ColumnVector.timestampDaysFromBoxedInts(
            -50,
            -500,
            null,
            50,
            500,
            null);
        ColumnVector expected = ColumnVector.fromBoxedInts(
            -1,
            -2,
            null,
            0,
            1,
            null);
        ColumnVector result = DateTimeUtils.computeYearDiff(input)) {
      assertColumnsAreEqual(expected, result);
    }

    // random test, use current day as seed
    long seed = LocalDate.now().toEpochDay();
    Random random = new Random(seed);
    int numRows = 1024;
    int[] days = new int[numRows];
    long[] micros = new long[numRows];
    for (int i = 0; i < numRows; ++i) {
      days[i] = random.nextInt();
      micros[i] = random.nextLong();
    }
    int[] expectedValues1 = new int[numRows];
    int[] expectedValues2 = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      expectedValues1[i] = daysToYearsOnCpu(days[i]);
      expectedValues2[i] = microsToYearsOnCpu(micros[i]);
    }
    try (
        ColumnVector input1 = ColumnVector.daysFromInts(days);
        ColumnVector expected1 = ColumnVector.fromInts(expectedValues1);
        ColumnVector result1 = DateTimeUtils.computeYearDiff(input1);
        ColumnVector input2 = ColumnVector.timestampMicroSecondsFromLongs(micros);
        ColumnVector expected2 = ColumnVector.fromInts(expectedValues2);
        ColumnVector result2 = DateTimeUtils.computeYearDiff(input2)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
    }
  }

  @Test
  void computeMonthDiffTest() {
    // basic test
    try (
        ColumnVector input = ColumnVector.timestampDaysFromBoxedInts(
            -50,
            -500,
            null,
            50,
            500,
            null);
        ColumnVector expected = ColumnVector.fromBoxedInts(
            -2,
            -17,
            null,
            1,
            16,
            null);
        ColumnVector result = DateTimeUtils.computeMonthDiff(input)) {
      assertColumnsAreEqual(expected, result);
    }

    // random test, use current day as seed
    long seed = LocalDate.now().toEpochDay();
    Random random = new Random(seed);
    int numRows = 1024;
    int[] days = new int[numRows];
    long[] micros = new long[numRows];
    for (int i = 0; i < numRows; ++i) {
      days[i] = random.nextInt();
      micros[i] = random.nextLong();
    }
    int[] expectedValues1 = new int[numRows];
    int[] expectedValues2 = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      expectedValues1[i] = daysToMonthsOnCpu(days[i]);
      expectedValues2[i] = microsToMonthsOnCpu(micros[i]);
    }
    try (
        ColumnVector input1 = ColumnVector.daysFromInts(days);
        ColumnVector expected1 = ColumnVector.fromInts(expectedValues1);
        ColumnVector result1 = DateTimeUtils.computeMonthDiff(input1);
        ColumnVector input2 = ColumnVector.timestampMicroSecondsFromLongs(micros);
        ColumnVector expected2 = ColumnVector.fromInts(expectedValues2);
        ColumnVector result2 = DateTimeUtils.computeMonthDiff(input2)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
    }
  }

  @Test
  void computeDaysDiffTest() {
    // random test, use current day as seed
    long seed = LocalDate.now().toEpochDay();
    Random random = new Random(seed);
    int numRows = 1024;
    int[] days = new int[numRows];
    long[] micros = new long[numRows];
    for (int i = 0; i < numRows; ++i) {
      days[i] = random.nextInt();
      micros[i] = random.nextLong();
    }
    int[] expectedValues1 = new int[numRows];
    int[] expectedValues2 = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      expectedValues1[i] = days[i]; // do nothing, days to days
      expectedValues2[i] = microsToDaysOnCpu(micros[i]);
    }
    try (
        ColumnVector input1 = ColumnVector.daysFromInts(days);
        ColumnVector input2 = ColumnVector.timestampMicroSecondsFromLongs(micros);
        ColumnVector expected1 = ColumnVector.fromInts(expectedValues1);
        ColumnVector expected2 = ColumnVector.fromInts(expectedValues2);
        ColumnVector resultOfDate = DateTimeUtils.computeDayDiff(input1);
        ColumnVector resultOfMicros = DateTimeUtils.computeDayDiff(input2)) {
      assertColumnsAreEqual(expected1, resultOfDate);
      assertColumnsAreEqual(expected2, resultOfMicros);
    }
  }

  @Test
  void computeHoursDiffTest() {
    // random test, use current day as seed
    long seed = LocalDate.now().toEpochDay();
    Random random = new Random(seed);
    int numRows = 1024;
    long[] micros = new long[numRows];
    for (int i = 0; i < numRows; ++i) {
      micros[i] = random.nextLong();
    }
    int[] expectedValues = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      expectedValues[i] = microsToHoursOnCpu(micros[i]);
    }
    try (
        ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(micros);
        ColumnVector expected = ColumnVector.fromInts(expectedValues);
        ColumnVector result = DateTimeUtils.computeHourDiff(input)) {
      assertColumnsAreEqual(expected, result);
    }
  }
}
