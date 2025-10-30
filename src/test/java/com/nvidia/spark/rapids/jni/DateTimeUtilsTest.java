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

    // random test
    // use current day as seed
    long seed = LocalDate.now().toEpochDay();
    Random random = new Random(seed);
    int numRows = 1024;
    int[] days = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      days[i] = random.nextInt();
    }
    int[] expectedValues = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      expectedValues[i] = daysToYearsOnCpu(days[i]);
    }
    try (
        ColumnVector input = ColumnVector.daysFromInts(days);
        ColumnVector expected = ColumnVector.fromInts(expectedValues);
        ColumnVector result = DateTimeUtils.computeYearDiff(input)) {
      assertColumnsAreEqual(expected, result);
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

    // random test
    // use current day as seed
    long seed = LocalDate.now().toEpochDay();
    Random random = new Random(seed);
    int numRows = 1024;
    int[] days = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      days[i] = random.nextInt();
    }
    int[] expectedValues = new int[numRows];
    for (int i = 0; i < numRows; ++i) {
      expectedValues[i] = daysToMonthsOnCpu(days[i]);
    }
    try (
        ColumnVector input = ColumnVector.daysFromInts(days);
        ColumnVector expected = ColumnVector.fromInts(expectedValues);
        ColumnVector result = DateTimeUtils.computeMonthDiff(input)) {
      assertColumnsAreEqual(expected, result);
    }
  }
}
