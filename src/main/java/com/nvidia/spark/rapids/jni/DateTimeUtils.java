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

import ai.rapids.cudf.*;

/**
 * Utility class for converting between column major and row major data
 */
public class DateTimeUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Convert the given timestamps as a number of days or microseconds since the epoch instant
   * 1970-01-01T00:00:00Z to a local date-time in Proleptic Gregorian calendar, reinterpreting
   * the result as in Julian calendar, then compute the number of days or microseconds since the
   * epoch from that Julian local date-time.
   * <p>
   * This is to match with Apache Spark's `localRebaseGregorianToJulianDays` and
   * `rebaseGregorianToJulianMicros` functions with timezone fixed to UTC.
   *
   * @param input The input column
   * @return A new column with the rebase applied
   */
  public static ColumnVector rebaseGregorianToJulian(ColumnView input) {
    return new ColumnVector(rebaseGregorianToJulian(input.getNativeView()));
  }

  /**
   * Convert the given timestamps as a number of days or microseconds since the epoch instant
   * 1970-01-01T00:00:00Z to a local date-time in Julian calendar, reinterpreting the result
   * as in Proleptic Gregorian calendar, then compute the number of days or microseconds since the
   * epoch from that Gregorian local date-time.
   * <p>
   * This is to match with Apache Spark's `localRebaseJulianToGregorianDays` and
   * `rebaseJulianToGregorianMicros` functions with timezone fixed to UTC.
   *
   * @param input The input column
   * @return A new column with the rebase applied
   */
  public static ColumnVector rebaseJulianToGregorian(ColumnView input) {
    return new ColumnVector(rebaseJulianToGregorian(input.getNativeView()));
  }

  /**
   * Truncate the given date or timestamp to the unit specified by the format string.
   * <p>
   * The input date/time must be of type TIMESTAMP_DAYS or TIMESTAMP_MICROSECONDS, and the format
   * be of type STRING. In addition, the format strings are case-insensitive.
   * <p>
   * For TIMESTAMP_DAYS, the valid format are:<br>
   *  - {@code "YEAR", "YYYY", "YY"}: truncate to the first date of the year.<br>
   *  - {@code "QUARTER"}: truncate to the first date of the quarter.<br>
   *  - {@code "MONTH", "MM", "MON"}: truncate to the first date of the month.<br>
   *  - {@code "WEEK"}: truncate to the Monday of the week.<br>
   * <br/>
   * For TIMESTAMP_MICROSECONDS, the valid format are:<br>
   *  - {@code "YEAR", "YYYY", "YY"}: truncate to the first date of the year.<br>
   *  - {@code "QUARTER"}: truncate to the first date of the quarter.<br>
   *  - {@code "MONTH", "MM", "MON"}: truncate to the first date of the month.<br>
   *  - {@code "WEEK"}: truncate to the Monday of the week.<br>
   *  - {@code "DAY", "DD"}: zero out the time part.<br>
   *  - {@code "HOUR"}: zero out the minute and second with fraction part.<br>
   *  - {@code "MINUTE"}: zero out the second with fraction part.<br>
   *  - {@code "SECOND"}: zero out the second fraction part.<br>
   *  - {@code "MILLISECOND"}: zero out the microseconds.<br>
   *  - {@code "MICROSECOND"}: keep everything.<br>
   *
   * @param datetime The input date/time
   * @param format The time component to truncate to
   * @return The truncated date/time
   */
  public static ColumnVector truncate(ColumnView datetime, ColumnView format) {
    return new ColumnVector(truncateWithColumnFormat(datetime.getNativeView(),
        format.getNativeView()));
  }

  /**
   * Truncate the given date or timestamp to the unit specified by the format string.
   * <p>
   * This function is similar to {@link #truncate(ColumnView, ColumnView)} but the input format
   * is a string literal instead of a column.
   *
   * @param datetime The input date/time
   * @param format The time component to truncate to
   * @return The truncated date/time
   */
  public static ColumnVector truncate(ColumnView datetime, String format) {
    return new ColumnVector(truncateWithScalarFormat(datetime.getNativeView(), format));
  }

  private static void checkTimestampOrDate(ColumnView input) {
    DType type = input.getType();
    if (type.getTypeId() != DType.DTypeEnum.TIMESTAMP_MICROSECONDS &&
        type.getTypeId() != DType.DTypeEnum.TIMESTAMP_DAYS) {
      throw new IllegalArgumentException("Input column must be of type " +
          "TIMESTAMP_MICROSECONDS or TIMESTAMP_DAYS");
    }
  }

  /**
   * Calculates the difference in years between the epoch year (1970) and the
   * given date/timestamp column. E.g.: for date '1971-01-01', the result would be
   * 1: (1 year after epoch year)
   *
   * @param input The input date/timestamp column.
   * @return A column of type INT32 containing the year differences from epoch.
   */
  public static ColumnVector computeYearDiff(ColumnView input) {
    checkTimestampOrDate(input);
    return new ColumnVector(computeYearDiff(input.getNativeView()));
  }

  /**
   * Calculates the difference in months between the epoch month (1970-01) and the
   * given date/timestamp column. E.g.: for date '1971-02-01', the result would be
   * 13: (1 year and 1 month after epoch month)
   *
   * @param input The input date/timestamp column.
   * @return A column of type INT32 containing the month differences from epoch.
   */
  public static ColumnVector computeMonthDiff(ColumnView input) {
    checkTimestampOrDate(input);
    return new ColumnVector(computeMonthDiff(input.getNativeView()));
  }

  /**
   * Calculates the difference in days between the epoch month (1970-01) and the
   * given date/timestamp column. E.g.: for date '1970-01-21', the result would be
   * 20: (20 days after epoch day)
   *
   * @param input The input date/timestamp column.
   * @return A column of type INT32 containing the day differences from epoch.
   */
  public static ColumnVector computeDayDiff(ColumnView input) {
    checkTimestampOrDate(input);
    return new ColumnVector(computeDayDiff(input.getNativeView()));
  }

  /**
   * Calculates the difference in hours between the epoch hour
   * (1970-01-01T00:00:00) and the given timestamp column.
   * E.g.: for timestamp '1970-01-01 01:00:00', the result would be 1
   * (1 hour after epoch hour)
   *
   * @param timestamp The input timestamp column.
   * @return A column of type INT32 containing the hour differences from epoch.
   */
  public static ColumnVector computeHourDiff(ColumnView input) {
    if (input.getType().getTypeId() != DType.DTypeEnum.TIMESTAMP_MICROSECONDS) {
      throw new IllegalArgumentException("Input column must be of type TIMESTAMP_MICROSECONDS");
    }
    return new ColumnVector(computeHourDiff(input.getNativeView()));
  }

  private static native long rebaseGregorianToJulian(long nativeHandle);

  private static native long rebaseJulianToGregorian(long nativeHandle);

  private static native long truncateWithColumnFormat(long datetimeHandle, long formatHandle);

  private static native long truncateWithScalarFormat(long datetimeHandle, String format);

  private static native long computeYearDiff(long nativeHandle);

  private static native long computeMonthDiff(long nativeHandle);

  private static native long computeDayDiff(long nativeHandle);

  private static native long computeHourDiff(long nativeHandle);
}
