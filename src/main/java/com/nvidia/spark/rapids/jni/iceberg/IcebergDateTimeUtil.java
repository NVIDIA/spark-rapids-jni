/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.jni.iceberg;

import ai.rapids.cudf.*;

/**
 * Gpu implementations for `org.apache.iceberg.util.DateTimeUtil` functions
 */
public class IcebergDateTimeUtil {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static void checkTimestampOrDateType(ColumnView input) {
    DType type = input.getType();
    if (type.getTypeId() != DType.DTypeEnum.TIMESTAMP_MICROSECONDS &&
        type.getTypeId() != DType.DTypeEnum.TIMESTAMP_DAYS) {
      throw new IllegalArgumentException("Input column must be of type " +
          "TIMESTAMP_MICROSECONDS or TIMESTAMP_DAYS");
    }
  }

  private static void checkTimestampType(ColumnView input) {
    DType type = input.getType();
    if (type.getTypeId() != DType.DTypeEnum.TIMESTAMP_MICROSECONDS) {
      throw new IllegalArgumentException("Input column must be of type " +
          "TIMESTAMP_MICROSECONDS");
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
  public static ColumnVector toYears(ColumnView input) {
    checkTimestampOrDateType(input);
    return new ColumnVector(toYears(input.getNativeView()));
  }

  /**
   * Calculates the difference in months between the epoch month (1970-01) and the
   * given date/timestamp column. E.g.: for date '1971-02-01', the result would be
   * 13: (1 year and 1 month after epoch month)
   *
   * @param input The input date/timestamp column.
   * @return A column of type INT32 containing the month differences from epoch.
   */
  public static ColumnVector toMonths(ColumnView input) {
    checkTimestampOrDateType(input);
    return new ColumnVector(toMonths(input.getNativeView()));
  }

  /**
   * Calculates the difference in days between the epoch month (1970-01) and the
   * given date/timestamp column. E.g.: for date '1970-01-21', the result would be
   * 20: (20 days after epoch day)
   *
   * @param input The input date/timestamp column.
   * @return A column of type Date.
   */
  public static ColumnVector toDays(ColumnView input) {
    checkTimestampOrDateType(input);
    return new ColumnVector(toDays(input.getNativeView()));
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
  public static ColumnVector toHours(ColumnView input) {
    checkTimestampType(input);
    return new ColumnVector(toHours(input.getNativeView()));
  }

  private static native long toYears(long nativeHandle);

  private static native long toMonths(long nativeHandle);

  private static native long toDays(long nativeHandle);

  private static native long toHours(long nativeHandle);
}
