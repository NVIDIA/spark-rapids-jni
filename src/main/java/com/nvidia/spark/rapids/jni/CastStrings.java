/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

/** Utility class for casting between string columns and native type columns */
public class CastStrings {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Convert a string column to an integer column of a specified type stripping away leading and
   * trailing spaces.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param type the type of the return column.
   * @return the converted column.
   */
  public static ColumnVector toInteger(ColumnView cv, boolean ansiMode, DType type) {
    return toInteger(cv, ansiMode, true, type);
  }

  /**
   * Convert a string column to an integer column of a specified type.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param strip true if leading and trailing spaces should be ignored when parsing.
   * @param type the type of the return column.
   * @return the converted column.
   */
  public static ColumnVector toInteger(ColumnView cv, boolean ansiMode, boolean strip, DType type) {
    return new ColumnVector(toInteger(cv.getNativeView(), ansiMode, strip,
        type.getTypeId().getNativeId()));
  }

  /**
   * Convert a string column to an integer column of a specified type stripping away leading and
   * trailing whitespace.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param precision the output precision.
   * @param scale the output scale.
   * @return the converted column.
   */
  public static ColumnVector toDecimal(ColumnView cv, boolean ansiMode, int precision, int scale) {
    return toDecimal(cv, ansiMode, true, precision, scale);
  }

  /**
   * Convert a string column to an integer column of a specified type.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param strip true if leading and trailing white space should be stripped.
   * @param precision the output precision.
   * @param scale the output scale.
   * @return the converted column.
   */
  public static ColumnVector toDecimal(ColumnView cv, boolean ansiMode, boolean strip,
      int precision, int scale) {
    return new ColumnVector(toDecimal(cv.getNativeView(), ansiMode, strip, precision, scale));
  }

  /**
   * Convert a float column to a formatted string column.
   *
   * @param cv the column data to process
   * @param digits the number of digits to display after the decimal point
   * @return the converted column
   */
  public static ColumnVector fromFloatWithFormat(ColumnView cv, int digits) {
    return new ColumnVector(fromFloatWithFormat(cv.getNativeView(), digits));
  }

  /**
   * Convert a float column to a string column.
   *
   * @param cv the column data to process
   * @return the converted column
   */
  public static ColumnVector fromFloat(ColumnView cv) {
    return new ColumnVector(fromFloat(cv.getNativeView()));
  }

  /**
   * Convert a decimal column to a string column.
   *
   * @param cv the column data to process
   * @return the converted column
   */
  public static ColumnVector fromDecimal(ColumnView cv) {
    return new ColumnVector(fromDecimal(cv.getNativeView()));
  }

  /**
   * Convert a string column to a given floating-point type column.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param type the type of the return column.
   * @return the converted column.
   */
  public static ColumnVector toFloat(ColumnView cv, boolean ansiMode, DType type) {
    return new ColumnVector(toFloat(cv.getNativeView(), ansiMode, type.getTypeId().getNativeId()));
  }


  public static ColumnVector toIntegersWithBase(ColumnView cv, int base,
    boolean ansiEnabled, DType type) {
    return new ColumnVector(toIntegersWithBase(cv.getNativeView(), base, ansiEnabled,
      type.getTypeId().getNativeId()));
  }

  /**
   * Converts an integer column to a string column by converting the underlying integers to the
   * specified base.
   *
   * Note: Right now we only support base 10 and 16. The hexadecimal values will be
   * returned without leading zeros or padding at the end
   * 
   * Example:
   * input = [123, -1, 0, 27, 342718233]
   * s = fromIntegersWithBase(input, 16)
   * s is [ '7B', 'FFFFFFFF', '0', '1B', '146D7719']
   * s = fromIntegersWithBase(input, 10)
   * s is ['123', '-1', '0', '27', '342718233']
   *
   * @param cv The input integer column to be converted.
   * @param base base that we want to convert to (currently only 10/16)
   * @return a new String ColumnVector
   */
  public static ColumnVector fromIntegersWithBase(ColumnView cv, int base) {
    return new ColumnVector(fromIntegersWithBase(cv.getNativeView(), base));
  }

  /**
   * Trims and parses a timestamp string column with time zone suffix to a
   * timestamp column.
   * Use the default time zone if string does not contain time zone.
   *
   * Supports the following formats:
   * `[+-]yyyy*`
   * `[+-]yyyy*-[m]m`
   * `[+-]yyyy*-[m]m-[d]d`
   * `[+-]yyyy*-[m]m-[d]d `
   * `[+-]yyyy*-[m]m-[d]d [h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
   * `[+-]yyyy*-[m]m-[d]dT[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
   * 
   * Supports the following time zones:
   * - Z - Zulu time zone UTC+0
   * - +|-[h]h:[m]m
   * - Region-based zone IDs in the form `area/city`, such as `Europe/Paris`
   *
   * Example:
   * input = [" 2023", "2023-01-01T08:00:00Asia/Shanghai "]
   * ts = toTimestamp(input, "UTC", allowSpecialExpressions = true, ansiEnabled =
   * false)
   * ts is: ['2023-01-01 00:00:00', '2023-01-01T00:00:00']
   * 
   * @param cv                      The input string column to be converted.
   * @param defaultTimeZone         Use the default time zone if string does not
   *                                contain time zone.
   * @param allowSpecialExpressions Whether allow: epoch, now, today, tomorrow
   * @param ansiEnabled             is Ansi mode
   * @return a timestamp column
   * @throws IllegalArgumentException if cv contains invalid value when
   *                                  ansiEnabled is true
   */
  public static ColumnVector toTimestamp(ColumnView cv, String defaultTimeZone,
      boolean allowSpecialExpressions, boolean ansiEnabled) {
    if (defaultTimeZone == null || defaultTimeZone.isEmpty()) {
      throw new IllegalArgumentException("Default time zone can not be empty.");
    }
    return new ColumnVector(toTimestamp(cv.getNativeView(), defaultTimeZone,
        allowSpecialExpressions, ansiEnabled));
  }

  /**
   * Trims and parses a timestamp string column with time zone suffix to a
   * timestamp column.
   * Do not use the time zones in timestamp strings.
   *
   * Supports the following formats:
   * `[+-]yyyy*`
   * `[+-]yyyy*-[m]m`
   * `[+-]yyyy*-[m]m-[d]d`
   * `[+-]yyyy*-[m]m-[d]d `
   * `[+-]yyyy*-[m]m-[d]d [h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
   * `[+-]yyyy*-[m]m-[d]dT[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
   * 
   * Supports the following time zones:
   * - Z - Zulu time zone UTC+0
   * - +|-[h]h:[m]m
   * - Region-based zone IDs in the form `area/city`, such as `Europe/Paris`
   *
   * Example:
   * input = [" 2023", "2023-01-01T08:00:00Asia/Shanghai "]
   * ts = toTimestampWithoutTimeZone(input, allowTimeZone = true,
   * allowSpecialExpressions = true, ansiEnabled = false)
   * ts is: ['2023-01-01 00:00:00', '2023-01-01T08:00:00']
   * 
   * @param cv                      The input string column to be converted.
   * @param allow_time_zone         whether allow time zone in the timestamp
   *                                string. e.g.:
   *                                1991-04-14T02:00:00Asia/Shanghai is invalid
   *                                when do not allow time zone.
   * @param allowSpecialExpressions Whether allow: epoch, now, today, tomorrow
   * @param ansiEnabled             is Ansi mode
   * @return a timestamp column
   * @throws IllegalArgumentException if cv contains invalid value when
   *                                  ansiEnabled is true
   */
  public static ColumnVector toTimestampWithoutTimeZone(ColumnView cv, boolean allowTimeZone,
      boolean allowSpecialExpressions, boolean ansiEnabled) {
    return new ColumnVector(toTimestampWithoutTimeZone(cv.getNativeView(), allowTimeZone,
        allowSpecialExpressions, ansiEnabled));
  }

  private static native long toInteger(long nativeColumnView, boolean ansi_enabled, boolean strip,
      int dtype);
  private static native long toDecimal(long nativeColumnView, boolean ansi_enabled, boolean strip,
      int precision, int scale);
  private static native long toFloat(long nativeColumnView, boolean ansi_enabled, int dtype);
  private static native long fromDecimal(long nativeColumnView);
  private static native long fromFloatWithFormat(long nativeColumnView, int digits);
  private static native long fromFloat(long nativeColumnView);
  private static native long toIntegersWithBase(long nativeColumnView, int base,
    boolean ansiEnabled, int dtype);
  private static native long fromIntegersWithBase(long nativeColumnView, int base);
  private static native long toTimestamp(long nativeColumnView, String defaultTimeZone,
      boolean allowSpecialExpressions, boolean ansiEnabled);
  private static native long toTimestampWithoutTimeZone(long nativeColumnView,
      boolean allowTimeZone, boolean allowSpecialExpressions, boolean ansiEnabled);
}
