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

  public static ColumnVector fromLongToBinary(ColumnView cv) {
    return new ColumnVector(fromLongToBinary(cv.getNativeView()));
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
   * Trims and parses strings to intermediate result.
   * This is the first phase of casting string with timezone to timestamp.
   * Intermediate result is a struct column with 7 sub-columns:
   * - Parse Result type: 0 Success, 1 invalid e.g. year is 7 digits 1234567
   * - UTC timestamp
   * - Just time in the ts string. If true, then UTC ts is at year 1970-01-01
   * - Timezone type: 0 unspecified, 1 fixed type, 2 other type, 3 invalid
   * - Timezone offset for fixed type, only applies to fixed type
   * - Timezone is DST, only applies to other type
   * - Timezone index to `GpuTimeZoneDB.transitions` table
   *
   * Refer to: https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
   * org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L544
   *
   * Use the default timezone if timestamp string does not contain timezone.
   *
   * Supports the following formats:
   * `[+-]yyyy[y][y]`
   * `[+-]yyyy[y][y]-m[m]`
   * `[+-]yyyy[y][y]-m[m]-d[d]`
   * `[+-]yyyy[y][y]-m[m]-d[d] `
   * `[+-]yyyy[y][y]-m[m]-d[d] [h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
   * `[+-]yyyy[y][y]-m[m]-d[d]T[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
   *
   * The max length of yyyy[y][y] is 6 digits.
   *
   * Supports the following zone id forms:
   * - Z - Zulu timezone UTC+0
   * - +|-[h]h:[m]m
   * - A short id, see
   * https://docs.oracle.com/javase/8/docs/api/java/time/ZoneId.html#SHORT_IDS
   * - An id with one of the prefixes UTC+, UTC-, GMT+, GMT-, UT+ or UT-,
   * and a suffix in the formats:
   * - +|-h[h]
   * - +|-hh[:]mm
   * - +|-hh:mm:ss
   * - +|-hhmmss
   * - Region-based zone IDs in the form `area/city`, such as `Europe/Paris`
   *
   * @param input                The input string column to be converted.
   * @param defaultTimeZoneIndex The default timezone index to transition table.
   * @param timeZoneInfo         The timezone information to be used for the
   *                             conversion, including
   *                             all the available timezone names, name to
   *                             transition index
   *                             mapping and name to `is_DST` mapping.
   * @return a struct column constains 7 columns described above.
   */
  static ColumnVector parseTimestampStrings(
      ColumnView input, int defaultTimeZoneIndex,
      ColumnView timeZoneInfo) {

    return new ColumnVector(parseTimestampStrings(
        input.getNativeView(), defaultTimeZoneIndex, timeZoneInfo.getNativeView()));
  }

  private static native long toInteger(long nativeColumnView, boolean ansi_enabled, boolean strip,
      int dtype);
  private static native long toDecimal(long nativeColumnView, boolean ansi_enabled, boolean strip,
      int precision, int scale);
  private static native long toFloat(long nativeColumnView, boolean ansi_enabled, int dtype);
  private static native long fromDecimal(long nativeColumnView);
  private static native long fromFloatWithFormat(long nativeColumnView, int digits);
  private static native long fromFloat(long nativeColumnView);
  private static native long fromLongToBinary(long nativeColumnView);
  private static native long toIntegersWithBase(long nativeColumnView, int base,
    boolean ansiEnabled, int dtype);
  private static native long fromIntegersWithBase(long nativeColumnView, int base);

  private static native long parseTimestampStrings(long input, int defaultTimezoneIndex,
      long timeZoneInfo);
}
