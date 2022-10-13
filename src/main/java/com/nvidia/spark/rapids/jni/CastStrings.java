/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
   * Convert a string column to an integer column of a specified type.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param type the type of the return column.
   * @return the converted column.
   */
  public static ColumnVector toInteger(ColumnView cv, boolean ansiMode, DType type) {
    return new ColumnVector(toInteger(cv.getNativeView(), ansiMode, type.getTypeId().getNativeId()));
  }

  /**
   * Convert a string column to an integer column of a specified type.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param type the type of the return column.
   * @return the converted column.
   */
  public static ColumnVector toDecimal(ColumnView cv, boolean ansiMode, int precision, int scale) {
    return new ColumnVector(toDecimal(cv.getNativeView(), ansiMode, precision, scale));
  }

  /**
   * Convert a string column to a float column of a specified type.
   *
   * @param cv the column data to process.
   * @param ansiMode true if invalid data are errors, false if they should be nulls.
   * @param type the type of the return column.
   * @return the converted column.
   */
  public static ColumnVector toFloat(ColumnView cv, boolean ansiMode, DType type) {
    return new ColumnVector(toFloat(cv.getNativeView(), ansiMode, type.getTypeId().getNativeId()));
  }

  private static native long toInteger(long nativeColumnView, boolean ansi_enabled, int dtype);
  private static native long toDecimal(long nativeColumnView, boolean ansi_enabled, int precision, int scale);
  private static native long toFloat(long nativeColumnView, boolean ansi_enabled, int dtype);
}