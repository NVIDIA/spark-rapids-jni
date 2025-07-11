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
package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.DType;

public class Arithmetic {

  private static void checkMultiply(DType leftType, DType rightType, 
      long leftRowCount, long rightRowCount, boolean isAnsiMode, boolean isTryMode) {
    
    if (!(leftType.getTypeId() == DType.DTypeEnum.INT8 ||
          leftType.getTypeId() == DType.DTypeEnum.INT16 ||
          leftType.getTypeId() == DType.DTypeEnum.INT32 ||
          leftType.getTypeId() == DType.DTypeEnum.INT64 ||
          leftType.getTypeId() == DType.DTypeEnum.FLOAT32 ||
          leftType.getTypeId() == DType.DTypeEnum.FLOAT64)) {
      throw new IllegalArgumentException("Multiplication types must be signed integral or float, " +
          "but get type: " + leftType);
    }

    if (leftType != rightType) {
      throw new IllegalArgumentException("Column types do not match: " + leftType
          + " vs " + rightType);
    }

    if (leftRowCount != rightRowCount) {
      throw new IllegalArgumentException("Row counts do not match: " + leftRowCount
          + " vs " + rightRowCount);
    }

    if (isAnsiMode && isTryMode) {
      throw new IllegalArgumentException("isAnsiMode and isTryMode cannot both be true");
    }
  }

  /**
   * Computes multiplication on two ColumnVectors.
   * If the types of the two vectors do not match, an IllegalArgumentException is
   * thrown. If the row counts of the two inputs do not match, an
   * IllegalArgumentException is thrown. If there is any overflow
   * in ANSI mode, an ExceptionWithRowIndex is thrown.
   * Only supports Spark data types: byte, short, integer, long, float32 and float64.
   * 
   * E.g.:
   * Integer.MAX_VALUE * 2 = -2 in regular mode, the result is wrong(overflow occurs).
   * Integer.MAX_VALUE * 2 = null in try mode
   * Integer.MAX_VALUE * 2 = throws exception in ansi mode
   * 
   * @param left       left input
   * @param right      right input
   * @param isAnsiMode is ANSI mode enabled
   * @param isTryMode  if true, set null when overflow occurs
   * @return a new ColumnVector containing the result of multiplying
   *
   * @throws ExceptionWithRowIndex if has any overflow in ANSI mode
   */
  public static ColumnVector multiply(ColumnVector left, ColumnVector right, boolean isAnsiMode,
      boolean isTryMode) {
    checkMultiply(left.getType(), right.getType(), left.getRowCount(),
        right.getRowCount(), isAnsiMode, isTryMode);

    return new ColumnVector(multiply(
        left.getNativeView(), true, right.getNativeView(), true, isAnsiMode,
        isTryMode));
  }

  /**
   * Computes multiplication on two ColumnVectors.
   * If the types of the two vectors do not match, an IllegalArgumentException is
   * thrown. If the row counts of the two inputs do not match, an
   * IllegalArgumentException is thrown. If there is any overflow
   * in ANSI mode, an ExceptionWithRowIndex is thrown.
   * Only supports Spark data types: byte, short, integer, long, float32 and float64.
   * 
   * E.g.:
   * Integer.MAX_VALUE * 2 = -2 in regular mode, the result is wrong(overflow occurs).
   * Integer.MAX_VALUE * 2 = null in try mode
   * Integer.MAX_VALUE * 2 = throws exception in ansi mode
   * 
   * @param left       left input
   * @param right      right input
   * @param isAnsiMode is ANSI mode enabled
   * @param isTryMode  if true, set null when overflow occurs
   * @return a new ColumnVector containing the result of multiplying
   *
   * @throws ExceptionWithRowIndex if has any overflow in ANSI mode
   */
  public static ColumnVector multiply(ColumnVector left, Scalar right, boolean isAnsiMode,
      boolean isTryMode) {
    checkMultiply(left.getType(), right.getType(), left.getRowCount(),
        left.getRowCount(), isAnsiMode, isTryMode);
    return new ColumnVector(multiply(
        left.getNativeView(), true, right.getScalarHandle(), false, isAnsiMode,
        isTryMode));
  }

  /**
   * Computes multiplication on two ColumnVectors.
   * If the types of the two vectors do not match, an IllegalArgumentException is
   * thrown. If the row counts of the two inputs do not match, an
   * IllegalArgumentException is thrown. If there is any overflow
   * in ANSI mode, an ExceptionWithRowIndex is thrown.
   * Only supports Spark data types: byte, short, integer, long, float32 and float64.
   * 
   * E.g.:
   * Integer.MAX_VALUE * 2 = -2 in regular mode, the result is wrong(overflow occurs).
   * Integer.MAX_VALUE * 2 = null in try mode
   * Integer.MAX_VALUE * 2 = throws exception in ansi mode
   * 
   * @param left       left input
   * @param right      right input
   * @param isAnsiMode is ANSI mode enabled
   * @param isTryMode  if true, set null when overflow occurs
   * @return a new ColumnVector containing the result of multiplying
   *
   * @throws ExceptionWithRowIndex if has any overflow in ANSI mode
   */
  public static ColumnVector multiply(Scalar left, ColumnVector right, boolean isAnsiMode,
      boolean isTryMode) {
    checkMultiply(left.getType(), right.getType(), right.getRowCount(),
        right.getRowCount(), isAnsiMode, isTryMode);
    return new ColumnVector(multiply(
        left.getScalarHandle(), false, right.getNativeView(), true, isAnsiMode,
        isTryMode));
  }

  private static native long multiply(long leftHandle, boolean isLeftCv, long rightHandle,
      boolean isRightCv, boolean isAnsiMode, boolean isTryMode);
}
