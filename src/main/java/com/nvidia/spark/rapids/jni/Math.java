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

public class Math {

  /**
   * Computes the element-wise multiplication of two ColumnVectors.
   * If the types of the two vectors do not match, an IllegalArgumentException is
   * thrown. If the row counts of the two vectors do not match, an
   * IllegalArgumentException is thrown. If the operation results in an overflow
   * in ANSI mode, an ExceptionAtRow is thrown.
   *
   * @param left       left column vector
   * @param right      right column vector
   * @param isAnsiMode is ANSI mode enabled
   *                   (if true, overflow will throw an ExceptionAtRow)
   *                   (if false, overflow will result in wrong values, e.g.
   *                   Integer.MAX_VALUE * 2 = -2)
   * @return a new ColumnVector containing the result of multiplying the two input
   *         vectors
   * @throws ExceptionAtRow if an overflow occurs in ANSI mode
   */
  public static ColumnVector multiply(ColumnVector left, ColumnVector right, boolean isAnsiMode) {
    if (left.getType() != right.getType()) {
      throw new IllegalArgumentException("Column types do not match: " + left.getType()
          + " vs " + right.getType());
    }
    if (left.getRowCount() != right.getRowCount()) {
      throw new IllegalArgumentException("Row counts do not match: " + left.getRowCount()
          + " vs " + right.getRowCount());
    }
    return new ColumnVector(multiply(
        left.getNativeView(), true, right.getNativeView(), true, isAnsiMode));
  }

  private static native long multiply(long leftHandle, boolean isLeftCv, long rightHandle,
      boolean isRightCv, boolean isAnsiMode);
}
