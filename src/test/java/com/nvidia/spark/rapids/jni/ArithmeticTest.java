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

import ai.rapids.cudf.*;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class ArithmeticTest {

  @Test
  void multiplyAnsiOffWithOverflow() {
    // Integer.MAX_VALUE * 2 = -2 in non-ANSI/non-try mode
    try (
        ColumnVector left = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE);
        ColumnVector right = ColumnVector.fromInts(0, 1, 2);
        ColumnVector expected = ColumnVector.fromInts(0, 1, -2);
        ColumnVector actual = Arithmetic.multiply(left, right, /* isAnsiMode */ false,
            /* isTryMode */ false)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void multiplyAnsiOnWithOverflow() {
    // Integer.MAX_VALUE * 2 throws exception in ANSI mode
    try (
        ColumnVector left = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE);
        ColumnVector right = ColumnVector.fromInts(0, 1, 2)) {
      Arithmetic.multiply(left, right, /* isAnsiMode */ true, /* isTryMode */ false);
      Assertions.fail("Expected ExceptionWithRowIndex due to overflow");
    } catch (ExceptionWithRowIndex e) {
      Assertions.assertEquals(2, e.getRowIndex());
    }
  }

  @Test
  void multiplyTryOnWithOverflow() {
    // Integer.MAX_VALUE * 2 = null in try mode
    try (
        ColumnVector left = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE);
        ColumnVector right = ColumnVector.fromInts(0, 1, 2);
        ColumnVector expected = ColumnVector.fromBoxedInts(0, 1, null);
        ColumnVector actual = Arithmetic.multiply(left, right, /* isAnsiMode */ false,
            /* isTryMode */ true)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void multiplyLongScalar() {
    // Integer.MAX_VALUE * 2 = -2 in non-ANSI/non-try mode
    try (
        ColumnVector left = ColumnVector.fromLongs(0, 1, Long.MAX_VALUE);
        Scalar right = Scalar.fromLong(2);
        ColumnVector expected = ColumnVector.fromBoxedLongs(0L, 2L, null);
        ColumnVector actual = Arithmetic.multiply(left, right, /* isAnsiMode */ false,
            /* isTryMode */ true)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void multiplyScalarInt() {
    // Integer.MAX_VALUE * 2 = -2 in non-ANSI/non-try mode
    try (
        Scalar left = Scalar.fromInt(2);
        ColumnVector right = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE);
        ColumnVector expected = ColumnVector.fromInts(0, 2, -2);
        ColumnVector actual = Arithmetic.multiply(left, right, /* isAnsiMode */ false,
            /* isTryMode */ false)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void multiplyScalarIntAnsi() {
    // Integer.MAX_VALUE * 2 = -2 in non-ANSI/non-try mode
    try (
        Scalar left = Scalar.fromInt(2);
        ColumnVector right = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE)) {
      Arithmetic.multiply(left, right, /* isAnsiMode */ true,
          /* isTryMode */ false);
      Assertions.fail("Expected ExceptionWithRowIndex due to overflow");
    } catch (ExceptionWithRowIndex e) {
      Assertions.assertEquals(2, e.getRowIndex());
    }
  }
}
