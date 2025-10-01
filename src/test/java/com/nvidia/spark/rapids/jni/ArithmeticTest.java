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
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.Scalar;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import com.nvidia.spark.rapids.jni.RoundMode;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import java.math.BigInteger;

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

  @Test
  void roundFloatsHalfUp() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(1.234f, 25.66f, null, 154.9f, 2346f);
         ColumnVector result1 = Arithmetic.round(v, 0, RoundMode.HALF_UP);
         ColumnVector result2 = Arithmetic.round(v, 1, RoundMode.HALF_UP);
         ColumnVector result3 = Arithmetic.round(v, -1, RoundMode.HALF_UP);
         ColumnVector expected1 = ColumnVector.fromBoxedFloats(1f, 26f, null, 155f, 2346f);
         ColumnVector expected2 = ColumnVector.fromBoxedFloats(1.2f, 25.7f, null, 154.9f, 2346f);
         ColumnVector expected3 = ColumnVector.fromBoxedFloats(0f, 30f, null, 150f, 2350f)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
      assertColumnsAreEqual(expected3, result3);
    }
  }

  @Test
  void roundFloatsHalfEven() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(1.5f, 2.5f, 1.35f, null, 1.25f, 15f, 25f);
         ColumnVector result1 = Arithmetic.round(v, RoundMode.HALF_EVEN);
         ColumnVector result2 = Arithmetic.round(v, 1, RoundMode.HALF_EVEN);
         ColumnVector result3 = Arithmetic.round(v, -1, RoundMode.HALF_EVEN);
         ColumnVector expected1 = ColumnVector.fromBoxedFloats(2f, 2f, 1f, null, 1f, 15f, 25f);
         ColumnVector expected2 = ColumnVector.fromBoxedFloats(1.5f, 2.5f, 1.4f, null, 1.2f, 15f, 25f);
         ColumnVector expected3 = ColumnVector.fromBoxedFloats(0f, 0f, 0f, null, 0f, 20f, 20f)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
      assertColumnsAreEqual(expected3, result3);
    }
  }

  @Test
  void roundIntsHalfUp() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(12, 135, 160, -1454, null, -1500, -140, -150);
         ColumnVector result1 = Arithmetic.round(v, 2, RoundMode.HALF_UP);
         ColumnVector result2 = Arithmetic.round(v, -2, RoundMode.HALF_UP);
         ColumnVector expected1 = ColumnVector.fromBoxedInts(12, 135, 160, -1454, null, -1500, -140, -150);
         ColumnVector expected2 = ColumnVector.fromBoxedInts(0, 100, 200, -1500, null, -1500, -100, -200)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
    }
  }

  @Test
  void roundIntsHalfEven() {
    try (ColumnVector v = ColumnVector.fromBoxedInts(12, 24, 135, 160, null, 1450, 1550, -1650);
         ColumnVector result1 = Arithmetic.round(v, 2, RoundMode.HALF_EVEN);
         ColumnVector result2 = Arithmetic.round(v, -2, RoundMode.HALF_EVEN);
         ColumnVector expected1 = ColumnVector.fromBoxedInts(12, 24, 135, 160, null, 1450, 1550, -1650);
         ColumnVector expected2 = ColumnVector.fromBoxedInts(0, 0, 100, 200, null, 1400, 1600, -1600)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
    }
  }

  @Test
  void roundDecimal() {
    final int dec32Scale1 = -2;
    final int resultScale1 = -3;

    final int[] DECIMAL32_1 = new int[]{14, 15, 16, 24, 25, 26} ;
    final int[] expectedHalfUp = new int[]{1, 2, 2, 2, 3, 3};
    final int[] expectedHalfEven = new int[]{1, 2, 2, 2, 2, 3};
    try (ColumnVector v = ColumnVector.decimalFromInts(-dec32Scale1, DECIMAL32_1);
         ColumnVector roundHalfUp = Arithmetic.round(v, -3, RoundMode.HALF_UP);
         ColumnVector roundHalfEven = Arithmetic.round(v, -3, RoundMode.HALF_EVEN);
         ColumnVector answerHalfUp = ColumnVector.decimalFromInts(-resultScale1, expectedHalfUp);
         ColumnVector answerHalfEven = ColumnVector.decimalFromInts(-resultScale1, expectedHalfEven)) {
      assertColumnsAreEqual(answerHalfUp, roundHalfUp);
      assertColumnsAreEqual(answerHalfEven, roundHalfEven);
    }
  }

  @Test
  void roundDefault() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(1.234f, 25.66f, null, 154.9f, 2346f);
         ColumnVector result = Arithmetic.round(v);
         ColumnVector expected = ColumnVector.fromBoxedFloats(1f, 26f, null, 155f, 2346f)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void roundWithDecimalPlaces() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(1.234f, 25.66f, null, 154.9f, 2346f);
         ColumnVector result = Arithmetic.round(v, 2);
         ColumnVector expected = ColumnVector.fromBoxedFloats(1.23f, 25.66f, null, 154.9f, 2346f)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void roundDoublesHalfUp() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(1.234, 25.66, null, 154.9, 2346.0);
         ColumnVector result1 = Arithmetic.round(v, 0, RoundMode.HALF_UP);
         ColumnVector result2 = Arithmetic.round(v, 1, RoundMode.HALF_UP);
         ColumnVector result3 = Arithmetic.round(v, -1, RoundMode.HALF_UP);
         ColumnVector expected1 = ColumnVector.fromBoxedDoubles(1.0, 26.0, null, 155.0, 2346.0);
         ColumnVector expected2 = ColumnVector.fromBoxedDoubles(1.2, 25.7, null, 154.9, 2346.0);
         ColumnVector expected3 = ColumnVector.fromBoxedDoubles(0.0, 30.0, null, 150.0, 2350.0)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
      assertColumnsAreEqual(expected3, result3);
    }
  }

  @Test
  void roundDoublesHalfEven() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(1.5, 2.5, 1.35, null, 1.25, 15.0, 25.0);
         ColumnVector result1 = Arithmetic.round(v, RoundMode.HALF_EVEN);
         ColumnVector result2 = Arithmetic.round(v, 1, RoundMode.HALF_EVEN);
         ColumnVector result3 = Arithmetic.round(v, -1, RoundMode.HALF_EVEN);
         ColumnVector expected1 = ColumnVector.fromBoxedDoubles(2.0, 2.0, 1.0, null, 1.0, 15.0, 25.0);
         ColumnVector expected2 = ColumnVector.fromBoxedDoubles(1.5, 2.5, 1.4, null, 1.2, 15.0, 25.0);
         ColumnVector expected3 = ColumnVector.fromBoxedDoubles(0.0, 0.0, 0.0, null, 0.0, 20.0, 20.0)) {
      assertColumnsAreEqual(expected1, result1);
      assertColumnsAreEqual(expected2, result2);
      assertColumnsAreEqual(expected3, result3);
    }
  }
}
