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

public class MathTest {

  @Test
  void multiplyAnsiOffWithOverflowWithoutThrow() {
    // Integer.MAX_VALUE * 2 = -2 in non-ANSI mode
    try (
        ColumnVector left = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE);
        ColumnVector right = ColumnVector.fromInts(0, 1, 2);
        ColumnVector expected = ColumnVector.fromInts(0, 1, -2)) {
      try (ColumnVector actual = Math.multiply(left, right, /* isAnsiMode */ false)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void multiplyAnsiOnWithThrow() {
    // Integer.MAX_VALUE * 2 throws exception in ANSI mode
    try (
        ColumnVector left = ColumnVector.fromInts(0, 1, Integer.MAX_VALUE);
        ColumnVector right = ColumnVector.fromInts(0, 1, 2)) {
      ColumnVector r = Math.multiply(left, right, /* isAnsiMode */ true);

      System.out.println(r.copyToHost().isNull(0));
      System.out.println(r.copyToHost().isNull(1));
      System.out.println(r.copyToHost().isNull(2));

      System.out.println(r.copyToHost().getInt(0));
      System.out.println(r.copyToHost().getInt(1));
      System.out.println(r.copyToHost().getInt(2));

      Assertions.fail("Expected ExceptionWithRowIndex due to overflow");
    } catch (ExceptionWithRowIndex e) {
      Assertions.assertEquals(2, e.getRowIndex());
    }
  }
}
