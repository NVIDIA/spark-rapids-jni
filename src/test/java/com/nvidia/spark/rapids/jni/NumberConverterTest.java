/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class NumberConverterTest {

  @Test
  void convertCvCvTest() {
    try (
        ColumnVector input = ColumnVector.fromStrings(
            "Z1", "34", " azc ");
        ColumnVector fromBase = ColumnVector.fromBoxedInts(
            36, 5, 34);
        ColumnVector toBase = ColumnVector.fromBoxedInts(
            10, 10, 9);
        ColumnVector expected = ColumnVector.fromStrings("1261", "19", "11")) {
      try (
          ColumnVector actual = NumberConverter.convertCvCvCv(input, fromBase, toBase)
      ) {
        assertColumnsAreEqual(expected, actual);
        boolean actualOverflow = NumberConverter.isConvertOverflowCvCvCv(input, fromBase, toBase);
        assertTrue(actualOverflow);
      }
    }
  }

  @Test
  void convertCvSTest() {
    final int constToBase = 27;

    try (
        ColumnVector input = ColumnVector.fromStrings(
            "Z1", "34", " azc ");
        ColumnVector fromBase = ColumnVector.fromBoxedInts(
            7, 36, 36);
        ColumnVector expected = ColumnVector.fromStrings("0", "44", "JE3")) {
      try (
          ColumnVector actual = NumberConverter.convertCvCvS(input, fromBase, constToBase)) {
        assertColumnsAreEqual(expected, actual);
        boolean actualOverflow = NumberConverter.isConvertOverflowCvCvS(input, fromBase, constToBase);
        assertTrue(actualOverflow);
      }
    }
  }

  @Test
  void convertSCvTest() {
    final int constFromBase = 4;

    try (
        ColumnVector input = ColumnVector.fromStrings(
          "Z1", "34", " azc ");
          ColumnVector toBase = ColumnVector.fromBoxedInts(
            7, 9, 36);
        ColumnVector expected = ColumnVector.fromStrings("0", "3", "0")) {
      try (ColumnVector actual = NumberConverter.convertCvSCv(input, constFromBase, toBase)) {
        assertColumnsAreEqual(expected, actual);
        boolean actualOverflow = NumberConverter.isConvertOverflowCvSCv(input, constFromBase, toBase);
        assertTrue(actualOverflow);
      }
    }
  }

  @Test
  void convertSSTest() {
    final int constFromBase = 9;
    final int constToBase = 27;

    try (
        ColumnVector input = ColumnVector.fromStrings(
          "Z1", "34", " azc ");
        ColumnVector expected = ColumnVector.fromStrings("0", "14", "0")) {
      try (ColumnVector actual = NumberConverter.convertCvSS(input, constFromBase, constToBase)) {
        assertColumnsAreEqual(expected, actual);
        boolean actualOverflow = NumberConverter.isConvertOverflowCvSS(input, constFromBase, constToBase);
        assertTrue(actualOverflow);
      }
    }
  }
}
