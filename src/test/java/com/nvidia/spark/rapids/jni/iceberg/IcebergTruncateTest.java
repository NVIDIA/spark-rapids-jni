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

import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import ai.rapids.cudf.*;
import ai.rapids.cudf.HostColumnVector.*;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

import org.apache.iceberg.util.TruncateUtil;
import org.apache.iceberg.util.UnicodeUtil;
import org.apache.iceberg.util.BinaryUtil;

import org.apache.commons.lang3.RandomStringUtils;

public class IcebergTruncateTest {

  private static long seed;

  @BeforeAll
  static void setup() {
    seed = System.currentTimeMillis();
    System.out.println("IcebergTruncateTest seed: " + seed);
  }

  @Test
  void testTruncateInt() {
    int width = 10;
    try (
        ColumnVector input = ColumnVector.fromBoxedInts(null,
            0,
            1,
            5,
            9,
            10,
            11,
            -1,
            -5,
            -10,
            -11,
            null);
        ColumnVector expected = ColumnVector.fromBoxedInts(
            null,
            0,
            0,
            0,
            0,
            10,
            10,
            -10,
            -10,
            -10,
            -20,
            null);
        ColumnVector result = IcebergTruncate.truncate(input, width)) {
      assertColumnsAreEqual(expected, result);
    }

    Random rand = new Random(seed);
    int numRows = 1024;
    Integer[] inputData = new Integer[numRows];
    Integer[] expectedData = new Integer[numRows];
    Integer[] expectedDataNegativeWidth = new Integer[numRows];
    for (int i = 0; i < numRows; i++) {
      int val = rand.nextInt();
      inputData[i] = val;
      // run on CPU to get expected value
      expectedData[i] = TruncateUtil.truncateInt(width, val);
      expectedDataNegativeWidth[i] = TruncateUtil.truncateInt(-width, val);
    }
    try (
        ColumnVector input = ColumnVector.fromBoxedInts(inputData);
        ColumnVector expected = ColumnVector.fromBoxedInts(expectedData);
        ColumnVector expectedNegativeWidth = ColumnVector.fromBoxedInts(expectedDataNegativeWidth);
        ColumnVector result = IcebergTruncate.truncate(input, width);
        ColumnVector resultNegativeWidth = IcebergTruncate.truncate(input, -width)) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expectedNegativeWidth, resultNegativeWidth);
    }
  }

  @Test
  void testTruncateLong() {
    int width = 10;
    try (
        ColumnVector input = ColumnVector.fromBoxedLongs(
            null,
            0L,
            1L,
            5L,
            9L,
            10L,
            11L,
            -1L,
            -5L,
            -10L,
            -11L,
            null);
        ColumnVector expected = ColumnVector.fromBoxedLongs(
            null,
            0L,
            0L,
            0L,
            0L,
            10L,
            10L,
            -10L,
            -10L,
            -10L,
            -20L,
            null);
        ColumnVector result = IcebergTruncate.truncate(input, width)) {
      assertColumnsAreEqual(expected, result);
    }

    Random rand = new Random(seed);
    int numRows = 1024;
    Long[] inputData = new Long[numRows];
    Long[] expectedData = new Long[numRows];
    Long[] expectedDataNegativeWidth = new Long[numRows];
    for (int i = 0; i < numRows; i++) {
      long val = rand.nextLong();
      inputData[i] = val;
      // run on CPU to get expected value
      expectedData[i] = TruncateUtil.truncateLong(width, val);
      expectedDataNegativeWidth[i] = TruncateUtil.truncateLong(-width, val);
    }
    try (
        ColumnVector input = ColumnVector.fromBoxedLongs(inputData);
        ColumnVector expected = ColumnVector.fromBoxedLongs(expectedData);
        ColumnVector expectedNegativeWidth = ColumnVector.fromBoxedLongs(expectedDataNegativeWidth);
        ColumnVector result = IcebergTruncate.truncate(input, width);
        ColumnVector resultNegativeWidth = IcebergTruncate.truncate(input, -width)) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expectedNegativeWidth, resultNegativeWidth);
    }
  }

  @Test
  void testTruncateString() {
    int width = 5;
    try (
        ColumnVector input = ColumnVector.fromStrings(
            null,
            "🚀23四😁678",
            "中华人民共和国",
            "",
            null);
        ColumnVector expected = ColumnVector.fromStrings(
            null,
            "🚀23四😁",
            "中华人民共",
            "",
            null);
        ColumnVector result = IcebergTruncate.truncate(input, width)) {
      assertColumnsAreEqual(expected, result);
    }

    int numRows = 1024;
    String[] inputData = new String[numRows];
    String[] expectedData = new String[numRows];
    for (int i = 0; i < numRows; i++) {
      String val = RandomStringUtils.randomPrint(i);
      inputData[i] = val;
      // run on CPU to get expected value
      expectedData[i] = (String) UnicodeUtil.truncateString(val, width);
    }
    try (
        ColumnVector input = ColumnVector.fromStrings(inputData);
        ColumnVector expected = ColumnVector.fromStrings(expectedData);
        ColumnVector result = IcebergTruncate.truncate(input, width)) {
      assertColumnsAreEqual(expected, result);
    }

    try (
        ColumnVector input = ColumnVector.fromStrings(inputData)) {
      // test negative width, should throw exception
      assertThrows(CudfException.class, () -> {
        IcebergTruncate.truncate(input, -width);
      });
    }
  }

  @Test
  @SuppressWarnings("unchecked")
  void testTruncateBinary() {
    try (
        ColumnVector input = ColumnVector.fromLists(
            new ListType(true, new BasicType(false, DType.UINT8)),
            Arrays.asList((byte) 1, (byte) 2, (byte) 3), // Normal case
            null, // Entire array is null
            Arrays.asList(), // Empty list
            Arrays.asList((byte) 11, (byte) 22, (byte) 33));
        ColumnVector expected = ColumnVector.fromLists(
            new ListType(true, new BasicType(false, DType.UINT8)),
            Arrays.asList((byte) 1, (byte) 2),
            null, // Entire array is null
            Collections.emptyList(), // Empty list
            Arrays.asList((byte) 11, (byte) 22));
        ColumnVector result = IcebergTruncate.truncate(input, 2)) {
      assertColumnsAreEqual(expected, result);
    }

    int numRows = 1024;
    List<Byte>[] inputData = new List[numRows];
    List<Byte>[] expectedData = new List[numRows];
    Random rand = new Random(seed);
    for (int i = 0; i < numRows; i++) {
      // generate random byte array
      int len = rand.nextInt(20); // length up to 20
      byte[] arr = new byte[len];
      rand.nextBytes(arr);
      inputData[i] = new ArrayList<>();
      for (byte b : arr) {
        inputData[i].add(b);
      }

      // run on CPU to get expected value
      ByteBuffer bb = ByteBuffer.wrap(arr);
      ByteBuffer fromCpu = BinaryUtil.truncateBinary(bb, 10);
      List<Byte> cpuRet = new ArrayList<>(fromCpu.remaining());
      while (fromCpu.hasRemaining()) {
        cpuRet.add(fromCpu.get());
      }
      expectedData[i] = cpuRet;
    }
    try (
        ColumnVector input = ColumnVector.fromLists(
            new ListType(true, new BasicType(false, DType.UINT8)),
            inputData);
        ColumnVector expected = ColumnVector.fromLists(
            new ListType(true, new BasicType(false, DType.UINT8)),
            expectedData);
        ColumnVector result = IcebergTruncate.truncate(input, 10)) {
      assertColumnsAreEqual(expected, result);
    }

    try (
        ColumnVector input = ColumnVector.fromLists(
            new ListType(true, new BasicType(false, DType.UINT8)),
            inputData)) {
      // test negative width, should throw exception
      assertThrows(CudfException.class, () -> {
        IcebergTruncate.truncate(input, -10);
      });
    }
  }

  // decimal 32 has max precision of 7
  private static final BigInteger MIN_DECIMAL_32_VALUE = new BigInteger("-9999999");
  private static final BigInteger MAX_DECIMAL_32_VALUE = new BigInteger("9999999");

  // decimal 64 has max precision of 16
  private static final BigInteger MIN_DECIMAL_64_VALUE = new BigInteger("-9999999999999999");
  private static final BigInteger MAX_DECIMAL_64_VALUE = new BigInteger("9999999999999999");

  // decimal 128 has max precision of 38
  private static final BigInteger MIN_DECIMAL_128_VALUE = new BigInteger(
      "-99999999999999999999999999999999999999");
  private static final BigInteger MAX_DECIMAL_128_VALUE = new BigInteger(
      "99999999999999999999999999999999999999");

  private static BigInteger randomBigInteger(BigInteger min, BigInteger max, Random random) {
    BigInteger range = max.subtract(min);
    BigInteger randomValue = new BigInteger(range.bitLength(), random).mod(range);
    return randomValue.add(min);
  }

  private static void compareDecimals(
      DType.DTypeEnum decimalType,
      BigDecimal[] expected,
      HostColumnVector actual,
      int numRows) {
    int minPrecision;
    int maxPrecision;
    switch (decimalType) {
      case DECIMAL32:
        minPrecision = 1;
        maxPrecision = 7;
        break;
      case DECIMAL64:
        minPrecision = 8;
        maxPrecision = 16;
        break;
      case DECIMAL128:
        minPrecision = 17;
        maxPrecision = 38;
        break;
      default:
        throw new IllegalArgumentException("Unsupported decimal type: " + decimalType);
    }

    for (int i = 0; i < numRows; i++) {
      if (expected[i] == null) {
        assert actual.isNull(i);
      } else {
        assertTrue(expected[i].precision() >= minPrecision
            && expected[i].precision() <= maxPrecision);
        assertTrue(expected[i].compareTo(actual.getBigDecimal(i)) == 0);
      }
    }
  }

  @Test
  void testTruncateDecimal32() {
    try (
        ColumnVector input = ColumnVector.decimalFromBoxedInts(-2,
            null,
            1234,
            1230,
            1229,
            5,
            -5,
            null);
        ColumnVector expected = ColumnVector.decimalFromBoxedInts(-2,
            null,
            1230,
            1230,
            1220,
            0,
            -10,
            null);
        ColumnVector result = IcebergTruncate.truncate(input, 10)) {
      assertColumnsAreEqual(expected, result);
    }

    int numRows = 1024;
    int[] inputData = new int[numRows];
    BigDecimal[] expectedData = new BigDecimal[numRows];
    BigDecimal[] expectedDataNegativeWidth = new BigDecimal[numRows];
    Random rand = new Random(seed);
    BigInteger width = new BigInteger("10");
    for (int i = 0; i < numRows; i++) {
      // generate random BigInteger within range of precision 7
      // GPU decimal32 supports precision up to 7
      BigInteger val = randomBigInteger(MIN_DECIMAL_32_VALUE, MAX_DECIMAL_32_VALUE, rand);
      inputData[i] = val.intValue();

      // run on CPU to get expected value
      BigDecimal v = new BigDecimal(val, 2);
      expectedData[i] = TruncateUtil.truncateDecimal(width, v);
      expectedDataNegativeWidth[i] = TruncateUtil.truncateDecimal(
          width.negate(), v);
    }

    try (
        ColumnVector input = ColumnVector.decimalFromInts(-2, inputData);
        ColumnVector result = IcebergTruncate.truncate(input, 10);
        ColumnVector resultNegativeWidth = IcebergTruncate.truncate(input, -10);
        HostColumnVector ret = result.copyToHost();
        HostColumnVector retNegativeWidth = resultNegativeWidth.copyToHost()) {

      assertTrue(input.getType().getTypeId() == DType.DTypeEnum.DECIMAL32);
      assertTrue(result.getType().getTypeId() == DType.DTypeEnum.DECIMAL32);
      compareDecimals(DType.DTypeEnum.DECIMAL32, expectedData, ret, numRows);

      assertTrue(resultNegativeWidth.getType().getTypeId() == DType.DTypeEnum.DECIMAL32);
      compareDecimals(DType.DTypeEnum.DECIMAL32, expectedDataNegativeWidth, retNegativeWidth, numRows);
    }
  }

  @Test
  void testTruncateDecimal64() {
    try (
        ColumnVector input = ColumnVector.decimalFromBoxedLongs(
            -2,
            null,
            1234L,
            1230L,
            1229L,
            5L,
            -5L,
            null);
        ColumnVector expected = ColumnVector.decimalFromBoxedLongs(
            -2,
            null,
            1230L,
            1230L,
            1220L,
            0L,
            -10L,
            null);
        ColumnVector result = IcebergTruncate.truncate(input, 10)) {
      assertColumnsAreEqual(expected, result);
    }

    int numRows = 1024;
    long[] inputData = new long[numRows];
    BigDecimal[] expectedData = new BigDecimal[numRows];
    BigDecimal[] expectedDataNegativeWidth = new BigDecimal[numRows];
    Random rand = new Random(seed);
    BigInteger width = new BigInteger("10");
    for (int i = 0; i < numRows; i++) {
      // generate random BigInteger within range of precision 16
      // GPU decimal128 supports precision up to 16
      BigInteger val = randomBigInteger(MIN_DECIMAL_64_VALUE, MAX_DECIMAL_64_VALUE, rand);
      inputData[i] = val.longValue();
      // run on CPU to get expected value
      BigDecimal v = new BigDecimal(val, 2);
      expectedData[i] = TruncateUtil.truncateDecimal(width, v);
      expectedDataNegativeWidth[i] = TruncateUtil.truncateDecimal(
          width.negate(), v);
    }

    try (
        ColumnVector input = ColumnVector.decimalFromLongs(-2, inputData);
        ColumnVector result = IcebergTruncate.truncate(input, 10);
        ColumnVector resultNegativeWidth = IcebergTruncate.truncate(input, -10);
        HostColumnVector ret = result.copyToHost();
        HostColumnVector retNegativeWidth = resultNegativeWidth.copyToHost()) {

      assertTrue(input.getType().getTypeId() == DType.DTypeEnum.DECIMAL64);
      assertTrue(result.getType().getTypeId() == DType.DTypeEnum.DECIMAL64);
      compareDecimals(DType.DTypeEnum.DECIMAL64, expectedData, ret, numRows);
      assertTrue(resultNegativeWidth.getType().getTypeId() == DType.DTypeEnum.DECIMAL64);
      compareDecimals(DType.DTypeEnum.DECIMAL64, expectedDataNegativeWidth, retNegativeWidth, numRows);
    }
  }

  @Test
  void testTruncateDecimal128() {
    try (
        ColumnVector input = ColumnVector.decimalFromBigInt(
            -2,
            null,
            new BigInteger("1234"),
            new BigInteger("1230"),
            new BigInteger("1229"),
            new BigInteger("5"),
            new BigInteger("-5"));
        ColumnVector expected = ColumnVector.decimalFromBigInt(
            -2,
            null,
            new BigInteger("1230"),
            new BigInteger("1230"),
            new BigInteger("1220"),
            new BigInteger("0"),
            new BigInteger("-10"));
        ColumnVector result = IcebergTruncate.truncate(input, 10)) {
      assertColumnsAreEqual(expected, result);
    }

    int numRows = 1024;
    BigInteger[] inputData = new BigInteger[numRows];
    BigDecimal[] expectedData = new BigDecimal[numRows];
    BigDecimal[] expectedDataNegativeWidth = new BigDecimal[numRows];
    Random rand = new Random(seed);
    BigInteger width = new BigInteger("10");
    for (int i = 0; i < numRows; i++) {
      // generate random BigInteger within range of precision 38
      // GPU decimal128 supports precision up to 38
      // Java BigInteger can support arbitrary precision
      BigInteger val = randomBigInteger(MIN_DECIMAL_128_VALUE, MAX_DECIMAL_128_VALUE, rand);
      inputData[i] = val;
      // run on CPU to get expected value
      BigDecimal v = new BigDecimal(val, 2);
      expectedData[i] = TruncateUtil.truncateDecimal(width, v);
      expectedDataNegativeWidth[i] = TruncateUtil.truncateDecimal(
          width.negate(), v);
    }

    try (
        ColumnVector input = ColumnVector.decimalFromBigInt(-2, inputData);
        ColumnVector result = IcebergTruncate.truncate(input, 10);
        ColumnVector resultNegativeWidth = IcebergTruncate.truncate(input, -10);
        HostColumnVector ret = result.copyToHost();
        HostColumnVector retNegativeWidth = resultNegativeWidth.copyToHost()) {
      assertTrue(input.getType().getTypeId() == DType.DTypeEnum.DECIMAL128);
      assertTrue(result.getType().getTypeId() == DType.DTypeEnum.DECIMAL128);
      compareDecimals(DType.DTypeEnum.DECIMAL128, expectedData, ret, numRows);
      assertTrue(resultNegativeWidth.getType().getTypeId() == DType.DTypeEnum.DECIMAL128);
      compareDecimals(DType.DTypeEnum.DECIMAL128, expectedDataNegativeWidth, retNegativeWidth, numRows);
    }
  }
}

