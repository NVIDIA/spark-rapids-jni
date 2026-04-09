/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.ListType;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CharsetDecodeTest {

  /** Helper: convert a byte array to a List<Byte> for ColumnVector.fromLists. */
  private static List<Byte> bytes(int... values) {
    List<Byte> list = new ArrayList<>(values.length);
    for (int v : values) {
      list.add((byte) v);
    }
    return list;
  }

  /** Helper: build a LIST<UINT8> column from multiple byte arrays. Null entries are supported. */
  @SuppressWarnings("unchecked")
  private static ColumnVector binaryColumn(byte[]... rows) {
    ListType type = new ListType(true, new BasicType(false, DType.UINT8));
    List<Byte>[] lists = new List[rows.length];
    for (int i = 0; i < rows.length; i++) {
      if (rows[i] == null) {
        lists[i] = null;
      } else {
        lists[i] = bytes(toIntArray(rows[i]));
      }
    }
    return ColumnVector.fromLists(type, lists);
  }

  private static int[] toIntArray(byte[] arr) {
    int[] result = new int[arr.length];
    for (int i = 0; i < arr.length; i++) {
      result[i] = arr[i] & 0xFF;
    }
    return result;
  }

  /** Decode bytes using Java's GBK charset (the CPU ground truth). */
  private static String decodeGbkJava(byte[] bytes) {
    try {
      Charset gbk = Charset.forName("GBK");
      CharsetDecoder decoder = gbk.newDecoder()
          .onMalformedInput(CodingErrorAction.REPLACE)
          .onUnmappableCharacter(CodingErrorAction.REPLACE);
      CharBuffer cb = decoder.decode(ByteBuffer.wrap(bytes));
      return cb.toString();
    } catch (java.nio.charset.CharacterCodingException e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  void testBasicChinese() {
    // "你好" in GBK: C4E3 BAC3
    byte[] nihao = {(byte) 0xC4, (byte) 0xE3, (byte) 0xBA, (byte) 0xC3};
    // "世界" in GBK: CAC0 BDE7
    byte[] shijie = {(byte) 0xCA, (byte) 0xC0, (byte) 0xBD, (byte) 0xE7};

    try (ColumnVector input = binaryColumn(nihao, shijie);
         ColumnVector result = CharsetDecode.decode(input, CharsetDecode.GBK);
         ColumnVector expected = ColumnVector.fromStrings("你好", "世界")) {
      AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testPureAscii() {
    byte[] ascii = "Hello, World!".getBytes();

    try (ColumnVector input = binaryColumn(ascii);
         ColumnVector result = CharsetDecode.decode(input, CharsetDecode.GBK);
         ColumnVector expected = ColumnVector.fromStrings("Hello, World!")) {
      AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testMixedAsciiAndChinese() {
    // "Hello你好World" in GBK
    byte[] mixed = {
        'H', 'e', 'l', 'l', 'o',
        (byte) 0xC4, (byte) 0xE3, (byte) 0xBA, (byte) 0xC3,  // 你好
        'W', 'o', 'r', 'l', 'd'
    };

    try (ColumnVector input = binaryColumn(mixed);
         ColumnVector result = CharsetDecode.decode(input, CharsetDecode.GBK);
         ColumnVector expected = ColumnVector.fromStrings("Hello你好World")) {
      AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testEmptyInput() {
    byte[] empty = {};

    try (ColumnVector input = binaryColumn(empty);
         ColumnVector result = CharsetDecode.decode(input, CharsetDecode.GBK);
         ColumnVector expected = ColumnVector.fromStrings("")) {
      AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testNullRows() {
    byte[] nihao = {(byte) 0xC4, (byte) 0xE3, (byte) 0xBA, (byte) 0xC3};

    try (ColumnVector input = binaryColumn(nihao, null, nihao);
         ColumnVector result = CharsetDecode.decode(input, CharsetDecode.GBK);
         ColumnVector expected = ColumnVector.fromStrings("你好", null, "你好")) {
      AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testInvalidBytes() {
    // 0xFF is not a valid GBK lead byte
    byte[] invalid = {(byte) 0xFF, (byte) 0xFF};
    String javaResult = decodeGbkJava(invalid);

    try (ColumnVector input = binaryColumn(invalid);
         ColumnVector result = CharsetDecode.decode(input, CharsetDecode.GBK);
         ColumnVector expected = ColumnVector.fromStrings(javaResult)) {
      AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testTruncatedDoubleByte() {
    // Lead byte 0x81 without a second byte
    byte[] truncated = {(byte) 0x81};
    String javaResult = decodeGbkJava(truncated);

    try (ColumnVector input = binaryColumn(truncated);
         ColumnVector result = CharsetDecode.decode(input, CharsetDecode.GBK);
         ColumnVector expected = ColumnVector.fromStrings(javaResult)) {
      AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testByte0x80() {
    // Byte 0x80 handling is JDK-dependent (some map to Euro sign, others to FFFD)
    byte[] euro = {(byte) 0x80};
    String javaResult = decodeGbkJava(euro);

    try (ColumnVector input = binaryColumn(euro);
         ColumnVector result = CharsetDecode.decode(input, CharsetDecode.GBK);
         ColumnVector expected = ColumnVector.fromStrings(javaResult)) {
      AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testAllGbkDoubleBytePairs() {
    // End-to-end verification: for every valid GBK double-byte pair,
    // compare GPU decode output against Java's GBK charset decoder.
    // We batch all pairs into a single column for efficiency.
    int firstMin = 0x81, firstMax = 0xFE;
    int secondMin = 0x40, secondMax = 0xFE;
    int numRows = (firstMax - firstMin + 1) * (secondMax - secondMin + 1);

    byte[][] allPairs = new byte[numRows][];
    String[] javaExpected = new String[numRows];
    Charset gbk = Charset.forName("GBK");

    int idx = 0;
    for (int first = firstMin; first <= firstMax; first++) {
      for (int second = secondMin; second <= secondMax; second++) {
        byte[] pair = {(byte) first, (byte) second};
        allPairs[idx] = pair;
        CharsetDecoder decoder = gbk.newDecoder()
            .onMalformedInput(CodingErrorAction.REPLACE)
            .onUnmappableCharacter(CodingErrorAction.REPLACE);
        try {
          javaExpected[idx] = decoder.decode(ByteBuffer.wrap(pair)).toString();
        } catch (java.nio.charset.CharacterCodingException e) {
          throw new RuntimeException(e);
        }
        idx++;
      }
    }

    try (ColumnVector input = binaryColumn(allPairs);
         ColumnVector result = CharsetDecode.decode(input, CharsetDecode.GBK);
         ColumnVector expected = ColumnVector.fromStrings(javaExpected)) {
      AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSlicedInput() {
    // Regression test: decode must handle sliced LIST<UINT8> columns correctly.
    // A sliced column has a non-zero parent offset; offsets_begin() must be used
    // instead of raw offsets().data() to avoid reading wrong row boundaries.
    byte[] nihao = {(byte) 0xC4, (byte) 0xE3, (byte) 0xBA, (byte) 0xC3};  // 你好
    byte[] shijie = {(byte) 0xCA, (byte) 0xC0, (byte) 0xBD, (byte) 0xE7}; // 世界
    byte[] ascii = {'A', 'B', 'C'};

    try (ColumnVector full = binaryColumn(ascii, nihao, shijie, ascii);
         // Slice rows [1, 3) -> should decode "你好", "世界"
         ColumnVector sliced = full.subVector(1, 3);
         ColumnVector result = CharsetDecode.decode(sliced, CharsetDecode.GBK);
         ColumnVector expected = ColumnVector.fromStrings("你好", "世界")) {
      AssertUtils.assertColumnsAreEqual(expected, result);
    }
  }
}
