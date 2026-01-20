/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector.*;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

/**
 * Tests for the Protobuf GPU decoder.
 * 
 * Test cases are inspired by Google's protobuf conformance test suite:
 * https://github.com/protocolbuffers/protobuf/tree/main/conformance
 */
public class ProtobufTest {

  // ============================================================================
  // Helper methods for encoding protobuf wire format
  // ============================================================================

  /** Encode a value as a varint (variable-length integer). */
  private static byte[] encodeVarint(long value) {
    long v = value;
    byte[] tmp = new byte[10];
    int idx = 0;
    while ((v & ~0x7FL) != 0) {
      tmp[idx++] = (byte) ((v & 0x7F) | 0x80);
      v >>>= 7;
    }
    tmp[idx++] = (byte) (v & 0x7F);
    byte[] out = new byte[idx];
    System.arraycopy(tmp, 0, out, 0, idx);
    return out;
  }

  /** 
   * Encode a varint with extra padding bytes (over-encoded but valid).
   * This is useful for testing that parsers accept non-canonical varints.
   */
  private static byte[] encodeLongVarint(long value, int extraBytes) {
    byte[] tmp = new byte[10];
    int idx = 0;
    long v = value;
    while ((v & ~0x7FL) != 0 || extraBytes > 0) {
      tmp[idx++] = (byte) ((v & 0x7F) | 0x80);
      v >>>= 7;
      if (v == 0 && extraBytes > 0) {
        extraBytes--;
      }
    }
    tmp[idx++] = (byte) (v & 0x7F);
    byte[] out = new byte[idx];
    System.arraycopy(tmp, 0, out, 0, idx);
    return out;
  }

  /** ZigZag encode a signed 32-bit integer, returning as unsigned long for varint encoding. */
  private static long zigzagEncode32(int n) {
    return Integer.toUnsignedLong((n << 1) ^ (n >> 31));
  }

  /** ZigZag encode a signed 64-bit integer. */
  private static long zigzagEncode64(long n) {
    return (n << 1) ^ (n >> 63);
  }

  /** Encode a 32-bit value in little-endian (fixed32). */
  private static byte[] encodeFixed32(int v) {
    return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array();
  }

  /** Encode a 64-bit value in little-endian (fixed64). */
  private static byte[] encodeFixed64(long v) {
    return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array();
  }

  /** Encode a float in little-endian (fixed32 wire type). */
  private static byte[] encodeFloat(float f) {
    return ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(f).array();
  }

  /** Encode a double in little-endian (fixed64 wire type). */
  private static byte[] encodeDouble(double d) {
    return ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putDouble(d).array();
  }

  /** Create a protobuf tag (field number + wire type). */
  private static byte[] tag(int fieldNumber, int wireType) {
    return encodeVarint((fieldNumber << 3) | wireType);
  }

  // Wire type constants
  private static final int WT_VARINT = 0;
  private static final int WT_64BIT = 1;
  private static final int WT_LEN = 2;
  private static final int WT_32BIT = 5;

  private static Byte[] box(byte[] bytes) {
    if (bytes == null) return null;
    Byte[] out = new Byte[bytes.length];
    for (int i = 0; i < bytes.length; i++) {
      out[i] = bytes[i];
    }
    return out;
  }

  private static Byte[] concat(Byte[]... parts) {
    int len = 0;
    for (Byte[] p : parts) if (p != null) len += p.length;
    Byte[] out = new Byte[len];
    int pos = 0;
    for (Byte[] p : parts) {
      if (p != null) {
        System.arraycopy(p, 0, out, pos, p.length);
        pos += p.length;
      }
    }
    return out;
  }

  // ============================================================================
  // Helper methods for calling the new API
  // ============================================================================

  /**
   * Helper method that wraps the new API for tests that decode all fields.
   * This simulates the old API behavior where all fields are decoded.
   */
  private static ColumnVector decodeAllFields(ColumnView binaryInput,
                                              int[] fieldNumbers,
                                              int[] typeIds,
                                              int[] encodings) {
    return decodeAllFields(binaryInput, fieldNumbers, typeIds, encodings, true);
  }

  /**
   * Helper method that wraps the new API for tests that decode all fields.
   * This simulates the old API behavior where all fields are decoded.
   */
  private static ColumnVector decodeAllFields(ColumnView binaryInput,
                                              int[] fieldNumbers,
                                              int[] typeIds,
                                              int[] encodings,
                                              boolean failOnErrors) {
    int numFields = fieldNumbers.length;
    // When decoding all fields, decodedFieldIndices is [0, 1, 2, ..., n-1]
    int[] decodedFieldIndices = new int[numFields];
    for (int i = 0; i < numFields; i++) {
      decodedFieldIndices[i] = i;
    }
    return Protobuf.decodeToStruct(binaryInput, numFields, decodedFieldIndices,
                                   fieldNumbers, typeIds, encodings, failOnErrors);
  }

  // ============================================================================
  // Basic Type Tests
  // ============================================================================

  @Test
  void decodeVarintAndStringToStruct() {
    // message Msg { int64 id = 1; string name = 2; }
    // Row0: id=100, name="alice"
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(100)),
        box(tag(2, WT_LEN)),
        box(encodeVarint(5)),
        box("alice".getBytes()));

    // Row1: id=200, name missing
    Byte[] row1 = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(200)));

    // Row2: null input message
    Byte[] row2 = null;

    try (Table input = new Table.TestBuilder().column(row0, row1, row2).build();
         ColumnVector expectedId = ColumnVector.fromBoxedLongs(100L, 200L, null);
         ColumnVector expectedName = ColumnVector.fromStrings("alice", null, null);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedId, expectedName);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT64.getTypeId().getNativeId(), DType.STRING.getTypeId().getNativeId()},
             new int[]{0, 0})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void decodeMoreTypes() {
    // message Msg { uint32 u32 = 1; sint64 s64 = 2; fixed32 f32 = 3; bytes b = 4; }
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(4000000000L)),
        box(tag(2, WT_VARINT)),
        box(encodeVarint(zigzagEncode64(-1234567890123L))),
        box(tag(3, WT_32BIT)),
        box(encodeFixed32(12345)),
        box(tag(4, WT_LEN)),
        box(encodeVarint(3)),
        box(new byte[]{1, 2, 3}));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0}).build();
         ColumnVector expectedU32 = ColumnVector.fromBoxedLongs(4000000000L);
         ColumnVector expectedS64 = ColumnVector.fromBoxedLongs(-1234567890123L);
         ColumnVector expectedF32 = ColumnVector.fromBoxedInts(12345);
         ColumnVector expectedB = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT8)),
             Arrays.asList((byte) 1, (byte) 2, (byte) 3));
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1, 2, 3, 4},
             new int[]{
                 DType.UINT32.getTypeId().getNativeId(),
                 DType.INT64.getTypeId().getNativeId(),
                 DType.INT32.getTypeId().getNativeId(),
                 DType.LIST.getTypeId().getNativeId()},
             new int[]{
                 Protobuf.ENC_DEFAULT,
                 Protobuf.ENC_ZIGZAG,
                 Protobuf.ENC_FIXED,
                 Protobuf.ENC_DEFAULT})) {
      try (ColumnVector expectedU32Correct = expectedU32.castTo(DType.UINT32);
           ColumnVector expectedStructCorrect = ColumnVector.makeStruct(
               expectedU32Correct, expectedS64, expectedF32, expectedB)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStructCorrect, actualStruct);
      }
    }
  }

  @Test
  void decodeFloatDoubleAndBool() {
    // message Msg { bool flag = 1; float f32 = 2; double f64 = 3; }
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), new Byte[]{(byte)0x01},  // bool=true
        box(tag(2, WT_32BIT)), box(encodeFloat(3.14f)),
        box(tag(3, WT_64BIT)), box(encodeDouble(2.71828)));

    Byte[] row1 = concat(
        box(tag(1, WT_VARINT)), new Byte[]{(byte)0x00},  // bool=false
        box(tag(2, WT_32BIT)), box(encodeFloat(-1.5f)),
        box(tag(3, WT_64BIT)), box(encodeDouble(0.0)));

    try (Table input = new Table.TestBuilder().column(row0, row1).build();
         ColumnVector expectedBool = ColumnVector.fromBoxedBooleans(true, false);
         ColumnVector expectedFloat = ColumnVector.fromBoxedFloats(3.14f, -1.5f);
         ColumnVector expectedDouble = ColumnVector.fromBoxedDoubles(2.71828, 0.0);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedBool, expectedFloat, expectedDouble);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1, 2, 3},
             new int[]{
                 DType.BOOL8.getTypeId().getNativeId(),
                 DType.FLOAT32.getTypeId().getNativeId(),
                 DType.FLOAT64.getTypeId().getNativeId()},
             new int[]{0, 0, 0})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Schema Projection Tests (new API feature)
  // ============================================================================

  @Test
  void testSchemaProjection() {
    // message Msg { int64 f1 = 1; string f2 = 2; int32 f3 = 3; }
    // Only decode f1 and f3, f2 should be null
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(100)),
        box(tag(2, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()),
        box(tag(3, WT_VARINT)), box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0}).build();
         // Expected: f1=100, f2=null (not decoded), f3=42
         ColumnVector expectedF1 = ColumnVector.fromBoxedLongs(100L);
         ColumnVector expectedF2 = ColumnVector.fromStrings((String)null);
         ColumnVector expectedF3 = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedF1, expectedF2, expectedF3);
         // Decode only fields at indices 0 and 2 (skip index 1)
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             3,  // total fields
             new int[]{0, 2},  // decode only indices 0 and 2
             new int[]{1, 3},  // field numbers for decoded fields
             new int[]{DType.INT64.getTypeId().getNativeId(),
                       DType.STRING.getTypeId().getNativeId(),
                       DType.INT32.getTypeId().getNativeId()},  // types for ALL fields
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},  // encodings for decoded fields
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testSchemaProjectionDecodeNone() {
    // Decode no fields - all should be null
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(100)),
        box(tag(2, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0}).build();
         ColumnVector expectedF1 = ColumnVector.fromBoxedLongs((Long)null);
         ColumnVector expectedF2 = ColumnVector.fromStrings((String)null);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedF1, expectedF2);
         ColumnVector actualStruct = Protobuf.decodeToStruct(
             input.getColumn(0),
             2,  // total fields
             new int[]{},  // decode no fields
             new int[]{},  // no field numbers
             new int[]{DType.INT64.getTypeId().getNativeId(),
                       DType.STRING.getTypeId().getNativeId()},  // types for ALL fields
             new int[]{},  // no encodings
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Varint Boundary Tests
  // ============================================================================

  @Test
  void testVarintMaxUint64() {
    // Max uint64 = 0xFFFFFFFFFFFFFFFF = 18446744073709551615
    // Encoded as 10 bytes: FF FF FF FF FF FF FF FF FF 01
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        new Byte[]{(byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF,
                   (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0x01});

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.UINT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      try (ColumnVector expectedU64 = ColumnVector.fromBoxedLongs(-1L);  // -1 as unsigned = max
           ColumnVector expectedU64Correct = expectedU64.castTo(DType.UINT64);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expectedU64Correct)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
      }
    }
  }

  @Test
  void testVarintZero() {
    // Zero encoded as single byte: 0x00
    Byte[] row = concat(box(tag(1, WT_VARINT)), new Byte[]{0x00});

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(0L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testVarintOverEncodedZero() {
    // Zero over-encoded as 10 bytes (all continuation bits except last)
    // This is valid per protobuf spec - parsers must accept non-canonical varints
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        new Byte[]{(byte)0x80, (byte)0x80, (byte)0x80, (byte)0x80, (byte)0x80,
                   (byte)0x80, (byte)0x80, (byte)0x80, (byte)0x80, (byte)0x00});

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(0L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testVarint10thByteInvalid() {
    // 10th byte with more than 1 significant bit is invalid
    // (uint64 can only hold 64 bits: 9*7=63 bits + 1 bit from 10th byte)
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        new Byte[]{(byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF,
                   (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0x02});  // 0x02 has 2nd bit set

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             false)) {
      try (ColumnVector expected = ColumnVector.fromBoxedLongs((Long)null);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  // ============================================================================
  // ZigZag Boundary Tests
  // ============================================================================

  @Test
  void testZigzagInt32Min() {
    // int32 min = -2147483648
    // zigzag encoded = 4294967295 = 0xFFFFFFFF
    int minInt32 = Integer.MIN_VALUE;
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(zigzagEncode32(minInt32))));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(minInt32);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_ZIGZAG})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testZigzagInt32Max() {
    // int32 max = 2147483647
    // zigzag encoded = 4294967294 = 0xFFFFFFFE
    int maxInt32 = Integer.MAX_VALUE;
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(zigzagEncode32(maxInt32))));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(maxInt32);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_ZIGZAG})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testZigzagInt64Min() {
    // int64 min = -9223372036854775808
    long minInt64 = Long.MIN_VALUE;
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(zigzagEncode64(minInt64))));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedLong = ColumnVector.fromBoxedLongs(minInt64);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedLong);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_ZIGZAG})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testZigzagInt64Max() {
    long maxInt64 = Long.MAX_VALUE;
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(zigzagEncode64(maxInt64))));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedLong = ColumnVector.fromBoxedLongs(maxInt64);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedLong);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_ZIGZAG})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testZigzagNegativeOne() {
    // -1 zigzag encoded = 1
    Byte[] row = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(zigzagEncode64(-1L))));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedLong = ColumnVector.fromBoxedLongs(-1L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedLong);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_ZIGZAG})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Truncated/Malformed Data Tests
  // ============================================================================

  @Test
  void testMalformedVarint() {
    // Varint that never terminates (all continuation bits set, 11 bytes)
    Byte[] malformed = new Byte[]{(byte)0x08, (byte)0xFF, (byte)0xFF, (byte)0xFF,
                                   (byte)0xFF, (byte)0xFF, (byte)0xFF,
                                   (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{malformed}).build();
         ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{0},
             false)) {
      try (ColumnVector expected = ColumnVector.fromBoxedLongs((Long)null);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  @Test
  void testTruncatedVarint() {
    // Single byte with continuation bit set but no following byte
    Byte[] truncated = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte)0x80});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build();
         ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{0},
             false)) {
      try (ColumnVector expected = ColumnVector.fromBoxedLongs((Long)null);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  @Test
  void testTruncatedLengthDelimited() {
    // String field with length=5 but no actual data
    Byte[] truncated = concat(box(tag(2, WT_LEN)), box(encodeVarint(5)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build();
         ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{2},
             new int[]{DType.STRING.getTypeId().getNativeId()},
             new int[]{0},
             false)) {
      try (ColumnVector expected = ColumnVector.fromStrings((String)null);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  @Test
  void testTruncatedFixed32() {
    // Fixed32 needs 4 bytes but only 3 provided
    Byte[] truncated = concat(box(tag(1, WT_32BIT)), new Byte[]{0x01, 0x02, 0x03});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build();
         ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_FIXED},
             false)) {
      try (ColumnVector expected = ColumnVector.fromBoxedInts((Integer)null);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  @Test
  void testTruncatedFixed64() {
    // Fixed64 needs 8 bytes but only 7 provided
    Byte[] truncated = concat(box(tag(1, WT_64BIT)), 
        new Byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build();
         ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_FIXED},
             false)) {
      try (ColumnVector expected = ColumnVector.fromBoxedLongs((Long)null);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  @Test
  void testPartialLengthDelimitedData() {
    // Length says 10 bytes but only 5 provided
    Byte[] partial = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(10)),
        box("hello".getBytes()));  // only 5 bytes
    try (Table input = new Table.TestBuilder().column(new Byte[][]{partial}).build();
         ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.STRING.getTypeId().getNativeId()},
             new int[]{0},
             false)) {
      try (ColumnVector expected = ColumnVector.fromStrings((String)null);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  // ============================================================================
  // Wrong Wire Type Tests
  // ============================================================================

  @Test
  void testWrongWireType() {
    // Expect varint (wire type 0) but provide fixed32 (wire type 5)
    Byte[] wrongType = concat(
        box(tag(1, WT_32BIT)),  // wire type 5 instead of 0
        box(encodeFixed32(100)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{wrongType}).build();
         ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},  // expects varint
             new int[]{Protobuf.ENC_DEFAULT},
             false)) {
      try (ColumnVector expected = ColumnVector.fromBoxedLongs((Long)null);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  @Test
  void testWrongWireTypeForString() {
    // Expect length-delimited (wire type 2) but provide varint (wire type 0)
    Byte[] wrongType = concat(
        box(tag(1, WT_VARINT)),
        box(encodeVarint(12345)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{wrongType}).build();
         ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.STRING.getTypeId().getNativeId()},  // expects LEN
             new int[]{Protobuf.ENC_DEFAULT},
             false)) {
      try (ColumnVector expected = ColumnVector.fromStrings((String)null);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  // ============================================================================
  // Unknown Field Skip Tests
  // ============================================================================

  @Test
  void testSkipUnknownVarintField() {
    // Unknown field 99 with varint, followed by known field 1
    Byte[] row = concat(
        box(tag(99, WT_VARINT)),
        box(encodeVarint(12345)),  // unknown field to skip
        box(tag(1, WT_VARINT)),
        box(encodeVarint(42)));    // known field

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testSkipUnknownFixed64Field() {
    // Unknown field 99 with fixed64, followed by known field 1
    Byte[] row = concat(
        box(tag(99, WT_64BIT)),
        box(encodeFixed64(0x123456789ABCDEF0L)),  // unknown field to skip
        box(tag(1, WT_VARINT)),
        box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testSkipUnknownLengthDelimitedField() {
    // Unknown field 99 with length-delimited data, followed by known field 1
    Byte[] row = concat(
        box(tag(99, WT_LEN)),
        box(encodeVarint(5)),
        box("hello".getBytes()),  // unknown field to skip
        box(tag(1, WT_VARINT)),
        box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testSkipUnknownFixed32Field() {
    // Unknown field 99 with fixed32, followed by known field 1
    Byte[] row = concat(
        box(tag(99, WT_32BIT)),
        box(encodeFixed32(12345)),  // unknown field to skip
        box(tag(1, WT_VARINT)),
        box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Last One Wins (Repeated Scalar Field) Tests
  // ============================================================================

  @Test
  void testLastOneWins() {
    // Same field appears multiple times - last value should win
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(100)),
        box(tag(1, WT_VARINT)), box(encodeVarint(200)),
        box(tag(1, WT_VARINT)), box(encodeVarint(300)));  // this should win

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs(300L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testLastOneWinsForString() {
    // Same string field appears multiple times
    Byte[] row = concat(
        box(tag(1, WT_LEN)), box(encodeVarint(5)), box("first".getBytes()),
        box(tag(1, WT_LEN)), box(encodeVarint(6)), box("second".getBytes()),
        box(tag(1, WT_LEN)), box(encodeVarint(4)), box("last".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("last");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.STRING.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Error Handling Tests
  // ============================================================================

  @Test
  void testFailOnErrorsTrue() {
    Byte[] malformed = new Byte[]{(byte)0x08, (byte)0xFF, (byte)0xFF, (byte)0xFF,
                                   (byte)0xFF, (byte)0xFF, (byte)0xFF,
                                   (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{malformed}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{0},
             true)) {
        }
      });
    }
  }

  @Test
  void testFieldNumberZeroInvalid() {
    // Field number 0 is reserved and invalid
    Byte[] invalid = concat(box(tag(0, WT_VARINT)), box(encodeVarint(123)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{invalid}).build();
         ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{0},
             false)) {
      try (ColumnVector expected = ColumnVector.fromBoxedLongs((Long)null);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  @Test
  void testEmptyMessage() {
    // Empty message should result in null/default values for all fields
    Byte[] empty = new Byte[0];
    try (Table input = new Table.TestBuilder().column(new Byte[][]{empty}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedLongs((Long)null);
         ColumnVector expectedStr = ColumnVector.fromStrings((String)null);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt, expectedStr);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT64.getTypeId().getNativeId(), DType.STRING.getTypeId().getNativeId()},
             new int[]{0, 0})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Float/Double Special Values Tests
  // ============================================================================

  @Test
  void testFloatSpecialValues() {
    Byte[] rowInf = concat(box(tag(1, WT_32BIT)), box(encodeFloat(Float.POSITIVE_INFINITY)));
    Byte[] rowNegInf = concat(box(tag(1, WT_32BIT)), box(encodeFloat(Float.NEGATIVE_INFINITY)));
    Byte[] rowNaN = concat(box(tag(1, WT_32BIT)), box(encodeFloat(Float.NaN)));
    Byte[] rowMin = concat(box(tag(1, WT_32BIT)), box(encodeFloat(Float.MIN_VALUE)));
    Byte[] rowMax = concat(box(tag(1, WT_32BIT)), box(encodeFloat(Float.MAX_VALUE)));

    try (Table input = new Table.TestBuilder().column(rowInf, rowNegInf, rowNaN, rowMin, rowMax).build();
         ColumnVector expectedFloat = ColumnVector.fromBoxedFloats(
             Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN, 
             Float.MIN_VALUE, Float.MAX_VALUE);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedFloat);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.FLOAT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDoubleSpecialValues() {
    Byte[] rowInf = concat(box(tag(1, WT_64BIT)), box(encodeDouble(Double.POSITIVE_INFINITY)));
    Byte[] rowNegInf = concat(box(tag(1, WT_64BIT)), box(encodeDouble(Double.NEGATIVE_INFINITY)));
    Byte[] rowNaN = concat(box(tag(1, WT_64BIT)), box(encodeDouble(Double.NaN)));
    Byte[] rowMin = concat(box(tag(1, WT_64BIT)), box(encodeDouble(Double.MIN_VALUE)));
    Byte[] rowMax = concat(box(tag(1, WT_64BIT)), box(encodeDouble(Double.MAX_VALUE)));

    try (Table input = new Table.TestBuilder().column(rowInf, rowNegInf, rowNaN, rowMin, rowMax).build();
         ColumnVector expectedDouble = ColumnVector.fromBoxedDoubles(
             Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN,
             Double.MIN_VALUE, Double.MAX_VALUE);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedDouble);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.FLOAT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Tests for Features Not Yet Implemented (Disabled)
  // ============================================================================

  @Disabled("Unpacked repeated fields not yet implemented")
  @Test
  void testUnpackedRepeatedInt32() {
    // Unpacked repeated: same field number appears multiple times
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(1, WT_VARINT)), box(encodeVarint(3)));

    // Expected: ARRAY<INT32> with values [1, 2, 3]
    // (Currently we implement "last one wins" semantics for scalars)
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      // TODO: implement unpacked repeated field decoding
    }
  }

  @Disabled("Nested messages not yet implemented")
  @Test
  void testNestedMessage() {
    // message Inner { int32 x = 1; }
    // message Outer { Inner inner = 1; }
    // Outer with inner.x = 42
    Byte[] innerMessage = concat(box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(innerMessage.length)),
        innerMessage);

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      // TODO: implement nested message decoding
      // Expected: STRUCT<inner: STRUCT<x: INT32>>
    }
  }

  @Disabled("Large field numbers not tested with current API")
  @Test
  void testLargeFieldNumber() {
    // Field numbers can be up to 2^29 - 1 = 536870911
    int largeFieldNum = 536870911;
    Byte[] row = concat(
        box(tag(largeFieldNum, WT_VARINT)),
        box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      // Current API uses int[] for field numbers, should work
      // But need to verify kernel handles large field numbers correctly
    }
  }

  // ============================================================================
  // FAILFAST Mode Tests (failOnErrors = true)
  // ============================================================================

  @Test
  void testFailfastMalformedVarint() {
    // Varint that never terminates (all continuation bits set)
    Byte[] malformed = new Byte[]{(byte)0x08, (byte)0xFF, (byte)0xFF, (byte)0xFF,
                                   (byte)0xFF, (byte)0xFF, (byte)0xFF,
                                   (byte)0xFF, (byte)0xFF, (byte)0xFF, (byte)0xFF};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{malformed}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = decodeAllFields(
            input.getColumn(0),
            new int[]{1},
            new int[]{DType.INT64.getTypeId().getNativeId()},
            new int[]{0},
            true)) {  // failOnErrors = true
        }
      });
    }
  }

  @Test
  void testFailfastTruncatedVarint() {
    // Single byte with continuation bit set but no following byte
    Byte[] truncated = concat(box(tag(1, WT_VARINT)), new Byte[]{(byte)0x80});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = decodeAllFields(
            input.getColumn(0),
            new int[]{1},
            new int[]{DType.INT64.getTypeId().getNativeId()},
            new int[]{0},
            true)) {
        }
      });
    }
  }

  @Test
  void testFailfastTruncatedString() {
    // String field with length=5 but no actual data
    Byte[] truncated = concat(box(tag(2, WT_LEN)), box(encodeVarint(5)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = decodeAllFields(
            input.getColumn(0),
            new int[]{2},
            new int[]{DType.STRING.getTypeId().getNativeId()},
            new int[]{0},
            true)) {
        }
      });
    }
  }

  @Test
  void testFailfastTruncatedFixed32() {
    // Fixed32 needs 4 bytes but only 3 provided
    Byte[] truncated = concat(box(tag(1, WT_32BIT)), new Byte[]{0x01, 0x02, 0x03});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = decodeAllFields(
            input.getColumn(0),
            new int[]{1},
            new int[]{DType.INT32.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_FIXED},
            true)) {
        }
      });
    }
  }

  @Test
  void testFailfastTruncatedFixed64() {
    // Fixed64 needs 8 bytes but only 5 provided
    Byte[] truncated = concat(box(tag(1, WT_64BIT)), new Byte[]{0x01, 0x02, 0x03, 0x04, 0x05});
    try (Table input = new Table.TestBuilder().column(new Byte[][]{truncated}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = decodeAllFields(
            input.getColumn(0),
            new int[]{1},
            new int[]{DType.INT64.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_FIXED},
            true)) {
        }
      });
    }
  }

  @Test
  void testFailfastWrongWireType() {
    // Field 1 with wire type 2 (length-delimited), but we request varint
    Byte[] row = concat(box(tag(1, WT_LEN)), box(encodeVarint(3)), box("abc".getBytes()));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = decodeAllFields(
            input.getColumn(0),
            new int[]{1},
            new int[]{DType.INT64.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT},
            true)) {
        }
      });
    }
  }

  @Test
  void testFailfastFieldNumberZero() {
    // Field number 0 is invalid in protobuf
    Byte[] row = concat(box(tag(0, WT_VARINT)), box(encodeVarint(42)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = decodeAllFields(
            input.getColumn(0),
            new int[]{1},
            new int[]{DType.INT64.getTypeId().getNativeId()},
            new int[]{0},
            true)) {
        }
      });
    }
  }

  @Test
  void testFailfastValidDataDoesNotThrow() {
    // Valid protobuf should not throw even with failOnErrors = true
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{0},
             true)) {
      try (ColumnVector expected = ColumnVector.fromBoxedLongs(42L);
           ColumnVector expectedStruct = ColumnVector.makeStruct(expected)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, result);
      }
    }
  }

  // ============================================================================
  // Performance Benchmark Tests (Multi-field)
  // ============================================================================

  @Test
  void testMultiFieldPerformance() {
    // Test with 6 fields to verify fused kernel efficiency
    // message Msg { bool f1=1; int32 f2=2; int64 f3=3; float f4=4; double f5=5; string f6=6; }
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), new Byte[]{0x01},
        box(tag(2, WT_VARINT)), box(encodeVarint(12345)),
        box(tag(3, WT_VARINT)), box(encodeVarint(9876543210L)),
        box(tag(4, WT_32BIT)), box(encodeFloat(3.14f)),
        box(tag(5, WT_64BIT)), box(encodeDouble(2.71828)),
        box(tag(6, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1, 2, 3, 4, 5, 6},
             new int[]{
                 DType.BOOL8.getTypeId().getNativeId(),
                 DType.INT32.getTypeId().getNativeId(),
                 DType.INT64.getTypeId().getNativeId(),
                 DType.FLOAT32.getTypeId().getNativeId(),
                 DType.FLOAT64.getTypeId().getNativeId(),
                 DType.STRING.getTypeId().getNativeId()},
             new int[]{0, 0, 0, 0, 0, 0})) {
      try (ColumnVector expectedBool = ColumnVector.fromBoxedBooleans(true);
           ColumnVector expectedInt = ColumnVector.fromBoxedInts(12345);
           ColumnVector expectedLong = ColumnVector.fromBoxedLongs(9876543210L);
           ColumnVector expectedFloat = ColumnVector.fromBoxedFloats(3.14f);
           ColumnVector expectedDouble = ColumnVector.fromBoxedDoubles(2.71828);
           ColumnVector expectedString = ColumnVector.fromStrings("hello");
           ColumnVector expectedStruct = ColumnVector.makeStruct(
               expectedBool, expectedInt, expectedLong, expectedFloat, expectedDouble, expectedString)) {
        AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
      }
    }
  }
}
