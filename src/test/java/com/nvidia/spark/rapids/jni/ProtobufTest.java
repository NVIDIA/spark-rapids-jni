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
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.HostColumnVector.*;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
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
    return encodeVarint(((long) fieldNumber << 3) | wireType);
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
  // Helper methods for calling the unified API
  // ============================================================================

  private static ProtobufSchemaDescriptor makeScalarSchema(int[] fieldNumbers, int[] typeIds,
                                                           int[] encodings) {
    int n = fieldNumbers.length;
    int[] parentIndices = new int[n];
    int[] depthLevels = new int[n];
    int[] wireTypes = new int[n];
    boolean[] isRepeated = new boolean[n];
    boolean[] isRequired = new boolean[n];
    boolean[] hasDefault = new boolean[n];
    long[] defaultInts = new long[n];
    double[] defaultFloats = new double[n];
    boolean[] defaultBools = new boolean[n];
    byte[][] defaultStrings = new byte[n][];
    int[][] enumValid = new int[n][];
    byte[][][] enumNames = new byte[n][][];

    java.util.Arrays.fill(parentIndices, -1);
    for (int i = 0; i < n; i++) {
      wireTypes[i] = deriveWireType(typeIds[i], encodings[i]);
    }
    return new ProtobufSchemaDescriptor(fieldNumbers, parentIndices, depthLevels,
        wireTypes, typeIds, encodings, isRepeated, isRequired, hasDefault,
        defaultInts, defaultFloats, defaultBools, defaultStrings, enumValid, enumNames);
  }

  private static int deriveWireType(int typeId, int encoding) {
    if (encoding == Protobuf.ENC_ENUM_STRING) return Protobuf.WT_VARINT;
    if (typeId == DType.FLOAT32.getTypeId().getNativeId()) return Protobuf.WT_32BIT;
    if (typeId == DType.FLOAT64.getTypeId().getNativeId()) return Protobuf.WT_64BIT;
    if (typeId == DType.STRING.getTypeId().getNativeId()) return Protobuf.WT_LEN;
    if (typeId == DType.LIST.getTypeId().getNativeId()) return Protobuf.WT_LEN;
    if (typeId == DType.STRUCT.getTypeId().getNativeId()) return Protobuf.WT_LEN;
    if (encoding == Protobuf.ENC_FIXED) {
      if (typeId == DType.INT64.getTypeId().getNativeId()
          || typeId == DType.UINT64.getTypeId().getNativeId()) return Protobuf.WT_64BIT;
      return Protobuf.WT_32BIT;
    }
    return Protobuf.WT_VARINT;
  }

  /**
   * Test-only convenience: wrap raw parallel arrays into a ProtobufSchemaDescriptor
   * and decode. Avoids verbose ProtobufSchemaDescriptor construction at every call site.
   */
  private static ColumnVector decodeRaw(ColumnView binaryInput,
                                        int[] fieldNumbers, int[] parentIndices, int[] depthLevels,
                                        int[] wireTypes, int[] outputTypeIds, int[] encodings,
                                        boolean[] isRepeated, boolean[] isRequired,
                                        boolean[] hasDefaultValue, long[] defaultInts,
                                        double[] defaultFloats, boolean[] defaultBools,
                                        byte[][] defaultStrings, int[][] enumValidValues,
                                        boolean failOnErrors) {
    return decodeRaw(binaryInput, fieldNumbers, parentIndices, depthLevels,
        wireTypes, outputTypeIds, encodings, isRepeated, isRequired,
        hasDefaultValue, defaultInts, defaultFloats, defaultBools,
        defaultStrings, enumValidValues, new byte[fieldNumbers.length][][], failOnErrors);
  }

  private static ColumnVector decodeRaw(ColumnView binaryInput,
                                        int[] fieldNumbers, int[] parentIndices, int[] depthLevels,
                                        int[] wireTypes, int[] outputTypeIds, int[] encodings,
                                        boolean[] isRepeated, boolean[] isRequired,
                                        boolean[] hasDefaultValue, long[] defaultInts,
                                        double[] defaultFloats, boolean[] defaultBools,
                                        byte[][] defaultStrings, int[][] enumValidValues,
                                        byte[][][] enumNames,
                                        boolean failOnErrors) {
    return Protobuf.decodeToStruct(binaryInput,
        new ProtobufSchemaDescriptor(fieldNumbers, parentIndices, depthLevels,
            wireTypes, outputTypeIds, encodings, isRepeated, isRequired,
            hasDefaultValue, defaultInts, defaultFloats, defaultBools,
            defaultStrings, enumValidValues, enumNames),
        failOnErrors);
  }

  /**
   * Helper to decode all scalar fields using the unified API.
   * Builds a flat schema (parentIndices=-1, depth=0, isRepeated=false for all fields).
   */
  private static ColumnVector decodeScalarFields(ColumnView binaryInput,
                                                 int[] fieldNumbers,
                                                 int[] typeIds,
                                                 int[] encodings,
                                                 boolean[] isRequired,
                                                 boolean[] hasDefaultValue,
                                                 long[] defaultInts,
                                                 double[] defaultFloats,
                                                 boolean[] defaultBools,
                                                 byte[][] defaultStrings,
                                                 int[][] enumValidValues,
                                                 boolean failOnErrors) {
    int numFields = fieldNumbers.length;
    int[] parentIndices = new int[numFields];
    int[] depthLevels = new int[numFields];
    int[] wireTypes = new int[numFields];
    boolean[] isRepeated = new boolean[numFields];

    java.util.Arrays.fill(parentIndices, -1);
    // depthLevels already initialized to 0
    // isRepeated already initialized to false
    for (int i = 0; i < numFields; i++) {
      wireTypes[i] = deriveWireType(typeIds[i], encodings[i]);
    }

    return Protobuf.decodeToStruct(binaryInput,
        new ProtobufSchemaDescriptor(fieldNumbers, parentIndices, depthLevels,
            wireTypes, typeIds, encodings, isRepeated, isRequired, hasDefaultValue,
            defaultInts, defaultFloats, defaultBools, defaultStrings, enumValidValues,
            new byte[fieldNumbers.length][][]),
        failOnErrors);
  }

  /**
   * Helper method that wraps the unified API for tests that decode all scalar fields.
   */
  private static ColumnVector decodeAllFields(ColumnView binaryInput,
                                              int[] fieldNumbers,
                                              int[] typeIds,
                                              int[] encodings) {
    return decodeAllFields(binaryInput, fieldNumbers, typeIds, encodings, true);
  }

  /**
   * Helper method that wraps the unified API for tests that decode all scalar fields.
   */
  private static ColumnVector decodeAllFields(ColumnView binaryInput,
                                              int[] fieldNumbers,
                                              int[] typeIds,
                                              int[] encodings,
                                              boolean failOnErrors) {
    int numFields = fieldNumbers.length;
    return decodeScalarFields(binaryInput, fieldNumbers, typeIds, encodings,
        new boolean[numFields], new boolean[numFields], new long[numFields],
        new double[numFields], new boolean[numFields], new byte[numFields][],
        new int[numFields][], failOnErrors);
  }

  private static void assertSingleNullStructRow(ColumnVector actual, String message) {
    try (HostColumnVector hostStruct = actual.copyToHost()) {
      assertEquals(1, actual.getNullCount(), message);
      assertTrue(hostStruct.isNull(0), "Row 0 should be null");
    }
  }

  /**
   * Helper method for tests with required field support.
   */
  private static ColumnVector decodeAllFieldsWithRequired(ColumnView binaryInput,
                                                          int[] fieldNumbers,
                                                          int[] typeIds,
                                                          int[] encodings,
                                                          boolean[] isRequired,
                                                          boolean failOnErrors) {
    int numFields = fieldNumbers.length;
    return decodeScalarFields(binaryInput, fieldNumbers, typeIds, encodings,
        isRequired, new boolean[numFields], new long[numFields],
        new double[numFields], new boolean[numFields], new byte[numFields][],
        new int[numFields][], failOnErrors);
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
      // Row 2 is a null input row — the output struct row should also be null.
      assertEquals(3, actualStruct.getRowCount());
      assertEquals(1, actualStruct.getNullCount());
      try (HostColumnVector hcv = actualStruct.copyToHost()) {
        assertFalse(hcv.isNull(0));
        assertFalse(hcv.isNull(1));
        assertTrue(hcv.isNull(2), "Null input row should produce null struct row");
      }
      // Children for non-null rows should still match
      try (ColumnVector childId = actualStruct.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostId = childId.copyToHost();
           ColumnVector childName = actualStruct.getChildColumnView(1).copyToColumnVector();
           HostColumnVector hostName = childName.copyToHost()) {
        assertEquals(100L, hostId.getLong(0));
        assertEquals(200L, hostId.getLong(1));
        assertEquals("alice", hostName.getJavaString(0));
        assertTrue(hostName.isNull(1));
      }
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
            new ListType(true, new BasicType(true, DType.UINT8)),
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
         // Expected: f1=100, f3=42 (schema projection: only decode these two)
         ColumnVector expectedF1 = ColumnVector.fromBoxedLongs(100L);
         ColumnVector expectedF3 = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedF1, expectedF3);
         // Decode only f1 (field_number=1) and f3 (field_number=3), skip f2
         // With the unified API, we only include the fields we want in the schema
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1, 3},  // field numbers for f1 and f3
             new int[]{DType.INT64.getTypeId().getNativeId(),
                       DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT})) {
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
         // With no fields in the schema, the GPU returns an empty struct
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{},  // no field numbers
             new int[]{},  // no types
             new int[]{})) {  // no encodings
      assertNotNull(actualStruct);
      assertEquals(DType.STRUCT, actualStruct.getType());
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

  // ============================================================================
  // Output shape tests — verify the stub produces correctly typed struct columns
  // ============================================================================

  @Test
  void testEmptySchemaProducesEmptyStruct() {
    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             makeScalarSchema(new int[]{}, new int[]{}, new int[]{}), true)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getRowCount());
      assertEquals(0, result.getNumChildren());
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
  void testSingleScalarFieldOutputShape() {
    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             makeScalarSchema(
                 new int[]{1},
                 new int[]{DType.INT64.getTypeId().getNativeId()},
                 new int[]{Protobuf.ENC_DEFAULT}), true)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getRowCount());
      assertEquals(1, result.getNumChildren());
      assertEquals(DType.INT64, result.getChildColumnView(0).getType());
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
      assertSingleNullStructRow(result, "Malformed varint should null the struct row");
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
      assertSingleNullStructRow(result, "Truncated varint should null the struct row");
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
      assertSingleNullStructRow(result,
          "Truncated length-delimited field should null the struct row");
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
      assertSingleNullStructRow(result, "Truncated fixed32 should null the struct row");
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
      assertSingleNullStructRow(result, "Truncated fixed64 should null the struct row");
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
      assertSingleNullStructRow(result,
          "Partial length-delimited payload should null the struct row");
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
      assertSingleNullStructRow(result, "Wrong wire type should null the struct row");
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
      assertSingleNullStructRow(result, "Wrong wire type for string should null the struct row");
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
      assertSingleNullStructRow(result, "Field number zero should null the struct row");
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
  // Enum Tests (enums.as.ints=true semantics)
  // ============================================================================

  @Test
  void testEnumAsInt() {
    // message Msg { enum Color { RED=0; GREEN=1; BLUE=2; } Color c = 1; }
    // c = GREEN (value 1) - encoded as varint
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(1);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumZeroValue() {
    // Enum with value 0 (first/default enum value)
    // c = RED (value 0)
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(0)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(0);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumUnknownValue() {
    // Protobuf allows unknown enum values - they should still be decoded as integers
    // c = 999 (unknown value not in enum definition)
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(999);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumNegativeValue() {
    // Negative enum values are valid in protobuf (stored as unsigned varint)
    // c = -1 (represented as 0xFFFFFFFF in protobuf wire format)
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(-1L & 0xFFFFFFFFL)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(-1);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumMultipleFields() {
    // message Msg { enum Status { OK=0; ERROR=1; } Status s1 = 1; int32 count = 2; Status s2 = 3; }
    // s1 = ERROR (1), count = 42, s2 = OK (0)
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),   // s1 = ERROR
        box(tag(2, WT_VARINT)), box(encodeVarint(42)),  // count = 42
        box(tag(3, WT_VARINT)), box(encodeVarint(0)));  // s2 = OK

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedS1 = ColumnVector.fromBoxedInts(1);
         ColumnVector expectedCount = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedS2 = ColumnVector.fromBoxedInts(0);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedS1, expectedCount, expectedS2);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1, 2, 3},
             new int[]{DType.INT32.getTypeId().getNativeId(),
                       DType.INT32.getTypeId().getNativeId(),
                       DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumMissingField() {
    // Enum field not present in message - should be null
    Byte[] row = concat(box(tag(2, WT_VARINT)), box(encodeVarint(42)));  // only count field

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedEnum = ColumnVector.fromBoxedInts((Integer) null);
         ColumnVector expectedCount = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedEnum, expectedCount);
         ColumnVector actualStruct = decodeAllFields(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT32.getTypeId().getNativeId(),
                       DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT})) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Required Field Tests
  // ============================================================================

  @Test
  void testRequiredFieldPresent() {
    // message Msg { required int64 id = 1; optional string name = 2; }
    // Both fields present - should decode successfully
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(42)),
        box(tag(2, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedId = ColumnVector.fromBoxedLongs(42L);
         ColumnVector expectedName = ColumnVector.fromStrings("hello");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedId, expectedName);
         ColumnVector actualStruct = decodeAllFieldsWithRequired(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT64.getTypeId().getNativeId(), DType.STRING.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new boolean[]{true, false},  // id is required, name is optional
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testRequiredFieldMissing_Permissive() {
    // Required field missing in permissive mode - should null the whole row without exception
    // message Msg { required int64 id = 1; optional string name = 2; }
    // Only name field present, required id is missing
    Byte[] row = concat(
        box(tag(2, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actualStruct = decodeAllFieldsWithRequired(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT64.getTypeId().getNativeId(), DType.STRING.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new boolean[]{true, false},  // id is required, name is optional
             false)) {  // permissive mode - don't fail on errors
      assertSingleNullStructRow(actualStruct,
          "Missing top-level required field should null the row in PERMISSIVE mode");
    }
  }

  @Test
  void testRequiredFieldMissing_Failfast() {
    // Required field missing in failfast mode - should throw exception
    // message Msg { required int64 id = 1; optional string name = 2; }
    // Only name field present, required id is missing
    Byte[] row = concat(
        box(tag(2, WT_LEN)), box(encodeVarint(5)), box("hello".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = decodeAllFieldsWithRequired(
            input.getColumn(0),
            new int[]{1, 2},
            new int[]{DType.INT64.getTypeId().getNativeId(), DType.STRING.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
            new boolean[]{true, false},  // id is required, name is optional
            true)) {  // failfast mode - should throw
        }
      });
    }
  }

  @Test
  void testMultipleRequiredFields_AllPresent() {
    // message Msg { required int32 a = 1; required int64 b = 2; required string c = 3; }
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(10)),
        box(tag(2, WT_VARINT)), box(encodeVarint(20)),
        box(tag(3, WT_LEN)), box(encodeVarint(3)), box("abc".getBytes()));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts(10);
         ColumnVector expectedB = ColumnVector.fromBoxedLongs(20L);
         ColumnVector expectedC = ColumnVector.fromStrings("abc");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB, expectedC);
         ColumnVector actualStruct = decodeAllFieldsWithRequired(
             input.getColumn(0),
             new int[]{1, 2, 3},
             new int[]{DType.INT32.getTypeId().getNativeId(),
                       DType.INT64.getTypeId().getNativeId(),
                       DType.STRING.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new boolean[]{true, true, true},  // all required
             true)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testMultipleRequiredFields_SomeMissing_Failfast() {
    // message Msg { required int32 a = 1; required int64 b = 2; required string c = 3; }
    // Only field a is present, b and c are missing
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(10)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = decodeAllFieldsWithRequired(
            input.getColumn(0),
            new int[]{1, 2, 3},
            new int[]{DType.INT32.getTypeId().getNativeId(),
                      DType.INT64.getTypeId().getNativeId(),
                      DType.STRING.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
            new boolean[]{true, true, true},  // all required
            true)) {
        }
      });
    }
  }

  @Test
  void testOptionalFieldsOnly_NoValidation() {
    // All fields optional - missing fields should not cause error
    // message Msg { optional int32 a = 1; optional int64 b = 2; }
    Byte[] row = new Byte[0];  // empty message

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts((Integer) null);
         ColumnVector expectedB = ColumnVector.fromBoxedLongs((Long) null);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB);
         ColumnVector actualStruct = decodeAllFieldsWithRequired(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT32.getTypeId().getNativeId(), DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new boolean[]{false, false},  // all optional
             true)) {  // even with failOnErrors=true, should succeed since all fields are optional
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testRequiredFieldWithMultipleRows() {
    // Test required field validation across multiple rows
    // Row 0: required field present
    // Row 1: required field missing (should cause error in failfast mode)
    Byte[] row0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    Byte[] row1 = new Byte[0];  // empty - required field missing

    try (Table input = new Table.TestBuilder().column(row0, row1).build()) {
      assertThrows(ai.rapids.cudf.CudfException.class, () -> {
        try (ColumnVector result = decodeAllFieldsWithRequired(
            input.getColumn(0),
            new int[]{1},
            new int[]{DType.INT64.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT},
            new boolean[]{true},  // required
            true)) {
        }
      });
    }
  }

  @Test
  void testRequiredFieldIgnoresNullInputRow_Failfast() {
    Byte[] row0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    Byte[] row1 = null;

    try (Table input = new Table.TestBuilder().column(row0, row1).build();
         ColumnVector actualStruct = decodeAllFieldsWithRequired(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{true},
             true);
         ColumnVector idCol = actualStruct.getChildColumnView(0).copyToColumnVector();
         HostColumnVector hostStruct = actualStruct.copyToHost();
         HostColumnVector hostId = idCol.copyToHost()) {
      assertEquals(1, actualStruct.getNullCount(), "Null input row should be null in output struct");
      assertFalse(hostStruct.isNull(0), "Present required field should keep row 0 valid");
      assertTrue(hostStruct.isNull(1), "Null input row should produce null struct row");
      assertEquals(1, idCol.getNullCount(), "The required child value should be null on the null input row");
      assertTrue(hostId.isNull(1), "Null input row should produce a null child value, not ERR_REQUIRED");
    }
  }

  // ============================================================================
  // Default Value Tests (API accepts parameters, CUDA fill not yet implemented)
  // ============================================================================

  /**
   * Helper method for tests with default value support.
   */
  private static ColumnVector decodeAllFieldsWithDefaults(ColumnView binaryInput,
                                                          int[] fieldNumbers,
                                                          int[] typeIds,
                                                          int[] encodings,
                                                          boolean[] isRequired,
                                                          boolean[] hasDefaultValue,
                                                          long[] defaultInts,
                                                          double[] defaultFloats,
                                                          boolean[] defaultBools,
                                                          byte[][] defaultStrings,
                                                          boolean failOnErrors) {
    int numFields = fieldNumbers.length;
    return decodeScalarFields(binaryInput, fieldNumbers, typeIds, encodings,
        isRequired, hasDefaultValue, defaultInts, defaultFloats, defaultBools,
        defaultStrings, new int[numFields][], failOnErrors);
  }

  @Test
  void testDefaultValueForMissingFields() {
    // Test that missing fields with default values return the defaults
    Byte[] row = new Byte[0];  // empty message

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         // With default values set, missing fields should return the default values
         ColumnVector expectedA = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedB = ColumnVector.fromBoxedLongs(100L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT32.getTypeId().getNativeId(), DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new boolean[]{false, false},  // not required
             new boolean[]{true, true},    // has default value
             new long[]{42, 100},          // default int values (42, 100)
             new double[]{0.0, 0.0},       // default float values (unused for int fields)
             new boolean[]{false, false},  // default bool values (unused)
             new byte[][]{null, null},     // default string values (unused)
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultValueFieldPresent_OverridesDefault() {
    // When field is present, use the actual value (not the default)
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(99)),
        box(tag(2, WT_VARINT)), box(encodeVarint(200)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts(99);
         ColumnVector expectedB = ColumnVector.fromBoxedLongs(200L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT32.getTypeId().getNativeId(), DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new boolean[]{false, false},  // not required
             new boolean[]{true, true},    // has default value
             new long[]{42, 100},          // default values - NOT used since field is present
             new double[]{0.0, 0.0},
             new boolean[]{false, false},
             new byte[][]{null, null},
             false)) {
      // Actual values should be used, not defaults
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultIntValue() {
    // optional int32 count = 1 [default = 42];
    // Empty message should return the default value
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{false},   // not required
             new boolean[]{true},    // has default
             new long[]{42},         // default = 42
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{null},
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultBoolValue() {
    // optional bool flag = 1 [default = true];
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedBool = ColumnVector.fromBoxedBooleans(true);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedBool);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.BOOL8.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{false},
             new boolean[]{true},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{true},  // default = true
             new byte[][]{null},
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultFloatValue() {
    // optional double rate = 1 [default = 3.14];
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedDouble = ColumnVector.fromBoxedDoubles(3.14);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedDouble);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.FLOAT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{false},
             new boolean[]{true},
             new long[]{0},
             new double[]{3.14},  // default = 3.14
             new boolean[]{false},
             new byte[][]{null},
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultInt64Value() {
    // optional int64 big_num = 1 [default = 9876543210];
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedLong = ColumnVector.fromBoxedLongs(9876543210L);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedLong);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{false},
             new boolean[]{true},
             new long[]{9876543210L},  // default = 9876543210
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{null},
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testMixedDefaultAndNonDefaultFields() {
    // optional int32 a = 1 [default = 42];
    // optional int64 b = 2; (no default)
    // optional bool c = 3 [default = true];
    // Empty message: a=42, b=null, c=true
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedB = ColumnVector.fromBoxedLongs((Long) null);  // no default
         ColumnVector expectedC = ColumnVector.fromBoxedBooleans(true);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB, expectedC);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1, 2, 3},
             new int[]{DType.INT32.getTypeId().getNativeId(),
                       DType.INT64.getTypeId().getNativeId(),
                       DType.BOOL8.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new boolean[]{false, false, false},  // not required
             new boolean[]{true, false, true},    // a and c have defaults, b doesn't
             new long[]{42, 0, 0},                // default for a
             new double[]{0.0, 0.0, 0.0},
             new boolean[]{false, false, true},   // default for c
             new byte[][]{null, null, null},
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultValueWithPartialMessage() {
    // optional int32 a = 1 [default = 42];
    // optional int64 b = 2 [default = 100];
    // Message has only field b set, a should use default
    Byte[] row = concat(
        box(tag(2, WT_VARINT)), box(encodeVarint(999)));  // b = 999

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedA = ColumnVector.fromBoxedInts(42);  // default
         ColumnVector expectedB = ColumnVector.fromBoxedLongs(999L);  // actual value
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedA, expectedB);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT32.getTypeId().getNativeId(), DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new boolean[]{false, false},  // not required
             new boolean[]{true, true},    // both have defaults
             new long[]{42, 100},
             new double[]{0.0, 0.0},
             new boolean[]{false, false},
             new byte[][]{null, null},
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultStringValue() {
    // optional string name = 1 [default = "hello"];
    // Empty message should return the default string
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("hello");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.STRING.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{false},   // not required
             new boolean[]{true},    // has default
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{"hello".getBytes(java.nio.charset.StandardCharsets.UTF_8)},
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultStringValueEmpty() {
    // optional string name = 1 [default = ""];
    // Empty message with empty default string
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.STRING.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{false},
             new boolean[]{true},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{new byte[0]},  // empty default string
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultStringValueWithPresent() {
    // optional string name = 1 [default = "default"];
    // Message has actual value, should override default
    byte[] strBytesRaw = "actual".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(strBytesRaw.length)),
        box(strBytesRaw));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("actual");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.STRING.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{false},
             new boolean[]{true},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{"default".getBytes(java.nio.charset.StandardCharsets.UTF_8)},
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultStringWithMixedFields() {
    // optional int32 count = 1 [default = 42];
    // optional string name = 2 [default = "test"];
    // Empty message should return both defaults
    Byte[] row = new Byte[0];

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedInt = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStr = ColumnVector.fromStrings("test");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedInt, expectedStr);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT32.getTypeId().getNativeId(), DType.STRING.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new boolean[]{false, false},
             new boolean[]{true, true},
             new long[]{42, 0},
             new double[]{0.0, 0.0},
             new boolean[]{false, false},
             new byte[][]{null, "test".getBytes(java.nio.charset.StandardCharsets.UTF_8)},
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testDefaultStringMultipleRows() {
    // optional string name = 1 [default = "default"];
    // Multiple rows: empty, has value, empty
    Byte[] row1 = new Byte[0];  // will use default
    byte[] strBytesRaw = "row2val".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    Byte[] row2 = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(strBytesRaw.length)),
        box(strBytesRaw));
    Byte[] row3 = new Byte[0];  // will use default

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row1, row2, row3}).build();
         ColumnVector expectedStr = ColumnVector.fromStrings("default", "row2val", "default");
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedStr);
         ColumnVector actualStruct = decodeAllFieldsWithDefaults(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.STRING.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{false},
             new boolean[]{true},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{"default".getBytes(java.nio.charset.StandardCharsets.UTF_8)},
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Tests for Nested and Repeated Fields (Phase 1-3 Implementation)
  // ============================================================================

  @Test
  void testUnpackedRepeatedInt32() {
    // Unpacked repeated: same field number appears multiple times
    // message TestMsg { repeated int32 ids = 1; }
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(1, WT_VARINT)), box(encodeVarint(3)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      // Use the new nested API for repeated fields
      // Field: ids (field_number=1, parent=-1, depth=0, wire_type=VARINT, type=INT32, repeated=true)
      try (ColumnVector result = decodeRaw(
          input.getColumn(0),
          new int[]{1},                    // fieldNumbers
          new int[]{-1},                   // parentIndices (-1 = top level)
          new int[]{0},                    // depthLevels
          new int[]{Protobuf.WT_VARINT},   // wireTypes
          new int[]{DType.INT32.getTypeId().getNativeId()},  // outputTypeIds (element type)
          new int[]{Protobuf.ENC_DEFAULT}, // encodings
          new boolean[]{true},             // isRepeated
          new boolean[]{false},            // isRequired
          new boolean[]{false},            // hasDefaultValue
          new long[]{0},                   // defaultInts
          new double[]{0.0},               // defaultFloats
          new boolean[]{false},            // defaultBools
          new byte[][]{null},              // defaultStrings
          new int[][]{null},               // enumValidValues
          false)) {                        // failOnErrors
        // Result should be STRUCT<ids: LIST<INT32>> containing [1, 2, 3]
        assertNotNull(result);
        assertEquals(DType.STRUCT, result.getType());
        try (ColumnVector listCol = result.getChildColumnView(0).copyToColumnVector();
             ColumnVector children = listCol.getChildColumnView(0).copyToColumnVector();
             HostColumnVector hostChildren = children.copyToHost()) {
          assertEquals(DType.LIST, listCol.getType());
          assertEquals(DType.INT32, children.getType());
          assertEquals(3, children.getRowCount());
          assertEquals(1, hostChildren.getInt(0));
          assertEquals(2, hostChildren.getInt(1));
          assertEquals(3, hostChildren.getInt(2));
        }
      }
    }
  }

  @Test
  void testPackedRepeatedDoubleWithMultipleFields() {
    // Test packed repeated fields with multiple types including edge cases.
    // message WithPackedRepeated {
    //   optional int32 id = 1;
    //   repeated int32 int_values = 2 [packed=true];
    //   repeated double double_values = 3 [packed=true];
    //   repeated bool bool_values = 4 [packed=true];
    // }

    // Helper to build packed int data (varints)
    java.io.ByteArrayOutputStream intBuf = new java.io.ByteArrayOutputStream();

    // Row 0: id=42, int_values=[1,-1,100] (12 bytes packed), double_values=[1.5,2.5], bool=[true,false]
    // Row 1: id=7, int_values=15x(-1) (150 bytes packed, 2-byte length varint!), double_values=[3.0,4.0], bool=[true]
    // Row 2: id=0, int_values=[] (field omitted), double_values=[5.0], bool=[] (field omitted)

    // --- Row 0 ---
    byte[] r0IntVarints = concatBytes(encodeVarint(1), encodeVarint(-1L & 0xFFFFFFFFFFFFFFFFL), encodeVarint(100));
    byte[] r0Doubles = concatBytes(encodeDouble(1.5), encodeDouble(2.5));
    byte[] r0Bools = new byte[]{0x01, 0x00};
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(42)),
        box(tag(2, WT_LEN)), box(encodeVarint(r0IntVarints.length)), box(r0IntVarints),
        box(tag(3, WT_LEN)), box(encodeVarint(r0Doubles.length)), box(r0Doubles),
        box(tag(4, WT_LEN)), box(encodeVarint(r0Bools.length)), box(r0Bools));

    // --- Row 1: 15 negative ints => 150 bytes packed (length varint is 2 bytes: 0x96 0x01) ---
    java.io.ByteArrayOutputStream buf1 = new java.io.ByteArrayOutputStream();
    byte[] negOneVarint = encodeVarint(-1L & 0xFFFFFFFFFFFFFFFFL); // 10 bytes
    for (int i = 0; i < 15; i++) {
      buf1.write(negOneVarint, 0, negOneVarint.length);
    }
    byte[] r1IntVarints = buf1.toByteArray(); // 150 bytes
    byte[] r1Doubles = concatBytes(encodeDouble(3.0), encodeDouble(4.0));
    byte[] r1Bools = new byte[]{0x01};
    Byte[] row1 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(2, WT_LEN)), box(encodeVarint(r1IntVarints.length)), box(r1IntVarints),
        box(tag(3, WT_LEN)), box(encodeVarint(r1Doubles.length)), box(r1Doubles),
        box(tag(4, WT_LEN)), box(encodeVarint(r1Bools.length)), box(r1Bools));

    // --- Row 2: no int_values, no bool_values ---
    byte[] r2Doubles = encodeDouble(5.0);
    Byte[] row2 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(0)),
        box(tag(3, WT_LEN)), box(encodeVarint(r2Doubles.length)), box(r2Doubles));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1, row2}).build()) {
      try (ColumnVector result = decodeRaw(
          input.getColumn(0),
          new int[]{1, 2, 3, 4},
          new int[]{-1, -1, -1, -1},
          new int[]{0, 0, 0, 0},
          new int[]{WT_VARINT, WT_VARINT, WT_64BIT, WT_VARINT},
          new int[]{
            DType.INT32.getTypeId().getNativeId(),
            DType.INT32.getTypeId().getNativeId(),
            DType.FLOAT64.getTypeId().getNativeId(),
            DType.BOOL8.getTypeId().getNativeId()
          },
          new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
          new boolean[]{false, true, true, true},
          new boolean[]{false, false, false, false},
          new boolean[]{false, false, false, false},
          new long[]{0, 0, 0, 0},
          new double[]{0.0, 0.0, 0.0, 0.0},
          new boolean[]{false, false, false, false},
          new byte[][]{null, null, null, null},
          new int[][]{null, null, null, null},
          false)) {
        assertNotNull(result);
        assertEquals(DType.STRUCT, result.getType());
        assertEquals(3, result.getRowCount());

        // Verify double_values child column has correct total count: 2 + 2 + 1 = 5
        try (ColumnVector doubleListCol = result.getChildColumnView(2).copyToColumnVector()) {
          assertEquals(DType.LIST, doubleListCol.getType());
          try (ColumnVector doubleChildren = doubleListCol.getChildColumnView(0).copyToColumnVector()) {
            assertEquals(DType.FLOAT64, doubleChildren.getType());
            assertEquals(5, doubleChildren.getRowCount(),
                "Total packed doubles across 3 rows should be 5, got " + doubleChildren.getRowCount());
            try (HostColumnVector hd = doubleChildren.copyToHost()) {
              assertEquals(1.5, hd.getDouble(0), 1e-10);
              assertEquals(2.5, hd.getDouble(1), 1e-10);
              assertEquals(3.0, hd.getDouble(2), 1e-10);
              assertEquals(4.0, hd.getDouble(3), 1e-10);
              assertEquals(5.0, hd.getDouble(4), 1e-10);
            }
          }
        }
      }
    }
  }

  /** Helper: concatenate byte arrays */
  private static byte[] concatBytes(byte[]... arrays) {
    int len = 0;
    for (byte[] a : arrays) len += a.length;
    byte[] out = new byte[len];
    int pos = 0;
    for (byte[] a : arrays) {
      System.arraycopy(a, 0, out, pos, a.length);
      pos += a.length;
    }
    return out;
  }

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
      // Flattened schema:
      // [0] inner: STRUCT, field_number=1, parent=-1, depth=0
      // [1] inner.x: INT32, field_number=1, parent=0, depth=1
      try (ColumnVector result = decodeRaw(
          input.getColumn(0),
          new int[]{1, 1},                 // fieldNumbers
          new int[]{-1, 0},                // parentIndices
          new int[]{0, 1},                 // depthLevels
          new int[]{Protobuf.WT_LEN, Protobuf.WT_VARINT},  // wireTypes
          new int[]{DType.STRUCT.getTypeId().getNativeId(), DType.INT32.getTypeId().getNativeId()},
          new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
          new boolean[]{false, false},     // isRepeated
          new boolean[]{false, false},     // isRequired
          new boolean[]{false, false},     // hasDefaultValue
          new long[]{0, 0},
          new double[]{0.0, 0.0},
          new boolean[]{false, false},
          new byte[][]{null, null},
          new int[][]{null, null},
          false)) {
        assertNotNull(result);
        assertEquals(DType.STRUCT, result.getType());
      }
    }
  }

  @Test
  void testPermissiveRepeatedWrongWireTypeDoesNotCorruptFollowingRow() {
    // message Msg { repeated int32 ids = 1; }
    // Row 0 has one valid element, then a malformed fixed32 occurrence for the same field,
    // then another valid varint that must be ignored once the row is marked malformed.
    // Row 1 must keep its own slot and not be overwritten by row 0's trailing occurrence.
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_32BIT)), box(encodeFixed32(77)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)));
    Byte[] row1 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(100)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector expectedIds = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             Arrays.asList(1),
             Arrays.asList(100));
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedIds);
         ColumnVector actualStruct = decodeRaw(
             input.getColumn(0),
             new int[]{1},
             new int[]{-1},
             new int[]{0},
             new int[]{WT_VARINT},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{true},
             new boolean[]{false},
             new boolean[]{false},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{null},
             new int[][]{null},
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testRepeatedUint32() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),
        box(tag(1, WT_VARINT)), box(encodeVarint(3)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = decodeRaw(
             input.getColumn(0),
             new int[]{1},
             new int[]{-1},
             new int[]{0},
             new int[]{WT_VARINT},
             new int[]{DType.UINT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{true},
             new boolean[]{false},
             new boolean[]{false},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{null},
             new int[][]{null},
             false)) {
      try (ColumnVector list = result.getChildColumnView(0).copyToColumnVector();
           ColumnVector vals = list.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostVals = vals.copyToHost()) {
        assertEquals(DType.UINT32, vals.getType());
        assertEquals(3, vals.getRowCount());
        assertEquals(1, Integer.toUnsignedLong(hostVals.getInt(0)));
        assertEquals(2, Integer.toUnsignedLong(hostVals.getInt(1)));
        assertEquals(3, Integer.toUnsignedLong(hostVals.getInt(2)));
      }
    }
  }

  @Test
  void testRepeatedUint64() {
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(11)),
        box(tag(1, WT_VARINT)), box(encodeVarint(22)),
        box(tag(1, WT_VARINT)), box(encodeVarint(33)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = decodeRaw(
             input.getColumn(0),
             new int[]{1},
             new int[]{-1},
             new int[]{0},
             new int[]{WT_VARINT},
             new int[]{DType.UINT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{true},
             new boolean[]{false},
             new boolean[]{false},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{null},
             new int[][]{null},
             false)) {
      try (ColumnVector list = result.getChildColumnView(0).copyToColumnVector();
           ColumnVector vals = list.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostVals = vals.copyToHost()) {
        assertEquals(DType.UINT64, vals.getType());
        assertEquals(3, vals.getRowCount());
        assertEquals(11L, hostVals.getLong(0));
        assertEquals(22L, hostVals.getLong(1));
        assertEquals(33L, hostVals.getLong(2));
      }
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
  void testFailfastFieldNumberAboveSpecLimit() {
    // Protobuf field numbers must be <= 2^29 - 1.
    Byte[] row = concat(box(tag(1 << 29, WT_VARINT)), box(encodeVarint(42)));
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
  void testUnknownEndGroupWireTypeNullsMalformedRow() {
    Byte[] row = concat(
        box(tag(5, 4)),
        box(tag(1, WT_VARINT)), box(encodeVarint(42)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = decodeAllFields(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT64.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             false)) {
      assertSingleNullStructRow(actual, "Unknown end-group wire type should null the struct row");
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

  // ============================================================================
  // Enum Validation Tests
  // ============================================================================

  /**
   * Helper method that wraps decodeToStruct with enum validation support.
   */
  private static ColumnVector decodeAllFieldsWithEnums(ColumnView binaryInput,
                                                        int[] fieldNumbers,
                                                        int[] typeIds,
                                                        int[] encodings,
                                                        int[][] enumValidValues,
                                                        boolean failOnErrors) {
    int numFields = fieldNumbers.length;
    return decodeScalarFields(binaryInput, fieldNumbers, typeIds, encodings,
        new boolean[numFields], new boolean[numFields], new long[numFields],
        new double[numFields], new boolean[numFields], new byte[numFields][],
        enumValidValues, failOnErrors);
  }

  /**
   * Helper that enables enum-as-string decoding by passing enum name mappings.
   */
  private static ColumnVector decodeAllFieldsWithEnumStrings(ColumnView binaryInput,
                                                             int[] fieldNumbers,
                                                             int[][] enumValidValues,
                                                             byte[][][] enumNames,
                                                             boolean failOnErrors) {
    int numFields = fieldNumbers.length;
    int[] typeIds = new int[numFields];
    int[] encodings = new int[numFields];
    for (int i = 0; i < numFields; i++) {
      typeIds[i] = DType.STRING.getTypeId().getNativeId();
      encodings[i] = Protobuf.ENC_ENUM_STRING;
    }
    int[] parentIndices = new int[numFields];
    int[] depthLevels = new int[numFields];
    int[] wireTypes = new int[numFields];
    boolean[] isRepeated = new boolean[numFields];
    java.util.Arrays.fill(parentIndices, -1);
    java.util.Arrays.fill(wireTypes, Protobuf.WT_VARINT);
    return Protobuf.decodeToStruct(binaryInput,
        new ProtobufSchemaDescriptor(fieldNumbers, parentIndices, depthLevels,
            wireTypes, typeIds, encodings, isRepeated,
            new boolean[numFields], new boolean[numFields], new long[numFields],
            new double[numFields], new boolean[numFields], new byte[numFields][],
            enumValidValues, enumNames),
        failOnErrors);
  }

  @Test
  void testEnumAsStringValidValue() {
    // enum Color { RED=0; GREEN=1; BLUE=2; }
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));  // GREEN

    byte[][][] enumNames = new byte[][][] {
        new byte[][] {
            "RED".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "GREEN".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "BLUE".getBytes(java.nio.charset.StandardCharsets.UTF_8)
        }
    };
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedField = ColumnVector.fromStrings("GREEN");
         ColumnVector expected = ColumnVector.makeStruct(expectedField);
         ColumnVector actual = decodeAllFieldsWithEnumStrings(
             input.getColumn(0),
             new int[]{1},
             new int[][]{{0, 1, 2}},
             enumNames,
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testEnumAsStringUnknownValueReturnsNullRow() {
    // Unknown enum value should null the entire struct row (PERMISSIVE behavior).
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));

    byte[][][] enumNames = new byte[][][] {
        new byte[][] {
            "RED".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "GREEN".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "BLUE".getBytes(java.nio.charset.StandardCharsets.UTF_8)
        }
    };
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = decodeAllFieldsWithEnumStrings(
             input.getColumn(0),
             new int[]{1},
             new int[][]{{0, 1, 2}},
             enumNames,
             false);
         HostColumnVector hostStruct = actual.copyToHost()) {
      assertEquals(1, actual.getNullCount(), "Struct row should be null for unknown enum value");
      assertTrue(hostStruct.isNull(0), "Row 0 should be null");
    }
  }

  @Test
  void testEnumAsStringMixedValidAndUnknown() {
    Byte[] row0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(0)));    // RED
    Byte[] row1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));  // unknown
    Byte[] row2 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(2)));    // BLUE

    byte[][][] enumNames = new byte[][][] {
        new byte[][] {
            "RED".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "GREEN".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "BLUE".getBytes(java.nio.charset.StandardCharsets.UTF_8)
        }
    };
    try (Table input = new Table.TestBuilder().column(row0, row1, row2).build();
         ColumnVector actual = decodeAllFieldsWithEnumStrings(
             input.getColumn(0),
             new int[]{1},
             new int[][]{{0, 1, 2}},
             enumNames,
             false);
         HostColumnVector hostStruct = actual.copyToHost()) {
      assertEquals(1, actual.getNullCount(), "Only the unknown enum row should be null");
      assertTrue(!hostStruct.isNull(0), "Row 0 should be valid");
      assertTrue(hostStruct.isNull(1), "Row 1 should be null");
      assertTrue(!hostStruct.isNull(2), "Row 2 should be valid");
    }
  }

  @Test
  void testEnumValidValue() {
    // enum Color { RED=0; GREEN=1; BLUE=2; }
    // message Msg { Color color = 1; }
    // Test with valid enum value (GREEN = 1)
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(1)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedColor = ColumnVector.fromBoxedInts(1);  // GREEN
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedColor);
         ColumnVector actualStruct = decodeAllFieldsWithEnums(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new int[][]{{0, 1, 2}},  // valid enum values: RED=0, GREEN=1, BLUE=2
             false)) {
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumUnknownValueReturnsNullRow() {
    // enum Color { RED=0; GREEN=1; BLUE=2; }
    // message Msg { Color color = 1; }
    // Test with unknown enum value (999 is not defined)
    // The entire struct row should be null (matching Spark CPU PERMISSIVE mode)
    Byte[] row = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actualStruct = decodeAllFieldsWithEnums(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new int[][]{{0, 1, 2}},  // valid enum values: RED=0, GREEN=1, BLUE=2
             false);
         HostColumnVector hostStruct = actualStruct.copyToHost()) {
      // The struct itself should be null (not just the field)
      assertEquals(1, actualStruct.getNullCount(), "Struct row should be null for unknown enum");
      assertTrue(hostStruct.isNull(0), "Row 0 should be null");
    }
  }

  @Test
  void testEnumMixedValidAndUnknown() {
    // Test multiple rows with mix of valid and unknown enum values
    // Rows with unknown enum values should have null struct (not just null field)
    Byte[] row0 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(0)));    // RED (valid) -> struct valid
    Byte[] row1 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(999)));  // unknown -> struct null
    Byte[] row2 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(2)));    // BLUE (valid) -> struct valid
    Byte[] row3 = concat(box(tag(1, WT_VARINT)), box(encodeVarint(-1)));   // negative (unknown) -> struct null

    try (Table input = new Table.TestBuilder().column(row0, row1, row2, row3).build();
         ColumnVector actualStruct = decodeAllFieldsWithEnums(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new int[][]{{0, 1, 2}},  // valid enum values
             false);
         HostColumnVector hostStruct = actualStruct.copyToHost()) {
      // Check struct-level nulls
      assertEquals(2, actualStruct.getNullCount(), "Should have 2 null rows (rows 1 and 3)");
      assertTrue(!hostStruct.isNull(0), "Row 0 should be valid");
      assertTrue(hostStruct.isNull(1), "Row 1 should be null (unknown enum 999)");
      assertTrue(!hostStruct.isNull(2), "Row 2 should be valid");
      assertTrue(hostStruct.isNull(3), "Row 3 should be null (unknown enum -1)");
    }
  }

  @Test
  void testEnumWithOtherFields_NullsEntireRow() {
    // message Msg { Color color = 1; int32 count = 2; }
    // Test that unknown enum value nulls the ENTIRE struct row (not just the enum field)
    // This matches Spark CPU PERMISSIVE mode behavior
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(999)),  // unknown enum value
        box(tag(2, WT_VARINT)), box(encodeVarint(42)));  // count = 42

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actualStruct = decodeAllFieldsWithEnums(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT32.getTypeId().getNativeId(), DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new int[][]{{0, 1, 2}, null},  // first field is enum, second is regular int (no validation)
             false);
         HostColumnVector hostStruct = actualStruct.copyToHost()) {
      // The entire struct row should be null
      assertEquals(1, actualStruct.getNullCount(), "Struct row should be null");
      assertTrue(hostStruct.isNull(0), "Row 0 should be null due to unknown enum");
    }
  }

  @Test
  void testEnumMissingFieldDoesNotNullRow() {
    // Missing enum field should return null for the field, but NOT null the entire row
    // Only unknown enum values (present but invalid) trigger row-level null
    Byte[] row = new Byte[0];  // empty message

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedColor = ColumnVector.fromBoxedInts((Integer) null);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedColor);
         ColumnVector actualStruct = decodeAllFieldsWithEnums(
             input.getColumn(0),
             new int[]{1},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new int[][]{{0, 1, 2}},  // valid enum values
             false)) {
      // Struct row should be valid (not null), only the field is null
      assertEquals(0, actualStruct.getNullCount(), "Struct row should NOT be null for missing field");
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  @Test
  void testEnumValidWithOtherFields() {
    // message Msg { Color color = 1; int32 count = 2; }
    // Test that valid enum value works correctly with other fields
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1)),    // GREEN (valid)
        box(tag(2, WT_VARINT)), box(encodeVarint(42)));  // count = 42

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector expectedColor = ColumnVector.fromBoxedInts(1);
         ColumnVector expectedCount = ColumnVector.fromBoxedInts(42);
         ColumnVector expectedStruct = ColumnVector.makeStruct(expectedColor, expectedCount);
         ColumnVector actualStruct = decodeAllFieldsWithEnums(
             input.getColumn(0),
             new int[]{1, 2},
             new int[]{DType.INT32.getTypeId().getNativeId(), DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new int[][]{{0, 1, 2}, null},  // first field is enum, second is regular int
             false)) {
      // Struct row should be valid with correct values
      assertEquals(0, actualStruct.getNullCount(), "Struct row should be valid");
      AssertUtils.assertStructColumnsAreEqual(expectedStruct, actualStruct);
    }
  }

  // ============================================================================
  // Repeated Enum-as-String Tests
  // ============================================================================

  @Test
  void testRepeatedEnumAsString() {
    // repeated Color colors = 1; with Color { RED=0; GREEN=1; BLUE=2; }
    // Row with three occurrences: RED, BLUE, GREEN
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(0)),   // RED
        box(tag(1, WT_VARINT)), box(encodeVarint(2)),   // BLUE
        box(tag(1, WT_VARINT)), box(encodeVarint(1)));  // GREEN

    byte[][][] enumNames = new byte[][][] {
        new byte[][] {
            "RED".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "GREEN".getBytes(java.nio.charset.StandardCharsets.UTF_8),
            "BLUE".getBytes(java.nio.charset.StandardCharsets.UTF_8)
        }
    };
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector actual = decodeRaw(
             input.getColumn(0),
             new int[]{1},
             new int[]{-1},
             new int[]{0},
             new int[]{Protobuf.WT_VARINT},
             new int[]{DType.STRING.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_ENUM_STRING},
             new boolean[]{true},   // isRepeated
             new boolean[]{false},
             new boolean[]{false},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{null},
             new int[][]{{0, 1, 2}},
             enumNames,
             false)) {
      assertNotNull(actual);
      assertEquals(DType.STRUCT, actual.getType());
      assertEquals(1, actual.getNumChildren());
      try (ColumnView listCol = actual.getChildColumnView(0)) {
        assertEquals(DType.LIST, listCol.getType());
        try (ColumnView strChild = listCol.getChildColumnView(0);
             HostColumnVector hostStrs = strChild.copyToHost()) {
          assertEquals(3, hostStrs.getRowCount());
          assertEquals("RED", hostStrs.getJavaString(0));
          assertEquals("BLUE", hostStrs.getJavaString(1));
          assertEquals("GREEN", hostStrs.getJavaString(2));
        }
      }
    }
  }

  // ============================================================================
  // Edge case and boundary tests
  // ============================================================================

  @Test
  void testPackedFixedMisaligned() {
    byte[] packedData = new byte[]{0x01, 0x02, 0x03, 0x04, 0x05};
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(packedData.length)),
        box(packedData));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(RuntimeException.class, () -> {
        try (ColumnVector result = decodeRaw(
            input.getColumn(0),
            new int[]{1},
            new int[]{-1},
            new int[]{0},
            new int[]{WT_32BIT},
            new int[]{DType.INT32.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_FIXED},
            new boolean[]{true},
            new boolean[]{false},
            new boolean[]{false},
            new long[]{0},
            new double[]{0.0},
            new boolean[]{false},
            new byte[][]{null},
            new int[][]{null},
            true)) {
        }
      });
    }
  }

  @Test
  void testPackedFixedMisaligned64() {
    byte[] packedData = new byte[]{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09};
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(packedData.length)),
        box(packedData));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(RuntimeException.class, () -> {
        try (ColumnVector result = decodeRaw(
            input.getColumn(0),
            new int[]{1},
            new int[]{-1},
            new int[]{0},
            new int[]{WT_64BIT},
            new int[]{DType.INT64.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_FIXED},
            new boolean[]{true},
            new boolean[]{false},
            new boolean[]{false},
            new long[]{0},
            new double[]{0.0},
            new boolean[]{false},
            new byte[][]{null},
            new int[][]{null},
            true)) {
        }
      });
    }
  }

  @Test
  void testLargeRepeatedField() throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    for (int i = 0; i < 100000; i++) {
      baos.write(tag(1, WT_VARINT));
      baos.write(encodeVarint(i));
    }
    Byte[] row = box(baos.toByteArray());

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = decodeRaw(
             input.getColumn(0),
             new int[]{1},
             new int[]{-1},
             new int[]{0},
             new int[]{WT_VARINT},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{true},
             new boolean[]{false},
             new boolean[]{false},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{null},
             new int[][]{null},
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      try (ColumnVector list = result.getChildColumnView(0).copyToColumnVector();
           ColumnVector vals = list.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostVals = vals.copyToHost()) {
        assertEquals(DType.LIST, list.getType());
        assertEquals(100000, vals.getRowCount(), "All 100K elements should be decoded");
        assertEquals(0, hostVals.getInt(0));
        assertEquals(50000, hostVals.getInt(50000));
        assertEquals(99999, hostVals.getInt(99999));
      }
    }
  }

  @Test
  void testMultiFieldOutputShape() {
    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             makeScalarSchema(
                 new int[]{1, 2, 3},
                 new int[]{DType.INT64.getTypeId().getNativeId(),
                           DType.STRING.getTypeId().getNativeId(),
                           DType.FLOAT32.getTypeId().getNativeId()},
                 new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT}),
             true)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getRowCount());
      assertEquals(3, result.getNumChildren());
      assertEquals(DType.INT64, result.getChildColumnView(0).getType());
      assertEquals(DType.STRING, result.getChildColumnView(1).getType());
      assertEquals(DType.FLOAT32, result.getChildColumnView(2).getType());
    }
  }

  @Test
  void testMultipleRowsOutputShape() {
    Byte[] row0 = new Byte[]{0x08, 0x01};
    Byte[] row1 = new Byte[]{0x08, 0x02};
    Byte[] row2 = new Byte[]{0x08, 0x03};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1, row2}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             makeScalarSchema(
                 new int[]{1},
                 new int[]{DType.INT64.getTypeId().getNativeId()},
                 new int[]{Protobuf.ENC_DEFAULT}), true)) {
      assertEquals(3, result.getRowCount());
      assertEquals(1, result.getNumChildren());
    }
  }

  // ============================================================================
  // Null input handling
  // ============================================================================

  @Test
  void testNullInputRowProducesNullStructRow() {
    Byte[] row0 = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, null}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             makeScalarSchema(
                 new int[]{1},
                 new int[]{DType.INT64.getTypeId().getNativeId()},
                 new int[]{Protobuf.ENC_DEFAULT}), true)) {
      assertEquals(2, result.getRowCount());
      try (HostColumnVector hcv = result.copyToHost()) {
        assertFalse(hcv.isNull(0), "Row 0 should not be null");
        assertTrue(hcv.isNull(1), "Row 1 (null input) should be null in output struct");
      }
    }
  }

  @Test
  void testMixedPackedUnpacked() {
    byte[] packedContent = concatBytes(encodeVarint(30), encodeVarint(40));
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(10)),
        box(tag(1, WT_VARINT)), box(encodeVarint(20)),
        box(tag(1, WT_LEN)), box(encodeVarint(packedContent.length)), box(packedContent));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = decodeRaw(
             input.getColumn(0),
             new int[]{1},
             new int[]{-1},
             new int[]{0},
             new int[]{WT_VARINT},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{true},
             new boolean[]{false},
             new boolean[]{false},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{null},
             new int[][]{null},
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      try (ColumnVector list = result.getChildColumnView(0).copyToColumnVector();
           ColumnVector vals = list.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostVals = vals.copyToHost()) {
        assertEquals(4, vals.getRowCount());
        assertEquals(10, hostVals.getInt(0));
        assertEquals(20, hostVals.getInt(1));
        assertEquals(30, hostVals.getInt(2));
        assertEquals(40, hostVals.getInt(3));
      }
    }
  }

  @Test
  void testAllNullInputRows() {
    try (Table input = new Table.TestBuilder().column(new Byte[][]{null, null, null}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             makeScalarSchema(
                 new int[]{1, 2},
                 new int[]{DType.INT64.getTypeId().getNativeId(),
                           DType.STRING.getTypeId().getNativeId()},
                 new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT}), true)) {
      assertEquals(3, result.getRowCount());
      assertEquals(2, result.getNumChildren());
      try (HostColumnVector hcv = result.copyToHost()) {
        for (int row = 0; row < 3; row++) {
          assertTrue(hcv.isNull(row), "Row " + row + " should be null");
        }
      }
    }
  }

  @Test
  void testLargeFieldNumber() {
    int maxFieldNumber = (1 << 29) - 1;
    Byte[] row = concat(
        box(tag(maxFieldNumber, WT_VARINT)),
        box(encodeVarint(42)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = decodeRaw(
             input.getColumn(0),
             new int[]{maxFieldNumber},
             new int[]{-1},
             new int[]{0},
             new int[]{WT_VARINT},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{false},
             new boolean[]{false},
             new boolean[]{false},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{null},
             new int[][]{null},
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      try (ColumnVector child = result.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostChild = child.copyToHost()) {
        assertEquals(42, hostChild.getInt(0));
      }
    }
  }

  private void verifyDeepNesting(int numLevels) {
    int numFields = 2 * numLevels - 1;

    byte[] current = concatBytes(tag(1, WT_VARINT), encodeVarint(1));
    for (int level = numLevels - 2; level >= 0; level--) {
      current = concatBytes(
          tag(1, WT_VARINT), encodeVarint(1),
          tag(2, WT_LEN), encodeVarint(current.length), current);
    }
    Byte[] row = box(current);

    int[] fieldNumbers = new int[numFields];
    int[] parentIndices = new int[numFields];
    int[] depthLevels = new int[numFields];
    int[] wireTypes = new int[numFields];
    int[] outputTypeIds = new int[numFields];
    int[] encodings = new int[numFields];
    boolean[] isRepeated = new boolean[numFields];
    boolean[] isRequired = new boolean[numFields];
    boolean[] hasDefaultValue = new boolean[numFields];
    long[] defaultInts = new long[numFields];
    double[] defaultFloats = new double[numFields];
    boolean[] defaultBools = new boolean[numFields];
    byte[][] defaultStrings = new byte[numFields][];
    int[][] enumValidValues = new int[numFields][];

    for (int level = 0; level < numLevels; level++) {
      int intIdx = 2 * level;
      int parentIdx = level == 0 ? -1 : 2 * (level - 1) + 1;

      fieldNumbers[intIdx] = 1;
      parentIndices[intIdx] = parentIdx;
      depthLevels[intIdx] = level;
      wireTypes[intIdx] = WT_VARINT;
      outputTypeIds[intIdx] = DType.INT32.getTypeId().getNativeId();
      encodings[intIdx] = Protobuf.ENC_DEFAULT;

      if (level < numLevels - 1) {
        int structIdx = 2 * level + 1;
        fieldNumbers[structIdx] = 2;
        parentIndices[structIdx] = parentIdx;
        depthLevels[structIdx] = level;
        wireTypes[structIdx] = WT_LEN;
        outputTypeIds[structIdx] = DType.STRUCT.getTypeId().getNativeId();
        encodings[structIdx] = Protobuf.ENC_DEFAULT;
      }
    }

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = decodeRaw(
             input.getColumn(0),
             fieldNumbers, parentIndices, depthLevels, wireTypes,
             outputTypeIds, encodings, isRepeated, isRequired,
             hasDefaultValue, defaultInts, defaultFloats, defaultBools,
             defaultStrings, enumValidValues, false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      // Verify the deepest leaf (encoded as 1 in this synthetic schema) actually decodes to 1.
      try (ColumnVector firstChild = result.getChildColumnView(0).copyToColumnVector();
           HostColumnVector hostChild = firstChild.copyToHost()) {
        assertEquals(DType.INT32, firstChild.getType());
        assertEquals(1, hostChild.getInt(0));
      }
    }
  }

  // ============================================================================
  // Empty-row (0 rows) handling
  // ============================================================================

  @Test
  void testZeroRowInput() {
    try (Table input = new Table.TestBuilder().column(new Byte[][]{}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0),
             makeScalarSchema(
                 new int[]{1, 2},
                 new int[]{DType.INT64.getTypeId().getNativeId(),
                           DType.STRING.getTypeId().getNativeId()},
                 new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT}), true)) {
      assertEquals(0, result.getRowCount());
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(2, result.getNumChildren());
      assertEquals(DType.INT64, result.getChildColumnView(0).getType());
      assertEquals(DType.STRING, result.getChildColumnView(1).getType());
    }
  }

  // ============================================================================
  // Nested schema shape tests (verifies correct column types without decode)
  // ============================================================================

  @Test
  void testNestedMessageOutputShape() {
    // Schema: message Outer { int32 a = 1; Inner b = 2; } message Inner { int32 x = 1; }
    int intType = DType.INT32.getTypeId().getNativeId();
    int structType = DType.STRUCT.getTypeId().getNativeId();
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptor(
        new int[]{1, 2, 1},           // field numbers
        new int[]{-1, -1, 1},         // parent indices
        new int[]{0, 0, 1},           // depth levels
        new int[]{Protobuf.WT_VARINT, Protobuf.WT_LEN, Protobuf.WT_VARINT},
        new int[]{intType, structType, intType},
        new int[]{0, 0, 0},           // encodings
        new boolean[]{false, false, false},
        new boolean[]{false, false, false},
        new boolean[]{false, false, false},
        new long[]{0, 0, 0},
        new double[]{0, 0, 0},
        new boolean[]{false, false, false},
        new byte[][]{null, null, null},
        new int[][]{null, null, null},
        new byte[][][]{null, null, null}
    );

    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getRowCount());
      assertEquals(2, result.getNumChildren());
      assertEquals(DType.INT32, result.getChildColumnView(0).getType());
      assertEquals(DType.STRUCT, result.getChildColumnView(1).getType());
      assertEquals(1, result.getChildColumnView(1).getNumChildren());
      assertEquals(DType.INT32, result.getChildColumnView(1).getChildColumnView(0).getType());
    }
  }

  @Test
  void testZeroLengthNestedMessage() {
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(0)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = decodeRaw(
             input.getColumn(0),
             new int[]{1, 1},
             new int[]{-1, 0},
             new int[]{0, 1},
             new int[]{WT_LEN, WT_VARINT},
             new int[]{DType.STRUCT.getTypeId().getNativeId(), DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
             new boolean[]{false, false},
             new boolean[]{false, false},
             new boolean[]{false, false},
             new long[]{0, 0},
             new double[]{0.0, 0.0},
             new boolean[]{false, false},
             new byte[][]{null, null},
             new int[][]{null, null},
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
    }
  }

  @Test
  void testRepeatedFieldOutputShape() {
    // Schema: message Msg { repeated int32 values = 1; }
    int intType = DType.INT32.getTypeId().getNativeId();
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptor(
        new int[]{1},
        new int[]{-1},
        new int[]{0},
        new int[]{Protobuf.WT_VARINT},
        new int[]{intType},
        new int[]{0},
        new boolean[]{true},          // is_repeated = true
        new boolean[]{false},
        new boolean[]{false},
        new long[]{0},
        new double[]{0},
        new boolean[]{false},
        new byte[][]{null},
        new int[][]{null},
        new byte[][][]{null}
    );

    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(1, result.getRowCount());
      assertEquals(1, result.getNumChildren());
      assertEquals(DType.LIST, result.getChildColumnView(0).getType());
    }
  }

  @Test
  void testEmptyPackedRepeated() {
    Byte[] row = concat(
        box(tag(1, WT_LEN)),
        box(encodeVarint(0)));

    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = decodeRaw(
             input.getColumn(0),
             new int[]{1},
             new int[]{-1},
             new int[]{0},
             new int[]{WT_VARINT},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{true},
             new boolean[]{false},
             new boolean[]{false},
             new long[]{0},
             new double[]{0.0},
             new boolean[]{false},
             new byte[][]{null},
             new int[][]{null},
             false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
    }
  }

  @Test
  void testZeroRowNestedSchemaShape() {
    // 0 rows with nested schema — verify correct type hierarchy
    int intType = DType.INT32.getTypeId().getNativeId();
    int structType = DType.STRUCT.getTypeId().getNativeId();
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptor(
        new int[]{1, 2, 1},
        new int[]{-1, -1, 1},
        new int[]{0, 0, 1},
        new int[]{Protobuf.WT_VARINT, Protobuf.WT_LEN, Protobuf.WT_VARINT},
        new int[]{intType, structType, intType},
        new int[]{0, 0, 0},
        new boolean[]{false, false, false},
        new boolean[]{false, false, false},
        new boolean[]{false, false, false},
        new long[]{0, 0, 0},
        new double[]{0, 0, 0},
        new boolean[]{false, false, false},
        new byte[][]{null, null, null},
        new int[][]{null, null, null},
        new byte[][][]{null, null, null}
    );

    try (Table input = new Table.TestBuilder().column(new Byte[][]{}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(0, result.getRowCount());
      assertEquals(2, result.getNumChildren());
      assertEquals(DType.INT32, result.getChildColumnView(0).getType());
      assertEquals(DType.STRUCT, result.getChildColumnView(1).getType());
      assertEquals(1, result.getChildColumnView(1).getNumChildren());
      assertEquals(DType.INT32, result.getChildColumnView(1).getChildColumnView(0).getType());
    }
  }

  @Test
  void testZeroRowRepeatedMessageShape() {
    // 0 rows with repeated message schema: repeated Inner inner = 1;
    int structType = DType.STRUCT.getTypeId().getNativeId();
    int intType = DType.INT32.getTypeId().getNativeId();
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptor(
        new int[]{1, 1},              // field numbers
        new int[]{-1, 0},             // parent indices: inner's child has parent=0
        new int[]{0, 1},              // depth levels
        new int[]{Protobuf.WT_LEN, Protobuf.WT_VARINT},
        new int[]{structType, intType},
        new int[]{0, 0},
        new boolean[]{true, false},   // inner is repeated
        new boolean[]{false, false},
        new boolean[]{false, false},
        new long[]{0, 0},
        new double[]{0, 0},
        new boolean[]{false, false},
        new byte[][]{null, null},
        new int[][]{null, null},
        new byte[][][]{null, null}
    );

    try (Table input = new Table.TestBuilder().column(new Byte[][]{}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(0, result.getRowCount());
      assertEquals(1, result.getNumChildren());
      assertEquals(DType.LIST, result.getChildColumnView(0).getType());
    }
  }

  @Test
  void testZeroRowRepeatedScalarShape() {
    int intType = DType.INT32.getTypeId().getNativeId();
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptor(
        new int[]{1},
        new int[]{-1},
        new int[]{0},
        new int[]{Protobuf.WT_VARINT},
        new int[]{intType},
        new int[]{0},
        new boolean[]{true},
        new boolean[]{false},
        new boolean[]{false},
        new long[]{0},
        new double[]{0},
        new boolean[]{false},
        new byte[][]{null},
        new int[][]{null},
        new byte[][][]{null}
    );

    try (Table input = new Table.TestBuilder().column(new Byte[][]{}).build();
         ColumnVector result = Protobuf.decodeToStruct(input.getColumn(0), schema, true)) {
      assertEquals(0, result.getRowCount());
      assertEquals(1, result.getNumChildren());
      assertEquals(DType.LIST, result.getChildColumnView(0).getType());
    }
  }

  // ============================================================================
  // Input validation tests
  // ============================================================================

  @Test
  void testNullBinaryInputThrows() {
    assertThrows(IllegalArgumentException.class, () ->
        Protobuf.decodeToStruct(null,
            makeScalarSchema(new int[]{1},
                new int[]{DType.INT64.getTypeId().getNativeId()},
                new int[]{0}), true));
  }

  @Test
  void testNullSchemaThrows() {
    Byte[] row = new Byte[]{0x08, 0x01};
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(IllegalArgumentException.class, () ->
          Protobuf.decodeToStruct(input.getColumn(0), null, true));
    }
  }

  @Test
  void testRepeatedString() {
    // Exercises the build_repeated_string_column non-enum path (CUB DeviceMemcpy::Batched
    // copy + length-extraction), which the existing testRepeatedEnumAsString does not cover.
    byte[] s1 = "hello".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    byte[] s2 = "world".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    byte[] s3 = "foo".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    Byte[] row0 = concat(
        box(tag(1, WT_LEN)), box(encodeVarint(s1.length)), box(s1),
        box(tag(1, WT_LEN)), box(encodeVarint(s2.length)), box(s2));
    Byte[] row1 = concat(
        box(tag(1, WT_LEN)), box(encodeVarint(s3.length)), box(s3));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, row1}).build();
         ColumnVector result = decodeRaw(
             input.getColumn(0),
             new int[]{1}, new int[]{-1}, new int[]{0},
             new int[]{Protobuf.WT_LEN},
             new int[]{DType.STRING.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{true}, new boolean[]{false}, new boolean[]{false},
             new long[]{0}, new double[]{0.0}, new boolean[]{false},
             new byte[][]{null}, new int[][]{null}, false)) {
      assertNotNull(result);
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(2, result.getRowCount());
      try (ColumnView listCol = result.getChildColumnView(0)) {
        assertEquals(DType.LIST, listCol.getType());
        try (ColumnView strChild = listCol.getChildColumnView(0);
             HostColumnVector hcv = strChild.copyToHost()) {
          assertEquals(3, hcv.getRowCount());
          assertEquals("hello", hcv.getJavaString(0));
          assertEquals("world", hcv.getJavaString(1));
          assertEquals("foo", hcv.getJavaString(2));
        }
      }
    }
  }

  @Test
  void testRepeatedSint32() {
    // sint32 zigzag encoding: zigzag(-1) = 1, zigzag(-2) = 3, zigzag(3) = 6.
    // Verifies the extract_varint_kernel<T, true, repeated_location_provider> instantiation.
    Byte[] row = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(1L)),
        box(tag(1, WT_VARINT)), box(encodeVarint(3L)),
        box(tag(1, WT_VARINT)), box(encodeVarint(6L)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build();
         ColumnVector result = decodeRaw(
             input.getColumn(0),
             new int[]{1}, new int[]{-1}, new int[]{0},
             new int[]{Protobuf.WT_VARINT},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_ZIGZAG},
             new boolean[]{true}, new boolean[]{false}, new boolean[]{false},
             new long[]{0}, new double[]{0.0}, new boolean[]{false},
             new byte[][]{null}, new int[][]{null}, false);
         ColumnView listCol = result.getChildColumnView(0);
         ColumnView vals = listCol.getChildColumnView(0);
         HostColumnVector hcv = vals.copyToHost()) {
      assertEquals(3, hcv.getRowCount());
      assertEquals(-1, hcv.getInt(0));
      assertEquals(-2, hcv.getInt(1));
      assertEquals(3, hcv.getInt(2));
    }
  }

  @Test
  void testNullInputRowProducesNullListForRepeatedField() {
    // Verifies make_list_column_with_input_nulls propagates the input null mask to the
    // output LIST column; previously only exercised by scalar (non-LIST) schemas.
    Byte[] row0 = concat(
        box(tag(1, WT_VARINT)), box(encodeVarint(7)),
        box(tag(1, WT_VARINT)), box(encodeVarint(8)));
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row0, null}).build();
         ColumnVector result = decodeRaw(
             input.getColumn(0),
             new int[]{1}, new int[]{-1}, new int[]{0},
             new int[]{Protobuf.WT_VARINT},
             new int[]{DType.INT32.getTypeId().getNativeId()},
             new int[]{Protobuf.ENC_DEFAULT},
             new boolean[]{true}, new boolean[]{false}, new boolean[]{false},
             new long[]{0}, new double[]{0.0}, new boolean[]{false},
             new byte[][]{null}, new int[][]{null}, false)) {
      assertEquals(DType.STRUCT, result.getType());
      assertEquals(2, result.getRowCount());
      try (ColumnView listCol = result.getChildColumnView(0);
           HostColumnVector hList = listCol.copyToHost()) {
        assertEquals(DType.LIST, hList.getType());
        assertFalse(hList.isNull(0), "row 0 should have a valid list");
        assertTrue(hList.isNull(1), "null input row 1 should produce a null list");
      }
    }
  }

  @Test
  void testSchemaWithTooManyRepeatedFields() {
    // Hits ERR_SCHEMA_TOO_LARGE: the scan_all_repeated_occurrences_kernel stack-array
    // guard rejects schemas with more than 32 top-level repeated fields.
    int n = 33;
    int[] fns = new int[n];
    int[] parents = new int[n];
    int[] depths = new int[n];
    int[] wts = new int[n];
    int[] types = new int[n];
    int[] encs = new int[n];
    boolean[] rep = new boolean[n];
    boolean[] req = new boolean[n];
    boolean[] hasDef = new boolean[n];
    long[] dInts = new long[n];
    double[] dFlts = new double[n];
    boolean[] dBools = new boolean[n];
    byte[][] dStrs = new byte[n][];
    int[][] enumVV = new int[n][];
    for (int i = 0; i < n; i++) {
      fns[i] = i + 1;
      parents[i] = -1;
      depths[i] = 0;
      wts[i] = WT_VARINT;
      types[i] = DType.INT32.getTypeId().getNativeId();
      encs[i] = Protobuf.ENC_DEFAULT;
      rep[i] = true;
    }
    // The MAX_STACK_FIELDS guard in scan_all_repeated_occurrences_kernel only fires when
    // every field actually has occurrences (zero-count fields are filtered out before the
    // kernel launch). Encode one occurrence per field so num_scan_fields == n.
    java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
    try {
      for (int i = 0; i < n; i++) {
        baos.write(tag(i + 1, WT_VARINT));
        baos.write(encodeVarint(1));
      }
    } catch (java.io.IOException e) {
      throw new RuntimeException(e);
    }
    Byte[] row = box(baos.toByteArray());
    try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
      assertThrows(RuntimeException.class, () -> {
        try (ColumnVector ignored = decodeRaw(input.getColumn(0),
            fns, parents, depths, wts, types, encs, rep, req,
            hasDef, dInts, dFlts, dBools, dStrs, enumVV, true)) {
          // unreachable: should throw
        }
      });
    }
  }
}
