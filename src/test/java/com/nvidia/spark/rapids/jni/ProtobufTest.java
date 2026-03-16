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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests for the Protobuf GPU decoder — framework PR.
 *
 * These tests verify the decode stub: schema validation, correct output shape,
 * null column construction, and empty-row handling. Actual data extraction tests
 * are added in follow-up PRs.
 */
public class ProtobufTest {

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
    if (typeId == DType.FLOAT32.getTypeId().getNativeId()) return Protobuf.WT_32BIT;
    if (typeId == DType.FLOAT64.getTypeId().getNativeId()) return Protobuf.WT_64BIT;
    if (typeId == DType.STRING.getTypeId().getNativeId()) return Protobuf.WT_LEN;
    if (typeId == DType.LIST.getTypeId().getNativeId()) return Protobuf.WT_LEN;
    if (typeId == DType.STRUCT.getTypeId().getNativeId()) return Protobuf.WT_LEN;
    if (encoding == Protobuf.ENC_FIXED) {
      if (typeId == DType.INT64.getTypeId().getNativeId()) return Protobuf.WT_64BIT;
      return Protobuf.WT_32BIT;
    }
    return Protobuf.WT_VARINT;
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
}
