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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ProtobufSchemaDescriptorTest {
  private ProtobufSchemaDescriptor makeDescriptor(
      boolean isRepeated,
      boolean hasDefaultValue,
      int encoding,
      int[] enumValidValues,
      byte[][] enumNames) {
    return new ProtobufSchemaDescriptor(
        new int[]{1},
        new int[]{-1},
        new int[]{0},
        new int[]{Protobuf.WT_VARINT},
        new int[]{ai.rapids.cudf.DType.INT32.getTypeId().getNativeId()},
        new int[]{encoding},
        new boolean[]{isRepeated},
        new boolean[]{false},
        new boolean[]{hasDefaultValue},
        new long[]{0},
        new double[]{0.0},
        new boolean[]{false},
        new byte[][]{null},
        new int[][]{enumValidValues},
        new byte[][][]{enumNames});
  }

  @Test
  void testRepeatedFieldCannotCarryDefaultValue() {
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(true, true, Protobuf.ENC_DEFAULT, null, null));
  }

  @Test
  void testEnumStringRequiresEnumMetadata() {
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(false, false, Protobuf.ENC_ENUM_STRING, null, null));
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(false, false, Protobuf.ENC_ENUM_STRING, new int[]{0, 1}, null));
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(false, false, Protobuf.ENC_ENUM_STRING, null,
            new byte[][]{"A".getBytes(), "B".getBytes()}));
  }

  @Test
  void testDuplicateFieldNumbersUnderSameParentRejected() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1, 7, 7},
            new int[]{-1, 0, 0},
            new int[]{0, 1, 1},
            new int[]{Protobuf.WT_LEN, Protobuf.WT_VARINT, Protobuf.WT_VARINT},
            new int[]{
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.INT32.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.INT32.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
            new boolean[]{false, false, false},
            new boolean[]{false, false, false},
            new boolean[]{false, false, false},
            new long[]{0, 0, 0},
            new double[]{0.0, 0.0, 0.0},
            new boolean[]{false, false, false},
            new byte[][]{null, null, null},
            new int[][]{null, null, null},
            new byte[][][]{null, null, null}));
  }

  @Test
  void testDuplicateFieldNumbersUnderDifferentParentsAllowed() {
    assertDoesNotThrow(() ->
        new ProtobufSchemaDescriptor(
            new int[]{1, 2, 7, 7},
            new int[]{-1, -1, 0, 1},
            new int[]{0, 0, 1, 1},
            new int[]{Protobuf.WT_LEN, Protobuf.WT_LEN, Protobuf.WT_VARINT, Protobuf.WT_VARINT},
            new int[]{
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.INT32.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.INT32.getTypeId().getNativeId()},
            new int[]{
                Protobuf.ENC_DEFAULT,
                Protobuf.ENC_DEFAULT,
                Protobuf.ENC_DEFAULT,
                Protobuf.ENC_DEFAULT},
            new boolean[]{false, false, false, false},
            new boolean[]{false, false, false, false},
            new boolean[]{false, false, false, false},
            new long[]{0, 0, 0, 0},
            new double[]{0.0, 0.0, 0.0, 0.0},
            new boolean[]{false, false, false, false},
            new byte[][]{null, null, null, null},
            new int[][]{null, null, null, null},
            new byte[][][]{null, null, null, null}));
  }
}
