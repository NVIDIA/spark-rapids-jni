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

import ai.rapids.cudf.DType;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ProtobufSchemaDescriptorTest {
  private ProtobufSchemaDescriptor makeDescriptor(
      boolean isRepeated,
      boolean hasDefaultValue,
      int encoding,
      int[] enumValidValues,
      byte[][] enumNames) {
    int outputType = (encoding == Protobuf.ENC_ENUM_STRING)
        ? ai.rapids.cudf.DType.STRING.getTypeId().getNativeId()
        : ai.rapids.cudf.DType.INT32.getTypeId().getNativeId();
    return new ProtobufSchemaDescriptor(
        new int[]{1},
        new int[]{-1},
        new int[]{0},
        new int[]{Protobuf.WT_VARINT},
        new int[]{outputType},
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
  void testFieldCannotBeBothRepeatedAndRequired() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1},
            new int[]{-1},
            new int[]{0},
            new int[]{Protobuf.WT_VARINT},
            new int[]{ai.rapids.cudf.DType.INT32.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT},
            new boolean[]{true},
            new boolean[]{true},
            new boolean[]{false},
            new long[]{0},
            new double[]{0.0},
            new boolean[]{false},
            new byte[][]{null},
            new int[][]{null},
            new byte[][][]{null}));
  }

  @Test
  void testStructFieldCannotCarryDefaultValue() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1},
            new int[]{-1},
            new int[]{0},
            new int[]{Protobuf.WT_LEN},
            new int[]{ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT},
            new boolean[]{false},
            new boolean[]{false},
            new boolean[]{true},
            new long[]{0},
            new double[]{0.0},
            new boolean[]{false},
            new byte[][]{null},
            new int[][]{null},
            new byte[][][]{null}));
  }

  @Test
  void testListFieldCannotCarryDefaultValue() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1},
            new int[]{-1},
            new int[]{0},
            new int[]{Protobuf.WT_LEN},
            new int[]{ai.rapids.cudf.DType.LIST.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT},
            new boolean[]{false},
            new boolean[]{false},
            new boolean[]{true},
            new long[]{0},
            new double[]{0.0},
            new boolean[]{false},
            new byte[][]{null},
            new int[][]{null},
            new byte[][][]{null}));
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
  void testEnumStringRejectsEmptyEnumArrays() {
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(false, false, Protobuf.ENC_ENUM_STRING, new int[]{}, new byte[][]{}));
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

  @Test
  void testChildParentMustBeStruct() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1, 2},
            new int[]{-1, 0},
            new int[]{0, 1},
            new int[]{Protobuf.WT_VARINT, Protobuf.WT_VARINT},
            new int[]{
                ai.rapids.cudf.DType.INT32.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.INT32.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
            new boolean[]{false, false},
            new boolean[]{false, false},
            new boolean[]{false, false},
            new long[]{0, 0},
            new double[]{0.0, 0.0},
            new boolean[]{false, false},
            new byte[][]{null, null},
            new int[][]{null, null},
            new byte[][][]{null, null}));
  }

  @Test
  void testEncodingCompatibilityValidation() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1},
            new int[]{-1},
            new int[]{0},
            new int[]{Protobuf.WT_32BIT},
            new int[]{ai.rapids.cudf.DType.INT32.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT},
            new boolean[]{false},
            new boolean[]{false},
            new boolean[]{false},
            new long[]{0},
            new double[]{0.0},
            new boolean[]{false},
            new byte[][]{null},
            new int[][]{null},
            new byte[][][]{null}));

    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1},
            new int[]{-1},
            new int[]{0},
            new int[]{Protobuf.WT_LEN},
            new int[]{ai.rapids.cudf.DType.STRING.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_ENUM_STRING},
            new boolean[]{false},
            new boolean[]{false},
            new boolean[]{false},
            new long[]{0},
            new double[]{0.0},
            new boolean[]{false},
            new byte[][]{null},
            new int[][]{{0, 1}},
            new byte[][][]{new byte[][]{"A".getBytes(), "B".getBytes()}}));
  }

  @Test
  void testDepthAboveSupportedLimitRejected() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
            new int[]{-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
            new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            new int[]{Protobuf.WT_LEN, Protobuf.WT_LEN, Protobuf.WT_LEN, Protobuf.WT_LEN,
                Protobuf.WT_LEN, Protobuf.WT_LEN, Protobuf.WT_LEN, Protobuf.WT_LEN,
                Protobuf.WT_LEN, Protobuf.WT_LEN, Protobuf.WT_VARINT},
            new int[]{
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.STRUCT.getTypeId().getNativeId(),
                ai.rapids.cudf.DType.INT32.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT,
                Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT,
                Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT,
                Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
            new boolean[]{false, false, false, false, false, false, false, false, false, false, false},
            new boolean[]{false, false, false, false, false, false, false, false, false, false, false},
            new boolean[]{false, false, false, false, false, false, false, false, false, false, false},
            new long[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            new boolean[]{false, false, false, false, false, false, false, false, false, false, false},
            new byte[][]{null, null, null, null, null, null, null, null, null, null, null},
            new int[][]{null, null, null, null, null, null, null, null, null, null, null},
            new byte[][][]{null, null, null, null, null, null, null, null, null, null, null}));
  }

  @Test
  void testSerializationRoundTripPreservesContentsAndIsolation() throws Exception {
    ProtobufSchemaDescriptor original = new ProtobufSchemaDescriptor(
        new int[]{1},
        new int[]{-1},
        new int[]{0},
        new int[]{Protobuf.WT_VARINT},
        new int[]{ai.rapids.cudf.DType.STRING.getTypeId().getNativeId()},
        new int[]{Protobuf.ENC_ENUM_STRING},
        new boolean[]{false},
        new boolean[]{false},
        new boolean[]{false},
        new long[]{7},
        new double[]{0.0},
        new boolean[]{false},
        new byte[][]{"def".getBytes()},
        new int[][]{{0, 1}},
        new byte[][][]{new byte[][]{"A".getBytes(), "B".getBytes()}});

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
      oos.writeObject(original);
    }

    ProtobufSchemaDescriptor roundTrip;
    try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()))) {
      roundTrip = (ProtobufSchemaDescriptor) ois.readObject();
    }

    assertEquals(original.numFields(), roundTrip.numFields());
    assertArrayEquals(original.fieldNumbers, roundTrip.fieldNumbers);
    assertArrayEquals(original.defaultStrings[0], roundTrip.defaultStrings[0]);
    assertArrayEquals(original.enumValidValues[0], roundTrip.enumValidValues[0]);
    assertArrayEquals(original.enumNames[0][0], roundTrip.enumNames[0][0]);
    assertArrayEquals(original.enumNames[0][1], roundTrip.enumNames[0][1]);
    assertNotSame(original.defaultStrings, roundTrip.defaultStrings);
    assertNotSame(original.defaultStrings[0], roundTrip.defaultStrings[0]);
    assertNotSame(original.enumValidValues, roundTrip.enumValidValues);
    assertNotSame(original.enumValidValues[0], roundTrip.enumValidValues[0]);
    assertNotSame(original.enumNames, roundTrip.enumNames);
    assertNotSame(original.enumNames[0], roundTrip.enumNames[0]);
    assertNotSame(original.enumNames[0][0], roundTrip.enumNames[0][0]);
  }

  @Test
  void testMismatchedArrayLengthsRejected() {
    // parentIndices has length 2 while every other array has length 1.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1},
            new int[]{-1, -1},
            new int[]{0},
            new int[]{Protobuf.WT_VARINT},
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
            new byte[][][]{null}));
  }

  @Test
  void testBackCompatConstructorMarksAllFieldsAsOutput() {
    // The 15-arg constructor is the back-compat path; verify it fills isOutput with all-true.
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptor(
        new int[]{1, 2},
        new int[]{-1, -1},
        new int[]{0, 0},
        new int[]{Protobuf.WT_VARINT, Protobuf.WT_LEN},
        new int[]{DType.INT32.getTypeId().getNativeId(),
                  DType.STRING.getTypeId().getNativeId()},
        new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
        new boolean[]{false, false},
        new boolean[]{false, false},
        new boolean[]{false, false},
        new long[]{0, 0},
        new double[]{0.0, 0.0},
        new boolean[]{false, false},
        new byte[][]{null, null},
        new int[][]{null, null},
        new byte[][][]{null, null});
    assertArrayEquals(new boolean[]{true, true}, schema.isOutput);
  }

  @Test
  void testNestedFieldMustShareOutputFlagWithParent() {
    int structType = DType.STRUCT.getTypeId().getNativeId();
    int intType = DType.INT32.getTypeId().getNativeId();
    // Hidden parent struct with a visible child -> illegal.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1, 1},
            new int[]{-1, 0},
            new int[]{0, 1},
            new int[]{Protobuf.WT_LEN, Protobuf.WT_VARINT},
            new int[]{structType, intType},
            new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
            new boolean[]{false, false},
            new boolean[]{false, false},
            new boolean[]{false, false},
            new boolean[]{false, true},     // hidden parent, visible child
            new long[]{0, 0},
            new double[]{0.0, 0.0},
            new boolean[]{false, false},
            new byte[][]{null, null},
            new int[][]{null, null},
            new byte[][][]{null, null}));

    // Reverse direction: visible parent struct with a hidden child -> also illegal.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1, 1},
            new int[]{-1, 0},
            new int[]{0, 1},
            new int[]{Protobuf.WT_LEN, Protobuf.WT_VARINT},
            new int[]{structType, intType},
            new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
            new boolean[]{false, false},
            new boolean[]{false, false},
            new boolean[]{false, false},
            new boolean[]{true, false},     // visible parent, hidden child
            new long[]{0, 0},
            new double[]{0.0, 0.0},
            new boolean[]{false, false},
            new byte[][]{null, null},
            new int[][]{null, null},
            new byte[][][]{null, null}));
  }

  @Test
  void testHiddenFieldRoundTripsThroughSerialization() throws Exception {
    int intType = DType.INT32.getTypeId().getNativeId();
    ProtobufSchemaDescriptor original = new ProtobufSchemaDescriptor(
        new int[]{1, 2},
        new int[]{-1, -1},
        new int[]{0, 0},
        new int[]{Protobuf.WT_VARINT, Protobuf.WT_VARINT},
        new int[]{intType, intType},
        new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
        new boolean[]{false, false},
        new boolean[]{false, false},
        new boolean[]{false, false},
        new boolean[]{true, false},  // second field hidden
        new long[]{0, 0},
        new double[]{0.0, 0.0},
        new boolean[]{false, false},
        new byte[][]{null, null},
        new int[][]{null, null},
        new byte[][][]{null, null});

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
      oos.writeObject(original);
    }
    ProtobufSchemaDescriptor roundTrip;
    try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()))) {
      roundTrip = (ProtobufSchemaDescriptor) ois.readObject();
    }
    assertArrayEquals(original.isOutput, roundTrip.isOutput);
  }

  @Test
  void testLegacyStreamWithoutIsOutputBackfillsAllOutput() throws Exception {
    int intType = DType.INT32.getTypeId().getNativeId();
    ProtobufSchemaDescriptor original = new ProtobufSchemaDescriptor(
        new int[]{1, 2},
        new int[]{-1, -1},
        new int[]{0, 0},
        new int[]{Protobuf.WT_VARINT, Protobuf.WT_VARINT},
        new int[]{intType, intType},
        new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
        new boolean[]{false, false},
        new boolean[]{false, false},
        new boolean[]{false, false},
        new boolean[]{true, true},
        new long[]{0, 0},
        new double[]{0.0, 0.0},
        new boolean[]{false, false},
        new byte[][]{null, null},
        new int[][]{null, null},
        new byte[][][]{null, null});

    // Simulate a stream written before isOutput existed: such a stream deserializes the field as
    // null. The constructor forbids a null isOutput, so null it out via reflection before
    // serializing; the resulting blob deserializes with isOutput == null and must hit the
    // readObject() backfill rather than failing validation.
    java.lang.reflect.Field f = ProtobufSchemaDescriptor.class.getDeclaredField("isOutput");
    f.setAccessible(true);
    f.set(original, null);

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
      oos.writeObject(original);
    }
    ProtobufSchemaDescriptor roundTrip;
    try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()))) {
      roundTrip = (ProtobufSchemaDescriptor) ois.readObject();
    }
    assertArrayEquals(new boolean[]{true, true}, roundTrip.isOutput);
  }
}
