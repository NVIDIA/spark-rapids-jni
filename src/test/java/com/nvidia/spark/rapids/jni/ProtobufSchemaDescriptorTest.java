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
    DType outputType = (encoding == Protobuf.ENC_ENUM_STRING) ? DType.STRING : DType.INT32;
    return new ProtobufSchemaDescriptorBuilder()
        .addField(1, outputType).encoding(encoding)
            .repeated(isRepeated).hasDefault(hasDefaultValue)
            .enumValidValues(enumValidValues).enumNames(enumNames)
        .build();
  }

  @Test
  void testRepeatedFieldCannotCarryDefaultValue() {
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(true, true, Protobuf.ENC_DEFAULT, null, null));
  }

  @Test
  void testFieldCannotBeBothRepeatedAndRequired() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT32).repeated().required()
            .build());
  }

  @Test
  void testStructFieldCannotCarryDefaultValue() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRUCT).hasDefault()
            .build());
  }

  @Test
  void testListFieldCannotCarryDefaultValue() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.LIST).hasDefault()
            .build());
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
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRUCT)
            .addField(7, DType.INT32).parent(0)
            .addField(7, DType.INT32).parent(0)  // duplicate field number under same parent
            .build());
  }

  @Test
  void testDuplicateFieldNumbersUnderDifferentParentsAllowed() {
    assertDoesNotThrow(() ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRUCT)
            .addField(2, DType.STRUCT)
            .addField(7, DType.INT32).parent(0)
            .addField(7, DType.INT32).parent(1)  // same number, different parents -> allowed
            .build());
  }

  @Test
  void testChildParentMustBeStruct() {
    // Field 2 is parented under field 1, but field 1 is INT32 (not STRUCT) -> illegal.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT32)
            .addField(2, DType.INT32).parent(0)
            .build());
  }

  @Test
  void testEncodingCompatibilityValidation() {
    // INT32 with a 32-bit wire type under default (non-fixed) encoding is incompatible.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT32).wireType(Protobuf.WT_32BIT)
            .build());

    // Enum-as-string must use the varint wire type; WT_LEN is incompatible.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRING).wireType(Protobuf.WT_LEN)
                .enumMetadata(new int[]{0, 1}, new byte[][]{"A".getBytes(), "B".getBytes()})
            .build());
  }

  @Test
  void testDepthAboveSupportedLimitRejected() {
    // Build a STRUCT chain reaching depth 10 (one past the limit), capped with an INT32 leaf.
    ProtobufSchemaDescriptorBuilder builder = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT);
    for (int i = 1; i <= 9; i++) {
      builder.addField(i + 1, DType.STRUCT).parent(i - 1);
    }
    builder.addField(11, DType.INT32).parent(9);  // depth 10 -> exceeds limit
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  void testSerializationRoundTripPreservesContentsAndIsolation() throws Exception {
    ProtobufSchemaDescriptor original = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRING)
            .enumMetadata(new int[]{0, 1}, new byte[][]{"A".getBytes(), "B".getBytes()})
            .defaultString("def".getBytes())
            .defaultInt(7)  // non-zero numeric default to exercise scalar round-trip
        .build();

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
    assertArrayEquals(original.defaultInts, roundTrip.defaultInts);
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
}
