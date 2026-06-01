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

import java.util.ArrayList;
import java.util.List;

/**
 * Test-only fluent builder for {@link ProtobufSchemaDescriptor}.
 *
 * <p>The descriptor is a set of parallel arrays, which is error-prone to assemble by hand: the
 * reader has to count positions across a dozen arrays to see what a single field looks like. This
 * builder lets each field be described in one place, with only the non-default attributes named:
 *
 * <pre>{@code
 * ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
 *     .addField(1, DType.INT32)
 *     .addField(2, DType.STRING).required()
 *     .build();
 * }</pre>
 *
 * <p>Each attribute setter applies to the most recently added field. Wire types are derived from
 * the output type and encoding unless overridden via {@link #wireType(int)}.
 */
public final class ProtobufSchemaDescriptorBuilder {
  private static final class Field {
    int fieldNumber;
    int outputTypeId;
    int parentIndex = -1;
    int depth = 0;
    int encoding = Protobuf.ENC_DEFAULT;
    Integer wireType = null;  // null => derive from output type + encoding
    boolean isRepeated = false;
    boolean isRequired = false;
    boolean hasDefaultValue = false;
    long defaultInt = 0L;
    double defaultFloat = 0.0;
    boolean defaultBool = false;
    byte[] defaultString = null;
    int[] enumValidValues = null;
    byte[][] enumNames = null;
  }

  private final List<Field> fields = new ArrayList<>();

  /** Start a new top-level field with the given field number and output type. */
  public ProtobufSchemaDescriptorBuilder addField(int fieldNumber, DType outputType) {
    Field f = new Field();
    f.fieldNumber = fieldNumber;
    f.outputTypeId = outputType.getTypeId().getNativeId();
    fields.add(f);
    return this;
  }

  /**
   * Mark the current field as a child of {@code parentIndex}, setting its depth to
   * parent depth + 1.
   */
  public ProtobufSchemaDescriptorBuilder parent(int parentIndex) {
    Field f = current();
    f.parentIndex = parentIndex;
    f.depth = fields.get(parentIndex).depth + 1;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder encoding(int encoding) {
    current().encoding = encoding;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder wireType(int wireType) {
    current().wireType = wireType;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder repeated() {
    current().isRepeated = true;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder required() {
    current().isRequired = true;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder defaultInt(long value) {
    Field f = current();
    f.hasDefaultValue = true;
    f.defaultInt = value;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder defaultFloat(double value) {
    Field f = current();
    f.hasDefaultValue = true;
    f.defaultFloat = value;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder defaultBool(boolean value) {
    Field f = current();
    f.hasDefaultValue = true;
    f.defaultBool = value;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder defaultString(byte[] value) {
    Field f = current();
    f.hasDefaultValue = true;
    f.defaultString = value;
    return this;
  }

  /**
   * Set the has-default flag without a value. Used by negative tests that pair a default with an
   * incompatible field (e.g. a repeated or STRUCT field) to exercise validation.
   */
  public ProtobufSchemaDescriptorBuilder hasDefault() {
    current().hasDefaultValue = true;
    return this;
  }

  /** Provide enum-as-string metadata for the current field (also sets ENC_ENUM_STRING). */
  public ProtobufSchemaDescriptorBuilder enumMetadata(int[] validValues, byte[][] names) {
    Field f = current();
    f.encoding = Protobuf.ENC_ENUM_STRING;
    f.enumValidValues = validValues;
    f.enumNames = names;
    return this;
  }

  /** Attach raw enum metadata without forcing the encoding (for validation-only tests). */
  public ProtobufSchemaDescriptorBuilder enumValidValues(int[] validValues) {
    current().enumValidValues = validValues;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder enumNames(byte[][] names) {
    current().enumNames = names;
    return this;
  }

  public ProtobufSchemaDescriptor build() {
    int n = fields.size();
    int[] fieldNumbers = new int[n];
    int[] parentIndices = new int[n];
    int[] depthLevels = new int[n];
    int[] wireTypes = new int[n];
    int[] outputTypeIds = new int[n];
    int[] encodings = new int[n];
    boolean[] isRepeated = new boolean[n];
    boolean[] isRequired = new boolean[n];
    boolean[] hasDefaultValue = new boolean[n];
    long[] defaultInts = new long[n];
    double[] defaultFloats = new double[n];
    boolean[] defaultBools = new boolean[n];
    byte[][] defaultStrings = new byte[n][];
    int[][] enumValidValues = new int[n][];
    byte[][][] enumNames = new byte[n][][];

    for (int i = 0; i < n; i++) {
      Field f = fields.get(i);
      fieldNumbers[i] = f.fieldNumber;
      parentIndices[i] = f.parentIndex;
      depthLevels[i] = f.depth;
      outputTypeIds[i] = f.outputTypeId;
      encodings[i] = f.encoding;
      wireTypes[i] = f.wireType != null ? f.wireType : deriveWireType(f.outputTypeId, f.encoding);
      isRepeated[i] = f.isRepeated;
      isRequired[i] = f.isRequired;
      hasDefaultValue[i] = f.hasDefaultValue;
      defaultInts[i] = f.defaultInt;
      defaultFloats[i] = f.defaultFloat;
      defaultBools[i] = f.defaultBool;
      defaultStrings[i] = f.defaultString;
      enumValidValues[i] = f.enumValidValues;
      enumNames[i] = f.enumNames;
    }

    return new ProtobufSchemaDescriptor(fieldNumbers, parentIndices, depthLevels, wireTypes,
        outputTypeIds, encodings, isRepeated, isRequired, hasDefaultValue,
        defaultInts, defaultFloats, defaultBools, defaultStrings, enumValidValues, enumNames);
  }

  private Field current() {
    if (fields.isEmpty()) {
      throw new IllegalStateException("addField must be called before setting field attributes");
    }
    return fields.get(fields.size() - 1);
  }

  static int deriveWireType(int typeId, int encoding) {
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
}
