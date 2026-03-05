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

/**
 * Immutable descriptor for a flattened protobuf schema, grouping the parallel arrays
 * that describe field structure, types, defaults, and enum metadata.
 *
 * <p>Use this class instead of passing 15+ individual arrays through the JNI boundary.
 * Validation is performed once in the constructor (and again on deserialization).
 *
 * <p>All arrays are defensively copied in the constructor to guarantee immutability.
 * Package-private field access from {@link Protobuf} is safe because the stored arrays
 * cannot be mutated by the original caller.
 */
public final class ProtobufSchemaDescriptor implements java.io.Serializable {
  private static final long serialVersionUID = 1L;
  private static final int MAX_FIELD_NUMBER = (1 << 29) - 1;

  final int[] fieldNumbers;
  final int[] parentIndices;
  final int[] depthLevels;
  final int[] wireTypes;
  final int[] outputTypeIds;
  final int[] encodings;
  final boolean[] isRepeated;
  final boolean[] isRequired;
  final boolean[] hasDefaultValue;
  final long[] defaultInts;
  final double[] defaultFloats;
  final boolean[] defaultBools;
  final byte[][] defaultStrings;
  final int[][] enumValidValues;
  final byte[][][] enumNames;

  /**
   * @throws IllegalArgumentException if any array is null, arrays have mismatched lengths,
   *         field numbers are out of range, or encoding values are invalid.
   */
  public ProtobufSchemaDescriptor(
      int[] fieldNumbers,
      int[] parentIndices,
      int[] depthLevels,
      int[] wireTypes,
      int[] outputTypeIds,
      int[] encodings,
      boolean[] isRepeated,
      boolean[] isRequired,
      boolean[] hasDefaultValue,
      long[] defaultInts,
      double[] defaultFloats,
      boolean[] defaultBools,
      byte[][] defaultStrings,
      int[][] enumValidValues,
      byte[][][] enumNames) {

    validate(fieldNumbers, parentIndices, depthLevels, wireTypes, outputTypeIds,
        encodings, isRepeated, isRequired, hasDefaultValue, defaultInts,
        defaultFloats, defaultBools, defaultStrings, enumValidValues, enumNames);

    this.fieldNumbers = fieldNumbers.clone();
    this.parentIndices = parentIndices.clone();
    this.depthLevels = depthLevels.clone();
    this.wireTypes = wireTypes.clone();
    this.outputTypeIds = outputTypeIds.clone();
    this.encodings = encodings.clone();
    this.isRepeated = isRepeated.clone();
    this.isRequired = isRequired.clone();
    this.hasDefaultValue = hasDefaultValue.clone();
    this.defaultInts = defaultInts.clone();
    this.defaultFloats = defaultFloats.clone();
    this.defaultBools = defaultBools.clone();
    this.defaultStrings = deepCopy(defaultStrings);
    this.enumValidValues = deepCopy(enumValidValues);
    this.enumNames = deepCopy(enumNames);
  }

  public int numFields() { return fieldNumbers.length; }

  private void readObject(java.io.ObjectInputStream in)
      throws java.io.IOException, ClassNotFoundException {
    in.defaultReadObject();
    try {
      validate(fieldNumbers, parentIndices, depthLevels, wireTypes, outputTypeIds,
          encodings, isRepeated, isRequired, hasDefaultValue, defaultInts,
          defaultFloats, defaultBools, defaultStrings, enumValidValues, enumNames);
    } catch (IllegalArgumentException e) {
      java.io.InvalidObjectException ioe = new java.io.InvalidObjectException(e.getMessage());
      ioe.initCause(e);
      throw ioe;
    }
  }

  private static byte[][] deepCopy(byte[][] src) {
    byte[][] dst = new byte[src.length][];
    for (int i = 0; i < src.length; i++) {
      dst[i] = src[i] != null ? src[i].clone() : null;
    }
    return dst;
  }

  private static int[][] deepCopy(int[][] src) {
    int[][] dst = new int[src.length][];
    for (int i = 0; i < src.length; i++) {
      dst[i] = src[i] != null ? src[i].clone() : null;
    }
    return dst;
  }

  private static byte[][][] deepCopy(byte[][][] src) {
    byte[][][] dst = new byte[src.length][][];
    for (int i = 0; i < src.length; i++) {
      if (src[i] == null) continue;
      dst[i] = new byte[src[i].length][];
      for (int j = 0; j < src[i].length; j++) {
        dst[i][j] = src[i][j] != null ? src[i][j].clone() : null;
      }
    }
    return dst;
  }

  private static void validate(
      int[] fieldNumbers, int[] parentIndices, int[] depthLevels,
      int[] wireTypes, int[] outputTypeIds, int[] encodings,
      boolean[] isRepeated, boolean[] isRequired, boolean[] hasDefaultValue,
      long[] defaultInts, double[] defaultFloats, boolean[] defaultBools,
      byte[][] defaultStrings, int[][] enumValidValues, byte[][][] enumNames) {

    if (fieldNumbers == null || parentIndices == null || depthLevels == null ||
        wireTypes == null || outputTypeIds == null || encodings == null ||
        isRepeated == null || isRequired == null || hasDefaultValue == null ||
        defaultInts == null || defaultFloats == null || defaultBools == null ||
        defaultStrings == null || enumValidValues == null || enumNames == null) {
      throw new IllegalArgumentException("All schema arrays must be non-null");
    }

    int n = fieldNumbers.length;
    if (parentIndices.length != n || depthLevels.length != n ||
        wireTypes.length != n || outputTypeIds.length != n ||
        encodings.length != n || isRepeated.length != n ||
        isRequired.length != n || hasDefaultValue.length != n ||
        defaultInts.length != n || defaultFloats.length != n ||
        defaultBools.length != n || defaultStrings.length != n ||
        enumValidValues.length != n || enumNames.length != n) {
      throw new IllegalArgumentException("All schema arrays must have the same length");
    }

    for (int i = 0; i < n; i++) {
      if (fieldNumbers[i] <= 0 || fieldNumbers[i] > MAX_FIELD_NUMBER) {
        throw new IllegalArgumentException(
            "Invalid field number at index " + i + ": " + fieldNumbers[i] +
            " (must be 1-" + MAX_FIELD_NUMBER + ")");
      }
      int wt = wireTypes[i];
      if (wt != 0 && wt != 1 && wt != 2 && wt != 5) {
        throw new IllegalArgumentException(
            "Invalid wire type at index " + i + ": " + wt +
            " (must be one of {0, 1, 2, 5})");
      }
      int enc = encodings[i];
      if (enc < Protobuf.ENC_DEFAULT || enc > Protobuf.ENC_ENUM_STRING) {
        throw new IllegalArgumentException(
            "Invalid encoding at index " + i + ": " + enc);
      }
      if (enumValidValues[i] != null) {
        int[] ev = enumValidValues[i];
        for (int j = 1; j < ev.length; j++) {
          if (ev[j] < ev[j - 1]) {
            throw new IllegalArgumentException(
                "enumValidValues[" + i + "] must be sorted in ascending order " +
                "(binary search requires it), but found " + ev[j - 1] + " before " + ev[j]);
          }
        }
        if (enumNames[i] != null && enumNames[i].length != ev.length) {
          throw new IllegalArgumentException(
              "enumNames[" + i + "].length (" + enumNames[i].length + ") must equal " +
              "enumValidValues[" + i + "].length (" + ev.length + ")");
        }
      }
    }
  }
}
