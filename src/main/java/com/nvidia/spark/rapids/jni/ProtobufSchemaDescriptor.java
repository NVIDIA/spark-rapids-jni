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
 * Validation is performed once in the constructor.
 */
public final class ProtobufSchemaDescriptor implements java.io.Serializable {
  private static final long serialVersionUID = 1L;
  private static final int MAX_FIELD_NUMBER = (1 << 29) - 1;

  public final int[] fieldNumbers;
  public final int[] parentIndices;
  public final int[] depthLevels;
  public final int[] wireTypes;
  public final int[] outputTypeIds;
  public final int[] encodings;
  public final boolean[] isRepeated;
  public final boolean[] isRequired;
  public final boolean[] hasDefaultValue;
  public final long[] defaultInts;
  public final double[] defaultFloats;
  public final boolean[] defaultBools;
  public final byte[][] defaultStrings;
  public final int[][] enumValidValues;
  public final byte[][][] enumNames;

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
      int enc = encodings[i];
      if (enc < Protobuf.ENC_DEFAULT || enc > Protobuf.ENC_ENUM_STRING) {
        throw new IllegalArgumentException(
            "Invalid encoding at index " + i + ": " + enc);
      }
    }

    this.fieldNumbers = fieldNumbers;
    this.parentIndices = parentIndices;
    this.depthLevels = depthLevels;
    this.wireTypes = wireTypes;
    this.outputTypeIds = outputTypeIds;
    this.encodings = encodings;
    this.isRepeated = isRepeated;
    this.isRequired = isRequired;
    this.hasDefaultValue = hasDefaultValue;
    this.defaultInts = defaultInts;
    this.defaultFloats = defaultFloats;
    this.defaultBools = defaultBools;
    this.defaultStrings = defaultStrings;
    this.enumValidValues = enumValidValues;
    this.enumNames = enumNames;
  }

  public int numFields() { return fieldNumbers.length; }
}
