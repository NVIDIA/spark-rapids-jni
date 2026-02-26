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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.NativeDepsLoader;

/**
 * GPU protobuf decoding utilities.
 *
 * This API uses a multi-pass approach for efficient decoding:
 * <ul>
 *   <li>Pass 1: Scan all messages, count nested elements and repeated field occurrences</li>
 *   <li>Pass 2: Prefix sum to compute output offsets for arrays and nested structs</li>
 *   <li>Pass 3: Extract data using pre-computed offsets</li>
 *   <li>Pass 4: Build nested column structure</li>
 * </ul>
 *
 * The schema is represented as a flattened array of field descriptors with parent-child
 * relationships. Top-level fields have parentIndices == -1 and depthLevels == 0.
 * For pure scalar schemas, all fields are top-level with isRepeated == false.
 *
 * Supported protobuf field types include:
 * <ul>
 *   <li>VARINT: {@code int32}, {@code int64}, {@code uint32}, {@code uint64}, {@code bool}</li>
 *   <li>ZIGZAG VARINT (encoding=2): {@code sint32}, {@code sint64}</li>
 *   <li>FIXED32 (encoding=1): {@code fixed32}, {@code sfixed32}, {@code float}</li>
 *   <li>FIXED64 (encoding=1): {@code fixed64}, {@code sfixed64}, {@code double}</li>
 *   <li>LENGTH_DELIMITED: {@code string}, {@code bytes}, nested {@code message}</li>
 *   <li>Nested messages and repeated fields</li>
 * </ul>
 */
public class Protobuf {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static final int ENC_DEFAULT = 0;
  public static final int ENC_FIXED   = 1;
  public static final int ENC_ZIGZAG  = 2;
  public static final int ENC_ENUM_STRING = 3;

  // Wire type constants
  public static final int WT_VARINT = 0;
  public static final int WT_64BIT  = 1;
  public static final int WT_LEN    = 2;
  public static final int WT_32BIT  = 5;
  private static final int MAX_FIELD_NUMBER = (1 << 29) - 1;

  /**
   * Decode protobuf messages into a STRUCT column using a flattened schema representation.
   *
   * The schema is represented as parallel arrays where nested fields have parent indices
   * pointing to their containing message field. For pure scalar schemas, all fields are
   * top-level (parentIndices == -1, depthLevels == 0, isRepeated == false).
   *
   * @param binaryInput column of type LIST&lt;INT8/UINT8&gt; where each row is one protobuf message.
   * @param fieldNumbers Protobuf field numbers for all fields in the flattened schema.
   * @param parentIndices Parent field index for each field (-1 for top-level fields).
   * @param depthLevels Nesting depth for each field (0 for top-level).
   * @param wireTypes Expected wire type for each field (WT_VARINT, WT_64BIT, WT_LEN, WT_32BIT).
   * @param outputTypeIds cudf native type ids for output columns.
   * @param encodings Encoding info for each field (0=default, 1=fixed, 2=zigzag,
   *                  3=enum-as-string).
   * @param isRepeated Whether each field is a repeated field (array).
   * @param isRequired Whether each field is required (proto2).
   * @param hasDefaultValue Whether each field has a default value.
   * @param defaultInts Default values for int/long/enum fields.
   * @param defaultFloats Default values for float/double fields.
   * @param defaultBools Default values for bool fields.
   * @param defaultStrings Default values for string/bytes fields as UTF-8 bytes.
   * @param enumValidValues Valid enum values for each field (null if not an enum).
   * @param enumNames Enum value names for enum-as-string fields (null if not enum-as-string).
   *                  For each field, this is a byte[][] containing UTF-8 enum names ordered by
   *                  the same sorted order as enumValidValues for that field.
   * @param failOnErrors if true, throw an exception on malformed protobuf messages.
   * @return a cudf STRUCT column with nested structure.
   */
  public static ColumnVector decodeToStruct(ColumnView binaryInput,
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
                                            byte[][][] enumNames,
                                            boolean failOnErrors) {
    // Parameter validation
    if (fieldNumbers == null || parentIndices == null || depthLevels == null ||
        wireTypes == null || outputTypeIds == null || encodings == null ||
        isRepeated == null || isRequired == null || hasDefaultValue == null ||
        defaultInts == null || defaultFloats == null || defaultBools == null ||
        defaultStrings == null || enumValidValues == null || enumNames == null) {
      throw new IllegalArgumentException("Arrays must be non-null");
    }

    int numFields = fieldNumbers.length;
    if (parentIndices.length != numFields ||
        depthLevels.length != numFields ||
        wireTypes.length != numFields ||
        outputTypeIds.length != numFields ||
        encodings.length != numFields ||
        isRepeated.length != numFields ||
        isRequired.length != numFields ||
        hasDefaultValue.length != numFields ||
        defaultInts.length != numFields ||
        defaultFloats.length != numFields ||
        defaultBools.length != numFields ||
        defaultStrings.length != numFields ||
        enumValidValues.length != numFields ||
        enumNames.length != numFields) {
      throw new IllegalArgumentException("All arrays must have the same length");
    }

    // Validate field numbers are positive and within protobuf spec range
    for (int i = 0; i < fieldNumbers.length; i++) {
      if (fieldNumbers[i] <= 0 || fieldNumbers[i] > MAX_FIELD_NUMBER) {
        throw new IllegalArgumentException(
            "Invalid field number at index " + i + ": " + fieldNumbers[i] +
            " (field numbers must be 1-" + MAX_FIELD_NUMBER + ")");
      }
    }

    // Validate encoding values
    for (int i = 0; i < encodings.length; i++) {
      int enc = encodings[i];
      if (enc < ENC_DEFAULT || enc > ENC_ENUM_STRING) {
        throw new IllegalArgumentException(
            "Invalid encoding value at index " + i + ": " + enc +
            " (expected " + ENC_DEFAULT + ", " + ENC_FIXED + ", " + ENC_ZIGZAG +
            ", or " + ENC_ENUM_STRING + ")");
      }
    }

    long handle = decodeToStruct(binaryInput.getNativeView(),
                                 fieldNumbers, parentIndices, depthLevels,
                                 wireTypes, outputTypeIds, encodings,
                                 isRepeated, isRequired, hasDefaultValue,
                                 defaultInts, defaultFloats, defaultBools,
                                 defaultStrings, enumValidValues, enumNames, failOnErrors);
    return new ColumnVector(handle);
  }

  /**
   * Backward-compatible overload for callers that don't provide enum name mappings.
   * This keeps existing JNI tests and call-sites source-compatible.
   */
  public static ColumnVector decodeToStruct(ColumnView binaryInput,
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
                                            boolean failOnErrors) {
    return decodeToStruct(binaryInput, fieldNumbers, parentIndices, depthLevels, wireTypes,
        outputTypeIds, encodings, isRepeated, isRequired, hasDefaultValue, defaultInts,
        defaultFloats, defaultBools, defaultStrings, enumValidValues,
        new byte[fieldNumbers.length][][], failOnErrors);
  }

  private static native long decodeToStruct(long binaryInputView,
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
                                            byte[][][] enumNames,
                                            boolean failOnErrors);
}
