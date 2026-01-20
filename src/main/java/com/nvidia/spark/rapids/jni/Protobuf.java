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
 * This API uses a two-pass approach for efficient decoding:
 * <ul>
 *   <li>Pass 1: Scan all messages once, recording (offset, length) for each requested field</li>
 *   <li>Pass 2: Extract data in parallel using the recorded locations</li>
 * </ul>
 *
 * This is significantly faster than per-field parsing when decoding multiple fields,
 * as each message is only parsed once regardless of the number of fields.
 *
 * Supported protobuf field types include scalar fields using the standard wire encodings:
 * <ul>
 *   <li>VARINT: {@code int32}, {@code int64}, {@code uint32}, {@code uint64}, {@code bool}</li>
 *   <li>ZIGZAG VARINT (encoding=2): {@code sint32}, {@code sint64}</li>
 *   <li>FIXED32 (encoding=1): {@code fixed32}, {@code sfixed32}, {@code float}</li>
 *   <li>FIXED64 (encoding=1): {@code fixed64}, {@code sfixed64}, {@code double}</li>
 *   <li>LENGTH_DELIMITED: {@code string}, {@code bytes}</li>
 * </ul>
 *
 * Nested messages, repeated fields, map fields, and oneof fields are out of scope for this API.
 */
public class Protobuf {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static final int ENC_DEFAULT = 0;
  public static final int ENC_FIXED   = 1;
  public static final int ENC_ZIGZAG  = 2;

  /**
   * Decode a protobuf message-per-row binary column into a STRUCT column.
   *
   * This method supports schema projection: only the fields specified in
   * {@code decodedFieldIndices} will be decoded. Other fields in the output
   * struct will contain all null values.
   *
   * @param binaryInput column of type LIST&lt;INT8/UINT8&gt; where each row is one protobuf message.
   * @param totalNumFields Total number of fields in the output struct (including null columns).
   * @param decodedFieldIndices Indices into the output struct for fields that should be decoded.
   *                            These must be sorted in ascending order.
   * @param fieldNumbers Protobuf field numbers for decoded fields (parallel to decodedFieldIndices).
   * @param allTypeIds cudf native type ids for ALL fields in the output struct (size = totalNumFields).
   * @param encodings Encoding info for decoded fields (parallel to decodedFieldIndices):
   *                  0=default (varint), 1=fixed, 2=zigzag.
   * @return a cudf STRUCT column with totalNumFields children. Decoded fields contain parsed data,
   *         other fields contain all nulls.
   */
  public static ColumnVector decodeToStruct(ColumnView binaryInput,
                                           int totalNumFields,
                                           int[] decodedFieldIndices,
                                           int[] fieldNumbers,
                                           int[] allTypeIds,
                                           int[] encodings) {
    return decodeToStruct(binaryInput, totalNumFields, decodedFieldIndices, fieldNumbers,
                          allTypeIds, encodings, true);
  }

  /**
   * Decode a protobuf message-per-row binary column into a STRUCT column.
   *
   * This method supports schema projection: only the fields specified in
   * {@code decodedFieldIndices} will be decoded. Other fields in the output
   * struct will contain all null values.
   *
   * @param binaryInput column of type LIST&lt;INT8/UINT8&gt; where each row is one protobuf message.
   * @param totalNumFields Total number of fields in the output struct (including null columns).
   * @param decodedFieldIndices Indices into the output struct for fields that should be decoded.
   *                            These must be sorted in ascending order.
   * @param fieldNumbers Protobuf field numbers for decoded fields (parallel to decodedFieldIndices).
   * @param allTypeIds cudf native type ids for ALL fields in the output struct (size = totalNumFields).
   * @param encodings Encoding info for decoded fields (parallel to decodedFieldIndices):
   *                  0=default (varint), 1=fixed, 2=zigzag.
   * @param failOnErrors if true, throw an exception on malformed protobuf messages.
   *                     If false, return nulls for fields that cannot be parsed.
   *                     Note: error checking is performed after all fields are processed,
   *                     not between fields, to avoid synchronization overhead.
   * @return a cudf STRUCT column with totalNumFields children. Decoded fields contain parsed data,
   *         other fields contain all nulls.
   */
  public static ColumnVector decodeToStruct(ColumnView binaryInput,
                                           int totalNumFields,
                                           int[] decodedFieldIndices,
                                           int[] fieldNumbers,
                                           int[] allTypeIds,
                                           int[] encodings,
                                           boolean failOnErrors) {
    // Parameter validation
    if (decodedFieldIndices == null || fieldNumbers == null ||
        allTypeIds == null || encodings == null) {
      throw new IllegalArgumentException("Arrays must be non-null");
    }
    if (totalNumFields < 0) {
      throw new IllegalArgumentException("totalNumFields must be non-negative");
    }
    if (allTypeIds.length != totalNumFields) {
      throw new IllegalArgumentException(
          "allTypeIds length (" + allTypeIds.length + ") must equal totalNumFields (" +
          totalNumFields + ")");
    }
    if (decodedFieldIndices.length != fieldNumbers.length ||
        decodedFieldIndices.length != encodings.length) {
      throw new IllegalArgumentException(
          "decodedFieldIndices/fieldNumbers/encodings must be the same length");
    }

    // Validate decoded field indices are in bounds and sorted
    int prevIdx = -1;
    for (int i = 0; i < decodedFieldIndices.length; i++) {
      int idx = decodedFieldIndices[i];
      if (idx < 0 || idx >= totalNumFields) {
        throw new IllegalArgumentException(
            "Invalid decoded field index at position " + i + ": " + idx +
            " (must be in range [0, " + totalNumFields + "))");
      }
      if (idx <= prevIdx) {
        throw new IllegalArgumentException(
            "decodedFieldIndices must be sorted in ascending order without duplicates");
      }
      prevIdx = idx;
    }

    // Validate field numbers are positive
    for (int i = 0; i < fieldNumbers.length; i++) {
      if (fieldNumbers[i] <= 0) {
        throw new IllegalArgumentException(
            "Invalid field number at index " + i + ": " + fieldNumbers[i] +
            " (field numbers must be positive)");
      }
    }

    // Validate encoding values
    for (int i = 0; i < encodings.length; i++) {
      int enc = encodings[i];
      if (enc < ENC_DEFAULT || enc > ENC_ZIGZAG) {
        throw new IllegalArgumentException(
            "Invalid encoding value at index " + i + ": " + enc +
            " (expected " + ENC_DEFAULT + ", " + ENC_FIXED + ", or " + ENC_ZIGZAG + ")");
      }
    }

    long handle = decodeToStruct(binaryInput.getNativeView(), totalNumFields,
                                 decodedFieldIndices, fieldNumbers, allTypeIds,
                                 encodings, failOnErrors);
    return new ColumnVector(handle);
  }

  private static native long decodeToStruct(long binaryInputView,
                                            int totalNumFields,
                                            int[] decodedFieldIndices,
                                            int[] fieldNumbers,
                                            int[] allTypeIds,
                                            int[] encodings,
                                            boolean failOnErrors);
}
