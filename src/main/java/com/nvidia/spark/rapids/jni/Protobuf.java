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
 * This API is intentionally limited to top-level scalar fields whose
 * values can be represented by a single cuDF scalar type. Supported protobuf field types
 * include scalar fields using the standard protobuf wire encodings:
 * <ul>
 *   <li>VARINT: {@code int32}, {@code int64}, {@code uint32}, {@code uint64}, {@code bool}</li>
 *   <li>ZIGZAG VARINT (encoding=2): {@code sint32}, {@code sint64}</li>
 *   <li>FIXED32 (encoding=1): {@code fixed32}, {@code sfixed32}, {@code float}</li>
 *   <li>FIXED64 (encoding=1): {@code fixed64}, {@code sfixed64}, {@code double}</li>
 *   <li>LENGTH_DELIMITED: {@code string}, {@code bytes}</li>
 * </ul>
 * Each decoded field becomes a child column of the resulting STRUCT, with its cuDF type
 * specified via the corresponding {@code typeIds} entry.
 * <p>
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
   * Decode a protobuf message-per-row binary column into a single STRUCT column.
   *
   * @param binaryInput column of type LIST&lt;INT8/UINT8&gt; where each row is one protobuf message.
   * @param fieldNumbers protobuf field numbers to decode (one per struct child)
   * @param typeIds cudf native type ids (one per struct child)
   * @param typeScales encoding info or decimal scales:
   *                   - For non-decimal types, this is the encoding: 0=default, 1=fixed, 2=zigzag.
   *                   - For decimal types, this is the scale (currently unsupported).
   * @return a cudf STRUCT column where children correspond 1:1 with {@code fieldNumbers}/{@code typeIds}.
   */
  public static ColumnVector decodeToStruct(ColumnView binaryInput,
                                           int[] fieldNumbers,
                                           int[] typeIds,
                                           int[] typeScales) {
    return decodeToStruct(binaryInput, fieldNumbers, typeIds, typeScales, true);
  }

  /**
   * Decode a protobuf message-per-row binary column into a single STRUCT column.
   *
   * @param binaryInput column of type LIST&lt;INT8/UINT8&gt; where each row is one protobuf message.
   * @param fieldNumbers protobuf field numbers to decode (one per struct child)
   * @param typeIds cudf native type ids (one per struct child)
   * @param typeScales encoding info or decimal scales:
   *                   - For non-decimal types, this is the encoding: 0=default, 1=fixed, 2=zigzag.
   *                   - For decimal types, this is the scale (currently unsupported).
   * @param failOnErrors if true, throw an exception on malformed protobuf messages.
   *                     If false, return nulls for fields that cannot be parsed.
   * @return a cudf STRUCT column where children correspond 1:1 with {@code fieldNumbers}/{@code typeIds}.
   */
  public static ColumnVector decodeToStruct(ColumnView binaryInput,
                                           int[] fieldNumbers,
                                           int[] typeIds,
                                           int[] typeScales,
                                           boolean failOnErrors) {
    if (fieldNumbers == null || typeIds == null || typeScales == null) {
      throw new IllegalArgumentException("fieldNumbers/typeIds/typeScales must be non-null");
    }
    if (fieldNumbers.length != typeIds.length || fieldNumbers.length != typeScales.length) {
      throw new IllegalArgumentException("fieldNumbers/typeIds/typeScales must be the same length");
    }
    // Validate field numbers are positive (protobuf field numbers must be 1-536870911)
    for (int i = 0; i < fieldNumbers.length; i++) {
      if (fieldNumbers[i] <= 0) {
        throw new IllegalArgumentException(
            "Invalid field number at index " + i + ": " + fieldNumbers[i]
                + " (field numbers must be positive)");
      }
    }
    // Validate encoding values are within valid range
    for (int i = 0; i < typeScales.length; i++) {
      int enc = typeScales[i];
      if (enc < ENC_DEFAULT || enc > ENC_ZIGZAG) {
        throw new IllegalArgumentException(
            "Invalid encoding value at index " + i + ": " + enc
                + " (expected " + ENC_DEFAULT + ", " + ENC_FIXED + ", or " + ENC_ZIGZAG + ")");
      }
    }
    long handle = decodeToStruct(binaryInput.getNativeView(), fieldNumbers, typeIds, typeScales, failOnErrors);
    return new ColumnVector(handle);
  }

  private static native long decodeToStruct(long binaryInputView,
                                            int[] fieldNumbers,
                                            int[] typeIds,
                                            int[] typeScales,
                                            boolean failOnErrors);
}

