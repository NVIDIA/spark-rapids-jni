/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
 * Simple GPU protobuf decoding utilities.
 *
 * This is intentionally limited to "simple types" (top-level scalar fields) as a first patch.
 * Nested/repeated/map/oneof are out of scope for this API.
 */
public class ProtobufSimple {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Decode a protobuf message-per-row binary column into a single STRUCT column.
   *
   * @param binaryInput column of type LIST&lt;INT8/UINT8&gt; where each row is one protobuf message.
   * @param fieldNumbers protobuf field numbers to decode (one per struct child)
   * @param typeIds cudf native type ids (one per struct child)
   * @param typeScales cudf decimal scales (unused for simple types; pass 0s)
   * @return a cudf STRUCT column where children correspond 1:1 with {@code fieldNumbers}/{@code typeIds}.
   */
  public static ColumnVector decodeToStruct(ColumnView binaryInput,
                                           int[] fieldNumbers,
                                           int[] typeIds,
                                           int[] typeScales) {
    if (fieldNumbers == null || typeIds == null || typeScales == null) {
      throw new IllegalArgumentException("fieldNumbers/typeIds/typeScales must be non-null");
    }
    if (fieldNumbers.length != typeIds.length || fieldNumbers.length != typeScales.length) {
      throw new IllegalArgumentException("fieldNumbers/typeIds/typeScales must be the same length");
    }
    long handle = decodeToStruct(binaryInput.getNativeView(), fieldNumbers, typeIds, typeScales);
    return new ColumnVector(handle);
  }

  private static native long decodeToStruct(long binaryInputView,
                                            int[] fieldNumbers,
                                            int[] typeIds,
                                            int[] typeScales);
}



