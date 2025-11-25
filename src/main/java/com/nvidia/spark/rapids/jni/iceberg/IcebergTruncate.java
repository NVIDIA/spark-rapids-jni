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
package com.nvidia.spark.rapids.jni.iceberg;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.NativeDepsLoader;

/** Utility class for Iceberg truncate transform operations */
public class IcebergTruncate {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Truncate integer values for Iceberg partitioning.
   *
   * For integer types, Iceberg truncation is: value - (value % width)
   * where width is the truncation parameter.
   *
   * @param input Integer column to truncate
   * @param width Truncation width, MUST be positive
   * @return Truncated integer column
   */
  public static ColumnVector truncate(ColumnView input, int width) {
    if (width <= 0) {
      throw new IllegalArgumentException("Truncation width must be positive");
    }
    return new ColumnVector(truncate(input.getNativeView(), width));
  }

  private static native long truncate(long nativeColumnView, int width);
}
