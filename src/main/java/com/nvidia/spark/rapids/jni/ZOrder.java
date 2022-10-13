/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Scalar;

public class ZOrder {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Interleave the bits going from the first column to the last column with the most significant
   * bit of each column. The output is a list of uint8 bytes (binary). This is designed to match
   * the InterleaveBits expression in DeltaLake. The input data should all be the same type and all
   * fixed with types. In general if you want good clustering/ordering then these should all
   * be positive integer values.
   * @param numRows the number of rows to output in a corner case where there are no input columns.
   *                This should never happen in practice, but the expression supports this so we
   *                should to.
   * @param inputColumns the data to process.
   * @return a binary column of the interleaved data.
   */
  public static ColumnVector interleaveBits(int numRows, ColumnVector ... inputColumns) {
    if (inputColumns.length == 0) {
      try (ColumnVector empty = ColumnVector.fromUnsignedBytes();
           Scalar emptyList = Scalar.listFromColumnView(empty)) {
        return ColumnVector.fromScalar(emptyList, numRows);
      }
    }
    long[] addrs = new long[inputColumns.length];
    for (int index = 0; index < inputColumns.length; index++) {
      ColumnVector cv = inputColumns[index];
      addrs[index] = cv.getNativeView();
    }

    return new ColumnVector(interleaveBits(addrs));
  }

  private static native long interleaveBits(long[] handles);
}
