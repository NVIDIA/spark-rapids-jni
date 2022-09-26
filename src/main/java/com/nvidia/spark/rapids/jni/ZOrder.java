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
