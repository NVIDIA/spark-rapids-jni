/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
import ai.rapids.cudf.CudfAccessor;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Scalar;

public class GpuSubstringIndexUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static ColumnVector substringIndex(ColumnView cv, Scalar delimiter, int count) {
    return new ColumnVector(
        substringIndex(cv.getNativeView(), CudfAccessor.getScalarHandle(delimiter), count));
  }

  private static native long substringIndex(long columnView, long delimiter, int count)
      throws CudfException;
}
