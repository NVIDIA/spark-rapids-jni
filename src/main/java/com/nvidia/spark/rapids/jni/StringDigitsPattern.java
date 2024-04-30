/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

import ai.rapids.cudf.*;

public class StringDigitsPattern {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static ColumnVector stringDigitsPattern(ColumnVector input, Scalar pattern, int d, int start, int end) {
    assert(input.getType().equals(DType.STRING)) : "column must be a String";
    return new ColumnVector(stringDigitsPattern(input.getNativeView(), CudfAccessor.getScalarHandle(pattern), d, start, end));
  }

  private static native long stringDigitsPattern(long input, long pattern, int d, int start, int end);
}
