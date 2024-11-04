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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.NativeDepsLoader;

/**
 * HyperLogLogPlusPlus
 */
public class HLLPP {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Compute the approximate count distinct value from sketch values.
   * <p>
   * The input sketch values must be given in the format `Struct<INT64, INT64, ...>`,
   * The num of children is: num_registers_per_sketch / 10 + 1, here 10 means a INT64 contains
   * max 10 registers. Register value is 6 bits. The input is columnar data, e.g.: sketch 0
   * is composed of by all the data of the children at index 0.
   *
   * @param input         The sketch column which constains Struct<INT64, INT64, ...> values.
   * @param precision     The num of bits for addressing.
   * @return A INT64 column with each value indicates the approximate count distinct value.
   */
  public static ColumnVector estimateDistinctValueFromSketches(ColumnView input, int precision) {
    return new ColumnVector(estimateDistinctValueFromSketches(input.getNativeView(), precision));
  }

  private static native long estimateDistinctValueFromSketches(long inputHandle, int precision);
}
