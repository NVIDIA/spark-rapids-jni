/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.NativeDepsLoader;

public class AggregationUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Compute percentiles from the given histograms and percentage values.
   * <p>
   * The input histograms must be given in the form of List<Struct<ElementType, LongType>>.
   *
   * @param input        The lists of input histograms
   * @param percentages  The input percentage values
   * @param outputAsList Specify whether the output percentiles will be wrapped into a list
   * @return A lists column, each list stores the percentile value(s) of the corresponding row in
   * the input column
   */
  public static ColumnVector percentileFromHistogram(ColumnView input, double[] percentages,
                                                     boolean outputAsList) {
    return new ColumnVector(percentileFromHistogram(input.getNativeView(), percentages,
        outputAsList));
  }


  private static native long percentileFromHistogram(long inputHandle, double[] percentage,
                                                     boolean outputAsList);
}
