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
import ai.rapids.cudf.NativeDepsLoader;

public class AggregationUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static ColumnVector createHistogramsIfValid(ColumnView values, ColumnView frequencies,
                                                     boolean outputAsLists) {
    return new ColumnVector(createHistogramsIfValid(values.getNativeView(),
        frequencies.getNativeView(), outputAsLists));
  }

  /**
   * Compute percentiles from the given histograms and percentage values.
   * <p>
   * The input histograms must be given in the form of List<Struct<ElementType, LongType>>.
   *
   * @param input         The lists of input histograms.
   * @param percentages   The input percentage values.
   * @param outputAsLists Specify whether the output percentiles will be wrapped into a list.
   * @return A lists column, each list stores the output percentile(s) computed for the
   * corresponding row in the input column.
   */
  public static ColumnVector percentileFromHistogram(ColumnView input, double[] percentages,
                                                     boolean outputAsLists) {
    return new ColumnVector(percentileFromHistogram(input.getNativeView(), percentages,
        outputAsLists));
  }


  private static native long createHistogramsIfValid(long valuesHandle, long frequenciesHandle,
                                                     boolean outputAsLists);

  private static native long percentileFromHistogram(long inputHandle, double[] percentages,
                                                     boolean outputAsLists);
}
