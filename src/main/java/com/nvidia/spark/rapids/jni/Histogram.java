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

public class Histogram {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Create histograms from the given values and frequencies if the frequencies are valid.
   * <p>
   * The input is valid if they satisfy the following conditions:
   * - Values and frequencies columns must have the same size.
   * - Frequencies column must be of type INT64, must not have nulls, and must not contain
   * negative numbers.
   * <p>
   * If the input columns are valid, a histogram will be created from them. The histogram data is
   * stored in a structs column in the form of `STRUCT<value, frequency>`.
   * If `output_as_lists == true`, each struct element is wrapped into a list, producing a
   * lists-of-structs column.
   * <p>
   * Note that only value-frequency pairs with positive frequencies will be copied into the output.
   *
   * @param values        The input values
   * @param frequencies   The frequencies corresponding to the input values
   * @param outputAsLists Specify whether to wrap each pair of <value, frequency> in the output
   *                      histogram into a separate list
   * @return A histogram column with data copied from the input
   */
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
