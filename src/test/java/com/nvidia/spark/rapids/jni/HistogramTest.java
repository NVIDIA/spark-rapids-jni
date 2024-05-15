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

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;

import org.junit.jupiter.api.Test;

public class HistogramTest {
  @Test
  void testZeroFrequency() {
    try (ColumnVector values = ColumnVector.fromInts(5, 10, 30);
         ColumnVector freqs = ColumnVector.fromLongs(1, 0, 1);
         ColumnVector histogram = Histogram.createHistogramIfValid(values, freqs, true);
         ColumnVector percentiles = Histogram.percentileFromHistogram(histogram, new double[]{1},
             false);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(5.0, null, 30.0)) {
      AssertUtils.assertColumnsAreEqual(percentiles, expected);
    }
  }

  @Test
  void testAllNulls() {
    try (ColumnVector values = ColumnVector.fromBoxedInts(null, null, null);
         ColumnVector freqs = ColumnVector.fromLongs(1, 2, 3);
         ColumnVector histogram = Histogram.createHistogramIfValid(values, freqs, true);
         ColumnVector percentiles = Histogram.percentileFromHistogram(histogram, new double[]{0.5},
             false);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(null, null, null)) {
      AssertUtils.assertColumnsAreEqual(percentiles, expected);
    }
  }
}
