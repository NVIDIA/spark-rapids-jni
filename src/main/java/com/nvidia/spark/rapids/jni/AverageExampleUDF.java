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

package com.nvidia.spark.rapids.jni;

import java.util.Objects;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.HostUDFWrapper;
import ai.rapids.cudf.NativeDepsLoader;

public class AverageExampleUDF extends HostUDFWrapper {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public AverageExampleUDF(AggregationType type) {
    this.type = type;
  }

  @Override
  public long createUDFInstance() {
    return createAverageHostUDF(type);
  }

  @Override
  public int computeHashCode() {
    return Objects.hash(this.getClass().getName(), type);
  }

  @Override
  public boolean isEqual(Object o) {
    if (this == o)
      return true;
    if (o == null || getClass() != o.getClass())
      return false;
    AverageExampleUDF other = (AverageExampleUDF) o;
    return type == other.type;
  }

  public enum AggregationType {

    // input: int column, output: struct(sum: int, count: int)
    GroupBy(2),

    // input: struct(sum: int, count: int), output: struct(sum: int, count: int)
    GroupByMerge(3),

    // input: int column, output: sum, count
    Reduction(0),

    // input: sum, count, output: sum, count
    ReductionMerge(1);

    final int nativeId;

    AggregationType(int nativeId) {
      this.nativeId = nativeId;
    }
  }

  private static long createAverageHostUDF(AggregationType type) {
    return createAverageHostUDF(type.nativeId);
  }

  // input struct(sum, count), output avg column
  public static ColumnVector computeAvg(ColumnView input) {
    return new ColumnVector(computeAvg(input.getNativeView()));
  }

  private static native long createAverageHostUDF(int type);

  private static native long computeAvg(long inputHandle);

  private AggregationType type;
}
