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

import ai.rapids.cudf.HostUDFWrapper;
import ai.rapids.cudf.NativeDepsLoader;

/**
 * CentralMoment groupby aggregation and its pairing aggregation to merge multiple grouped
 * CentralMoment results.
 * <p>
 * This CentralMoment aggregation is a specialized version of Apache Spark's CentralMomentAgg class.
 * In particular, it computes and outputs the following values:
 *  - Non-null count (n),
 *  - Grouped mean value (avg),
 *  - Sum of squares of differences from the mean value of the input grouped numbers (m2).
 */
public class CentralMomentHostUDF extends HostUDFWrapper {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Aggregation types.
   * TODO: Extract this as a common enum to use for all HOST_UDF aggregations, including
   *       HyperLogLogPlusPlus. This is a follow up work.
   */
  public enum AggregationType {
    // TODO: Reduction(0),
    // TODO: ReductionMerge(1),
    GroupBy(2),
    GroupByMerge(3);

    final int nativeId;
    AggregationType(int nativeId) {
      this.nativeId = nativeId;
    }
  }

  public CentralMomentHostUDF(AggregationType type) {
    this.type = type;
  }

  @Override
  public long createUDFInstance() {
    return createNativeUDFInstance(type.nativeId);
  }

  @Override
  public int computeHashCode() {
    return Objects.hash(this.getClass().getName(), type);
  }

  @Override
  public boolean isEqual(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    CentralMomentHostUDF other = (CentralMomentHostUDF) o;
    return type == other.type;
  }

  private final AggregationType type;

  private static native long createNativeUDFInstance(int type);
}
