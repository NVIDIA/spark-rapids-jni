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

  /**
   * Compute the population standard deviation from the given non-null count and sum of squares
   * of differences from the mean.
   *
   * @param n The column representing the non-null count of grouped values.
   * @param m2 The column representing the sum of squares of differences from the mean.
   * @return A column vector containing the population standard deviation for each group.
   */
  public static ColumnVector stddevPop(ColumnView n, ColumnView m2) {
    return new ColumnVector(stddev_pop(n.getNativeView(), m2.getNativeView()));
  }

  /**
   * Compute the sample standard deviation from the given non-null count and sum of squares
   * of differences from the mean.
   *
   * @param n The column representing the non-null count of grouped values.
   * @param m2 The column representing the sum of squares of differences from the mean.
   * @param nullOnDivideByZero If true, returns null when the number of sample in the group is one;
   *        otherwise returns a NaN value.
   * @return A column vector containing the sample standard deviation for each group.
   */
  public static ColumnVector stddevSamp(ColumnView n, ColumnView m2, boolean nullOnDivideByZero) {
    return new ColumnVector(stddev_samp(n.getNativeView(), m2.getNativeView(), nullOnDivideByZero));
  }

  /**
   * Compute the population variance from the given non-null count and sum of squares
   * of differences from the mean.
   *
   * @param n The column representing the non-null count of grouped values.
   * @param m2 The column representing the sum of squares of differences from the mean.
   * @return A column vector containing the population variance for each group.
   */
  public static ColumnVector varPop(ColumnView n, ColumnView m2) {
    return new ColumnVector(var_pop(n.getNativeView(), m2.getNativeView()));
  }

  /**
   * Compute the sample variance from the given non-null count and sum of squares
   * of differences from the mean.
   *
   * @param n The column representing the non-null count of grouped values.
   * @param m2 The column representing the sum of squares of differences from the mean.
   * @param nullOnDivideByZero If true, returns null when the number of sample in the group is one;
   *        otherwise returns a NaN value.
   * @return A column vector containing the sample variance for each group.
   */
  public static ColumnVector varSamp(ColumnView n, ColumnView m2, boolean nullOnDivideByZero) {
    return new ColumnVector(var_samp(n.getNativeView(), m2.getNativeView(), nullOnDivideByZero));
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

  private static native long stddev_pop(long nHandle, long m2Handle);
  private static native long stddev_samp(long nHandle, long m2Handle, boolean nullOnDivideByZero);
  private static native long var_pop(long nHandle, long m2Handle);
  private static native long var_samp(long nHandle, long m2Handle, boolean nullOnDivideByZero);
}
