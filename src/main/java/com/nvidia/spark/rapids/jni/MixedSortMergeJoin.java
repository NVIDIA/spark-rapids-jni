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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.ast.CompiledExpression;

/**
 * Mixed sort-merge join operations combining equality keys with conditional expressions.
 * <p>
 * These joins perform a sort-merge join on the equality keys first, then filter the results
 * using the conditional expression evaluated on the conditional tables. This approach is
 * more memory-efficient and scalable than hash-based mixed joins for large datasets.
 * </p>
 */
public class MixedSortMergeJoin {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Helper to convert gather map data from JNI to GatherMap array
   */
  private static GatherMap[] gatherMapsFromJNI(long[] gatherMapData) {
    long bufferSize = gatherMapData[0];
    long leftAddr = gatherMapData[1];
    long leftHandle = gatherMapData[2];
    long rightAddr = gatherMapData[3];
    long rightHandle = gatherMapData[4];
    GatherMap[] maps = new GatherMap[2];
    maps[0] = new GatherMap(DeviceMemoryBuffer.fromRmm(leftAddr, bufferSize, leftHandle));
    maps[1] = new GatherMap(DeviceMemoryBuffer.fromRmm(rightAddr, bufferSize, rightHandle));
    return maps;
  }

  /**
   * Helper to convert single gather map data from JNI to GatherMap
   */
  private static GatherMap gatherMapFromJNI(long bufferAddr, long bufferSize, long bufferHandle) {
    return new GatherMap(DeviceMemoryBuffer.fromRmm(bufferAddr, bufferSize, bufferHandle));
  }

  /**
   * Performs a mixed sort-merge inner join.
   * <p>
   * Returns gather maps for all pairs of rows where the equality keys match
   * AND the conditional expression evaluates to true.
   * </p>
   *
   * @param leftEquality The left table for equality key comparison
   * @param rightEquality The right table for equality key comparison
   * @param leftConditional The left table for conditional expression evaluation
   * @param rightConditional The right table for conditional expression evaluation
   * @param condition The compiled conditional expression (AST) to evaluate
   * @param isLeftSorted Whether the left equality table is pre-sorted
   * @param isRightSorted Whether the right equality table is pre-sorted
   * @param compareNullsEqual Whether nulls in equality keys should be considered equal
   * @return An array of two GatherMaps: [left_map, right_map]
   */
  public static GatherMap[] innerJoin(Table leftEquality,
                                      Table rightEquality,
                                      Table leftConditional,
                                      Table rightConditional,
                                      CompiledExpression condition,
                                      boolean isLeftSorted,
                                      boolean isRightSorted,
                                      boolean compareNullsEqual) {
    long[] result = mixedSortMergeInnerJoin(
      leftEquality.getNativeView(),
      rightEquality.getNativeView(),
      leftConditional.getNativeView(),
      rightConditional.getNativeView(),
      condition.getNativeHandle(),
      isLeftSorted,
      isRightSorted,
      compareNullsEqual);
    
    return gatherMapsFromJNI(result);
  }

  /**
   * Performs a mixed sort-merge left join.
   * <p>
   * Returns row index vectors for all left rows. Rows that have no match (either no
   * equality match or conditional expression evaluates to false) will have an
   * out-of-bounds right index equal to {@code rightEquality.getRowCount()}.
   * When used with gather operations and {@code OutOfBoundsPolicy.NULLIFY},
   * these out-of-bounds indices will produce nulls.
   * </p>
   *
   * @param leftEquality The left table for equality key comparison
   * @param rightEquality The right table for equality key comparison
   * @param leftConditional The left table for conditional expression evaluation
   * @param rightConditional The right table for conditional expression evaluation
   * @param condition The compiled conditional expression (AST) to evaluate
   * @param isLeftSorted Whether the left equality table is pre-sorted
   * @param isRightSorted Whether the right equality table is pre-sorted
   * @param compareNullsEqual Whether nulls in equality keys should be considered equal
   * @return An array of two GatherMaps: [left_map, right_map]
   */
  public static GatherMap[] leftJoin(Table leftEquality,
                                     Table rightEquality,
                                     Table leftConditional,
                                     Table rightConditional,
                                     CompiledExpression condition,
                                     boolean isLeftSorted,
                                     boolean isRightSorted,
                                     boolean compareNullsEqual) {
    long[] result = mixedSortMergeLeftJoin(
      leftEquality.getNativeView(),
      rightEquality.getNativeView(),
      leftConditional.getNativeView(),
      rightConditional.getNativeView(),
      condition.getNativeHandle(),
      isLeftSorted,
      isRightSorted,
      compareNullsEqual);
    
    return gatherMapsFromJNI(result);
  }

  /**
   * Performs a mixed sort-merge left semi join.
   * <p>
   * Returns row indices of left table rows that have at least one match
   * (both equality and conditional) in the right table. Each left row appears
   * at most once in the result.
   * </p>
   *
   * @param leftEquality The left table for equality key comparison
   * @param rightEquality The right table for equality key comparison
   * @param leftConditional The left table for conditional expression evaluation
   * @param rightConditional The right table for conditional expression evaluation
   * @param condition The compiled conditional expression (AST) to evaluate
   * @param isLeftSorted Whether the left equality table is pre-sorted
   * @param isRightSorted Whether the right equality table is pre-sorted
   * @param compareNullsEqual Whether nulls in equality keys should be considered equal
   * @return A ColumnVector of left table indices
   */
  public static GatherMap leftSemiJoin(Table leftEquality,
                                       Table rightEquality,
                                       Table leftConditional,
                                       Table rightConditional,
                                       CompiledExpression condition,
                                       boolean isLeftSorted,
                                       boolean isRightSorted,
                                       boolean compareNullsEqual) {
    long[] result = mixedSortMergeLeftSemiJoin(
      leftEquality.getNativeView(),
      rightEquality.getNativeView(),
      leftConditional.getNativeView(),
      rightConditional.getNativeView(),
      condition.getNativeHandle(),
      isLeftSorted,
      isRightSorted,
      compareNullsEqual);
    
    return gatherMapFromJNI(result[1], result[0], result[2]);
  }

  /**
   * Performs a mixed sort-merge left anti join.
   * <p>
   * Returns row indices of left table rows that have NO matches
   * (neither equality nor conditional) in the right table. Each left row appears
   * at most once in the result.
   * </p>
   *
   * @param leftEquality The left table for equality key comparison
   * @param rightEquality The right table for equality key comparison
   * @param leftConditional The left table for conditional expression evaluation
   * @param rightConditional The right table for conditional expression evaluation
   * @param condition The compiled conditional expression (AST) to evaluate
   * @param isLeftSorted Whether the left equality table is pre-sorted
   * @param isRightSorted Whether the right equality table is pre-sorted
   * @param compareNullsEqual Whether nulls in equality keys should be considered equal
   * @return A ColumnVector of left table indices
   */
  public static GatherMap leftAntiJoin(Table leftEquality,
                                       Table rightEquality,
                                       Table leftConditional,
                                       Table rightConditional,
                                       CompiledExpression condition,
                                       boolean isLeftSorted,
                                       boolean isRightSorted,
                                       boolean compareNullsEqual) {
    long[] result = mixedSortMergeLeftAntiJoin(
      leftEquality.getNativeView(),
      rightEquality.getNativeView(),
      leftConditional.getNativeView(),
      rightConditional.getNativeView(),
      condition.getNativeHandle(),
      isLeftSorted,
      isRightSorted,
      compareNullsEqual);
    
    return gatherMapFromJNI(result[1], result[0], result[2]);
  }

  // Native method declarations
  private static native long[] mixedSortMergeInnerJoin(
    long leftEquality,
    long rightEquality,
    long leftConditional,
    long rightConditional,
    long condition,
    boolean isLeftSorted,
    boolean isRightSorted,
    boolean compareNullsEqual);

  private static native long[] mixedSortMergeLeftJoin(
    long leftEquality,
    long rightEquality,
    long leftConditional,
    long rightConditional,
    long condition,
    boolean isLeftSorted,
    boolean isRightSorted,
    boolean compareNullsEqual);

  private static native long[] mixedSortMergeLeftSemiJoin(
    long leftEquality,
    long rightEquality,
    long leftConditional,
    long rightConditional,
    long condition,
    boolean isLeftSorted,
    boolean isRightSorted,
    boolean compareNullsEqual);

  private static native long[] mixedSortMergeLeftAntiJoin(
    long leftEquality,
    long rightEquality,
    long leftConditional,
    long rightConditional,
    long condition,
    boolean isLeftSorted,
    boolean isRightSorted,
    boolean compareNullsEqual);
}

