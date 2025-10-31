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

import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/**
 * Equality-only sort-merge join operations.
 * <p>
 * These joins perform a sort-merge join on the equality keys without any conditional filtering.
 * This is more efficient than conditional joins when only equality conditions are needed.
 * These methods complement cuDF's sort_merge_inner_join by adding left outer, left semi,
 * and left anti join types.
 * </p>
 */
public class SortMergeJoin {

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
   * Performs a sort-merge inner join.
   * <p>
   * Returns gather maps for rows that have matches in both tables.
   * Only matching rows are included in the result.
   * </p>
   *
   * @param leftKeys The left table for equality key comparison
   * @param rightKeys The right table for equality key comparison
   * @param isLeftSorted Whether the left table is pre-sorted
   * @param isRightSorted Whether the right table is pre-sorted
   * @param compareNullsEqual Whether nulls in equality keys should be considered equal
   * @return An array of two GatherMaps: [left_map, right_map]
   */
  public static GatherMap[] innerJoin(Table leftKeys,
                                      Table rightKeys,
                                      boolean isLeftSorted,
                                      boolean isRightSorted,
                                      boolean compareNullsEqual) {
    long[] result = sortMergeInnerJoin(
      leftKeys.getNativeView(),
      rightKeys.getNativeView(),
      isLeftSorted,
      isRightSorted,
      compareNullsEqual);
    
    return gatherMapsFromJNI(result);
  }

  /**
   * Performs a sort-merge left outer join.
   * <p>
   * Returns gather maps for all left rows. Rows with no match in the right table
   * will have an out-of-bounds right index equal to {@code rightKeys.getRowCount()}.
   * When used with gather operations and {@code OutOfBoundsPolicy.NULLIFY},
   * these out-of-bounds indices will produce nulls.
   * </p>
   *
   * @param leftKeys The left table for equality key comparison
   * @param rightKeys The right table for equality key comparison
   * @param isLeftSorted Whether the left table is pre-sorted
   * @param isRightSorted Whether the right table is pre-sorted
   * @param compareNullsEqual Whether nulls in equality keys should be considered equal
   * @return An array of two GatherMaps: [left_map, right_map]
   */
  public static GatherMap[] leftJoin(Table leftKeys,
                                     Table rightKeys,
                                     boolean isLeftSorted,
                                     boolean isRightSorted,
                                     boolean compareNullsEqual) {
    long[] result = sortMergeLeftJoin(
      leftKeys.getNativeView(),
      rightKeys.getNativeView(),
      isLeftSorted,
      isRightSorted,
      compareNullsEqual);
    
    return gatherMapsFromJNI(result);
  }

  /**
   * Performs a sort-merge left semi join.
   * <p>
   * Returns row indices of left table rows that have at least one match in the
   * right table. Each left row appears at most once in the result.
   * </p>
   *
   * @param leftKeys The left table for equality key comparison
   * @param rightKeys The right table for equality key comparison
   * @param isLeftSorted Whether the left table is pre-sorted
   * @param isRightSorted Whether the right table is pre-sorted
   * @param compareNullsEqual Whether nulls in equality keys should be considered equal
   * @return A GatherMap of left table indices
   */
  public static GatherMap leftSemiJoin(Table leftKeys,
                                       Table rightKeys,
                                       boolean isLeftSorted,
                                       boolean isRightSorted,
                                       boolean compareNullsEqual) {
    long[] result = sortMergeLeftSemiJoin(
      leftKeys.getNativeView(),
      rightKeys.getNativeView(),
      isLeftSorted,
      isRightSorted,
      compareNullsEqual);
    
    return gatherMapFromJNI(result[1], result[0], result[2]);
  }

  /**
   * Performs a sort-merge left anti join.
   * <p>
   * Returns row indices of left table rows that have NO matches in the right table.
   * Each left row appears at most once in the result.
   * </p>
   *
   * @param leftKeys The left table for equality key comparison
   * @param rightKeys The right table for equality key comparison
   * @param isLeftSorted Whether the left table is pre-sorted
   * @param isRightSorted Whether the right table is pre-sorted
   * @param compareNullsEqual Whether nulls in equality keys should be considered equal
   * @return A GatherMap of left table indices
   */
  public static GatherMap leftAntiJoin(Table leftKeys,
                                       Table rightKeys,
                                       boolean isLeftSorted,
                                       boolean isRightSorted,
                                       boolean compareNullsEqual) {
    long[] result = sortMergeLeftAntiJoin(
      leftKeys.getNativeView(),
      rightKeys.getNativeView(),
      isLeftSorted,
      isRightSorted,
      compareNullsEqual);
    
    return gatherMapFromJNI(result[1], result[0], result[2]);
  }

  // Native method declarations
  private static native long[] sortMergeInnerJoin(
    long leftKeys,
    long rightKeys,
    boolean isLeftSorted,
    boolean isRightSorted,
    boolean compareNullsEqual);

  private static native long[] sortMergeLeftJoin(
    long leftKeys,
    long rightKeys,
    boolean isLeftSorted,
    boolean isRightSorted,
    boolean compareNullsEqual);

  private static native long[] sortMergeLeftSemiJoin(
    long leftKeys,
    long rightKeys,
    boolean isLeftSorted,
    boolean isRightSorted,
    boolean compareNullsEqual);

  private static native long[] sortMergeLeftAntiJoin(
    long leftKeys,
    long rightKeys,
    boolean isLeftSorted,
    boolean isRightSorted,
    boolean compareNullsEqual);
}

