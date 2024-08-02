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

import ai.rapids.cudf.*;

import java.util.List;

public class JSONUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static final int MAX_PATH_DEPTH = getMaxJSONPathDepth();

  public enum PathInstructionType {
    WILDCARD,
    INDEX,
    NAMED
  }

  public static class PathInstructionJni {
    // type: byte, name: String, index: int
    private final byte type;
    private final String name;
    private final int index;

    public PathInstructionJni(PathInstructionType type, String name, long index) {
      this.type = (byte)type.ordinal();
      this.name = name;
      if (index > Integer.MAX_VALUE) {
        throw new IllegalArgumentException("index is too large " + index);
      }
      this.index = (int)index;
    }

    public PathInstructionJni(PathInstructionType type, String name, int index) {
      this.type = (byte)type.ordinal();
      this.name = name;
      this.index = index;
    }
  }

  /**
   * Extract a JSON path from a JSON column. The path is processed in a Spark compatible way.
   * @param input the string column containing JSON
   * @param pathInstructions the instructions for the path processing
   * @return the result of processing the path
   */
  public static ColumnVector getJsonObject(ColumnVector input, PathInstructionJni[] pathInstructions) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    int numTotalInstructions = pathInstructions.length;
    byte[] typeNums = new byte[numTotalInstructions];
    String[] names = new String[numTotalInstructions];
    int[] indexes = new int[numTotalInstructions];

    for (int i = 0; i < pathInstructions.length; i++) {
      PathInstructionJni current = pathInstructions[i];
      typeNums[i] = current.type;
      names[i] = current.name;
      indexes[i] = current.index;
    }
    return new ColumnVector(getJsonObject(input.getNativeView(), typeNums, names, indexes));
  }

  /**
   * Extract multiple JSON paths from a JSON column. The paths are processed in a Spark
   * compatible way.
   * @param input the string column containing JSON
   * @param paths the instructions for multiple paths
   * @return the result of processing each path in the order that they were passed in
   */
  public static ColumnVector[] getJsonObjectMultiplePaths(ColumnVector input,
                                                          List<List<PathInstructionJni>> paths) {
    return getJsonObjectMultiplePaths(input, paths, -1, -1);
  }

  /**
   * Extract multiple JSON paths from a JSON column. The paths are processed in a Spark
   * compatible way.
   * @param input the string column containing JSON
   * @param paths the instructions for multiple paths
   * @param memoryBudgetBytes a budget that is used to limit the amount of memory
   *                          that is used when processing the paths. This is a soft limit.
   *                          A value <= 0 disables this and all paths will be processed in parallel.
   * @param parallelOverride Set a maximum number of paths to be processed in parallel. The memory
   *                         budget can limit how many paths can be processed in parallel. This overrides
   *                         that automatically calculated value with a set value for benchmarking purposes.
   *                         A value <= 0 disables this.
   * @return the result of processing each path in the order that they were passed in
   */
  public static ColumnVector[] getJsonObjectMultiplePaths(ColumnVector input,
                                                          List<List<PathInstructionJni>> paths,
                                                          long memoryBudgetBytes,
                                                          int parallelOverride) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    int[] pathOffsets = new int[paths.size() + 1];
    int offset = 0;
    for (int i = 0; i < paths.size(); i++) {
      pathOffsets[i] = offset;
      offset += paths.get(i).size();
    }
    pathOffsets[paths.size()] = offset;
    int numTotalInstructions = offset;
    byte[] typeNums = new byte[numTotalInstructions];
    String[] names = new String[numTotalInstructions];
    int[] indexes = new int[numTotalInstructions];

    for (int i = 0; i < paths.size(); i++) {
      for (int j = 0; j < paths.get(i).size(); j++) {
        PathInstructionJni current = paths.get(i).get(j);
        typeNums[pathOffsets[i] + j] = current.type;
        names[pathOffsets[i] + j] = current.name;
        indexes[pathOffsets[i] + j] = current.index;
      }
    }
    long[] ptrs = getJsonObjectMultiplePaths(input.getNativeView(), typeNums,
        names, indexes, pathOffsets, memoryBudgetBytes, parallelOverride);
    ColumnVector[] ret = new ColumnVector[ptrs.length];
    for (int i = 0; i < ptrs.length; i++) {
      ret[i] = new ColumnVector(ptrs[i]);
    }
    return ret;
  }

  private static native int getMaxJSONPathDepth();

  private static native long getJsonObject(long input,
                                           byte[] typeNums,
                                           String[] names,
                                           int[] indexes);

  private static native long[] getJsonObjectMultiplePaths(long input,
                                                          byte[] typeNums,
                                                          String[] names,
                                                          int[] indexes,
                                                          int[] pathOffsets,
                                                          long memoryBudgetBytes,
                                                          int parallelOverride);
}
