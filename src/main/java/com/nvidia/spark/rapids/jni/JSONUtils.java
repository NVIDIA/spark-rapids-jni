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
    // type: Int, name: String, index: Long
    private final int type;
    private final String name;
    private final long index;

    public PathInstructionJni(PathInstructionType type, String name, long index) {
      this.type = type.ordinal();
      this.name = name;
      this.index = index;
    }
  }

  public static ColumnVector getJsonObject(ColumnVector input, PathInstructionJni[] path_instructions) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    return new ColumnVector(getJsonObject(input.getNativeView(), path_instructions));
  }

  public static ColumnVector[] getJsonObjectMultiplePaths(ColumnVector input,
                                                          List<List<PathInstructionJni>> paths) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    int[] pathOffsets = new int[paths.size() + 1];
    int offset = 0;
    for (int i = 0; i < paths.size(); i++) {
      pathOffsets[i] = offset;
      offset += paths.get(i).size();
    }
    pathOffsets[paths.size()] = offset;

    int numTotalInstructions = pathOffsets[paths.size()];
    PathInstructionJni[] pathsArray = new PathInstructionJni[numTotalInstructions];
    for (int i = 0; i < paths.size(); i++) {
      for (int j = 0; j < paths.get(i).size(); j++) {
        pathsArray[pathOffsets[i] + j] = paths.get(i).get(j);
      }
    }
    long[] ptrs = getJsonObjectMultiplePaths(input.getNativeView(), pathsArray, pathOffsets);
    ColumnVector[] ret = new ColumnVector[ptrs.length];
    for (int i = 0; i < ptrs.length; i++) {
      ret[i] = new ColumnVector(ptrs[i]);
    }
    return ret;
  }

  private static native int getMaxJSONPathDepth();

  private static native long getJsonObject(long input, PathInstructionJni[] path_instructions);

  private static native long[] getJsonObjectMultiplePaths(long input, PathInstructionJni[] paths,
                                                          int[] pathOffsets);
}
