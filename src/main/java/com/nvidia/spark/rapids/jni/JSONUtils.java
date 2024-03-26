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

public class JSONUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public enum PathInstructionType {
    SUBSCRIPT,
    WILDCARD,
    KEY,
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
    assert(input.getType().equals(DType.STRING)) : "column must be a String";
    return new ColumnVector(getJsonObject(input.getNativeView(), path_instructions));
  }

  private static native long getJsonObject(long input, PathInstructionJni[] path_instructions);
}
