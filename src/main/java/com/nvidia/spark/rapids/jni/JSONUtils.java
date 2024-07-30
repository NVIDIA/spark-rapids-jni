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
      this.type = (byte) type.ordinal();
      this.name = name;
      if (index > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(String.format("index %d is too large.", index));
      }
      this.index = (int) index;
    }

    public PathInstructionJni(PathInstructionType type, String name, int index) {
      this.type = (byte) type.ordinal();
      this.name = name;
      this.index = index;
    }
  }

  public static class GpuJSONPath implements AutoCloseable {
    private long nativeHandle;

    public GpuJSONPath(long nativeHandle) {
      if (nativeHandle == 0) {
        throw new IllegalStateException("Cannot create native GpuJSONPath object.");
      }
      this.nativeHandle = nativeHandle;
    }

    public long getNativeHandle() {
      return nativeHandle;
    }

    @Override
    public void close() {
      if (nativeHandle != 0) {
        closeGpuJSONPath(nativeHandle);
        nativeHandle = 0;
      }
    }
  }

  public static GpuJSONPath[] createGpuJSONPaths(List<List<PathInstructionJni>> paths) {
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
    long[] ptrs = createGpuJSONPaths(typeNums, names, indexes, pathOffsets);
    GpuJSONPath[] ret = new GpuJSONPath[ptrs.length];
    for (int i = 0; i < ptrs.length; i++) {
      ret[i] = new GpuJSONPath(ptrs[i]);
    }
    return ret;
  }

  public static ColumnVector getJsonObject(ColumnVector input, GpuJSONPath path) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    return new ColumnVector(getJsonObject(input.getNativeView(), path.getNativeHandle()));
  }

  public static ColumnVector[] getJsonObjectMultiplePaths(ColumnVector input, GpuJSONPath[] paths) {
    assert (input.getType().equals(DType.STRING)) : "Input must be of STRING type";
    long[] pathNativeHandles = new long[paths.length];
    for (int i = 0; i < paths.length; i++) {
      pathNativeHandles[i] = paths[i].getNativeHandle();
    }
    long[] outputHandles = getJsonObjectMultiplePaths(input.getNativeView(), pathNativeHandles);
    ColumnVector[] ret = new ColumnVector[outputHandles.length];
    for (int i = 0; i < outputHandles.length; i++) {
      ret[i] = new ColumnVector(outputHandles[i]);
    }
    return ret;
  }

  private static native int getMaxJSONPathDepth();

  private static native long[] createGpuJSONPaths(byte[] typeNums, String[] names, int[] indexes,
                                                  int[] pathOffsets);

  private static native void closeGpuJSONPath(long nativeHandle);

  private static native long getJsonObject(long input, long pathNativeHandle);

  private static native long[] getJsonObjectMultiplePaths(long input, long[] pathNativeHandles);
}
