/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

public class ParquetFooter implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long nativeHandle;

  private ParquetFooter(long handle) {
    nativeHandle = handle;
  }

  public HostMemoryBuffer serializeThriftFile() {
    return serializeThriftFile(nativeHandle);
  }

  @Override
  public void close() throws Exception {
    if (nativeHandle != 0) {
      close(nativeHandle);
      nativeHandle = 0;
    }
  }

  public static ParquetFooter readAndFilter(HostMemoryBuffer buffer,
      long partOffset, long partLength,
      String[] names,
      int[] numChildren,
      int parentNumChildren,
      boolean ignoreCase) {
    return new ParquetFooter(
        readAndFilter
            (buffer.getAddress(), buffer.getLength(),
            partOffset, partLength,
            names, numChildren,
            parentNumChildren,
            ignoreCase));
  }

  // Native APIS
  private static native long readAndFilter(long address, long length,
      long partOffset, long partLength,
      String[] names,
      int[] numChildren,
      int parentNumChildren,
      boolean ignoreCase) throws CudfException;

  private static native void close(long nativeHandle);

  private static native HostMemoryBuffer serializeCustom(long nativeHandle);

  private static native HostMemoryBuffer serializeThriftFile(long nativeHandle);
}
