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

/**
 * Represents a footer for a parquet file that can be parsed using native code.
 */
public class ParquetFooter implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long nativeHandle;

  private ParquetFooter(long handle) {
    nativeHandle = handle;
  }

  /**
   * Write the filtered footer back out in a format that is compatible with a parquet
   * footer file. This will include the MAGIC PAR1 at the beginning and end and also the
   * length of the footer just before the PAR1 at the end.
   */
  public HostMemoryBuffer serializeThriftFile() {
    return serializeThriftFile(nativeHandle);
  }

  /**
   * Get the number of rows in the footer after filtering.
   */
  public long getNumRows() {
    return getNumRows(nativeHandle);
  }

  /**
   * Get the number of top level columns in the footer after filtering.
   */
  public int getNumColumns() {
    return getNumColumns(nativeHandle);
  }

  @Override
  public void close() throws Exception {
    if (nativeHandle != 0) {
      close(nativeHandle);
      nativeHandle = 0;
    }
  }

  /**
   * Read a parquet thrift footer from a buffer and filter it like the java code would. The buffer
   * should only include the thrift footer itself. This includes filtering out row groups that do
   * not fall within the partition and pruning columns that are not needed.
   * @param buffer the buffer to parse the footer out from.
   * @param partOffset for a split the start of the split
   * @param partLength the length of the split
   * @param names the names of the nodes in the tree to keep, flattened in a depth first way. The
   *              root node should be skipped and the names of maps and lists needs to match what
   *              parquet writes in.
   * @param numChildren the number of children for each item in name.
   * @param parentNumChildren the number of children in the root nodes
   * @param ignoreCase should case be ignored when matching column names. If this is true then
   *                   names should be converted to lower case before being passed to this.
   * @return a reference to the parsed footer.
   */
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

  private static native long getNumRows(long nativeHandle);

  private static native int getNumColumns(long nativeHandle);

  private static native HostMemoryBuffer serializeCustom(long nativeHandle);

  private static native HostMemoryBuffer serializeThriftFile(long nativeHandle);
}
