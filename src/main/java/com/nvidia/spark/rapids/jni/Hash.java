/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

public class Hash {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Create a new vector containing spark's 32-bit murmur3 hash of each row in the table.
   * Spark's murmur3 hash uses a different tail processing algorithm.
   *
   * @param seed integer seed for the murmur3 hash function
   * @param columns array of columns to hash, must have identical number of rows.
   * @return the new ColumnVector of 32-bit values representing each row's hash value.
   */
  public static ColumnVector murmurHash32(int seed, ColumnView columns[]) {
    if (columns.length < 1) {
      throw new IllegalArgumentException("Murmur3 hashing requires at least 1 column of input");
    }
    long[] columnViews = new long[columns.length];
    long size = columns[0].getRowCount();

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must be the same size";
      assert !columns[i].getType().isDurationType() : "Unsupported column type Duration";
      columnViews[i] = columns[i].getNativeView(); 
    }
    return new ColumnVector(murmurHash32(seed, columnViews));
  }

  public static ColumnVector murmurHash32(ColumnView columns[]) {
    return murmurHash32(0, columns);
  }

  private static native long murmurHash32(int seed, long[] viewHandles) throws CudfException;
}
