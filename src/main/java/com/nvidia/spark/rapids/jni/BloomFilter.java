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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfAccessor;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.NativeDepsLoader;

public class BloomFilter {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Create a bloom filter with the specified number of hashes and bloom filter bits.
   * @param numHashes The number of hashes to use when inserting values into the bloom filter or
   * when probing.
   * @param bloomFilterBits Size of the bloom filter in bits.
   * @return a Scalar object which encapsulates the bloom filter.
   */
  public static Scalar create(int numHashes, long bloomFilterBits){
    if(numHashes <= 0){
      throw new IllegalArgumentException("Bloom filters must have a positive hash count");
    }
    if(bloomFilterBits <= 0){
      throw new IllegalArgumentException("Bloom filters must have a positive number of bits");
    }
    return CudfAccessor.scalarFromHandle(DType.LIST, creategpu(numHashes, bloomFilterBits));
  }

  /**
   * Insert a column of longs into a bloom filter.
   * @param bloomFilter The bloom filter to which values will be inserted.
   * @param cv The column containing the values to add.
   */
  public static void put(Scalar bloomFilter, ColumnVector cv){
    put(CudfAccessor.getScalarHandle(bloomFilter), cv.getNativeView());
  }

  /**
   * Merge one or more bloom filters into a new bloom filter.
   * @param bloomFilters A ColumnVector containing a bloom filter per row. 
   * @return A new bloom filter containing the merged inputs.
   */
  public static Scalar merge(ColumnVector bloomFilters){
    return CudfAccessor.scalarFromHandle(DType.LIST, merge(bloomFilters.getNativeView()));
  }

  /**
   * Probe a bloom filter with a column of longs. Returns a column of booleans. For 
   * each row in the output; a value of true indicates that the corresponding input value
   * -may- be in the set of values used to build the bloom filter; a value of false indicates
   * that the corresponding input value is conclusively not in the set of values used to build
   * the bloom filter. 
   * @param bloomFilter The bloom filter to be probed.
   * @param cv The column containing the values to check.
   * @return A boolean column indicating the results of the probe.
   */
  public static ColumnVector probe(Scalar bloomFilter, ColumnVector cv){
    return new ColumnVector(probe(CudfAccessor.getScalarHandle(bloomFilter), cv.getNativeView()));
  }

  /**
   * Probe a bloom filter with a column of longs. Returns a column of booleans. For 
   * each row in the output; a value of true indicates that the corresponding input value
   * -may- be in the set of values used to build the bloom filter; a value of false indicates
   * that the corresponding input value is conclusively not in the set of values used to build
   * the bloom filter. 
   * @param bloomFilter The bloom filter to be probed. This buffer is expected to be the 
   * fully packed Spark bloom filter, including header.
   * @param cv The column containing the values to check.
   * @return A boolean column indicating the results of the probe.
   */
  public static ColumnVector probe(BaseDeviceMemoryBuffer bloomFilter, ColumnVector cv){
    return new ColumnVector(probebuffer(bloomFilter.getAddress(), bloomFilter.getLength(), cv.getNativeView()));
  }
  
  private static native long creategpu(int numHashes, long bloomFilterBits) throws CudfException;
  private static native int put(long bloomFilter, long cv) throws CudfException;
  private static native long merge(long bloomFilters) throws CudfException;
  private static native long probe(long bloomFilter, long cv) throws CudfException;  
  private static native long probebuffer(long bloomFilter, long bloomFilterSize, long cv) throws CudfException;  
}
