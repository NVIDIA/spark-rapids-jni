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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.MemoryCleaner;
import ai.rapids.cudf.NativeDepsLoader;

public class BloomFilter implements AutoCloseable {
  private static final Logger log = LoggerFactory.getLogger(BloomFilter.class);
  private final int numHashes;
  private final long bloomFilterBits;
  private DeviceMemoryBuffer bloomFilter;

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Construct an empty BloomFilter which uses the specified number of hashes and bits.
   * @param numHashes The number of hashes to use when adding values and probing.
   * @param bloomFilterBits The size of the bloom filter in bits.
   */
  public BloomFilter(int numHashes, long bloomFilterBits){
    if(numHashes <= 0){
      throw new IllegalArgumentException("Bloom filters must have a positive hash count");
    }
    if(bloomFilterBits <= 0){
      throw new IllegalArgumentException("Bloom filters must have a positive number of bits");
    }

    this.numHashes = numHashes;    
    this.bloomFilterBits = bloomFilterBits;
    
    long bloomFilterBytes = bloomFilterByteSize(bloomFilterBits);
    bloomFilter = DeviceMemoryBuffer.allocate(bloomFilterBytes);
    Cuda.asyncMemset(bloomFilter.getAddress(), (byte)0, bloomFilterBytes);
  }  

  /**
   * Construct a BloomFilter from a pre-existing buffer, using the specified number of
   * hashes and bits.
   * @param numHashes The number of hashes to use when adding values and probing.
   * @param bloomFilterBits The size of the bloom filter in bits.
   * @param buffer The pre-existing buffer of.
   */
  public BloomFilter(int numHashes, long bloomFilterBits, DeviceMemoryBuffer buffer){
    if(numHashes <= 0){
      throw new IllegalArgumentException("Bloom filters must have a positive hash count");
    }
    if(bloomFilterBits <= 0){
      throw new IllegalArgumentException("Bloom filters must have a positive number of bits");
    }
    if(buffer.getLength() != bloomFilterByteSize(bloomFilterBits)){
      throw new IllegalArgumentException("Invalid pre-existing buffer passed. Size mismatch");
    }

    this.numHashes = numHashes;
    this.bloomFilterBits = bloomFilterBits;
    bloomFilter = buffer;
  }

  @Override
  public void close() {
    bloomFilter.close();
  }

  /**
   * Insert a column of longs into the bloom filter.
   * @param cv The column containing the values to add.
   */
  public void put(ColumnView cv){
    put(bloomFilter.getAddress(), bloomFilter.getLength(), bloomFilterBits, cv.getNativeView(), numHashes);
  }

  /**
   * Probe the bloom filter with a column of longs. Returns a column of booleans. For 
   * each row in the output; a value of true indicates that the corresponding input value
   * -may- be in the set of values used to build the bloom filter; a value of false indicates
   * that the corresponding input value is conclusively not in the set of values used to build
   * the bloom filter.
   * @param cv The column containing the values to check.
   * @return A boolean column indicating the results of the probe.
   */
  public ColumnVector probe(ColumnView cv){
     return new ColumnVector(probe(cv.getNativeView(), bloomFilter.getAddress(), bloomFilter.getLength(), bloomFilterBits, numHashes));
  }

  /**
   * Retrieve the underlying device memory buffer
   * @return The buffer containing the bloom filter.
   */
  public DeviceMemoryBuffer getBuffer(){
    return bloomFilter;
  }

  /**
   * Merge one or more bloom filters into a new bloom filter.
   * @param bloomFilters The bloom filters to be merged
   * @return A new bloom filter containing the merged inputs.
   */
  public static BloomFilter merge(BloomFilter bloomFilters[]){
    if(bloomFilters.length < 1){
      throw new IllegalArgumentException("Must pass at least 1 bloom filters to merge");
    }    
    // verify all the bloom filters match in size
    for (int i = 1; i < bloomFilters.length; i++) {
      if(bloomFilters[i].numHashes != bloomFilters[0].numHashes){
        throw new IllegalArgumentException("All bloom filters must use the same number of hashes");
      }
      if(bloomFilters[i].bloomFilterBits != bloomFilters[0].bloomFilterBits){
        throw new IllegalArgumentException("All bloom filters must have the same number of bits");
      }
    }

    long buffers[] = new long[bloomFilters.length];
    for (int i = 0; i < bloomFilters.length; i++) {
      buffers[i] = bloomFilters[i].getBuffer().getAddress();
    }

    int numHashes = bloomFilters[0].numHashes;
    long bloomFilterBits = bloomFilters[0].bloomFilterBits;
    long bloomFilterBytes = bloomFilterByteSize(bloomFilterBits);

    long[] merged = merge(buffers, bloomFilterBytes);
    return new BloomFilter(numHashes, bloomFilterBits, DeviceMemoryBuffer.fromRmm(merged[0], bloomFilterBytes, merged[1]));
  }

  /**
   * Return the size in bytes of the underlying device buffer.  It is important
   * to note that this value may not correspond exactly to the number of bits in the
   * bloom filter itself.  The total byte size is rounded up to the number of 32 bit
   * words needed to represent the specified bit size.
   * @return The size of the bloom filter in bytes.
   */
  static long bloomFilterByteSize(long numBits){
    return ((numBits + 31) / 32) * 4;
  }

  private static native void put(long bloomFilter, long bloomFilterBytes, long bloomFilterBits, long cv, int numHashes) throws CudfException;
  private static native long probe(long cv, long bloomFilter, long bloomFilterBytes, long bloomFilterBits, int numHashes) throws CudfException;
  private static native long[] merge(long bloomFilter[], long bloomFilterBytes) throws CudfException;
}
