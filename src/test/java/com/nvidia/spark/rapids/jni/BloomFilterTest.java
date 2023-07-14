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

import com.nvidia.spark.rapids.jni.BloomFilter;

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DeviceMemoryBuffer;

import static org.junit.jupiter.api.Assertions.assertThrows;
import org.junit.jupiter.api.Test;

public class BloomFilterTest {  
  @Test
  void testBuildAndProbe(){
    int numHashes = 3;
    long bloomFilterBits = 4 * 1024 * 1024;

    try (ColumnVector input = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000);
         BloomFilter bloomFilter = new BloomFilter(numHashes, bloomFilterBits)){
      
      bloomFilter.put(input);
      try(ColumnVector probe = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3);
          ColumnVector expected = ColumnVector.fromBooleans(true, true, true, true, true, true, true, false, false, false, false);
          ColumnVector result = bloomFilter.probe(probe)){
        AssertUtils.assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void testBuildFromBufferAndProbe(){
    long bloomFilterBits = 4 * 1024 * 1024;
    long bloomFilterBytes = BloomFilter.bloomFilterByteSize(bloomFilterBits);

    try (ColumnVector input = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000)){
      DeviceMemoryBuffer bloomFilterBuf = DeviceMemoryBuffer.allocate(bloomFilterBytes);
      Cuda.asyncMemset(bloomFilterBuf.getAddress(), (byte)0, bloomFilterBytes);
      BloomFilter bloomFilter = new BloomFilter(3, bloomFilterBits, bloomFilterBuf);
      bloomFilter.put(input);
      try(ColumnVector probe = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000, -10, 1, 2, 3);
          ColumnVector expected = ColumnVector.fromBooleans(true, true, true, true, true, true, true, false, false, false, false);
          ColumnVector result = bloomFilter.probe(probe)){
        AssertUtils.assertColumnsAreEqual(expected, result);
      }
    }
  }

    @Test
  void testBuildMergeProbe(){
    int numHashes = 3;
    long bloomFilterBits = 4 * 1024 * 1024;

    try (ColumnVector colA = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000);
         ColumnVector colB = ColumnVector.fromLongs(100, 200, 300, 400);
         ColumnVector colC = ColumnVector.fromLongs(-100, -200, -300, -400);
         BloomFilter bloomFilterA = new BloomFilter(numHashes, bloomFilterBits);
         BloomFilter bloomFilterB = new BloomFilter(numHashes, bloomFilterBits);
         BloomFilter bloomFilterC = new BloomFilter(numHashes, bloomFilterBits)){

      bloomFilterA.put(colA);
      bloomFilterB.put(colB);
      bloomFilterC.put(colC);

      try(ColumnVector probe = ColumnVector.fromLongs(-9, 200, 300, 6000, -2546, 99, 65535, 0, -100, -200, -300, -400);
          ColumnVector expected = ColumnVector.fromBooleans(true, true, true, false, false, true, false, false, true, true, true, true);
          BloomFilter merged = BloomFilter.merge(new BloomFilter[]{bloomFilterA, bloomFilterB, bloomFilterC});
          ColumnVector result = merged.probe(probe)){
          AssertUtils.assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void testBuildTrivialMergeProbe(){
    int numHashes = 3;
    long bloomFilterBits = 4 * 1024 * 1024;

    try (ColumnVector colA = ColumnVector.fromLongs(20, 80, 100, 99, 47, -9, 234000000);
         BloomFilter bloomFilterA = new BloomFilter(numHashes, bloomFilterBits)){

      bloomFilterA.put(colA);

      try(ColumnVector probe = ColumnVector.fromLongs(-9, 200, 300, 6000, -2546, 99, 65535, 0, -100, -200, -300, -400);
          ColumnVector expected = ColumnVector.fromBooleans(true, false, false, false, false, true, false, false, false, false, false, false);
          BloomFilter merged = BloomFilter.merge(new BloomFilter[]{bloomFilterA});
          ColumnVector result = merged.probe(probe)){
          AssertUtils.assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void testBuildExpectedFailures(){
    // bloom filter with no hashes
    assertThrows(IllegalArgumentException.class, () -> {
      try (BloomFilter bloomFilter = new BloomFilter(0, 64)){}
    });

    // bloom filter with no size
    assertThrows(IllegalArgumentException.class, () -> {
      try (BloomFilter bloomFilter = new BloomFilter(3, 0)){}
    });
    assertThrows(IllegalArgumentException.class, () -> {
      DeviceMemoryBuffer bloomFilterBuf = DeviceMemoryBuffer.allocate(0);
      try (BloomFilter bloomFilter = new BloomFilter(3, 0, bloomFilterBuf)){}
    });

    // bloom filter with an incorrectly sized pre-existing buffer
    assertThrows(IllegalArgumentException.class, () -> {
      DeviceMemoryBuffer bloomFilterBuf = DeviceMemoryBuffer.allocate(8);
      try (BloomFilter bloomFilter = new BloomFilter(3, 7, bloomFilterBuf)){}
    });

    // empty merge
    assertThrows(IllegalArgumentException.class, () -> {
      try (BloomFilter merged = BloomFilter.merge(new BloomFilter[]{})){}
    });

    // merge with mixed hash counts
    assertThrows(IllegalArgumentException.class, () -> {
      try (BloomFilter bloomFilterA = new BloomFilter(3, 1024);
           BloomFilter bloomFilterB = new BloomFilter(4, 1024);
           BloomFilter bloomFilterC = new BloomFilter(4, 1024);
           BloomFilter merged = BloomFilter.merge(new BloomFilter[]{bloomFilterA, bloomFilterB, bloomFilterC})){}
    });

    // merge with mixed hash bit sizes
    assertThrows(IllegalArgumentException.class, () -> {
      try (BloomFilter bloomFilterA = new BloomFilter(3, 1024);
           BloomFilter bloomFilterB = new BloomFilter(3, 1024);
           BloomFilter bloomFilterC = new BloomFilter(3, 2048);
           BloomFilter merged = BloomFilter.merge(new BloomFilter[]{bloomFilterA, bloomFilterB, bloomFilterC})){}
    });
  }
}
