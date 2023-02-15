/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.RmmEventHandler;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

public class RmmSparkTest {
  @BeforeEach
  public void setup() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }

  @AfterEach
  public void teardown() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }

  @Test
  public void testBasicInitAndTeardown() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler());
  }

  @Test
  public void testInsertOOMs() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler());
    long threadId = RmmSpark.getCurrentThreadId();
    long taskid = 0; // This is arbitrary
    RmmSpark.associateThreadWithTask(threadId, taskid);
    try {
      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();

      // Force an exception
      RmmSpark.forceRetryOOM(threadId);
      assertThrows(RetryOOM.class, () -> Rmm.alloc(100).close());
      // Verify that injecting OOM does not cause the block to actually happen
      RmmSpark.blockThreadUntilReady();

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();

      // Force another exception
      RmmSpark.forceSplitAndRetryOOM(threadId);
      assertThrows(SplitAndRetryOOM.class, () -> Rmm.alloc(100).close());
      // Verify that injecting OOM does not cause the block to actually happen
      RmmSpark.blockThreadUntilReady();

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
    } finally {
      RmmSpark.removeThreadAssociation(threadId);
    }
  }

  @Test
  public void testInsertMultipleOOMs() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler());
    long threadId = RmmSpark.getCurrentThreadId();
    long taskid = 0; // This is arbitrary
    RmmSpark.associateThreadWithTask(threadId, taskid);
    try {
      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();

      // Force an exception
      int numRetryOOMs = 3;
      RmmSpark.forceRetryOOM(threadId, numRetryOOMs);
      for (int i = 0; i < numRetryOOMs; i++) {
        assertThrows(RetryOOM.class, () -> Rmm.alloc(100).close());
        // Verify that injecting OOM does not cause the block to actually happen
        RmmSpark.blockThreadUntilReady();
      }

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();

      // Force another exception
      int numSplitAndRetryOOMs = 5;
      RmmSpark.forceSplitAndRetryOOM(threadId, numSplitAndRetryOOMs);
      for (int i = 0; i < numSplitAndRetryOOMs; i++) {
        assertThrows(SplitAndRetryOOM.class, () -> Rmm.alloc(100).close());
        // Verify that injecting OOM does not cause the block to actually happen
        RmmSpark.blockThreadUntilReady();
      }

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
    } finally {
      RmmSpark.removeThreadAssociation(threadId);
    }
  }

  @Test
  public void testCudfException() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler());
    long threadId = RmmSpark.getCurrentThreadId();
    long taskid = 0; // This is arbitrary
    RmmSpark.associateThreadWithTask(threadId, taskid);
    try {
      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();

      // Force an exception
      int numCudfExceptions = 3;
      RmmSpark.forceCudfException(threadId, numCudfExceptions);
      for (int i = 0; i < numCudfExceptions; i++) {
        assertThrows(CudfException.class, () -> Rmm.alloc(100).close());
        // Verify that injecting OOM does not cause the block to actually happen
        RmmSpark.blockThreadUntilReady();
      }

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
    } finally {
      RmmSpark.removeThreadAssociation(threadId);
    }
  }

  private static class BaseRmmEventHandler implements RmmEventHandler {
    @Override
    public long[] getAllocThresholds() {
      return null;
    }

    @Override
    public long[] getDeallocThresholds() {
      return null;
    }

    @Override
    public void onAllocThreshold(long totalAllocSize) {
    }

    @Override
    public void onDeallocThreshold(long totalAllocSize) {
    }

    @Override
    public boolean onAllocFailure(long sizeRequested, int retryCount) {
      // This is just a test for now, no spilling...
      return false;
    }
  }
}
