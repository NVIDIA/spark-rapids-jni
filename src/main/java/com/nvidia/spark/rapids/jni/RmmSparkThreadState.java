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

/**
 * The state of a given thread according to RmmSpark. This is intended really for debugging and
 * testing only.
 */
public enum RmmSparkThreadState {
  UNKNOWN(-1), // thread is not associated with anything...
  TASK_RUNNING(0), // task thread running normally
  TASK_WAIT_ON_SHUFFLE(1), // task thread waiting on shuffle
  TASK_BUFN_WAIT_ON_SHUFFLE(2), // task thread waiting on shuffle, but marked as BUFN
  TASK_ALLOC(3), // task thread in the middle of doing an allocation
  TASK_ALLOC_FREE(4), // task thread in the middle of doing an allocation and a free happened
  TASK_BLOCKED(5), // task thread that is temporarily blocked
  TASK_BUFN_THROW(6), // task thread that should throw an exception to roll back before blocking
  TASK_BUFN_WAIT(7), // task thread that threw an exception to roll back and now should
  // block the next time alloc is called
  TASK_BUFN(8), // task thread that is blocked until higher priority tasks start to succeed
  TASK_SPLIT_THROW(9), // task thread that should throw an exception to split input and retry
  TASK_REMOVE_THROW(10), // task thread that is being removed and needs to throw an exception
  SHUFFLE_RUNNING(11), // shuffle thread that is running normally
  SHUFFLE_ALLOC(12), // shuffle thread that is in the middle of doing an alloc
  SHUFFLE_ALLOC_FREE(13), // shuffle thread that is doing an alloc and a free happened.
  SHUFFLE_BLOCKED(14), // shuffle thread that is temporarily blocked
  SHUFFLE_THROW(15), // shuffle thread that needs to throw an OOM
  SHUFFLE_REMOVE_THROW(16); // shuffle thread that is being removed and needs to throw an exception

  private final int nativeId;

  RmmSparkThreadState(int nativeId) {
    this.nativeId = nativeId;
  }

  static RmmSparkThreadState fromNativeId(int nativeId) {
    for (RmmSparkThreadState ts : RmmSparkThreadState.values()) {
      if (ts.nativeId == nativeId) {
        return ts;
      }
    }
    throw new IllegalArgumentException("Could not find an ID for " + nativeId);
  }
}
