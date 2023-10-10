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

import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.RmmDeviceMemoryResource;
import ai.rapids.cudf.RmmEventHandlerResourceAdaptor;
import ai.rapids.cudf.RmmWrappingDeviceMemoryResource;

public class SparkResourceAdaptor
    extends RmmWrappingDeviceMemoryResource<RmmEventHandlerResourceAdaptor<RmmDeviceMemoryResource>> {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private long handle = 0;
  private Thread watchDog;

  /**
   * Create a new tracking resource adaptor.
   * @param wrapped the memory resource to track allocations. This should not be reused.
   */
  public SparkResourceAdaptor(RmmEventHandlerResourceAdaptor<RmmDeviceMemoryResource> wrapped) {
    this(wrapped, null);
  }

  /**
   * Create a new tracking resource adaptor.
   * @param wrapped the memory resource to track allocations. This should not be reused.
   * @param logLoc the location that logs should go. "stderr" is treated as going to stderr
   *               "stdout" is treated as going to stdout. null will disable logging and
   *               anything else is treated as a file name.
   */
  public SparkResourceAdaptor(RmmEventHandlerResourceAdaptor<RmmDeviceMemoryResource> wrapped,
      String logLoc) {
    super(wrapped);
    watchDog = new Thread(() -> {
      try {
        while (handle > 0) {
          checkAndBreakDeadlocks();
          Thread.sleep(100);
        }
      } catch (InterruptedException e) {
        // Ignored we are going to exit.
      }
    }, "SparkResourceAdaptor WatchDog");
    // Do a little normalization before setting up logging...
    if ("stderr".equalsIgnoreCase(logLoc)) {
      logLoc = "stderr";
    } else if ("stdout".equalsIgnoreCase(logLoc)) {
      logLoc = "stdout";
    }
    handle = createNewAdaptor(wrapped.getHandle(), logLoc);
    watchDog.setDaemon(true);
    watchDog.start();
  }

  @Override
  public long getHandle() {
    return handle;
  }

  @Override
  public void close() {
    if (handle != 0) {
      releaseAdaptor(handle);
      handle = 0;
    }
    super.close();
  }


  public boolean isOpen() {
    return handle != 0;
  }

  /**
   * Start a dedicated task thread. There can be more than one thread for a task. It is also
   * possible for this thread to be in a thread pool. It is just that there is no way if this
   * thread blocks that it will transitively block any other active tasks.
   * @param threadId the thread ID to use (not java thread id)
   * @param taskId the task ID this thread is associated with.
   */
  public void startDedicatedTaskThread(long threadId, long taskId) {
    startDedicatedTaskThread(getHandle(), threadId, taskId);
  }

  public void startRetryBlock(long threadId) {
    startRetryBlock(getHandle(), threadId);
  }

  public void endRetryBlock(long threadId) {
    endRetryBlock(getHandle(), threadId);
  }

  public void checkAndBreakDeadlocks() {
    checkAndBreakDeadlocks(getHandle());
  }

  /**
   * A thread in a shared thread pool has picked up some work for a set of tasks.
   * @param isForShuffle true if this is for shuffle, else false. Shuffle allows
   *                     for multiple task ids to be active at once, and also has
   *                     the highest priority to run. Other pool threads will have
   *                     a priority based off of the tasks they are working for.
   * @param threadId the thread that will be doing the work.
   * @param taskIds the ids of the tasks that it will be working on.
   */
  public void poolThreadWorkingOnTasks(boolean isForShuffle, long threadId, long[] taskIds) {
    if (taskIds.length > 0) {
      poolThreadWorkingOnTasks(getHandle(), isForShuffle, threadId, taskIds);
    }
  }

  public void poolThreadFinishedForTasks(long threadId, long[] taskIds) {
    if (taskIds.length > 0) {
      poolThreadFinishedForTasks(getHandle(), threadId, taskIds);
    }
  }

  /**
   * Remove the given thread ID from any association.
   * @param threadId the ID of the thread that is no longer a part of a task or shuffle (not java thread id).
   */
  public void removeThreadAssociation(long threadId) {
    removeThreadAssociation(getHandle(), threadId);
  }

  /**
   * Indicate that a given task is done and if there are any threads still associated with it
   * then they should also be removed.
   * @param taskId the ID of the task that has completed.
   */
  public void taskDone(long taskId) {
    taskDone(getHandle(), taskId);
  }

  /**
   * A dedicated task thread is going to submit work to a pool.
   * @param threadId the ID of the thread that will submit the work.
   */
  public void submittingToPool(long threadId) {
    submittingToPool(getHandle(), threadId);
  }

  /**
   * A dedicated task thread is going to wait on work in a pool to complete.
   * @param threadId the ID of the thread that will submit the work.
   */
  public void waitingOnPool(long threadId) {
    waitingOnPool(getHandle(), threadId);
  }

  /**
   * A dedicated task thread is done waiting on a pool. This could be because of submitting
   * something to the pool or waiting on a result from the pool.
   * @param threadId the ID of the thread that is done.
   */
  public void doneWaitingOnPool(long threadId) {
    doneWaitingOnPool(getHandle(), threadId);
  }

  /**
   * Force the thread with the given ID to throw a GpuRetryOOM on their next allocation attempt.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   * @param numOOMs the number of times the GpuRetryOOM should be thrown
   */
  public void forceRetryOOM(long threadId, int numOOMs) {
    forceRetryOOM(getHandle(), threadId, numOOMs);
  }

  /**
   * Force the thread with the given ID to throw a GpuSplitAndRetryOOM on their next allocation attempt.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   * @param numOOMs the number of times the GpuSplitAndRetryOOM should be thrown
   */
  public void forceSplitAndRetryOOM(long threadId, int numOOMs) {
    forceSplitAndRetryOOM(getHandle(), threadId, numOOMs);
  }

  /**
   * Force the thread with the given ID to throw a GpuSplitAndRetryOOM on their next allocation attempt.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   * @param numTimes the number of times the CudfException should be thrown
   */
  public void forceCudfException(long threadId, int numTimes) {
    forceCudfException(getHandle(), threadId, numTimes);
  }

  /**
   * Block the current thread until the resource adaptor thinks it is ready to continue.
   */
  public void blockThreadUntilReady() {
    blockThreadUntilReady(getHandle());
  }

  public RmmSparkThreadState getStateOf(long threadId) {
    return RmmSparkThreadState.fromNativeId(getStateOf(getHandle(), threadId));
  }

  public int getAndResetNumRetryThrow(long taskId) {
    return getAndResetRetryThrowInternal(getHandle(), taskId);
  }

  public int getAndResetNumSplitRetryThrow(long taskId) {
    return getAndResetSplitRetryThrowInternal(getHandle(), taskId);
  }

  public long getAndResetBlockTime(long taskId) {
    return getAndResetBlockTimeInternal(getHandle(), taskId);
  }

  public long getAndResetComputeTimeLostToRetry(long taskId) {
    return getAndResetComputeTimeLostToRetry(getHandle(), taskId);
  }


  /**
   * Called before doing an allocation on the CPU. This could throw an injected exception to help
   * with testing.
   * @param amount the amount of memory being requested
   * @param blocking is this for a blocking allocate or a non-blocking one.
   */
  public boolean preCpuAlloc(long amount, boolean blocking) {
    return preCpuAlloc(getHandle(), amount, blocking);
  }

  /**
   * The allocation that was going to be done succeeded.
   * @param ptr a pointer to the memory that was allocated.
   * @param amount the amount of memory that was allocated.
   * @param blocking is this for a blocking allocate or a non-blocking one.
   * @param wasRecursive the result of calling preCpuAlloc.
   */
  public void postCpuAllocSuccess(long ptr, long amount, boolean blocking, boolean wasRecursive) {
    postCpuAllocSuccess(getHandle(), ptr, amount, blocking, wasRecursive);
  }

  /**
   * The allocation failed, and spilling didn't save it.
   * @param wasOom wat the failure caused by an OOM or something else.
   * @param blocking is this for a blocking allocate or a non-blocking one.
   * @param wasRecursive the result of calling preCpuAlloc
   * @return true if the allocation should be retried else false if the state machine
   * thinks that a retry would not help.
   */
  public boolean postCpuAllocFailed(boolean wasOom, boolean blocking, boolean wasRecursive) {
    return postCpuAllocFailed(getHandle(), wasOom, blocking, wasRecursive);
  }

  /**
   * Some CPU memory was freed.
   * @param ptr a pointer to the memory being deallocated.
   * @param amount the amount that was made available.
   */
  public void cpuDeallocate(long ptr, long amount) {
    cpuDeallocate(getHandle(), ptr, amount);
  }

  /**
   * Get the ID of the current thread that can be used with the other SparkResourceAdaptor APIs.
   * Don't use the java thread ID. They are not related.
   */
  public static native long getCurrentThreadId();

  private native static long createNewAdaptor(long wrappedHandle, String logLoc);
  private native static void releaseAdaptor(long handle);
  private static native void startDedicatedTaskThread(long handle, long threadId, long taskId);
  private static native void poolThreadWorkingOnTasks(long handle, boolean isForShuffle, long threadId, long[] taskIds);
  private static native void poolThreadFinishedForTasks(long handle, long threadId, long[] taskIds);
  private static native void removeThreadAssociation(long handle, long threadId);
  private static native void taskDone(long handle, long taskId);
  private static native void submittingToPool(long handle, long threadId);
  private static native void waitingOnPool(long handle, long threadId);
  private static native void doneWaitingOnPool(long handle, long threadId);
  private static native void forceRetryOOM(long handle, long threadId, int numOOMs);
  private static native void forceSplitAndRetryOOM(long handle, long threadId, int numOOMs);
  private static native void forceCudfException(long handle, long threadId, int numTimes);
  private static native void blockThreadUntilReady(long handle);
  private static native int getStateOf(long handle, long threadId);
  private static native int getAndResetRetryThrowInternal(long handle, long taskId);
  private static native int getAndResetSplitRetryThrowInternal(long handle, long taskId);
  private static native long getAndResetBlockTimeInternal(long handle, long taskId);
  private static native long getAndResetComputeTimeLostToRetry(long handle, long taskId);
  private static native void startRetryBlock(long handle, long threadId);
  private static native void endRetryBlock(long handle, long threadId);
  private static native void checkAndBreakDeadlocks(long handle);
  private static native boolean preCpuAlloc(long handle, long amount, boolean blocking);
  private static native void postCpuAllocSuccess(long handle, long ptr, long amount,
                                                 boolean blocking, boolean wasRecursive);
  private static native boolean postCpuAllocFailed(long handle, boolean wasOom,
                                                   boolean blocking, boolean wasRecursive);
  private static native void cpuDeallocate(long handle, long ptr, long amount);
}
