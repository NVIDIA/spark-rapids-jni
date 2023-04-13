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
    // Do a little normalization before setting up logging...
    if ("stderr".equalsIgnoreCase(logLoc)) {
      logLoc = "stderr";
    } else if ("stdout".equalsIgnoreCase(logLoc)) {
      logLoc = "stdout";
    }
    handle = createNewAdaptor(wrapped.getHandle(), logLoc);
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
   * Associate a thread with a given task id.
   * @param threadId the thread ID to use (not java thread id)
   * @param taskId the task ID this thread is associated with.
   */
  public void associateThreadWithTask(long threadId, long taskId) {
    associateThreadWithTask(getHandle(), threadId, taskId);
  }

  /**
   * Associate a thread with shuffle.
   * @param threadId the thread ID to associate (not java thread id).
   */
  public void associateThreadWithShuffle(long threadId) {
    associateThreadWithShuffle(getHandle(), threadId);
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
   * Indicate that the given thread could block on shuffle.
   * @param threadId the id of the thread that could block (not java thread id).
   */
  public void threadCouldBlockOnShuffle(long threadId) {
    threadCouldBlockOnShuffle(getHandle(), threadId);
  }

  /**
   * Indicate that the given thread can no longer block on shuffle.
   * @param threadId the ID of the thread that o longer can block on shuffle (not java thread id).
   */
  public void threadDoneWithShuffle(long threadId) {
    threadDoneWithShuffle(getHandle(), threadId);
  }

  /**
   * Force the thread with the given ID to throw a RetryOOM on their next allocation attempt.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   * @param numOOMs the number of times the RetryOOM should be thrown
   */
  public void forceRetryOOM(long threadId, int numOOMs) {
    forceRetryOOM(getHandle(), threadId, numOOMs);
  }

  /**
   * Force the thread with the given ID to throw a SplitAndRetryOOM on their next allocation attempt.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   * @param numOOMs the number of times the SplitAndRetryOOM should be thrown
   */
  public void forceSplitAndRetryOOM(long threadId, int numOOMs) {
    forceSplitAndRetryOOM(getHandle(), threadId, numOOMs);
  }

  /**
   * Force the thread with the given ID to throw a SplitAndRetryOOM on their next allocation attempt.
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

  /**
   * Get the ID of the current thread that can be used with the other SparkResourceAdaptor APIs.
   * Don't use the java thread ID. They are not related.
   */
  public static native long getCurrentThreadId();

  private native static long createNewAdaptor(long wrappedHandle, String logLoc);
  private native static void releaseAdaptor(long handle);
  private static native void associateThreadWithTask(long handle, long threadId, long taskId);
  private static native void associateThreadWithShuffle(long handle, long threadId);
  private static native void removeThreadAssociation(long handle, long threadId);
  private static native void taskDone(long handle, long taskId);
  private static native void threadCouldBlockOnShuffle(long handle, long threadId);
  private static native void threadDoneWithShuffle(long handle, long threadId);
  private static native void forceRetryOOM(long handle, long threadId, int numOOMs);
  private static native void forceSplitAndRetryOOM(long handle, long threadId, int numOOMs);
  private static native void forceCudfException(long handle, long threadId, int numTimes);
  private static native void blockThreadUntilReady(long handle);
  private static native int getStateOf(long handle, long threadId);
  private static native int getAndResetRetryThrowInternal(long handle, long taskId);
  private static native int getAndResetSplitRetryThrowInternal(long handle, long taskId);
  private static native long getAndResetBlockTimeInternal(long handle, long taskId);

}
