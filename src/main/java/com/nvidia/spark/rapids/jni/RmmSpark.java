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

import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmDeviceMemoryResource;
import ai.rapids.cudf.RmmEventHandler;
import ai.rapids.cudf.RmmEventHandlerResourceAdaptor;
import ai.rapids.cudf.RmmException;
import ai.rapids.cudf.RmmTrackingResourceAdaptor;

/**
 * Initialize RMM in ways that are specific to Spark.
 */
public class RmmSpark {

  private static volatile SparkResourceAdaptor sra = null;

  /**
   * Set the event handler in a way that Spark wants it. For now this is the same as RMM, but in
   * the future it is likely to change.
   */
  public static void setEventHandler(RmmEventHandler handler) throws RmmException {
    // synchronize with RMM not RmmSpark to stay in sync with Rmm itself.
    synchronized (Rmm.class) {
      // RmmException constructor is not public, so we have to use a different exception
      if (!Rmm.isInitialized()) {
        throw new RuntimeException("RMM has not been initialized");
      }
      RmmDeviceMemoryResource deviceResource = Rmm.getCurrentDeviceResource();
      if (deviceResource instanceof RmmEventHandlerResourceAdaptor ||
          deviceResource instanceof SparkResourceAdaptor) {
        throw new RuntimeException("Another event handler is already set");
      }
      RmmTrackingResourceAdaptor<RmmDeviceMemoryResource> tracker = Rmm.getTracker();
      if (tracker == null) {
        // This is just to be safe it should always be true if this is initialized.
        throw new RuntimeException("A tracker must be set for the event handler to work");
      }
      RmmEventHandlerResourceAdaptor<RmmDeviceMemoryResource> eventHandler =
          new RmmEventHandlerResourceAdaptor<>(deviceResource, tracker, handler, false);
      sra = new SparkResourceAdaptor(eventHandler);
      boolean success = false;
      try {
        Rmm.setCurrentDeviceResource(sra, deviceResource, false);
        success = true;
      } finally {
        if (!success) {
          sra.releaseWrapped();
          eventHandler.releaseWrapped();
        }
      }
    }
  }

  /**
   * Clears the active RMM event handler and anything else that is needed for spark to behave
   * properly.
   */
  public static void clearEventHandler() throws RmmException {
    // synchronize with RMM not RmmSpark to stay in sync with Rmm itself.
    synchronized (Rmm.class) {
      RmmDeviceMemoryResource deviceResource = Rmm.getCurrentDeviceResource();
      if (deviceResource != null && deviceResource instanceof SparkResourceAdaptor) {
        SparkResourceAdaptor sra = (SparkResourceAdaptor) deviceResource;
        RmmEventHandlerResourceAdaptor<RmmDeviceMemoryResource> event = sra.getWrapped();
        boolean success = false;
        try {
          Rmm.setCurrentDeviceResource(event.getWrapped(), sra, false);
          success = true;
        } finally {
          if (success) {
            RmmSpark.sra = null;
            sra.releaseWrapped();
            event.releaseWrapped();
          }
        }
      }
    }
  }

  public static long getCurrentThreadId() {
    return SparkResourceAdaptor.getCurrentThreadId();
  }

  /**
   * Associate a thread with a given task id.
   * @param threadId the thread ID to use
   * @param taskId the task ID this thread is associated with.
   */
  public static void associateThreadWithTask(long threadId, long taskId) {
    synchronized (Rmm.class) {
      if (sra != null) {
        sra.associateThreadWithTask(threadId, taskId);
      }
    }
  }

  /**
   * Associate a thread with shuffle.
   * @param threadId the thread ID to associate (not java thread id).
   */
  public static void associateThreadWithShuffle(long threadId) {
    synchronized (Rmm.class) {
      if (sra != null) {
        sra.associateThreadWithShuffle(threadId);
      }
    }
  }

  /**
   * Remove the given thread ID from any association.
   * @param threadId the ID of the thread that is no longer a part of a task or shuffle
   *                 (not java thread id).
   */
  public static void removeThreadAssociation(long threadId) {
    synchronized (Rmm.class) {
      if (sra != null) {
        sra.removeThreadAssociation(threadId);
      }
    }
  }

  /**
   * Indicate that a given task is done and if there are any threads still associated with it
   * then they should also be removed.
   * @param taskId the ID of the task that has completed.
   */
  public static void taskDone(long taskId) {
    synchronized (Rmm.class) {
      if (sra != null) {
        sra.taskDone(taskId);
      }
    }
  }

  /**
   * Indicate that the given thread could block on shuffle.
   * @param threadId the id of the thread that could block (not java thread id).
   */
  public static void threadCouldBlockOnShuffle(long threadId) {
    synchronized (Rmm.class) {
      if (sra != null) {
        sra.threadCouldBlockOnShuffle(threadId);
      }
    }
  }

  /**
   * Indicate that the current thread could block on shuffle.
   */
  public static void threadCouldBlockOnShuffle() {
    threadCouldBlockOnShuffle(getCurrentThreadId());
  }

  /**
   * Indicate that the given thread can no longer block on shuffle.
   * @param threadId the ID of the thread that o longer can block on shuffle (not java thread id).
   */
  public static void threadDoneWithShuffle(long threadId) {
    synchronized (Rmm.class) {
      if (sra != null) {
        sra.threadDoneWithShuffle(threadId);
      }
    }
  }

  /**
   * Indicate that the current thread can no longer block on shuffle.
   */
  public static void threadDoneWithShuffle() {
    threadDoneWithShuffle(getCurrentThreadId());
  }

  /**
   * Force the thread with the given ID to throw a RetryOOM on their next allocation attempt.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   */
  public static void forceRetryOOM(long threadId) {
    synchronized (Rmm.class) {
      if (sra != null) {
        sra.forceRetryOOM(threadId);
      } else {
        throw new IllegalStateException("RMM has not been configured for OOM injection");
      }
    }
  }

  /**
   * Force the thread with the given ID to throw a SplitAndRetryOOM on their next allocation attempt.
   * @param threadId the ID of the thread to throw the exception (not java thread id).
   */
  public static void forceSplitAndRetryOOM(long threadId) {
    synchronized (Rmm.class) {
      if (sra != null) {
        sra.forceSplitAndRetryOOM(threadId);
      } else {
        throw new IllegalStateException("RMM has not been configured for OOM injection");
      }
    }
  }
}
