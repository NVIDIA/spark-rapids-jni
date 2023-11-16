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

import java.util.HashMap;
import java.util.HashSet;

/**
 * This is used to allow us to map a native thread id to a java thread so we can look at the
 * state from a java perspective.
 */
class ThreadStateRegistry {
  private static final Logger LOG = LoggerFactory.getLogger(ThreadStateRegistry.class);

  private static final HashMap<Long, Thread> knownThreads = new HashMap<>();
  private static final HashSet<Long> deadThreads = new HashSet<>();

  public static void clearDeadThreads() {
    HashSet<Long> copy;
    synchronized(ThreadStateRegistry.class) {
      copy = new HashSet<>(deadThreads);
      deadThreads.clear();
    }
    for (long id : copy) {
      RmmSpark.removeThreadAssociation(id);
    }
  }

  public static void addThread(long nativeId, Thread t) {
    clearDeadThreads();
    synchronized (ThreadStateRegistry.class) {
      knownThreads.put(nativeId, t);
    }
  }

  // Typically called from JNI
  public static synchronized void removeThread(long threadId) {
    knownThreads.remove(threadId);
    deadThreads.remove(threadId);
  }

  // This is likely called from JNI
  public static synchronized boolean isThreadBlocked(long nativeId) {
    Thread t = knownThreads.get(nativeId);
    if (t == null || !t.isAlive()) {
      deadThreads.add(nativeId);
      LOG.warn("Thread " + nativeId + " was not cleaned up properly.");
      // Dead is as good as blocked. This is mostly for tests, not so much for
      // production
      return true;
    }
    Thread.State state = t.getState();
    switch (state) {
      case BLOCKED:
        // fall through
      case WAITING:
        // fall through
      case TIMED_WAITING:
        return true;
      case TERMINATED:
        // Technically there is a race with `!t.isAlive` check above
        deadThreads.add(nativeId);
        LOG.warn("Thread " + nativeId + " was not cleaned up properly.");
        // Dead is as good as blocked. This is mostly for tests, not so much for
        // production
        return true;
      default:
        return false;
    }
  }
}
