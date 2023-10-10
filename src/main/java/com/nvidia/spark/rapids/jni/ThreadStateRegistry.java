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

import java.util.HashMap;
import java.util.HashSet;

/**
 * This is used to allow us to map a native thread id to a java thread so we can look at the
 * state from a java perspective.
 */
class ThreadStateRegistry {

  // TODO will need a background thread to act as a watchdog for this because we cannot solve all
  //  races with just polling :(

  private static HashMap<Long, Thread> knownThreads = new HashMap<>();

  public static synchronized void addThread(long nativeId, Thread t) {
//    System.err.println("ADD THREAD " + nativeId + " " + t);
    knownThreads.put(nativeId, t);
  }

  // Typically called from JNI
  public static synchronized void removeThread(long threadId) {
    Thread t = knownThreads.remove(threadId);
//    System.err.println("REMOVING THREAD " + threadId + " " + t);
  }

  // This is likely called from JNI
  public static synchronized boolean isThreadBlocked(long nativeId) {
    Thread t = knownThreads.get(nativeId);
    if (t == null) {
      throw new IllegalStateException("Thread " + nativeId + " could not be found.");
    } else if (!t.isAlive()) {
      throw new IllegalStateException("Thread " + nativeId + " is not longer alive.");
    }
    Thread.State state = t.getState();
//    System.err.println("CHECK THREAD STATE " + nativeId + " " + t + " " + state);
    switch (state) {
      case BLOCKED:
        // fall through
      case WAITING:
        // fall through
      case TIMED_WAITING:
        return true;
      default:
        return false;
    }
  }

  public static synchronized void purgeCleanedThreads() {
    HashSet<Long> keysToRemove = new HashSet<>();
    for (Long k : knownThreads.keySet()) {
      Thread t = knownThreads.get(k);
      if (t == null || !t.isAlive()) {
        keysToRemove.add(k);
      }
    }

    for (Long k : keysToRemove) {
      knownThreads.remove(k);
    }
  }
}
