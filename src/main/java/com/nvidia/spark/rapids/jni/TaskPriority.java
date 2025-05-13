/*
 *
 *  Copyright (c) 2025, NVIDIA CORPORATION.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.NativeDepsLoader;

/**
 * Get the priority for any task. If the priority for one task is larger than the priority for another task,
 * then it means that the task first task (larger number) should get access to resources before the task with
 * the lower priority value.
 * This is currently tied into the SparkResourceAdaptor for task life cycle
 */
public class TaskPriority {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Get the priority for a given task
   * @param taskAttemptId the id of the task to get the priority for.
   */
  public static native long getTaskPriority(long taskAttemptId);

  static native void taskDone(long taskAttemptId);
}
