/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"
#include "task_priority.hpp"
#include <unordered_map>
#include <mutex>

namespace spark_rapids_jni {

// Map to store attempt_id to priority mappings
static std::unordered_map<long, long> attempt_to_priority;
static std::mutex priority_mutex;
static long next_priority = std::numeric_limits<long>::max();

long get_task_priority(long attempt_id) {
  std::lock_guard<std::mutex> lock(priority_mutex);
  auto it = attempt_to_priority.find(attempt_id);
  if (it != attempt_to_priority.end()) {
    return it->second;
  }
  // First time seeing this attempt_id, assign next highest priority
  long priority = next_priority--;
  attempt_to_priority[attempt_id] = priority;
  return priority;
}

void task_done(long attempt_id) {
  std::lock_guard<std::mutex> lock(priority_mutex);
  attempt_to_priority.erase(attempt_id);
}

} // namespace spark_rapids_jni

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_TaskPriority_getTaskPriority(
    JNIEnv* env, jclass, jlong task_attempt_id)
{
  return spark_rapids_jni::get_task_priority(task_attempt_id);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_TaskPriority_taskDone(
  JNIEnv* env, jclass, jlong task_attempt_id)
{
  spark_rapids_jni::task_done(task_attempt_id);
}

}
