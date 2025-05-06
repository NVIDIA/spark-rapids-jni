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

namespace spark_rapids_jni {

long get_task_priority(long attempt_id) {
  return std::numeric_limits<long>::max() - (attempt_id + 1);
}

void task_done(long attempt_id) {
  // noop for now will change soon
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
