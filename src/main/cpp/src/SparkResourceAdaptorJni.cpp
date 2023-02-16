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
#include <exception>
#include <map>
#include <set>
#include <pthread.h>

#include <rmm/mr/device/device_memory_resource.hpp>

#include <cudf_jni_apis.hpp>

namespace {

constexpr char const *RMM_EXCEPTION_CLASS = "ai/rapids/cudf/RmmException";
constexpr char const *RETRY_OOM_CLASS = "com/nvidia/spark/rapids/jni/RetryOOM";
constexpr char const *SPLIT_AND_RETRY_OOM_CLASS = "com/nvidia/spark/rapids/jni/SplitAndRetryOOM";
constexpr char const *CUDF_EXCEPTION_CLASS = "ai/rapids/cudf/CudfException";

class thread_state {
public:
    int retry_oom_injected = 0;
    int split_and_retry_oom_injected = 0;
    int cudf_exception_injected = 0;
    long task_id = -1;
};

class spark_resource_adaptor final : public rmm::mr::device_memory_resource {
public:
  spark_resource_adaptor(JNIEnv *env, rmm::mr::device_memory_resource *mr)
      : resource{mr} {
    if (env->GetJavaVM(&jvm) < 0) {
      throw std::runtime_error("GetJavaVM failed");
    }
  }

  rmm::mr::device_memory_resource *get_wrapped_resource() { return resource; }

  bool supports_get_mem_info() const noexcept override { return resource->supports_get_mem_info(); }

  bool supports_streams() const noexcept override { return resource->supports_streams(); }

  void associate_thread_with_task(long thread_id, long task_id) {
    std::scoped_lock lock(state_mutex);
    auto was_threads_inserted = threads.insert({thread_id, {0, 0, 0, task_id}});
    if (was_threads_inserted.second == false) {
      throw std::invalid_argument("a thread can only be added if it is in the unknown state");
    }
    try {
      auto was_inserted = task_to_threads.insert({task_id, {thread_id}});
      if (was_inserted.second == false) {
        // task_to_threads already has a task_id for this, so insert the
        // thread_id
        was_inserted.first->second.insert(thread_id);
      }
    } catch (const std::exception &) {
      if (was_threads_inserted.second == true) {
        threads.erase(thread_id);
      }
      throw;
    }
  }

  void remove_thread_association(long thread_id) {
    std::scoped_lock lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      auto task_id = threads_at->second.task_id;
      if (task_id > 0) {
        auto task_at = task_to_threads.find(task_id);
        if (task_at != task_to_threads.end()) {
          task_at->second.erase(thread_id);
        }
      }
      threads.erase(threads_at);
    }
  }

  void force_retry_oom(long thread_id, int num_ooms) {
    std::scoped_lock lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      threads_at->second.retry_oom_injected = num_ooms;
    } else {
      throw std::invalid_argument("the thread is not associated with any task.");
    }
  }

  void force_split_and_retry_oom(long thread_id, int num_ooms) {
    std::scoped_lock lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      threads_at->second.split_and_retry_oom_injected = num_ooms;
    } else {
      throw std::invalid_argument("the thread is not associated with any task.");
    }
  }

  void force_cudf_exception(long thread_id, int num_times) {
    std::scoped_lock lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      threads_at->second.cudf_exception_injected = num_times;
    } else {
      throw std::invalid_argument("the thread is not associated with any task.");
    }
  }

  void block_thread_until_ready() {
    // TODO actually do this once we have state that can block
    //auto current_thread = static_cast<int64_t>(pthread_self());
    //std::scoped_lock lock(state_mutex)
  }

private:
  rmm::mr::device_memory_resource *const resource;
  // The state mutex must be held when modifying the state of threads or tasks
  // it must never be held when calling into the child resource or after returning
  // from an operation.
  std::mutex state_mutex;
  std::map<long, thread_state> threads;
  std::map<long, std::set<long>> task_to_threads;
  JavaVM *jvm;

  void throw_java_exception(const char* ex_class_name, const char* msg) {
    JNIEnv *env = cudf::jni::get_jni_env(jvm);
    cudf::jni::throw_java_exception(env, ex_class_name, msg);
  }

  void *do_allocate(std::size_t num_bytes, rmm::cuda_stream_view stream) override {
    auto tid = static_cast<long>(pthread_self());
    {
      std::scoped_lock lock(state_mutex);
      // pre allocate checks
      auto thread = threads.find(tid);
      if (thread != threads.end()) {
        if (thread->second.retry_oom_injected > 0) {
          thread->second.retry_oom_injected--;
          throw_java_exception(RETRY_OOM_CLASS, "injected RetryOOM");
        }

        if (thread->second.split_and_retry_oom_injected > 0) {
          thread->second.split_and_retry_oom_injected--;
          throw_java_exception(SPLIT_AND_RETRY_OOM_CLASS, "injected SplitAndRetryOOM");
        }

        if (thread->second.cudf_exception_injected > 0) {
          thread->second.cudf_exception_injected--;
          throw_java_exception(CUDF_EXCEPTION_CLASS, "injected CudfException");
        }
      }
    }
    return resource->allocate(num_bytes, stream);
  }

  void do_deallocate(void *p, std::size_t size, rmm::cuda_stream_view stream) override {
    resource->deallocate(p, size, stream);
  }

  std::pair<size_t, size_t> do_get_mem_info(rmm::cuda_stream_view stream) const override {
    return resource->get_mem_info(stream);
  }
};

} // empty namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getCurrentThreadId(
        JNIEnv *env, jclass) {
  try {
    cudf::jni::auto_set_device(env);
    return static_cast<jlong>(pthread_self());
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_createNewAdaptor(
        JNIEnv *env, jclass, jlong child) {
  JNI_NULL_CHECK(env, child, "child is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto wrapped = reinterpret_cast<rmm::mr::device_memory_resource *>(child);
    auto ret = new spark_resource_adaptor(env, wrapped);
    return cudf::jni::ptr_as_jlong(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_releaseAdaptor(
        JNIEnv *env, jclass, jlong ptr) {
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    delete mr;
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_associateThreadWithTask(
        JNIEnv *env, jclass, jlong ptr, jlong thread_id, jlong task_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->associate_thread_with_task(thread_id, task_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_removeThreadAssociation(
        JNIEnv *env, jclass, jlong ptr, jlong thread_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->remove_thread_association(thread_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_forceRetryOOM(
        JNIEnv *env, jclass, jlong ptr, jlong thread_id, jint num_ooms) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->force_retry_oom(thread_id, num_ooms);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_forceSplitAndRetryOOM(
        JNIEnv *env, jclass, jlong ptr, jlong thread_id, jint num_ooms) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->force_split_and_retry_oom(thread_id, num_ooms);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_forceCudfException(
        JNIEnv *env, jclass, jlong ptr, jlong thread_id, jint num_times) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->force_cudf_exception(thread_id, num_times);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_blockThreadUntilReady(
        JNIEnv *env, jclass, jlong ptr) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->block_thread_until_ready();
  }
  CATCH_STD(env, )
}

}
