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

class thread_state {
public:
    bool retry_oom_injected = false;
    bool split_and_retry_oom_injected = false;
    long task_id = -1;
};

class rollback {
    using on_error_type = const std::function<void()>;
public:
    rollback(on_error_type & on_error): on_error(on_error) {}

    ~rollback() {
      if (std::uncaught_exceptions() > 0) {
        on_error();
      }
    }
private:
    on_error_type on_error;
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
    auto was_threads_inserted = threads.insert({thread_id, {false, false, task_id}});
    if (was_threads_inserted.second == false) {
      throw std::invalid_argument("a thread can only be added if it is in the unknown state");
    }
    {
      rollback rb([this, thread_id]() {
        threads.erase(thread_id);
      });

      auto was_inserted = task_to_threads.insert({task_id, {thread_id}});
      if (was_inserted.second == false) {
        // task_to_threads already has a task_id for this, so insert the
        // thread_id
        was_inserted.first->second.insert(thread_id);
      }
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

  void force_retry_oom(long thread_id) {
    std::scoped_lock lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      threads_at->second.retry_oom_injected = true;
    } else {
      throw std::invalid_argument("the thread is not associated with any task.");
    }
  }

  void force_split_and_retry_oom(long thread_id) {
    std::scoped_lock lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      threads_at->second.split_and_retry_oom_injected = true;
    } else {
      throw std::invalid_argument("the thread is not associated with any task.");
    }
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

  void *do_allocate(std::size_t num_bytes, rmm::cuda_stream_view stream) override {
    auto tid = static_cast<long>(pthread_self());
    {
      std::scoped_lock lock(state_mutex);
      // pre allocate checks
      auto thread = threads.find(tid);
      if (thread != threads.end()) {
        if (thread->second.retry_oom_injected) {
          thread->second.retry_oom_injected = false;
          JNIEnv *env = cudf::jni::get_jni_env(jvm);
          // TODO cache what is needed for this...
          jclass ex_class = env->FindClass("com/nvidia/spark/rapids/jni/RetryOOM");
          if (ex_class != nullptr) {
            env->ThrowNew(ex_class, "OOM injected");
          }
          throw cudf::jni::jni_exception("injected RetryOOM");
        }

        if (thread->second.split_and_retry_oom_injected) {
          thread->second.split_and_retry_oom_injected = false;
          JNIEnv *env = cudf::jni::get_jni_env(jvm);
          // TODO cache what is needed for this...
          jclass ex_class = env->FindClass("com/nvidia/spark/rapids/jni/SplitAndRetryOOM");
          if (ex_class != nullptr) {
            env->ThrowNew(ex_class, "OOM injected");
          }
          throw cudf::jni::jni_exception("injected SplitAndRetryOOM");
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
        JNIEnv *env, jclass, jlong ptr, jlong thread_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->force_retry_oom(thread_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_forceSplitAndRetryOOM(
        JNIEnv *env, jclass, jlong ptr, jlong thread_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->force_split_and_retry_oom(thread_id);
  }
  CATCH_STD(env, )
}

}
