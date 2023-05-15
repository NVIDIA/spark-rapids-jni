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
#include <chrono>
#include <exception>
#include <map>
#include <set>
#include <sstream>

#include <pthread.h>

#include <cudf_jni_apis.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

namespace {

constexpr char const *RETRY_OOM_CLASS = "com/nvidia/spark/rapids/jni/RetryOOM";
constexpr char const *SPLIT_AND_RETRY_OOM_CLASS = "com/nvidia/spark/rapids/jni/SplitAndRetryOOM";
constexpr char const *JAVA_OOM_CLASS = "java/lang/OutOfMemoryError";

// In the task states BUFN means Block Until Further Notice.
// Meaning the thread should be blocked until another task finishes.
// The reasoning is that spilling, and even pausing threads with large allocations
// was not enough to avoid an out of memory error, so we want to not start
// again until we know that progress has been made. We might add an API
// in the future to know when a retry section has passed, which would
// probably be a preferable time to restart all BUFN threads.
enum thread_state {
  UNKNOWN = -1, // unknown state, this is really here for logging and anything transitioning to
                // this state should actually be accomplished by deleting the thread from the state
  TASK_RUNNING = 0,              // task thread running normally
  TASK_WAIT_ON_SHUFFLE = 1,      // task thread waiting on shuffle
  TASK_BUFN_WAIT_ON_SHUFFLE = 2, // task thread waiting on shuffle, but marked as BUFN
  TASK_ALLOC = 3,                // task thread in the middle of doing an allocation
  TASK_ALLOC_FREE = 4,  // task thread in the middle of doing an allocation and a free happened
  TASK_BLOCKED = 5,     // task thread that is temporarily blocked
  TASK_BUFN_THROW = 6,  // task thread that should throw an exception to roll back before blocking
  TASK_BUFN_WAIT = 7,   // task thread that threw an exception to roll back and now should
                        // block the next time alloc or block_until_ready is called
  TASK_BUFN = 8,        // task thread that is blocked until higher priority tasks start to succeed
  TASK_SPLIT_THROW = 9, // task thread that should throw an exception to split input and retry
  TASK_REMOVE_THROW = 10,   // task thread that is being removed and needs to throw an exception
                            // to start the blocked thread running again.
  SHUFFLE_RUNNING = 11,     // shuffle thread that is running normally
  SHUFFLE_ALLOC = 12,       // shuffle thread that is in the middle of doing an alloc
  SHUFFLE_ALLOC_FREE = 13,  // shuffle thread that is doing an alloc and a free happened.
  SHUFFLE_BLOCKED = 14,     // shuffle thread that is temporarily blocked
  SHUFFLE_THROW = 15,       // shuffle thread that needs to throw an OOM
  SHUFFLE_REMOVE_THROW = 16 // shuffle thread that is being removed and needs to throw an exception
};

/**
 * Convert a state to a string representation for logging.
 */
const char *as_str(thread_state state) {
  switch (state) {
    case TASK_RUNNING: return "TASK_RUNNING";
    case TASK_WAIT_ON_SHUFFLE: return "TASK_WAIT_ON_SHUFFLE";
    case TASK_BUFN_WAIT_ON_SHUFFLE: return "TASK_BUFN_WAIT_ON_SHUFFLE";
    case TASK_ALLOC: return "TASK_ALLOC";
    case TASK_ALLOC_FREE: return "TASK_ALLOC_FREE";
    case TASK_BLOCKED: return "TASK_BLOCKED";
    case TASK_BUFN_THROW: return "TASK_BUFN_THROW";
    case TASK_BUFN_WAIT: return "TASK_BUFN_WAIT";
    case TASK_BUFN: return "TASK_BUFN";
    case TASK_SPLIT_THROW: return "TASK_SPLIT_THROW";
    case TASK_REMOVE_THROW: return "TASK_REMOVE_THROW";
    case SHUFFLE_RUNNING: return "SHUFFLE_RUNNING";
    case SHUFFLE_ALLOC: return "SHUFFLE_ALLOC";
    case SHUFFLE_ALLOC_FREE: return "SHUFFLE_ALLOC_FREE";
    case SHUFFLE_BLOCKED: return "SHUFFLE_BLOCKED";
    case SHUFFLE_THROW: return "SHUFFLE_THROW";
    case SHUFFLE_REMOVE_THROW: return "SHUFFLE_REMOVE_THROW";
    default: return "UNKNOWN";
  }
}

static std::shared_ptr<spdlog::logger> make_logger(std::ostream &stream) {
  return std::make_shared<spdlog::logger>("SPARK_RMM",
                                          std::make_shared<spdlog::sinks::ostream_sink_mt>(stream));
}

static std::shared_ptr<spdlog::logger> make_logger() {
  return std::make_shared<spdlog::logger>("SPARK_RMM",
                                          std::make_shared<spdlog::sinks::null_sink_mt>());
}

static auto make_logger(std::string const &filename) {
  return std::make_shared<spdlog::logger>(
      "SPARK_RMM",
      std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true /*truncate file*/));
}

/**
 * The priority of a thread is primarily based off of the task id. The thread id (PID on Linux) is
 * only used as a tie breaker if a task has more than a single thread associated with it.
 * In Spark task ids increase sequentially as they are assigned in an application. We want to give
 * priority to tasks that came first. This is to avoid situations where the first task stays as
 * the lowest priority task and is constantly retried while newer tasks move to the front of the
 * line. So a higher task_id should be a lower priority.
 *
 * We also want all shuffle threads to have the highest priority possible. So we assign them
 * a task id of -1. The problem is overflow on a long, so for the priority of a task the formula
 * will be MAX_LONG - (task_id + 1).
 */
class thread_priority {
public:
  thread_priority(long tsk_id, long t_id) : task_id(tsk_id), thread_id(t_id) {}

  long get_thread_id() const { return thread_id; }

  long get_task_id() const { return task_id; }

  bool operator<(const thread_priority &other) const {
    long task_priority = this->task_priority();
    long other_task_priority = other.task_priority();
    if (task_priority < other_task_priority) {
      return true;
    } else if (task_priority == other_task_priority) {
      return thread_id < other.thread_id;
    }
    return false;
  }

  bool operator>(const thread_priority &other) const {
    long task_priority = this->task_priority();
    long other_task_priority = other.task_priority();
    if (task_priority > other_task_priority) {
      return true;
    } else if (task_priority == other_task_priority) {
      return thread_id > other.thread_id;
    }
    return false;
  }

  void operator=(const thread_priority &other) {
    task_id = other.task_id;
    thread_id = other.thread_id;
  }

private:
  long task_id;
  long thread_id;

  long task_priority() const { return std::numeric_limits<long>::max() - (task_id + 1); }
};

/**
 * This is the full state of a thread. Some things like the thread_id and task_id
 * should not change after the state is set up. Everything else is up for change,
 * but some things should generally be changed with caution. Access to anything in
 * this should be accessed with a lock held.
 */
class full_thread_state {
public:
  full_thread_state(thread_state state, long thread_id) : state(state), thread_id(thread_id) {}
  full_thread_state(thread_state state, long thread_id, long task_id)
      : state(state), thread_id(thread_id), task_id(task_id) {}
  thread_state state;
  long thread_id;
  long task_id = -1;
  int retry_oom_injected = 0;
  int split_and_retry_oom_injected = 0;
  int cudf_exception_injected = 0;
  // watchdog limit on maximum number of retries to avoid unexpected live lock situations
  int num_times_retried = 0;
  // metric for being able to report how many times each type of exception was thrown,
  // and some timings
  int num_times_retry_throw = 0;
  int num_times_split_retry_throw = 0;
  long time_blocked_nanos = 0;

  std::chrono::time_point<std::chrono::steady_clock> block_start;

  std::unique_ptr<std::condition_variable> wake_condition =
      std::make_unique<std::condition_variable>();

  /**
   * Transition to a new state. Ideally this is what is called when doing a state transition instead
   * of setting the state directly.
   */
  void transition_to(thread_state new_state) {
    if (new_state == thread_state::UNKNOWN) {
      throw std::runtime_error(
          "Going to UNKNOWN state should delete the thread state, not call transition_to");
    }
    state = new_state;
  }

  void before_block() {
    block_start = std::chrono::steady_clock::now();
  }

  void after_block() {
    auto end = std::chrono::steady_clock::now();
    auto diff = end - block_start;
    time_blocked_nanos += std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();
  }

  /**
   * Get the priority of this thread.
   */
  thread_priority priority() { return thread_priority(task_id, thread_id); }
};

/**
 * A resource adaptor that is expected to come before the spill resource adaptor
 * when setting up RMM. This will handle tracking state for threads/tasks to
 * decide on when to pause a thread after a failed allocation and what other
 * mitigation we might want to do to avoid killing a task with an out of
 * memory error.
 */
class spark_resource_adaptor final : public rmm::mr::device_memory_resource {
public:
  spark_resource_adaptor(JNIEnv *env, rmm::mr::device_memory_resource *mr,
                         std::shared_ptr<spdlog::logger> &logger)
      : resource{mr}, logger{logger} {
    if (env->GetJavaVM(&jvm) < 0) {
      throw std::runtime_error("GetJavaVM failed");
    }
    logger->flush_on(spdlog::level::info);
    logger->set_pattern("%v");
    logger->info("time,op,current thread,op thread,op task,from state,to state,notes");
    logger->set_pattern("%H:%M:%S.%f,%v");
  }

  rmm::mr::device_memory_resource *get_wrapped_resource() { return resource; }

  bool supports_get_mem_info() const noexcept override { return resource->supports_get_mem_info(); }

  bool supports_streams() const noexcept override { return resource->supports_streams(); }

  /**
   * Update the internal state so that a specific thread is associated with a task.
   * This may be called multiple times for a given thread and if the thread is already
   * associated with the task, then most of the time this is a noop. The only exception
   * is if the thread is marked that it is shutting down, but has not completed yet.
   * This should never happen in practice with Spark because the only time we would
   * shut down a task thread on a thread that is different from itself is if there
   * was an error and the entire executor is shutting down. So there should be no
   * reuse.
   */
  void associate_thread_with_task(long thread_id, long task_id) {
    std::unique_lock<std::mutex> lock(state_mutex);
    if (shutting_down) {
      throw std::runtime_error("spark_resource_adaptor is shutting down");
    }
    auto was_threads_inserted = threads.emplace(
        thread_id, full_thread_state(thread_state::TASK_RUNNING, thread_id, task_id));
    if (was_threads_inserted.second == false) {
      if (was_threads_inserted.first->second.task_id != task_id) {
        throw std::invalid_argument("a thread can only be associated with a single task.");
      }

      if (was_threads_inserted.first->second.state == thread_state::TASK_REMOVE_THROW) {
        throw std::invalid_argument("the thread is in the process of shutting down.");
      }
    }

    try {
      auto was_inserted = task_to_threads.insert({task_id, {thread_id}});
      if (was_inserted.second == false) {
        // task_to_threads already has a task_id for this, so insert the thread_id
        was_inserted.first->second.insert(thread_id);
      }
    } catch (const std::exception &) {
      if (was_threads_inserted.second == true) {
        // roll back the thread insertion
        threads.erase(thread_id);
      }
      throw;
    }
    if (was_threads_inserted.second == true) {
      log_transition(thread_id, task_id, thread_state::UNKNOWN, thread_state::TASK_RUNNING);
    }
  }

  /**
   * Update the internal state so that a specific thread is associated with shuffle.
   * This may be called multiple times for a given thread and if the thread is already
   * associated with shuffle, the this is a noop in most cases. The only time
   * this is an error is if the thread is already marked as shutting down and has
   * not completed that transition yet.
   */
  void associate_thread_with_shuffle(long thread_id) {
    std::unique_lock<std::mutex> lock(state_mutex);
    if (shutting_down) {
      throw std::runtime_error("spark_resource_adaptor is shutting down");
    }

    auto was_inserted =
        threads.emplace(thread_id, full_thread_state(thread_state::SHUFFLE_RUNNING, thread_id));
    if (was_inserted.second == true) {
      log_transition(thread_id, -1, thread_state::UNKNOWN, thread_state::SHUFFLE_RUNNING);
    } else if (was_inserted.first->second.task_id != -1) {
      throw std::invalid_argument("the thread is associated with a non-shuffle task already");
    } else if (was_inserted.first->second.state == thread_state::SHUFFLE_REMOVE_THROW) {
      throw std::invalid_argument("the thread is in the process of shutting down.");
    }
  }

  /**
   * Update the internal state so that a specific thread is no longer associated with
   * a task or with shuffle. If that thread is currently blocked/waiting, then the
   * thread will not be immediately removed, but is instead marked that it needs to wake
   * up and throw an exception. At that point the thread's state will be completely
   * removed.
   */
  void remove_thread_association(long thread_id) {
    std::unique_lock<std::mutex> lock(state_mutex);
    if (remove_thread_association(thread_id, lock)) {
      wake_up_threads_after_task_finishes(lock);
    }
  }

  /**
   * Update the internal state so that all threads associated with a task are
   * cleared. Just like with remove_thread_association if one or more of these
   * threads are currently blocked/waiting then the state will not be totally
   * removed until the thread is woken.
   */
  void task_done(long task_id) {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto task_at = task_to_threads.find(task_id);
    if (task_at != task_to_threads.end()) {
      // we want to make a copy so there is no conflict here...
      std::set<long> threads_to_remove = task_at->second;
      bool run_checks = false;
      for (auto thread_id : threads_to_remove) {
        run_checks = remove_thread_association(thread_id, lock) || run_checks;
      }
      if (run_checks) {
        wake_up_threads_after_task_finishes(lock);
      }
    }
    task_to_threads.erase(task_id);
  }

  /**
   * This should be called before shutting down the adaptor. It will try
   * to shut down everything in an orderly way and wait for all of the
   * threads to be done.
   */
  void all_done() {
    {
      std::unique_lock<std::mutex> lock(state_mutex);
      // 1. Mark all threads that need to be removed as such
      // make a copy of the ids so we don't modify threads while walking it
      std::vector<long> threads_to_remove;
      for (auto thread = threads.begin(); thread != threads.end(); thread++) {
        threads_to_remove.push_back(thread->first);
      }

      for (auto thread_id : threads_to_remove) {
        remove_thread_association(thread_id, lock);
      }
      shutting_down = true;
    }

    // 2. release the semaphore so they can run
    {
      // 3. grab the semaphore again and see if wait until threads is empty.
      // There is a low risk of deadlock because new threads cannot be added
      // asking all of the existing threads to exit. The only problem would
      // be if the threads did not wake up or notify us properly.
      // So we have a timeout just in case.
      std::unique_lock<std::mutex> lock(state_mutex);
      // This should be fast, just wake up threads, change state and do
      // some notifications.
      std::chrono::milliseconds timeout{1000};
      task_has_woken_condition.wait_for(lock, timeout, [this] { return !threads.empty(); });
    }
    // No need to check for BUFN here, we are shutting down.
  }

  /**
   * Force a specific thread to throw one or more RetryOOM exceptions when an
   * alloc is called. This is intended only for testing.
   */
  void force_retry_oom(long thread_id, int num_ooms) {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      threads_at->second.retry_oom_injected = num_ooms;
    } else {
      throw std::invalid_argument("the thread is not associated with any task/shuffle");
    }
  }

  /**
   * Force a specific thread to throw one or more SplitAndRetryOOM exceptions
   * when an alloc is called. This is intended only for testing.
   */
  void force_split_and_retry_oom(long thread_id, int num_ooms) {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      threads_at->second.split_and_retry_oom_injected = num_ooms;
    } else {
      throw std::invalid_argument("the thread is not associated with any task/shuffle");
    }
  }

  /**
   * force a specific thread to throw one or more CudfExceptions when an
   * alloc is called. This is intended only for testing.
   */
  void force_cudf_exception(long thread_id, int num_times) {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      threads_at->second.cudf_exception_injected = num_times;
    } else {
      throw std::invalid_argument("the thread is not associated with any task/shuffle");
    }
  }

  /**
   * get the number of times a retry was thrown and reset the value to 0.
   */
  int get_n_reset_num_retry(long task_id) {
    std::unique_lock<std::mutex> lock(state_mutex);
    int ret = 0;
    auto task_at = task_to_threads.find(task_id);
    if (task_at != task_to_threads.end()) {
      for (auto thread_id : task_at->second) {
        auto threads_at = threads.find(thread_id);
        if (threads_at != threads.end()) {
          ret += threads_at->second.num_times_retry_throw;
          threads_at->second.num_times_retry_throw = 0;
        }
      }
    }
    return ret;
  }

  /**
   * get the number of times a split and retry was thrown and reset the value to 0.
   */
  int get_n_reset_num_split_retry(long task_id) {
    std::unique_lock<std::mutex> lock(state_mutex);
    int ret = 0;
    auto task_at = task_to_threads.find(task_id);
    if (task_at != task_to_threads.end()) {
      for (auto thread_id : task_at->second) {
        auto threads_at = threads.find(thread_id);
        if (threads_at != threads.end()) {
          ret += threads_at->second.num_times_split_retry_throw;
          threads_at->second.num_times_split_retry_throw = 0;
        }
      }
    }
    return ret;
  }

  /**
   * get the time in ns that the task was blocked for.
   */
  long get_n_reset_block_time(long task_id) {
    std::unique_lock<std::mutex> lock(state_mutex);
    long ret = 0;
    auto task_at = task_to_threads.find(task_id);
    if (task_at != task_to_threads.end()) {
      for (auto thread_id : task_at->second) {
        auto threads_at = threads.find(thread_id);
        if (threads_at != threads.end()) {
          ret += threads_at->second.time_blocked_nanos;
          threads_at->second.time_blocked_nanos = 0;
        }
      }
    }
    return ret;
  }

  /**
   * Update the internal state so that this thread is known that it is going to enter a
   * shuffle stage and could indirectly block on a shuffle thread (UCX).
   */
  void thread_could_block_on_shuffle(long thread_id) {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      switch (threads_at->second.state) {
        case TASK_RUNNING:
          transition(threads_at->second, thread_state::TASK_WAIT_ON_SHUFFLE);
          break;
        case TASK_BUFN_WAIT:
          transition(threads_at->second, thread_state::TASK_BUFN_WAIT_ON_SHUFFLE);
          break;
        case TASK_WAIT_ON_SHUFFLE:
        // fall through
        case TASK_BUFN_WAIT_ON_SHUFFLE:
          // noop already in an expected state...
          break;
        default: {
          std::stringstream ss;
          ss << "thread  " << thread_id << " is in an unexpected state "
             << as_str(threads_at->second.state) << " to start shuffle";
          throw std::invalid_argument(ss.str());
        }
      }
      check_and_update_for_bufn(lock);
    } else {
      throw std::invalid_argument("the thread is not associated with any task/shuffle");
    }
  }

  /**
   * Indicate that the thread no longer will block indirectly on a shuffle thread.
   */
  void thread_done_with_shuffle(long thread_id) {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      switch (threads_at->second.state) {
        case TASK_WAIT_ON_SHUFFLE:
          transition(threads_at->second, thread_state::TASK_RUNNING);
          break;
        case TASK_BUFN_WAIT_ON_SHUFFLE:
          transition(threads_at->second, thread_state::TASK_BUFN_WAIT);
          break;
        case TASK_RUNNING:
        // fall through
        case TASK_BUFN_WAIT:
          // noop already in an expected state...
          break;
        default: {
          std::stringstream ss;
          ss << "thread  " << thread_id << " is in an unexpected state "
             << as_str(threads_at->second.state) << " to end shuffle";
          throw std::invalid_argument(ss.str());
        }
      }
    } else {
      throw std::invalid_argument("the thread is not associated with any task/shuffle");
    }
  }

  /**
   * Called after a RetryOOM is thrown to wait until it is okay to start processing
   * data again. This is here mostly to prevent spillable code becoming unspillable
   * before an alloc is called.  If this is not called alloc will also call into the
   * same code and block if needed until the task is ready to keep going.
   */
  void block_thread_until_ready() {
    auto thread_id = static_cast<long>(pthread_self());
    std::unique_lock<std::mutex> lock(state_mutex);
    block_thread_until_ready(thread_id, lock);
  }

  /**
   * This is really here just for testing. It provides a way to look at the
   * current state of a thread.
   */
  int get_thread_state_as_int(long thread_id) {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      return static_cast<int>(threads_at->second.state);
    } else {
      return -1;
    }
  }

private:
  rmm::mr::device_memory_resource *const resource;
  std::shared_ptr<spdlog::logger> logger; ///< spdlog logger object

  // The state mutex must be held when modifying the state of threads or tasks
  // it must never be held when calling into the child resource or after returning
  // from an operation.
  std::mutex state_mutex;
  std::condition_variable task_has_woken_condition;
  std::map<long, full_thread_state> threads;
  std::map<long, std::set<long>> task_to_threads;
  bool shutting_down = false;
  JavaVM *jvm;

  /**
   * log a status change that does not involve a state transition.
   */
  void log_status(const char *op, long thread_id, long task_id, thread_state state,
                  const char *notes = nullptr) {
    auto this_id = static_cast<long>(pthread_self());
    logger->info("{},{},{},{},{},,{}", op, this_id, thread_id, task_id, as_str(state),
                 (notes == nullptr ? "" : notes));
  }

  /**
   * log that a state transition happened.
   */
  void log_transition(long thread_id, long task_id, thread_state from, thread_state to,
                      const char *notes = nullptr) {
    auto this_id = static_cast<long>(pthread_self());
    logger->info("TRANSITION,{},{},{},{},{},{}", this_id, thread_id, task_id, as_str(from),
                 as_str(to), (notes == nullptr ? "" : notes));
  }

  /**
   * Transition to a new state. Ideally this is what is called when doing a state transition instead
   * of setting the state directly. This will log the transition and do a little bit of
   * verification.
   */
  void transition(full_thread_state &state, thread_state new_state, const char *message = nullptr) {
    thread_state original = state.state;
    state.transition_to(new_state);
    log_transition(state.thread_id, state.task_id, original, new_state, message);
  }

  /**
   * throw a java exception using the cached jvm/env.
   */
  void throw_java_exception(const char *ex_class_name, const char *msg) {
    JNIEnv *env = cudf::jni::get_jni_env(jvm);
    cudf::jni::throw_java_exception(env, ex_class_name, msg);
  }

  /**
   * This is a watchdog to prevent us from live locking. It should be called before we throw an
   * RetryOOM or a SplitAndRetryOOM to know if we actually should throw something else.
   */
  void check_before_oom(full_thread_state &state, const std::unique_lock<std::mutex> &lock) {
    // The limit is an arbitrary number, large enough that we should not hit it in "normal"
    // operation, but also small enough that we can detect a livelock fairly quickly.
    // In testing it looks like it is a few ms if in a tight loop, not including spill
    // overhead
    if (state.num_times_retried + 1 > 500) {
      throw_java_exception(JAVA_OOM_CLASS, "GPU OutOfMemory: retry limit exceeded");
    }
    state.num_times_retried++;
  }

  void throw_retry_oom(const char *msg, full_thread_state &state,
                       const std::unique_lock<std::mutex> &lock) {
    state.num_times_retry_throw++;
    check_before_oom(state, lock);
    throw_java_exception(RETRY_OOM_CLASS, "GPU OutOfMemory");
  }

  void throw_split_n_retry_oom(const char *msg, full_thread_state &state,
                               const std::unique_lock<std::mutex> &lock) {
    state.num_times_split_retry_throw++;
    check_before_oom(state, lock);
    throw_java_exception(SPLIT_AND_RETRY_OOM_CLASS, "GPU OutOfMemory");
  }

  bool is_blocked(thread_state state) {
    switch (state) {
      case TASK_BLOCKED:
      // fall through
      case TASK_BUFN:
      // fall through
      case SHUFFLE_BLOCKED: return true;
      default: return false;
    }
  }

  /**
   * Internal implementation that will block a thread until it is ready to continue.
   */
  void block_thread_until_ready(long thread_id, std::unique_lock<std::mutex> &lock) {
    bool done = false;
    bool first_time = true;
    // Because this is called from alloc as well as from the public facing block_thread_until_ready
    // there are states that should only show up in relation to alloc failing. These include
    // TASK_BUFN_THROW and TASK_SPLIT_THROW. They should never happen unless this is being called
    // from within an alloc.
    while (!done) {
      auto thread = threads.find(thread_id);
      if (thread != threads.end()) {
        switch (thread->second.state) {
          case TASK_BLOCKED:
          // fall through
          case TASK_BUFN:
          // fall through
          case SHUFFLE_BLOCKED:
            log_status("WAITING", thread_id, thread->second.task_id, thread->second.state);
            thread->second.before_block();
            do {
              thread->second.wake_condition->wait(lock);
              thread = threads.find(thread_id);
            } while (thread != threads.end() && is_blocked(thread->second.state));
            thread->second.after_block();
            task_has_woken_condition.notify_all();
            break;
          case SHUFFLE_THROW:
            transition(thread->second, thread_state::SHUFFLE_RUNNING);
            throw_java_exception(JAVA_OOM_CLASS, "GPU OutOfMemory: could not allocate enough for shuffle");
            break;
          case TASK_BUFN_THROW:
            transition(thread->second, thread_state::TASK_BUFN_WAIT);
            throw_retry_oom("rollback and retry operation", thread->second, lock);
            break;
          case TASK_BUFN_WAIT:
            transition(thread->second, thread_state::TASK_BUFN);
            // Before we can wait it is possible that the throw didn't release anything
            // and the other threads didn't get unblocked by this, so we need to
            // check again to see if this was fixed or not.
            check_and_update_for_bufn(lock);
            log_status("WAITING", thread_id, thread->second.task_id, thread->second.state);
            thread->second.before_block();
            do {
              thread->second.wake_condition->wait(lock);
              thread = threads.find(thread_id);
            } while (thread != threads.end() && is_blocked(thread->second.state));
            thread->second.after_block();
            task_has_woken_condition.notify_all();
            break;
          case TASK_SPLIT_THROW:
            transition(thread->second, thread_state::TASK_RUNNING);
            throw_split_n_retry_oom("rollback, split input, and retry operation", thread->second,
                                    lock);
            break;
          case TASK_REMOVE_THROW:
          // fall through
          case SHUFFLE_REMOVE_THROW:
            log_transition(thread_id, thread->second.task_id, thread->second.state,
                           thread_state::UNKNOWN);
            threads.erase(thread);
            task_has_woken_condition.notify_all();
            throw std::runtime_error("thread removed while blocked");
          default:
            if (!first_time) {
              log_status("DONE WAITING", thread_id, thread->second.task_id, thread->second.state);
            }
            done = true;
        }
      } else {
        // the thread is not registered any more, or never was, but don't block...
        done = true;
      }
      first_time = false;
    }
  }

  /**
   * Wake up threads after a task finished. The task finishing successfully means
   * that progress was made. So we want to restart some tasks to see if they can
   * make progress. Right now the idea is to wake up all blocked threads first
   * and if there are no blocked threads, then we wake up all BUFN threads.
   * Hopefully the frees have already woken up all the blocked threads anyways.
   */
  void wake_up_threads_after_task_finishes(const std::unique_lock<std::mutex> &lock) {
    bool are_any_tasks_just_blocked = false;
    for (auto thread = threads.begin(); thread != threads.end(); thread++) {
      switch (thread->second.state) {
        case TASK_BLOCKED:
          transition(thread->second, thread_state::TASK_RUNNING);
          thread->second.wake_condition->notify_all();
          are_any_tasks_just_blocked = true;
          break;
        case SHUFFLE_BLOCKED:
          transition(thread->second, thread_state::SHUFFLE_RUNNING);
          thread->second.wake_condition->notify_all();
          break;
        default: break;
      }
    }

    if (!are_any_tasks_just_blocked) {
      // wake up all of the BUFN tasks.
      for (auto thread = threads.begin(); thread != threads.end(); thread++) {
        switch (thread->second.state) {
          case TASK_BUFN:
          // fall through
          case TASK_BUFN_THROW:
          // fall through
          case TASK_BUFN_WAIT:
            transition(thread->second, thread_state::TASK_RUNNING);
            thread->second.wake_condition->notify_all();
            break;
          default: break;
        }
      }
    }
  }

  /**
   * Internal implementation that removes a threads association with a task/shuffle.
   * returns true if the thread that ended was a normally running task thread.
   * This should be used to decide if wake_up_threads_after_task_finishes is called or not.
   */
  bool remove_thread_association(long thread_id, const std::unique_lock<std::mutex> &lock) {
    bool ret = false;
    auto threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      auto task_id = threads_at->second.task_id;
      if (task_id >= 0) {
        auto task_at = task_to_threads.find(task_id);
        if (task_at != task_to_threads.end()) {
          task_at->second.erase(thread_id);
        }
      }

      switch (threads_at->second.state) {
        case TASK_BLOCKED:
        // fall through
        case TASK_BUFN:
          transition(threads_at->second, thread_state::TASK_REMOVE_THROW);
          threads_at->second.wake_condition->notify_all();
          break;
        case SHUFFLE_BLOCKED:
          transition(threads_at->second, thread_state::SHUFFLE_REMOVE_THROW);
          threads_at->second.wake_condition->notify_all();
          break;
        case TASK_RUNNING:
          ret = true;
          // fall through;
        default:
          log_transition(thread_id, threads_at->second.task_id, threads_at->second.state,
                         thread_state::UNKNOWN);
          threads.erase(threads_at);
      }
    }
    return ret;
  }

  /**
   * Called prior to processing an alloc attempt. This will throw any injected exception and
   * wait until the thread is ready to actually do/retry the allocation. That blocking API may
   * throw other exceptions if rolling back or splitting the input is considered needed.
   * 
   * @return true if the call finds our thread in an ALLOC state, meaning that we recursively
   *         entered the state machine. The only known case is GPU memory required for setup in
   *         cuDF for a spill operation.
   */
  bool pre_alloc(long thread_id) {
    std::unique_lock<std::mutex> lock(state_mutex);

    auto thread = threads.find(thread_id);
    if (thread != threads.end()) {
      switch(thread->second.state) {
        // If the thread is in one of the ALLOC or ALLOC_FREE states, we have detected a loop
        // likely due to spill setup required in cuDF. We will treat this allocation differently
        // and skip transitions.
        case TASK_ALLOC:
        case SHUFFLE_ALLOC:
        case TASK_ALLOC_FREE:
        case SHUFFLE_ALLOC_FREE:
          return true;

        default: break;
      }

      if (thread->second.retry_oom_injected > 0) {
        thread->second.retry_oom_injected--;
        thread->second.num_times_retry_throw++;
        log_status("INJECTED_RETRY_OOM", thread_id, thread->second.task_id, thread->second.state);
        throw_java_exception(RETRY_OOM_CLASS, "injected RetryOOM");
      }

      if (thread->second.cudf_exception_injected > 0) {
        thread->second.cudf_exception_injected--;
        log_status("INJECTED_CUDF_EXCEPTION", thread_id, thread->second.task_id, thread->second.state);
        throw_java_exception(cudf::jni::CUDF_ERROR_CLASS, "injected CudfException");
      }

      if (thread->second.split_and_retry_oom_injected > 0) {
        thread->second.split_and_retry_oom_injected--;
        thread->second.num_times_split_retry_throw++;
        log_status("INJECTED_SPLIT_AND_RETRY_OOM", thread_id, thread->second.task_id, thread->second.state);
        throw_java_exception(SPLIT_AND_RETRY_OOM_CLASS, "injected SplitAndRetryOOM");
      }

      block_thread_until_ready(thread_id, lock);

      switch (thread->second.state) {
        case TASK_RUNNING: transition(thread->second, thread_state::TASK_ALLOC); break;
        case SHUFFLE_RUNNING:
          transition(thread->second, thread_state::SHUFFLE_ALLOC);
          break;

        // TODO I don't think there are other states that we need to handle, but
        // this needs more testing.
        default: {
          std::stringstream ss;
          ss << "thread " << thread_id << " in unexpected state pre alloc "
             << as_str(thread->second.state);

          throw std::invalid_argument(ss.str());
        }
      }
    }
    return false;
  }

  /**
   * Handle any state changes that happen after an alloc request succeeded.
   * No code in here should throw an exception or we are going to leak
   * GPU memory. I don't want to mark it as nothrow, because we can throw an
   * exception on an internal error, and I would rather see that we got the internal
   * error and leak something instead of getting a segfault.
   * 
   * `likely_spill` if this allocation should be treated differently, because
   * we detected recursion while handling a prior allocation in this thread.
   */
  void post_alloc_success(long thread_id, bool likely_spill) {
    std::unique_lock<std::mutex> lock(state_mutex);
    // pre allocate checks
    auto thread = threads.find(thread_id);
    if (!likely_spill && thread != threads.end()) {
      switch (thread->second.state) {
        case TASK_ALLOC:
          // fall through
        case TASK_ALLOC_FREE: transition(thread->second, thread_state::TASK_RUNNING); break;
        case SHUFFLE_ALLOC:
          // fall through
        case SHUFFLE_ALLOC_FREE: transition(thread->second, thread_state::SHUFFLE_RUNNING); break;
        default: break;
      }
      wake_next_highest_priority_regular_blocked(lock);
    }
  }

  /**
   * Wake the highest priority blocked (not BUFN) thread so it can make progress.
   * This is typically called when a free happens, or an alloc succeeds.
   */
  void wake_next_highest_priority_regular_blocked(const std::unique_lock<std::mutex> &lock) {
    // 1. Find the highest priority blocked thread, including shuffle.
    thread_priority to_wake(-1, -1);
    bool is_to_wake_set = false;
    for (auto thread = threads.begin(); thread != threads.end(); thread++) {
      thread_state state = thread->second.state;
      if (state == thread_state::TASK_BLOCKED || state == thread_state::SHUFFLE_BLOCKED) {
        thread_priority current = thread->second.priority();
        if (!is_to_wake_set || to_wake < current) {
          to_wake = current;
          is_to_wake_set = true;
        }
      }
    }
    // 2. wake up that thread
    long thread_id_to_wake = to_wake.get_thread_id();
    if (thread_id_to_wake > 0) {
      auto thread = threads.find(thread_id_to_wake);
      if (thread != threads.end()) {
        switch (thread->second.state) {
          case TASK_BLOCKED:
            transition(thread->second, thread_state::TASK_RUNNING);
            thread->second.wake_condition->notify_all();
            break;
          case SHUFFLE_BLOCKED:
            transition(thread->second, thread_state::SHUFFLE_RUNNING);
            thread->second.wake_condition->notify_all();
            break;
          default: {
            std::stringstream ss;
            ss << "internal error expected to only wake up blocked threads " << thread_id_to_wake
               << " " << as_str(thread->second.state);
            throw std::runtime_error(ss.str());
          }
        }
      }
    }
  }

  /**
   * Check to see if any threads need to move to BUFN. This should be
   * called when a task or shuffle thread becomes blocked so that we can
   * check to see if one of them needs to become BUFN or do a split and rollback.
   */
  void check_and_update_for_bufn(const std::unique_lock<std::mutex> &lock) {
    // We want to know if all active tasks have at least one thread that
    // is effectively blocked or not.  We could change the definitions here,
    // but for now this sounds like a good starting point.
    std::set<long> tasks_with_threads;
    std::set<long> tasks_with_threads_effectively_blocked;
    bool is_any_shuffle_thread_blocked = false;
    // To keep things simple we are going to do multiple passes through
    // the state. The first is to find out if any shuffle thread is blocked
    // because if it is, then there is a possibility that any task thread
    // in a shuffle could also be blocked.
    for (auto thread = threads.begin(); thread != threads.end(); thread++) {
      switch (thread->second.state) {
        case SHUFFLE_BLOCKED: is_any_shuffle_thread_blocked = true; break;
        default: break;
      }
    }

    for (auto thread = threads.begin(); thread != threads.end(); thread++) {
      if (thread->second.task_id >= 0) {
        tasks_with_threads.insert(thread->second.task_id);
      }

      switch (thread->second.state) {
        case TASK_WAIT_ON_SHUFFLE:
        // fall through
        case TASK_BUFN_WAIT_ON_SHUFFLE:
          if (is_any_shuffle_thread_blocked) {
            tasks_with_threads_effectively_blocked.insert(thread->second.task_id);
          }
          break;
        case TASK_BLOCKED:
        // fall through
        case TASK_BUFN_THROW:
        // fall through
        case TASK_BUFN_WAIT:
        // fall through
        case TASK_BUFN:
          tasks_with_threads_effectively_blocked.insert(thread->second.task_id);
          break;
        default: break;
      }
    }

    bool need_to_break_deadlock =
        tasks_with_threads.size() == tasks_with_threads_effectively_blocked.size();
    if (need_to_break_deadlock) {
      // Find the task thread with the lowest priority that is not already BUFN
      thread_priority to_bufn(-1, -1);
      bool is_to_bufn_set = false;
      for (auto thread = threads.begin(); thread != threads.end(); thread++) {
        switch (thread->second.state) {
          case TASK_BLOCKED: {
            thread_priority current = thread->second.priority();
            if (!is_to_bufn_set || current < to_bufn) {
              to_bufn = current;
              is_to_bufn_set = true;
            }
          } break;
          default: break;
        }
      }
      if (is_to_bufn_set) {
        long thread_id_to_bufn = to_bufn.get_thread_id();
        auto thread = threads.find(thread_id_to_bufn);
        if (thread != threads.end()) {
          transition(thread->second, thread_state::TASK_BUFN_THROW);
          thread->second.wake_condition->notify_all();
        }
      }

      // Now we need to check if all of the threads are BUFN
      // Are all BUFN??
      bool all_bufn_or_shuffle = true;
      thread_priority to_wake(-1, -1);
      bool is_to_wake_set = false;
      for (auto thread = threads.begin(); thread != threads.end(); thread++) {
        if (thread->second.task_id >= 0) {
          switch (thread->second.state) {
            case TASK_BUFN:
            // fall through
            case TASK_BUFN_WAIT:
            // fall through
            case TASK_BUFN_THROW: {
              thread_priority current = thread->second.priority();
              if (!is_to_wake_set || to_wake < current) {
                to_wake = current;
                is_to_wake_set = true;
              }
            } break;
            case TASK_WAIT_ON_SHUFFLE:
            // fall through
            case TASK_BUFN_WAIT_ON_SHUFFLE:
              if (!is_any_shuffle_thread_blocked) {
                all_bufn_or_shuffle = false;
              }
              break;
            default: all_bufn_or_shuffle = false; break;
          }
        }
      }
      if (all_bufn_or_shuffle) {
        long thread_id = to_wake.get_thread_id();
        auto found_thread = threads.find(thread_id);
        if (found_thread != threads.end()) {
          transition(found_thread->second, thread_state::TASK_SPLIT_THROW);
          found_thread->second.wake_condition->notify_all();
        } else {
          // the only threads left are blocked on shuffle. No way for shuffle
          // to split and throw, and ideally all of the data for those threads
          // should already be spillable, so at this point shuffle needs to
          // throw an OOM.
          for (auto thread = threads.begin(); thread != threads.end(); thread++) {
            switch (thread->second.state) {
              case SHUFFLE_BLOCKED:
                transition(thread->second, thread_state::SHUFFLE_THROW);
                thread->second.wake_condition->notify_all();
                break;
              default: break;
            }
          }
        }
      }
    }
  }

  /**
   * alloc failed so handle any state changes needed with that. Blocking will
   * typically happen after this has run, and we loop around to retry the alloc
   * if the state says we should.
   */
  bool post_alloc_failed(long thread_id, bool is_oom, bool likely_spill) {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto thread = threads.find(thread_id);
    // only retry if this was due to an out of memory exception.
    bool ret = true;
    if (!likely_spill && thread != threads.end()) {
      switch (thread->second.state) {
        case TASK_ALLOC_FREE: transition(thread->second, thread_state::TASK_RUNNING); break;
        case TASK_ALLOC:
          if (is_oom) {
            transition(thread->second, thread_state::TASK_BLOCKED);
          } else {
            // don't block unless it is OOM
            transition(thread->second, thread_state::TASK_RUNNING);
          }
          break;
        case SHUFFLE_ALLOC_FREE: transition(thread->second, thread_state::SHUFFLE_RUNNING); break;
        case SHUFFLE_ALLOC:
          if (is_oom) {
            transition(thread->second, thread_state::SHUFFLE_BLOCKED);
          } else {
            // don't block unless it is OOM
            transition(thread->second, thread_state::SHUFFLE_RUNNING);
          }
          break;
        default: {
          std::stringstream ss;
          ss << "Internal error: unexpected state after alloc failed " << thread_id << " "
             << as_str(thread->second.state);
          throw std::runtime_error(ss.str());
        }
      }
    } else {
      // do not retry if the thread is not registered...
      ret = false;
    }
    check_and_update_for_bufn(lock);
    return ret;
  }

  void *do_allocate(std::size_t num_bytes, rmm::cuda_stream_view stream) override {
    auto tid = static_cast<long>(pthread_self());
    while (true) {
      bool likely_spill = pre_alloc(tid);
      try {
        void *ret = resource->allocate(num_bytes, stream);
        post_alloc_success(tid, likely_spill);
        return ret;
      } catch (const std::bad_alloc &e) {
        if (!post_alloc_failed(tid, true, likely_spill)) {
          throw;
        }
      } catch (const std::exception &e) {
        post_alloc_failed(tid, false, likely_spill);
        throw;
      }
    }
    // we should never reach this point, but just in case
    throw std::bad_alloc();
  }

  void do_deallocate(void *p, std::size_t size, rmm::cuda_stream_view stream) override {
    resource->deallocate(p, size, stream);
    // deallocate success
    if (size > 0) {
      std::unique_lock<std::mutex> lock(state_mutex);

      auto tid = static_cast<long>(pthread_self());
      auto thread = threads.find(tid);
      if (thread != threads.end()) {
        log_status("DEALLOC", tid, thread->second.task_id, thread->second.state);
      } else {
        log_status("DEALLOC", tid, -2, thread_state::UNKNOWN);
      }

      for (auto thread = threads.begin(); thread != threads.end(); thread++) {
        // Only update state for _other_ threads. We update only other threads, for the case
        // where we are handling a free from the recursive case: when an allocation/free 
        // happened while handling an allocation failure in onAllocFailed.
        //
        // If we moved all threads to *_ALLOC_FREE, after we exit the recursive state and
        // are back handling the original allocation failure, we are left with a thread
        // in a state that won't be retried in `post_alloc_failed`.
        //
        // By not changing our thread's state to TASK_ALLOC_FREE, we keep the state
        // the same, but we still let other threads know that there was a free and they should
        // handle accordingly.
        if (thread->second.thread_id != tid) {
          switch (thread->second.state) {
            case TASK_ALLOC: 
              transition(thread->second, thread_state::TASK_ALLOC_FREE); break;
            case SHUFFLE_ALLOC: 
              transition(thread->second, thread_state::SHUFFLE_ALLOC_FREE); break;
            default: break;
          }
        }
      }
      wake_next_highest_priority_regular_blocked(lock);
    }
  }

  std::pair<size_t, size_t> do_get_mem_info(rmm::cuda_stream_view stream) const override {
    return resource->get_mem_info(stream);
  }
};

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getCurrentThreadId(JNIEnv *env, jclass) {
  try {
    cudf::jni::auto_set_device(env);
    return static_cast<jlong>(pthread_self());
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_createNewAdaptor(
    JNIEnv *env, jclass, jlong child, jstring log_loc) {
  JNI_NULL_CHECK(env, child, "child is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto wrapped = reinterpret_cast<rmm::mr::device_memory_resource *>(child);
    cudf::jni::native_jstring nlogloc(env, log_loc);
    std::shared_ptr<spdlog::logger> logger;
    if (nlogloc.is_null()) {
      logger = make_logger();
    } else {
      std::string slog_loc(nlogloc.get());
      if (slog_loc == "stderr") {
        logger = make_logger(std::cerr);
      } else if (slog_loc == "stdout") {
        logger = make_logger(std::cout);
      } else {
        logger = make_logger(slog_loc);
      }
    }

    auto ret = new spark_resource_adaptor(env, wrapped, logger);
    return cudf::jni::ptr_as_jlong(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_releaseAdaptor(
    JNIEnv *env, jclass, jlong ptr) {
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->all_done();
    delete mr;
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_associateThreadWithTask(JNIEnv *env, jclass,
                                                                              jlong ptr,
                                                                              jlong thread_id,
                                                                              jlong task_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->associate_thread_with_task(thread_id, task_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_associateThreadWithShuffle(JNIEnv *env,
                                                                                 jclass, jlong ptr,
                                                                                 jlong thread_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->associate_thread_with_shuffle(thread_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_removeThreadAssociation(JNIEnv *env, jclass,
                                                                              jlong ptr,
                                                                              jlong thread_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->remove_thread_association(thread_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_taskDone(
    JNIEnv *env, jclass, jlong ptr, jlong task_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->task_done(task_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_threadCouldBlockOnShuffle(JNIEnv *env, jclass,
                                                                                jlong ptr,
                                                                                jlong thread_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->thread_could_block_on_shuffle(thread_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_threadDoneWithShuffle(
    JNIEnv *env, jclass, jlong ptr, jlong thread_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    mr->thread_done_with_shuffle(thread_id);
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

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getStateOf(
    JNIEnv *env, jclass, jlong ptr, jlong thread_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    return mr->get_thread_state_as_int(thread_id);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getAndResetRetryThrowInternal(
    JNIEnv *env, jclass, jlong ptr, jlong task_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    return mr->get_n_reset_num_retry(task_id);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getAndResetSplitRetryThrowInternal(
    JNIEnv *env, jclass, jlong ptr, jlong task_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    return mr->get_n_reset_num_split_retry(task_id);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getAndResetBlockTimeInternal(
    JNIEnv *env, jclass, jlong ptr, jlong task_id) {
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor *>(ptr);
    return mr->get_n_reset_block_time(task_id);
  }
  CATCH_STD(env, 0)
}
}
