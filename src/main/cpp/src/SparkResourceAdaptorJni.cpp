/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <rmm/mr/device/device_memory_resource.hpp>

#include <cudf_jni_apis.hpp>
#include <pthread.h>
#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <map>
#include <set>
#include <sstream>
#include <unordered_set>

namespace {

constexpr char const* GPU_RETRY_OOM_CLASS = "com/nvidia/spark/rapids/jni/GpuRetryOOM";
constexpr char const* GPU_SPLIT_AND_RETRY_OOM_CLASS =
  "com/nvidia/spark/rapids/jni/GpuSplitAndRetryOOM";
constexpr char const* CPU_RETRY_OOM_CLASS = "com/nvidia/spark/rapids/jni/CpuRetryOOM";
constexpr char const* CPU_SPLIT_AND_RETRY_OOM_CLASS =
  "com/nvidia/spark/rapids/jni/CpuSplitAndRetryOOM";
constexpr char const* THREAD_REG_CLASS      = "com/nvidia/spark/rapids/jni/ThreadStateRegistry";
constexpr char const* IS_THREAD_BLOCKED     = "isThreadBlocked";
constexpr char const* IS_THREAD_BLOCKED_SIG = "(J)Z";
constexpr char const* REMOVE_THREAD         = "removeThread";
constexpr char const* REMOVE_THREAD_SIG     = "(J)V";

// This is a bit of a hack to cache the methods because CUDF is getting java to do an onload
// there.
std::mutex jni_mutex;
bool is_jni_loaded = false;
jclass ThreadStateRegistry_jclass;
jmethodID removeThread_method;
jmethodID isThreadBlocked_method;

void cache_thread_reg_jni(JNIEnv* env)
{
  std::unique_lock<std::mutex> lock(jni_mutex);
  if (is_jni_loaded) { return; }
  jclass cls = env->FindClass(THREAD_REG_CLASS);
  if (cls == nullptr) { return; }

  removeThread_method = env->GetStaticMethodID(cls, REMOVE_THREAD, REMOVE_THREAD_SIG);
  if (removeThread_method == nullptr) { return; }

  isThreadBlocked_method = env->GetStaticMethodID(cls, IS_THREAD_BLOCKED, IS_THREAD_BLOCKED_SIG);
  if (isThreadBlocked_method == nullptr) { return; }

  // Convert local reference to global so it cannot be garbage collected.
  ThreadStateRegistry_jclass = static_cast<jclass>(env->NewGlobalRef(cls));
  if (ThreadStateRegistry_jclass == nullptr) { return; }
  is_jni_loaded = true;
}

// In the task states BUFN means Block Until Further Notice.
// Meaning the thread should be blocked until another task finishes.
// The reasoning is that spilling, and even pausing threads with large allocations
// was not enough to avoid an out of memory error, so we want to not start
// again until we know that progress has been made. We might add an API
// in the future to know when a retry section has passed, which would
// probably be a preferable time to restart all BUFN threads.
enum class thread_state {
  UNKNOWN = -1,  // unknown state, this is really here for logging and anything transitioning to
                 // this state should actually be accomplished by deleting the thread from the state
  THREAD_RUNNING    = 0,  // task thread running normally
  THREAD_ALLOC      = 1,  // task thread in the middle of doing an allocation
  THREAD_ALLOC_FREE = 2,  // task thread in the middle of doing an allocation and a free happened
  THREAD_BLOCKED    = 3,  // task thread that is temporarily blocked
  THREAD_BUFN_THROW = 4,  // task thread that should throw an exception to roll back before blocking
  THREAD_BUFN_WAIT  = 5,  // task thread that threw an exception to roll back and now should
                          // block the next time alloc or block_until_ready is called
  THREAD_BUFN = 6,  // task thread that is blocked until higher priority tasks start to succeed
  THREAD_SPLIT_THROW  = 7,  // task thread that should throw an exception to split input and retry
  THREAD_REMOVE_THROW = 8,  // task thread that is being removed and needs to throw an exception
};

/**
 * Convert a state to a string representation for logging.
 */
const char* as_str(thread_state state)
{
  switch (state) {
    case thread_state::THREAD_RUNNING: return "THREAD_RUNNING";
    case thread_state::THREAD_ALLOC: return "THREAD_ALLOC";
    case thread_state::THREAD_ALLOC_FREE: return "THREAD_ALLOC_FREE";
    case thread_state::THREAD_BLOCKED: return "THREAD_BLOCKED";
    case thread_state::THREAD_BUFN_THROW: return "THREAD_BUFN_THROW";
    case thread_state::THREAD_BUFN_WAIT: return "THREAD_BUFN_WAIT";
    case thread_state::THREAD_BUFN: return "THREAD_BUFN";
    case thread_state::THREAD_SPLIT_THROW: return "THREAD_SPLIT_THROW";
    case thread_state::THREAD_REMOVE_THROW: return "THREAD_REMOVE_THROW";
    default: return "UNKNOWN";
  }
}

static std::shared_ptr<spdlog::logger> make_logger(std::ostream& stream)
{
  return std::make_shared<spdlog::logger>("SPARK_RMM",
                                          std::make_shared<spdlog::sinks::ostream_sink_mt>(stream));
}

static std::shared_ptr<spdlog::logger> make_logger()
{
  return std::make_shared<spdlog::logger>("SPARK_RMM",
                                          std::make_shared<spdlog::sinks::null_sink_mt>());
}

static auto make_logger(std::string const& filename)
{
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
 * We also want all non-task threads to have the highest priority possible. So we assign them
 * a task id of -1. The problem is overflow on a long, so for the priority of a task the formula
 * will be MAX_LONG - (task_id + 1).
 */
class thread_priority {
 public:
  thread_priority(long const tsk_id, long const t_id) : task_id(tsk_id), thread_id(t_id) {}

  long get_thread_id() const { return thread_id; }

  long get_task_id() const { return task_id; }

  bool operator<(const thread_priority& other) const
  {
    long task_priority       = this->task_priority();
    long other_task_priority = other.task_priority();
    if (task_priority < other_task_priority) {
      return true;
    } else if (task_priority == other_task_priority) {
      return thread_id < other.thread_id;
    }
    return false;
  }

  bool operator>(const thread_priority& other) const
  {
    long task_priority       = this->task_priority();
    long other_task_priority = other.task_priority();
    if (task_priority > other_task_priority) {
      return true;
    } else if (task_priority == other_task_priority) {
      return thread_id > other.thread_id;
    }
    return false;
  }

  void operator=(const thread_priority& other)
  {
    task_id   = other.task_id;
    thread_id = other.thread_id;
  }

 private:
  long task_id;
  long thread_id;

  long task_priority() const { return std::numeric_limits<long>::max() - (task_id + 1); }
};

/**
 * Holds metrics for a given task/thread about retry counts and times. It is here
 * because the mapping between tasks and threads can be complicated and can span
 * different time ranges too.
 */
struct task_metrics {
  // metric for being able to report how many times each type of exception was thrown,
  // and some timings
  int num_times_retry_throw       = 0;
  int num_times_split_retry_throw = 0;
  long time_blocked_nanos         = 0;
  // The amount of time that this thread has lost due to retries (not including blocked time)
  long time_lost_nanos = 0;

  long gpu_max_memory_allocated = 0;

  void take_from(task_metrics& other)
  {
    add(other);
    other.clear();
  }

  void add(task_metrics const& other)
  {
    this->num_times_retry_throw += other.num_times_retry_throw;
    this->num_times_split_retry_throw += other.num_times_split_retry_throw;
    this->time_blocked_nanos += other.time_blocked_nanos;
    this->time_lost_nanos += other.time_lost_nanos;
    this->gpu_max_memory_allocated =
      std::max(this->gpu_max_memory_allocated, other.gpu_max_memory_allocated);
  }

  void clear()
  {
    num_times_retry_throw       = 0;
    num_times_split_retry_throw = 0;
    time_blocked_nanos          = 0;
    time_lost_nanos             = 0;
  }
};

enum class oom_type {
  CPU_OR_GPU = 0,
  CPU,
  GPU,
};

struct oom_state_type {
  int hit_count   = 0;
  int skip_count  = 0;
  oom_type filter = oom_type::CPU_OR_GPU;

  void init(int const num_ooms, int const skip_count, int const oom_type_id)
  {
    if (num_ooms < 0) { throw std::invalid_argument("num_ooms cannot be negative"); }
    if (skip_count < 0) { throw std::invalid_argument("skip_count cannot be negative"); }
    if (oom_type_id < 0 || oom_type_id > 2) {
      throw std::invalid_argument("oom_filter must be between 0 and 2");
    }
    this->hit_count  = num_ooms;
    this->skip_count = skip_count;
    this->filter     = static_cast<oom_type>(oom_type_id);
  }

  bool matches(bool is_for_cpu)
  {
    return filter == oom_type::CPU_OR_GPU || (is_for_cpu && filter == oom_type::CPU) ||
           ((!is_for_cpu) && filter == oom_type::GPU);
  }
};

/**
 * This is the full state of a thread. Some things like the thread_id and task_id
 * should not change after the state is set up. Everything else is up for change,
 * but some things should generally be changed with caution. Access to anything in
 * this should be accessed with a lock held.
 */
class full_thread_state {
 public:
  full_thread_state(thread_state const state, long const thread_id)
    : state(state), thread_id(thread_id)
  {
  }
  full_thread_state(thread_state const state, long const thread_id, long const task_id)
    : state(state), thread_id(thread_id), task_id(task_id)
  {
  }
  thread_state state;
  long thread_id;
  long task_id        = -1;
  bool is_for_shuffle = false;
  std::unordered_set<long> pool_task_ids;
  bool is_cpu_alloc = false;
  // Is the thread transitively blocked on a pool or not.
  bool pool_blocked = false;

  oom_state_type retry_oom;
  oom_state_type split_and_retry_oom;

  int cudf_exception_injected = 0;
  // watchdog limit on maximum number of retries to avoid unexpected live lock situations
  int num_times_retried = 0;
  // When did the retry time for this thread start, or when did the block time end.
  std::chrono::time_point<std::chrono::steady_clock> retry_start_or_block_end;
  // Is this thread currently in a marked retry block. This is only used for metrics.
  bool is_in_retry = false;
  // The amount of time that this thread has spent in the current retry block (not including block
  // time)
  long time_retry_running_nanos = 0;
  std::chrono::time_point<std::chrono::steady_clock> block_start;
  long gpu_memory_allocated_bytes = 0;

  // metrics for the current thread
  task_metrics metrics;

  std::unique_ptr<std::condition_variable> wake_condition =
    std::make_unique<std::condition_variable>();

  /**
   * Transition to a new state. Ideally this is what is called when doing a state transition instead
   * of setting the state directly.
   */
  void transition_to(thread_state const new_state)
  {
    if (new_state == thread_state::UNKNOWN) {
      throw std::runtime_error(
        "Going to UNKNOWN state should delete the thread state, not call transition_to");
    }
    state = new_state;
  }

  void before_block()
  {
    block_start = std::chrono::steady_clock::now();
    // Don't record running time lost while we are blocked...
    record_and_reset_pending_retry_time();
  }

  void after_block()
  {
    auto const end  = std::chrono::steady_clock::now();
    auto const diff = end - block_start;
    metrics.time_blocked_nanos +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();
    if (is_in_retry) { retry_start_or_block_end = end; }
  }

  void record_failed_retry_time()
  {
    if (is_in_retry) {
      record_and_reset_pending_retry_time();
      metrics.time_lost_nanos += time_retry_running_nanos;
      time_retry_running_nanos = 0;
    }
  }

  void record_and_reset_pending_retry_time()
  {
    if (is_in_retry) {
      auto const end  = std::chrono::steady_clock::now();
      auto const diff = end - retry_start_or_block_end;
      time_retry_running_nanos +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count();
      retry_start_or_block_end = end;
    }
  }

  void reset_retry_state(bool const is_in_retry)
  {
    time_retry_running_nanos = 0;
    if (is_in_retry) { retry_start_or_block_end = std::chrono::steady_clock::now(); }
    this->is_in_retry = is_in_retry;
  }

  /**
   * Get the priority of this thread.
   */
  thread_priority priority() const
  {
    if (task_id < 0 && !is_for_shuffle) {
      // The task id for a non-shuffle pool thread is the same as the lowest task id
      // it is currently working on.
      auto const min_id = std::min_element(pool_task_ids.begin(), pool_task_ids.end());
      if (min_id != pool_task_ids.end()) {
        return thread_priority(*min_id, thread_id);
      } else {
        return thread_priority(-1, thread_id);
      }
    } else {
      return thread_priority(task_id, thread_id);
    }
  }
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
  spark_resource_adaptor(JNIEnv* env,
                         rmm::mr::device_memory_resource* mr,
                         std::shared_ptr<spdlog::logger>& logger,
                         bool const is_log_enabled)
    : resource{mr}, logger{logger}, is_log_enabled{is_log_enabled}
  {
    if (env->GetJavaVM(&jvm) < 0) { throw std::runtime_error("GetJavaVM failed"); }
    logger->flush_on(spdlog::level::info);
    logger->set_pattern("%v");
    logger->info("time,op,current thread,op thread,op task,from state,to state,notes");
    logger->set_pattern("%H:%M:%S.%f,%v");
  }

  rmm::mr::device_memory_resource* get_wrapped_resource() { return resource; }

  /**
   * Update the internal state so that a specific thread is dedicated to a task.
   * This may be called multiple times for a given thread and if the thread is already
   * dedicated to the task, then most of the time this is a noop. The only exception
   * is if the thread is marked that it is shutting down, but has not completed yet.
   * This should never happen in practice with Spark because the only time we would
   * shut down a task thread on a thread that is different from itself is if there
   * was an error and the entire executor is shutting down. So there should be no
   * reuse.
   */
  void start_dedicated_task_thread(long const thread_id, long const task_id)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    if (shutting_down) { throw std::runtime_error("spark_resource_adaptor is shutting down"); }
    auto const found = threads.find(thread_id);
    if (found != threads.end()) {
      if (found->second.task_id >= 0 && found->second.task_id != task_id) {
        if (is_log_enabled) {
          std::stringstream ss;
          ss << "desired task_id " << task_id;

          log_status("FIXUP", thread_id, found->second.task_id, found->second.state, ss.str());
        }
        remove_thread_association(thread_id, found->second.task_id, lock);
      }
    }
    auto const was_threads_inserted = threads.emplace(
      thread_id, full_thread_state(thread_state::THREAD_RUNNING, thread_id, task_id));
    if (was_threads_inserted.second == false) {
      if (was_threads_inserted.first->second.state == thread_state::THREAD_REMOVE_THROW) {
        std::stringstream ss;
        ss << "A thread " << thread_id << " is shutting down "
           << was_threads_inserted.first->second.task_id << " vs " << task_id;

        auto const msg = ss.str();
        log_status("ERROR",
                   thread_id,
                   was_threads_inserted.first->second.task_id,
                   was_threads_inserted.first->second.state,
                   msg);
        throw std::invalid_argument(msg);
      }

      if (was_threads_inserted.first->second.task_id != task_id) {
        std::stringstream ss;
        ss << "A thread " << thread_id << " can only be dedicated to a single task."
           << was_threads_inserted.first->second.task_id << " != " << task_id;
        auto const msg = ss.str();
        log_status("ERROR",
                   thread_id,
                   was_threads_inserted.first->second.task_id,
                   was_threads_inserted.first->second.state,
                   msg);
        throw std::invalid_argument(msg);
      }
    }

    try {
      auto const was_inserted = task_to_threads.insert({task_id, {thread_id}});
      if (was_inserted.second == false) {
        // task_to_threads already has a task_id for this, so insert the thread_id
        was_inserted.first->second.insert(thread_id);
      }
    } catch (std::exception const&) {
      if (was_threads_inserted.second == true) {
        // roll back the thread insertion
        threads.erase(thread_id);
      }
      throw;
    }
    if (was_threads_inserted.second == true) {
      log_transition(thread_id, task_id, thread_state::UNKNOWN, thread_state::THREAD_RUNNING);
    }
  }

  void start_retry_block(long const thread_id)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto const thread = threads.find(thread_id);
    if (thread != threads.end()) { thread->second.reset_retry_state(true); }
  }

  void end_retry_block(long const thread_id)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto const thread = threads.find(thread_id);
    if (thread != threads.end()) { thread->second.reset_retry_state(false); }
  }

  /**
   * Update the internal state so that a specific thread is associated with transitive
   * thread pools and is working on a set of tasks.
   * This may be called multiple times for a given thread and the set of tasks will be
   * updated accordingly.
   */
  void pool_thread_working_on_tasks(bool const is_for_shuffle,
                                    long const thread_id,
                                    std::unordered_set<long> const& task_ids)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    if (shutting_down) { throw std::runtime_error("spark_resource_adaptor is shutting down"); }

    auto const was_inserted =
      threads.emplace(thread_id, full_thread_state(thread_state::THREAD_RUNNING, thread_id));
    if (was_inserted.second == true) {
      was_inserted.first->second.is_for_shuffle = is_for_shuffle;
      log_transition(thread_id, -1, thread_state::UNKNOWN, thread_state::THREAD_RUNNING);
    } else if (was_inserted.first->second.task_id != -1) {
      throw std::invalid_argument("the thread is associated with a non-pool task already");
    } else if (was_inserted.first->second.state == thread_state::THREAD_REMOVE_THROW) {
      throw std::invalid_argument("the thread is in the process of shutting down.");
    } else if (was_inserted.first->second.is_for_shuffle != is_for_shuffle) {
      if (is_for_shuffle) {
        throw std::invalid_argument(
          "the thread is marked as a non-shuffle thread, and we cannot change it while there are "
          "active tasks");
      } else {
        throw std::invalid_argument(
          "the thread is marked as a shuffle thread, and we cannot change it while there are "
          "active tasks");
      }
    }

    // save the metrics for all tasks before we add any new ones.
    checkpoint_metrics(was_inserted.first->second);

    was_inserted.first->second.pool_task_ids.insert(task_ids.begin(), task_ids.end());
    if (is_log_enabled) {
      std::stringstream ss;
      ss << "CURRENT IDs ";
      for (const auto& task_id : was_inserted.first->second.pool_task_ids) {
        ss << task_id << " ";
      }
      log_status("ADD_TASKS", thread_id, -1, was_inserted.first->second.state, ss.str());
    }
  }

  void pool_thread_finished_for_tasks(long const thread_id,
                                      std::unordered_set<long> const& task_ids)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    if (shutting_down) { throw std::runtime_error("spark_resource_adaptor is shutting down"); }

    auto const thread = threads.find(thread_id);
    if (thread != threads.end()) {
      // save the metrics for all tasks before we remove any of them.
      checkpoint_metrics(thread->second);

      // Now drop the tasks from the pool
      for (auto const& id : task_ids) {
        thread->second.pool_task_ids.erase(id);
      }
      if (is_log_enabled) {
        std::stringstream ss;
        ss << "CURRENT IDs ";
        for (const auto& id : thread->second.pool_task_ids) {
          ss << id << " ";
        }
        log_status("REMOVE_TASKS", thread_id, -1, thread->second.state, ss.str());
      }
      if (thread->second.pool_task_ids.empty()) {
        if (remove_thread_association(thread_id, -1, lock)) {
          wake_up_threads_after_task_finishes(lock);
        }
      }
    }
  }

  /**
   * Update the internal state so that a specific thread is no longer associated with
   * a task or with shuffle. If that thread is currently blocked/waiting, then the
   * thread will not be immediately removed, but is instead marked that it needs to wake
   * up and throw an exception. At that point the thread's state will be completely
   * removed.
   */
  void remove_thread_association(long const thread_id, long const task_id)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    if (remove_thread_association(thread_id, task_id, lock)) {
      wake_up_threads_after_task_finishes(lock);
    }
  }

  /**
   * Update the internal state so that all threads associated with a task are
   * cleared. Just like with remove_thread_association if one or more of these
   * threads are currently blocked/waiting then the state will not be totally
   * removed until the thread is woken.
   */
  void task_done(long const task_id)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    bool run_checks    = false;
    auto const task_at = task_to_threads.find(task_id);
    if (task_at != task_to_threads.end()) {
      // we want to make a copy so there is no conflict here...
      std::set<long> const threads_to_remove = task_at->second;
      for (auto const thread_id : threads_to_remove) {
        run_checks = remove_thread_association(thread_id, task_id, lock) || run_checks;
      }
    }
    std::unordered_set<long> thread_ids;
    for (auto const& [thread_id, ignored] : threads) {
      thread_ids.insert(thread_id);
    }
    for (auto const& thread_id : thread_ids) {
      auto const thread = threads.find(thread_id);
      if (thread != threads.end()) {
        if (thread->second.pool_task_ids.erase(task_id) != 0) {
          if (is_log_enabled) {
            std::stringstream ss;
            ss << "CURRENT IDs ";
            for (const auto& id : thread->second.pool_task_ids) {
              ss << id << " ";
            }
            log_status("REMOVE_TASKS", thread_id, -1, thread->second.state, ss.str());
          }
          if (thread->second.pool_task_ids.empty()) {
            run_checks = remove_thread_association(thread_id, task_id, lock) || run_checks;
          }
        }
      }
    }

    if (run_checks) { wake_up_threads_after_task_finishes(lock); }
    task_to_threads.erase(task_id);
  }

  /**
   * A dedicated task thread is submitting to a pool.
   */
  void submitting_to_pool(long const thread_id) { waiting_on_pool_status_changed(thread_id, true); }

  /**
   * A dedicated task thread is waiting on a result from a pool.
   */
  void waiting_on_pool(long const thread_id) { waiting_on_pool_status_changed(thread_id, true); }

  /**
   * A dedicated task thread is no longer blocked on a pool.
   * It got the answer, an exception, or it submitted the
   * work successfully.
   */
  void done_waiting_on_pool(long const thread_id)
  {
    waiting_on_pool_status_changed(thread_id, false);
  }

  /**
   * This should be called before shutting down the adaptor. It will try
   * to shut down everything in an orderly way and wait for all of the
   * threads to be done.
   */
  void all_done()
  {
    {
      std::unique_lock<std::mutex> lock(state_mutex);
      // 1. Mark all threads that need to be removed as such
      // make a copy of the ids so we don't modify threads while walking it
      std::vector<long> threads_to_remove;
      for (auto const& thread : threads) {
        threads_to_remove.push_back(thread.first);
      }

      for (auto const thread_id : threads_to_remove) {
        remove_thread_association(thread_id, -1, lock);
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
  void force_retry_oom(long const thread_id,
                       int const num_ooms,
                       int const oom_filter,
                       int const skip_count)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto const threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      threads_at->second.retry_oom.init(num_ooms, skip_count, oom_filter);
    } else {
      throw std::invalid_argument("the thread is not associated with any task/shuffle");
    }
  }

  /**
   * Force a specific thread to throw one or more SplitAndRetryOOM exceptions
   * when an alloc is called. This is intended only for testing.
   */
  void force_split_and_retry_oom(long const thread_id,
                                 int const num_ooms,
                                 int const oom_filter,
                                 int const skip_count)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto const threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      threads_at->second.split_and_retry_oom.init(num_ooms, skip_count, oom_filter);
    } else {
      throw std::invalid_argument("the thread is not associated with any task/shuffle");
    }
  }

  /**
   * force a specific thread to throw one or more CudfExceptions when an
   * alloc is called. This is intended only for testing.
   */
  void force_cudf_exception(long const thread_id, int const num_times)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto const threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      threads_at->second.cudf_exception_injected = num_times;
    } else {
      throw std::invalid_argument("the thread is not associated with any task/shuffle");
    }
  }

  // Some C++ magic to get and reset a single metric.
  // Metrics are recorded on a per-thread basis, but are reported per-task
  // But the life time of threads and tasks are not directly tied together
  // so they are check-pointed periodically. This reads and resets
  // the metric for both the threads and the tasks
  template <class T>
  T get_and_reset_metric(long const task_id, T task_metrics::*MetricPtr)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    T ret              = 0;
    auto const task_at = task_to_threads.find(task_id);
    if (task_at != task_to_threads.end()) {
      for (auto const thread_id : task_at->second) {
        auto const threads_at = threads.find(thread_id);
        if (threads_at != threads.end()) {
          ret += (threads_at->second.metrics.*MetricPtr);
          (threads_at->second.metrics.*MetricPtr) = 0;
        }
      }
    }

    auto const metrics_at = task_to_metrics.find(task_id);
    if (metrics_at != task_to_metrics.end()) {
      ret += (metrics_at->second.*MetricPtr);
      (metrics_at->second.*MetricPtr) = 0;
    }
    return ret;
  }

  /**
   * get the number of times a retry was thrown and reset the value to 0.
   */
  int get_and_reset_num_retry(long const task_id)
  {
    return get_and_reset_metric(task_id, &task_metrics::num_times_retry_throw);
  }

  /**
   * get the number of times a split and retry was thrown and reset the value to 0.
   */
  int get_and_reset_num_split_retry(long const task_id)
  {
    return get_and_reset_metric(task_id, &task_metrics::num_times_split_retry_throw);
  }

  /**
   * get the time in ns that the task was blocked for.
   */
  long get_and_reset_block_time(long const task_id)
  {
    return get_and_reset_metric(task_id, &task_metrics::time_blocked_nanos);
  }

  /**
   * get the time in ns that was lost because a retry was thrown.
   */
  long get_and_reset_lost_time(long const task_id)
  {
    return get_and_reset_metric(task_id, &task_metrics::time_lost_nanos);
  }

  long get_and_reset_gpu_max_memory_allocated(long const task_id)
  {
    return get_and_reset_metric(task_id, &task_metrics::gpu_max_memory_allocated);
  }

  void check_and_break_deadlocks()
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    check_and_update_for_bufn(lock);
  }

  bool cpu_prealloc(size_t const amount, bool const blocking)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto const thread_id = static_cast<long>(pthread_self());
    return pre_alloc_core(thread_id, true, blocking, lock);
  }

  void cpu_postalloc_success(void const* addr,
                             size_t const amount,
                             bool const blocking,
                             bool const was_recursive)
  {
    // addr is not used yet, but is here in case we want it in the future.
    // amount is not used yet, but is here in case we want it for debugging/metrics.
    std::unique_lock<std::mutex> lock(state_mutex);
    auto const thread_id = static_cast<long>(pthread_self());
    post_alloc_success_core(thread_id, true, was_recursive, amount, lock);
  }

  bool cpu_postalloc_failed(bool const was_oom, bool const blocking, bool const was_recursive)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto const thread_id = static_cast<long>(pthread_self());
    return post_alloc_failed_core(thread_id, true, was_oom, blocking, was_recursive, lock);
  }

  void cpu_dealloc(void const* addr, size_t const amount)
  {
    // addr is not used yet, but is here in case we want it in the future.
    // amount is not used yet, but is here in case we want it for debugging/metrics.
    std::unique_lock<std::mutex> lock(state_mutex);
    dealloc_core(true, lock, amount);
  }

  /**
   * Called after a RetryOOM is thrown to wait until it is okay to start processing
   * data again. This is here mostly to prevent spillable code becoming unspillable
   * before an alloc is called.  If this is not called alloc will also call into the
   * same code and block if needed until the task is ready to keep going.
   */
  void block_thread_until_ready()
  {
    auto const thread_id = static_cast<long>(pthread_self());
    std::unique_lock<std::mutex> lock(state_mutex);
    block_thread_until_ready(thread_id, lock);
  }

  /**
   * This is really here just for testing. It provides a way to look at the
   * current state of a thread.
   */
  int get_thread_state_as_int(long const thread_id)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto const threads_at = threads.find(thread_id);
    if (threads_at != threads.end()) {
      return static_cast<int>(threads_at->second.state);
    } else {
      return -1;
    }
  }

 private:
  rmm::mr::device_memory_resource* const resource;
  std::shared_ptr<spdlog::logger> logger;  ///< spdlog logger object
  bool const is_log_enabled;

  // The state mutex must be held when modifying the state of threads or tasks
  // it must never be held when calling into the child resource or after returning
  // from an operation.
  std::mutex state_mutex;
  std::condition_variable task_has_woken_condition;
  std::map<long, full_thread_state> threads;
  std::map<long, std::set<long>> task_to_threads;
  // Metrics are a little complicated. Spark reports metrics at a task level
  // but we track and collect them at a thread level. The life time of a thread
  // and a task are not tied to each other, and a thread can work on things for
  // multiple tasks at the same time. So whenever a thread changes status
  // the metrics for the tasks it is working on are aggregated here. When a task
  // finishes the metrics for that task are then deleted.
  std::map<long, task_metrics> task_to_metrics;
  bool shutting_down = false;
  JavaVM* jvm;

  /**
   * log a status change that does not involve a state transition.
   */
  void log_status(std::string const& op,
                  long const thread_id,
                  long const task_id,
                  thread_state const state,
                  std::string const& notes = "") const
  {
    auto const this_id = static_cast<long>(pthread_self());
    logger->info("{},{},{},{},{},,{}", op, this_id, thread_id, task_id, as_str(state), notes);
  }

  /**
   * log that a state transition happened.
   */
  void log_transition(long const thread_id,
                      long const task_id,
                      thread_state const from,
                      thread_state const to,
                      std::string const& notes = "") const
  {
    auto const this_id = static_cast<long>(pthread_self());
    logger->info(
      "TRANSITION,{},{},{},{},{},{}", this_id, thread_id, task_id, as_str(from), as_str(to), notes);
  }

  /**
   * Transition to a new state. Ideally this is what is called when doing a state transition instead
   * of setting the state directly. This will log the transition and do a little bit of
   * verification.
   */
  void transition(full_thread_state& state,
                  thread_state const new_state,
                  std::string const& message = "")
  {
    thread_state original = state.state;
    state.transition_to(new_state);
    log_transition(state.thread_id, state.task_id, original, new_state, message);
  }

  /**
   * throw a java exception using the cached jvm/env.
   */
  void throw_java_exception(char const* ex_class_name, char const* msg)
  {
    JNIEnv* env = cudf::jni::get_jni_env(jvm);
    cudf::jni::throw_java_exception(env, ex_class_name, msg);
  }

  void waiting_on_pool_status_changed(long const thread_id, bool const pool_blocked)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    auto const thread = threads.find(thread_id);
    long task_id      = -1;
    if (thread != threads.end()) { task_id = thread->second.task_id; }

    if (task_id < 0) {
      std::stringstream ss;
      ss << "thread " << thread_id << " is not a dedicated task thread";
      throw std::invalid_argument(ss.str());
    }

    thread->second.pool_blocked = pool_blocked;
  }

  /**
   * Checkpoint all of the metrics for a thread.
   */
  void checkpoint_metrics(full_thread_state& state)
  {
    if (state.task_id < 0) {
      // save the metrics for all tasks before we add any new ones.
      for (auto const task_id : state.pool_task_ids) {
        auto const metrics_at = task_to_metrics.try_emplace(task_id, task_metrics());
        metrics_at.first->second.add(state.metrics);
      }
      state.metrics.clear();
    } else {
      auto const metrics_at = task_to_metrics.try_emplace(state.task_id, task_metrics());
      metrics_at.first->second.take_from(state.metrics);
    }
  }

  /**
   * This is a watchdog to prevent us from live locking. It should be called before we throw an
   * RetryOOM or a SplitAndRetryOOM to know if we actually should throw something else.
   */
  void check_before_oom(full_thread_state& state, std::unique_lock<std::mutex> const& lock)
  {
    // The limit is an arbitrary number, large enough that we should not hit it in "normal"
    // operation, but also small enough that we can detect a livelock fairly quickly.
    // In testing it looks like it is a few ms if in a tight loop, not including spill
    // overhead
    if (state.num_times_retried + 1 > 500) {
      state.record_failed_retry_time();
      throw_java_exception(cudf::jni::OOM_CLASS, "GPU OutOfMemory: retry limit exceeded");
    }
    state.num_times_retried++;
  }

  void throw_retry_oom(char const* msg,
                       full_thread_state& state,
                       std::unique_lock<std::mutex> const& lock)
  {
    state.metrics.num_times_retry_throw++;
    check_before_oom(state, lock);
    state.record_failed_retry_time();
    if (state.is_cpu_alloc) {
      throw_java_exception(CPU_RETRY_OOM_CLASS, "CPU OutOfMemory");
    } else {
      throw_java_exception(GPU_RETRY_OOM_CLASS, "GPU OutOfMemory");
    }
  }

  void throw_split_and_retry_oom(char const* msg,
                                 full_thread_state& state,
                                 std::unique_lock<std::mutex> const& lock)
  {
    state.metrics.num_times_split_retry_throw++;
    check_before_oom(state, lock);
    state.record_failed_retry_time();
    if (state.is_cpu_alloc) {
      throw_java_exception(CPU_SPLIT_AND_RETRY_OOM_CLASS, "CPU OutOfMemory");
    } else {
      throw_java_exception(GPU_SPLIT_AND_RETRY_OOM_CLASS, "GPU OutOfMemory");
    }
  }

  bool is_blocked(thread_state state) const
  {
    switch (state) {
      case thread_state::THREAD_BLOCKED:
      // fall through
      case thread_state::THREAD_BUFN: return true;
      default: return false;
    }
  }

  /**
   * Internal implementation that will block a thread until it is ready to continue.
   */
  void block_thread_until_ready(long const thread_id, std::unique_lock<std::mutex>& lock)
  {
    bool done       = false;
    bool first_time = true;
    // Because this is called from alloc as well as from the public facing block_thread_until_ready
    // there are states that should only show up in relation to alloc failing. These include
    // THREAD_BUFN_THROW and THREAD_SPLIT_THROW. They should never happen unless this is being
    // called from within an alloc.
    while (!done) {
      auto thread = threads.find(thread_id);
      if (thread != threads.end()) {
        switch (thread->second.state) {
          case thread_state::THREAD_BLOCKED:
          // fall through
          case thread_state::THREAD_BUFN:
            log_status("WAITING", thread_id, thread->second.task_id, thread->second.state);
            thread->second.before_block();
            do {
              thread->second.wake_condition->wait(lock);
              thread = threads.find(thread_id);
            } while (thread != threads.end() && is_blocked(thread->second.state));
            thread->second.after_block();
            task_has_woken_condition.notify_all();
            break;
          case thread_state::THREAD_BUFN_THROW:
            transition(thread->second, thread_state::THREAD_BUFN_WAIT);
            thread->second.record_failed_retry_time();
            throw_retry_oom("rollback and retry operation", thread->second, lock);
            break;
          case thread_state::THREAD_BUFN_WAIT:
            transition(thread->second, thread_state::THREAD_BUFN);
            // Before we can wait it is possible that the throw didn't release anything
            // and the other threads didn't get unblocked by this, so we need to
            // check again to see if this was fixed or not.
            check_and_update_for_bufn(lock);
            // If that caused us to transition to a new state, then we need to adjust to it
            // appropriately...
            if (is_blocked(thread->second.state)) {
              log_status("WAITING", thread_id, thread->second.task_id, thread->second.state);
              thread->second.before_block();
              do {
                thread->second.wake_condition->wait(lock);
                thread = threads.find(thread_id);
              } while (thread != threads.end() && is_blocked(thread->second.state));
              thread->second.after_block();
              task_has_woken_condition.notify_all();
            }
            break;
          case thread_state::THREAD_SPLIT_THROW:
            transition(thread->second, thread_state::THREAD_RUNNING);
            thread->second.record_failed_retry_time();
            throw_split_and_retry_oom(
              "rollback, split input, and retry operation", thread->second, lock);
            break;
          case thread_state::THREAD_REMOVE_THROW:
            log_transition(
              thread_id, thread->second.task_id, thread->second.state, thread_state::UNKNOWN);
            // don't need to record failed time metric the thread is already gone...
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
  void wake_up_threads_after_task_finishes(const std::unique_lock<std::mutex>& lock)
  {
    bool are_any_tasks_just_blocked = false;
    for (auto& [thread_id, t_state] : threads) {
      switch (t_state.state) {
        case thread_state::THREAD_BLOCKED:
          transition(t_state, thread_state::THREAD_RUNNING);
          t_state.wake_condition->notify_all();
          are_any_tasks_just_blocked = true;
          break;
        default: break;
      }
    }

    if (!are_any_tasks_just_blocked) {
      // wake up all of the BUFN tasks.
      for (auto& [thread_id, t_state] : threads) {
        switch (t_state.state) {
          case thread_state::THREAD_BUFN:
          // fall through
          case thread_state::THREAD_BUFN_THROW:
          // fall through
          case thread_state::THREAD_BUFN_WAIT:
            transition(t_state, thread_state::THREAD_RUNNING);
            t_state.wake_condition->notify_all();
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
  bool remove_thread_association(long thread_id,
                                 long remove_task_id,
                                 const std::unique_lock<std::mutex>& lock)
  {
    bool thread_should_be_removed = false;
    bool ret                      = false;
    auto const threads_at         = threads.find(thread_id);
    if (threads_at != threads.end()) {
      // save the metrics no matter what
      checkpoint_metrics(threads_at->second);

      if (remove_task_id < 0) {
        thread_should_be_removed = true;
      } else {
        auto const task_id = threads_at->second.task_id;
        if (task_id >= 0) {
          if (task_id == remove_task_id) { thread_should_be_removed = true; }
        } else {
          threads_at->second.pool_task_ids.erase(remove_task_id);
          if (threads_at->second.pool_task_ids.empty()) { thread_should_be_removed = true; }
        }
      }

      if (thread_should_be_removed) {
        JNIEnv* env = nullptr;
        if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) == JNI_OK) {
          cache_thread_reg_jni(env);
          env->CallStaticVoidMethod(ThreadStateRegistry_jclass, removeThread_method, thread_id);
        }
        if (remove_task_id >= 0) {
          auto const task_at = task_to_threads.find(remove_task_id);
          if (task_at != task_to_threads.end()) { task_at->second.erase(thread_id); }
        }

        switch (threads_at->second.state) {
          case thread_state::THREAD_BLOCKED:
          // fall through
          case thread_state::THREAD_BUFN:
            transition(threads_at->second, thread_state::THREAD_REMOVE_THROW);
            threads_at->second.wake_condition->notify_all();
            break;
          case thread_state::THREAD_RUNNING:
            ret = true;
            // fall through;
          default:
            log_transition(thread_id,
                           threads_at->second.task_id,
                           threads_at->second.state,
                           thread_state::UNKNOWN);
            threads.erase(threads_at);
        }
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
  bool pre_alloc(long const thread_id)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    return pre_alloc_core(thread_id, false, true, lock);
  }

  /**
   * Called prior to processing an alloc attempt (CPU or GPU). This will throw any injected
   * exception and wait until the thread is ready to actually do/retry the allocation (if
   * the allocation is blocking). That blocking API may throw other exceptions if rolling
   * back or splitting the input is considered needed.
   *
   * @return true if the call finds our thread in an ALLOC state, meaning that we recursively
   *         entered the state machine. This happens when we need to spill in a few cases for
   *         the CPU.
   */
  bool pre_alloc_core(long const thread_id,
                      bool const is_for_cpu,
                      bool const blocking,
                      std::unique_lock<std::mutex>& lock)
  {
    auto const thread = threads.find(thread_id);
    if (thread != threads.end()) {
      switch (thread->second.state) {
        // If the thread is in one of the ALLOC or ALLOC_FREE states, we have detected a loop
        // likely due to spill setup required in cuDF. We will treat this allocation differently
        // and skip transitions.
        case thread_state::THREAD_ALLOC:
        // fall through
        case thread_state::THREAD_ALLOC_FREE:
          if (is_for_cpu && blocking) {
            // On the CPU we want the spill code to be explicit so we don't have to detect it
            // on the GPU we detect it and adjust dynamically
            std::stringstream ss;
            ss << "thread " << thread_id
               << " is trying to do a blocking allocate while already in the state "
               << as_str(thread->second.state);

            throw std::invalid_argument(ss.str());
          }
          // We are in a recursive allocation.
          return true;
        default: break;
      }

      if (thread->second.retry_oom.matches(is_for_cpu)) {
        if (thread->second.retry_oom.skip_count > 0) {
          thread->second.retry_oom.skip_count--;
        } else if (thread->second.retry_oom.hit_count > 0) {
          thread->second.retry_oom.hit_count--;
          thread->second.metrics.num_times_retry_throw++;
          std::string const op_prefix = "INJECTED_RETRY_OOM_";
          std::string const op        = op_prefix + (is_for_cpu ? "CPU" : "GPU");
          log_status(op, thread_id, thread->second.task_id, thread->second.state);
          thread->second.record_failed_retry_time();
          throw_java_exception(is_for_cpu ? CPU_RETRY_OOM_CLASS : GPU_RETRY_OOM_CLASS,
                               "injected RetryOOM");
        }
      }

      if (thread->second.cudf_exception_injected > 0) {
        thread->second.cudf_exception_injected--;
        log_status(
          "INJECTED_CUDF_EXCEPTION", thread_id, thread->second.task_id, thread->second.state);
        thread->second.record_failed_retry_time();
        throw_java_exception(cudf::jni::CUDF_ERROR_CLASS, "injected CudfException");
      }

      if (thread->second.split_and_retry_oom.matches(is_for_cpu)) {
        if (thread->second.split_and_retry_oom.skip_count > 0) {
          thread->second.split_and_retry_oom.skip_count--;
        } else if (thread->second.split_and_retry_oom.hit_count > 0) {
          thread->second.split_and_retry_oom.hit_count--;
          thread->second.metrics.num_times_split_retry_throw++;
          std::string const op_prefix = "INJECTED_SPLIT_AND_RETRY_OOM_";
          std::string const op        = op_prefix + (is_for_cpu ? "CPU" : "GPU");
          log_status(op, thread_id, thread->second.task_id, thread->second.state);
          thread->second.record_failed_retry_time();
          if (is_for_cpu) {
            throw_java_exception(CPU_SPLIT_AND_RETRY_OOM_CLASS, "injected SplitAndRetryOOM");
          } else {
            throw_java_exception(GPU_SPLIT_AND_RETRY_OOM_CLASS, "injected SplitAndRetryOOM");
          }
        }
      }

      if (blocking) { block_thread_until_ready(thread_id, lock); }

      switch (thread->second.state) {
        case thread_state::THREAD_RUNNING:
          transition(thread->second, thread_state::THREAD_ALLOC);
          thread->second.is_cpu_alloc = is_for_cpu;
          break;
        default: {
          std::stringstream ss;
          ss << "thread " << thread_id << " in unexpected state pre alloc "
             << as_str(thread->second.state);

          throw std::invalid_argument(ss.str());
        }
      }
    }
    // Not a recursive allocation
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
  void post_alloc_success(long const thread_id,
                          bool const likely_spill,
                          std::size_t const num_bytes)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    post_alloc_success_core(thread_id, false, likely_spill, num_bytes, lock);
  }

  void post_alloc_success_core(long const thread_id,
                               bool const is_for_cpu,
                               bool const was_recursive,
                               std::size_t const num_bytes,
                               std::unique_lock<std::mutex>& lock)
  {
    // pre allocate checks
    auto const thread = threads.find(thread_id);
    if (!was_recursive && thread != threads.end()) {
      switch (thread->second.state) {
        case thread_state::THREAD_ALLOC:
          // fall through
        case thread_state::THREAD_ALLOC_FREE:
          if (thread->second.is_cpu_alloc != is_for_cpu) {
            std::stringstream ss;
            ss << "thread " << thread_id << " has a mismatch on CPU vs GPU post alloc "
               << as_str(thread->second.state);

            throw std::invalid_argument(ss.str());
          }
          transition(thread->second, thread_state::THREAD_RUNNING);
          thread->second.is_cpu_alloc = false;
          // num_bytes is likely not padded, which could cause slight inaccuracies
          // but for now it shouldn't matter for watermark purposes
          if (!is_for_cpu) {
            thread->second.gpu_memory_allocated_bytes += num_bytes;
            thread->second.metrics.gpu_max_memory_allocated =
              std::max(thread->second.metrics.gpu_max_memory_allocated,
                       thread->second.gpu_memory_allocated_bytes);
          }
          break;
        default: break;
      }
      wake_next_highest_priority_blocked(lock, false, is_for_cpu);
    }
  }

  /**
   * Wake the highest priority blocked (not BUFN) thread so it can make progress,
   * or the highest priority BUFN thread if all of the tasks are in some form of BUFN
   * and this was triggered by a free.
   *
   * This is typically called when a free happens, or an alloc succeeds.
   * @param is_from_free true if a free happen.
   * @param is_for_cpu true if it was a CPU operation (free or alloc)
   */
  void wake_next_highest_priority_blocked(std::unique_lock<std::mutex> const& lock,
                                          bool const is_from_free,
                                          bool const is_for_cpu)
  {
    // 1. Find the highest priority blocked thread, for the alloc that matches
    thread_priority to_wake(-1, -1);
    bool is_to_wake_set = false;
    for (auto const& [thread_d, t_state] : threads) {
      thread_state const& state = t_state.state;
      if (state == thread_state::THREAD_BLOCKED && is_for_cpu == t_state.is_cpu_alloc) {
        thread_priority current = t_state.priority();
        if (!is_to_wake_set || to_wake < current) {
          to_wake        = current;
          is_to_wake_set = true;
        }
      }
    }
    // 2. wake up that thread
    long const thread_id_to_wake = to_wake.get_thread_id();
    if (thread_id_to_wake > 0) {
      auto const thread = threads.find(thread_id_to_wake);
      if (thread != threads.end()) {
        switch (thread->second.state) {
          case thread_state::THREAD_BLOCKED:
            transition(thread->second, thread_state::THREAD_RUNNING);
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
    } else if (is_from_free) {
      // 3. Otherwise look to see if we are in a BUFN deadlock state.
      //
      // Memory was freed and if all of the tasks are in a BUFN state,
      // then we want to wake up the highest priority one so it can make progress
      // instead of trying to split its input. But we only do this if it
      // is a different thread that is freeing memory from the one we want to wake up.
      // This is because if the threads are the same no new memory is being added
      // to what that task has access to and the task may never throw a retry and split.
      // Instead it would just keep retrying and freeing the same memory each time.
      std::map<long, long> pool_bufn_task_thread_count;
      std::map<long, long> pool_task_thread_count;
      std::unordered_set<long> bufn_task_ids;
      std::unordered_set<long> all_task_ids;
      is_in_deadlock(
        pool_bufn_task_thread_count, pool_task_thread_count, bufn_task_ids, all_task_ids, lock);
      bool const all_bufn = all_task_ids.size() == bufn_task_ids.size();
      if (all_bufn) {
        thread_priority to_wake(-1, -1);
        bool is_to_wake_set = false;
        for (auto const& [thread_id, t_state] : threads) {
          switch (t_state.state) {
            case thread_state::THREAD_BUFN: {
              if (is_for_cpu == t_state.is_cpu_alloc) {
                thread_priority current = t_state.priority();
                if (!is_to_wake_set || to_wake < current) {
                  to_wake        = current;
                  is_to_wake_set = true;
                }
              }
            } break;
            default: break;
          }
        }
        // 4. Wake up the BUFN thread if we should
        if (is_to_wake_set) {
          long const thread_id_to_wake = to_wake.get_thread_id();
          if (thread_id_to_wake > 0) {
            // Don't wake up yourself on a free. It is not adding more memory for this thread
            // to use on a retry and we might need a split instead to break a deadlock
            auto const this_id = static_cast<long>(pthread_self());
            auto const thread  = threads.find(thread_id_to_wake);
            if (thread != threads.end() && thread->first != this_id) {
              switch (thread->second.state) {
                case thread_state::THREAD_BUFN:
                  transition(thread->second, thread_state::THREAD_RUNNING);
                  thread->second.wake_condition->notify_all();
                  break;
                case thread_state::THREAD_BUFN_WAIT:
                  transition(thread->second, thread_state::THREAD_RUNNING);
                  // no need to notify anyone, we will just retry without blocking...
                  break;
                case thread_state::THREAD_BUFN_THROW:
                  // This should really never happen, this is a temporary state that is here only
                  // while the lock is held, but just in case we don't want to mess it up, or throw
                  // an exception.
                  break;
                default: {
                  std::stringstream ss;
                  ss << "internal error expected to only wake up blocked threads "
                     << thread_id_to_wake << " " << as_str(thread->second.state);
                  throw std::runtime_error(ss.str());
                }
              }
            }
          }
        }
      }
    }
  }

  bool is_thread_bufn_or_above(JNIEnv* env, full_thread_state const& state)
  {
    bool ret = false;
    if (state.pool_blocked) {
      ret = true;
    } else {
      switch (state.state) {
        case thread_state::THREAD_BLOCKED: ret = false; break;
        case thread_state::THREAD_BUFN:
          // empty we are looking for even a single thread that is not blocked
          ret = true;
          break;
        default:
          ret = env->CallStaticBooleanMethod(
            ThreadStateRegistry_jclass, isThreadBlocked_method, state.thread_id);
          break;
      }
    }
    return ret;
  }

  bool is_in_deadlock(std::map<long, long>& pool_bufn_task_thread_count,
                      std::map<long, long>& pool_task_thread_count,
                      std::unordered_set<long>& bufn_task_ids,
                      std::unordered_set<long>& all_task_ids,
                      std::unique_lock<std::mutex> const& lock)
  {
    JNIEnv* env = nullptr;
    if (jvm->GetEnv(reinterpret_cast<void**>(&env), cudf::jni::MINIMUM_JNI_VERSION) != JNI_OK) {
      throw std::runtime_error("Cloud not init JNI callbacks");
    }
    cache_thread_reg_jni(env);

    // If all of the tasks are blocked, then we are in a deadlock situation
    // and we need to wake something up. In theory if any one thread is still
    // doing something, then we are not deadlocked. But the problem is detecting
    // if a thread is blocked cheaply and accurately. We can tell if this code has
    // blocked a thread. We can also have code we control inform us if a thread is
    // blocked. We even have a callback to the JVM to see if the state of the java
    // thread indicates if it is blocked or not. But I/O in java most of the time
    // shows the thread as RUNNABLE. We also don't want to look at stack traces if
    // we can avoid it as it is expensive. The reason this matters is because of
    // python UDFs. When a python process runs to execute UDFs at least two dedicated
    // task threads are used for a single task. One will write data to the python
    // process and another will read results from it. Because both involve
    // I/O we need a solution. For now we assume that a task is blocked if any
    // one of the dedicated task threads are blocked and if all of the pool
    // threads working on that task are also blocked. This is because the pool
    // threads, even if they are blocked on I/O will eventually finish without
    // needing to worry about it.
    //
    // We also need a way to detect if we need to split the input and retry.
    // This happens when all of the tasks are also blocked until
    // further notice. So we are going to treat a task as blocked until
    // further notice if any of the dedicated threads for it are blocked until
    // further notice, or all of the pool threads working on things for it are
    // blocked until further notice.
    std::unordered_set<long> blocked_task_ids;

    // We are going to do two passes through the threads to deal with this.
    // First pass is to look at the dedicated task threads
    for (auto const& [thread_id, t_state] : threads) {
      long const task_id = t_state.task_id;
      if (task_id >= 0) {
        all_task_ids.insert(task_id);
        bool const is_bufn_plus = is_thread_bufn_or_above(env, t_state);
        if (is_bufn_plus) { bufn_task_ids.insert(task_id); }
        if (is_bufn_plus || t_state.state == thread_state::THREAD_BLOCKED) {
          blocked_task_ids.insert(task_id);
        }
      }
    }

    // Second pass is to look at the pool threads
    for (auto const& [thread_id, t_state] : threads) {
      long const is_pool_thread = t_state.task_id < 0;
      if (is_pool_thread) {
        for (auto const& task_id : t_state.pool_task_ids) {
          auto const it = pool_task_thread_count.find(task_id);
          if (it != pool_task_thread_count.end()) {
            it->second += 1;
          } else {
            pool_task_thread_count[task_id] = 1;
          }
        }

        bool const is_bufn_plus = is_thread_bufn_or_above(env, t_state);
        if (is_bufn_plus) {
          for (auto const& task_id : t_state.pool_task_ids) {
            auto const it = pool_bufn_task_thread_count.find(task_id);
            if (it != pool_bufn_task_thread_count.end()) {
              it->second += 1;
            } else {
              pool_bufn_task_thread_count[task_id] = 1;
            }
          }
        }
        if (!is_bufn_plus && t_state.state != thread_state::THREAD_BLOCKED) {
          for (auto const& task_id : t_state.pool_task_ids) {
            blocked_task_ids.erase(task_id);
          }
        }
      }
    }
    // Now if all of the tasks are blocked, then we need to break a deadlock
    return all_task_ids.size() == blocked_task_ids.size();
  }

  /**
   * Check to see if any threads need to move to BUFN. This should be
   * called when a task or shuffle thread becomes blocked so that we can
   * check to see if one of them needs to become BUFN or do a split and rollback.
   */
  void check_and_update_for_bufn(const std::unique_lock<std::mutex>& lock)
  {
    std::map<long, long> pool_bufn_task_thread_count;
    std::map<long, long> pool_task_thread_count;
    std::unordered_set<long> bufn_task_ids;
    std::unordered_set<long> all_task_ids;
    bool const need_to_break_deadlock = is_in_deadlock(
      pool_bufn_task_thread_count, pool_task_thread_count, bufn_task_ids, all_task_ids, lock);
    if (need_to_break_deadlock) {
      // Find the task thread with the lowest priority that is not already BUFN
      thread_priority to_bufn(-1, -1);
      bool is_to_bufn_set = false;
      for (auto const& [thread_id, t_state] : threads) {
        switch (t_state.state) {
          case thread_state::THREAD_BLOCKED: {
            thread_priority const& current = t_state.priority();
            if (!is_to_bufn_set || current < to_bufn) {
              to_bufn        = current;
              is_to_bufn_set = true;
            }
          } break;
          default: break;
        }
      }
      if (is_to_bufn_set) {
        long const thread_id_to_bufn = to_bufn.get_thread_id();
        auto const thread            = threads.find(thread_id_to_bufn);
        if (thread != threads.end()) {
          transition(thread->second, thread_state::THREAD_BUFN_THROW);
          thread->second.wake_condition->notify_all();
          // We are explicitly not going to update the state around BUFN
          // here, because we really want to wait for the retry to run
          // it's course instead of doing a split right away.
        }
      }
      // We now need a way to detect if we need to split the input and retry.
      // This happens when all of the tasks are also blocked until
      // further notice. So we are going to treat a task as blocked until
      // further notice if any of the dedicated threads for it are blocked until
      // further notice, or all of the pool threads working on things for it are
      // blocked until further notice.

      for (auto const& [task_id, bufn_count] : pool_bufn_task_thread_count) {
        auto const pttc = pool_task_thread_count.find(task_id);
        if (pttc != pool_task_thread_count.end() && pttc->second <= bufn_count) {
          bufn_task_ids.insert(task_id);
        }
      }

      bool const all_bufn = all_task_ids.size() == bufn_task_ids.size();

      if (all_bufn) {
        thread_priority to_wake(-1, -1);
        bool is_to_wake_set = false;
        for (auto const& [thread_id, t_state] : threads) {
          switch (t_state.state) {
            case thread_state::THREAD_BUFN: {
              thread_priority const& current = t_state.priority();
              if (!is_to_wake_set || to_wake < current) {
                to_wake        = current;
                is_to_wake_set = true;
              }
            } break;
            default: break;
          }
        }
        long const thread_id    = to_wake.get_thread_id();
        auto const found_thread = threads.find(thread_id);
        if (found_thread != threads.end()) {
          transition(found_thread->second, thread_state::THREAD_SPLIT_THROW);
          found_thread->second.wake_condition->notify_all();
        }
      }
    }
  }

  /**
   * alloc failed so handle any state changes needed with that. Blocking will
   * typically happen after this has run, and we loop around to retry the alloc
   * if the state says we should.
   */
  bool post_alloc_failed(long const thread_id, bool const is_oom, bool const likely_spill)
  {
    std::unique_lock<std::mutex> lock(state_mutex);
    return post_alloc_failed_core(thread_id, false, is_oom, true, likely_spill, lock);
  }

  bool post_alloc_failed_core(long const thread_id,
                              bool const is_for_cpu,
                              bool const is_oom,
                              bool const blocking,
                              bool const was_recursive,
                              std::unique_lock<std::mutex>& lock)
  {
    auto const thread = threads.find(thread_id);
    // only retry if this was due to an out of memory exception.
    bool ret = true;
    if (!was_recursive && thread != threads.end()) {
      if (thread->second.is_cpu_alloc != is_for_cpu) {
        std::stringstream ss;
        ss << "thread " << thread_id << " has a mismatch on CPU vs GPU post alloc "
           << as_str(thread->second.state);

        throw std::invalid_argument(ss.str());
      }

      switch (thread->second.state) {
        case thread_state::THREAD_ALLOC_FREE:
          transition(thread->second, thread_state::THREAD_RUNNING);
          break;
        case thread_state::THREAD_ALLOC:
          if (is_oom && blocking) {
            transition(thread->second, thread_state::THREAD_BLOCKED);
          } else {
            // don't block unless it is OOM on a blocking allocation
            transition(thread->second, thread_state::THREAD_RUNNING);
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

  void* do_allocate(std::size_t const num_bytes, rmm::cuda_stream_view stream) override
  {
    auto const tid = static_cast<long>(pthread_self());
    while (true) {
      bool const likely_spill = pre_alloc(tid);
      try {
        void* ret = resource->allocate(num_bytes, stream);
        post_alloc_success(tid, likely_spill, num_bytes);
        return ret;
      } catch (rmm::out_of_memory const& e) {
        // rmm::out_of_memory is what is thrown when an allocation failed
        // but there are other rmm::bad_alloc exceptions that could be
        // thrown as well, which are handled by the std::exception case.
        if (!post_alloc_failed(tid, true, likely_spill)) { throw; }
      } catch (std::exception const& e) {
        post_alloc_failed(tid, false, likely_spill);
        throw;
      }
    }
    // we should never reach this point, but just in case
    throw rmm::bad_alloc("Internal Error");
  }

  void dealloc_core(bool const is_for_cpu,
                    std::unique_lock<std::mutex>& lock,
                    std::size_t const num_bytes)
  {
    auto const tid    = static_cast<long>(pthread_self());
    auto const thread = threads.find(tid);
    if (thread != threads.end()) {
      log_status("DEALLOC", tid, thread->second.task_id, thread->second.state);
    } else {
      log_status("DEALLOC", tid, -2, thread_state::UNKNOWN);
    }

    for (auto& [thread_id, t_state] : threads) {
      // Only update state for _other_ threads. We update only other threads, for the case
      // where we are handling a free from the recursive case: when an allocation/free
      // happened while handling an allocation failure in onAllocFailed.
      //
      // If we moved all threads to *_ALLOC_FREE, after we exit the recursive state and
      // are back handling the original allocation failure, we are left with a thread
      // in a state that won't be retried in `post_alloc_failed`.
      //
      // By not changing our thread's state to THREAD_ALLOC_FREE, we keep the state
      // the same, but we still let other threads know that there was a free and they should
      // handle accordingly.
      if (t_state.thread_id != tid) {
        switch (t_state.state) {
          case thread_state::THREAD_ALLOC:
            if (is_for_cpu == t_state.is_cpu_alloc) {
              transition(t_state, thread_state::THREAD_ALLOC_FREE);
            }
            if (!is_for_cpu) { t_state.gpu_memory_allocated_bytes -= num_bytes; }
            break;
          default: break;
        }
      }
    }
    wake_next_highest_priority_blocked(lock, true, is_for_cpu);
  }

  void do_deallocate(void* p, std::size_t size, rmm::cuda_stream_view stream) override
  {
    resource->deallocate(p, size, stream);
    // deallocate success
    if (size > 0) {
      std::unique_lock<std::mutex> lock(state_mutex);
      dealloc_core(false, lock, size);
    }
  }
};

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getCurrentThreadId(JNIEnv* env, jclass)
{
  try {
    cudf::jni::auto_set_device(env);
    return static_cast<jlong>(pthread_self());
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_createNewAdaptor(
  JNIEnv* env, jclass, jlong child, jstring log_loc)
{
  JNI_NULL_CHECK(env, child, "child is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto wrapped = reinterpret_cast<rmm::mr::device_memory_resource*>(child);
    cudf::jni::native_jstring nlogloc(env, log_loc);
    std::shared_ptr<spdlog::logger> logger;
    bool is_log_enabled;
    if (nlogloc.is_null()) {
      logger         = make_logger();
      is_log_enabled = false;
    } else {
      is_log_enabled = true;
      std::string slog_loc(nlogloc.get());
      if (slog_loc == "stderr") {
        logger = make_logger(std::cerr);
      } else if (slog_loc == "stdout") {
        logger = make_logger(std::cout);
      } else {
        logger = make_logger(slog_loc);
      }
    }

    auto ret = new spark_resource_adaptor(env, wrapped, logger, is_log_enabled);
    return cudf::jni::ptr_as_jlong(ret);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_releaseAdaptor(JNIEnv* env, jclass, jlong ptr)
{
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->all_done();
    delete mr;
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_startDedicatedTaskThread(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id, jlong task_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->start_dedicated_task_thread(thread_id, task_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_poolThreadWorkingOnTasks(
  JNIEnv* env, jclass, jlong ptr, jboolean is_for_shuffle, jlong thread_id, jlongArray task_ids)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  JNI_NULL_CHECK(env, task_ids, "task_ids is null", );
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jlongArray jtask_ids(env, task_ids);
    std::unordered_set<long> task_set(jtask_ids.begin(), jtask_ids.end());
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->pool_thread_working_on_tasks(is_for_shuffle, thread_id, task_set);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_poolThreadFinishedForTasks(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id, jlongArray task_ids)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  JNI_NULL_CHECK(env, task_ids, "task_ids is null", );
  try {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jlongArray jtask_ids(env, task_ids);
    std::unordered_set<long> task_set(jtask_ids.begin(), jtask_ids.end());
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->pool_thread_finished_for_tasks(thread_id, task_set);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_removeThreadAssociation(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id, jlong task_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->remove_thread_association(thread_id, task_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_taskDone(JNIEnv* env,
                                                                                      jclass,
                                                                                      jlong ptr,
                                                                                      jlong task_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->task_done(task_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_submittingToPool(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->submitting_to_pool(thread_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_waitingOnPool(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->waiting_on_pool(thread_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_doneWaitingOnPool(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->done_waiting_on_pool(thread_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_forceRetryOOM(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id, jint num_ooms, jint oom_filter, jint skip_count)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->force_retry_oom(thread_id, num_ooms, oom_filter, skip_count);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_forceSplitAndRetryOOM(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id, jint num_ooms, jint oom_filter, jint skip_count)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->force_split_and_retry_oom(thread_id, num_ooms, oom_filter, skip_count);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_forceCudfException(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id, jint num_times)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->force_cudf_exception(thread_id, num_times);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_blockThreadUntilReady(
  JNIEnv* env, jclass, jlong ptr)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->block_thread_until_ready();
  }
  CATCH_STD(env, )
}

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getStateOf(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    return mr->get_thread_state_as_int(thread_id);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jint JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getAndResetRetryThrowInternal(JNIEnv* env,
                                                                                    jclass,
                                                                                    jlong ptr,
                                                                                    jlong task_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    return mr->get_and_reset_num_retry(task_id);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jint JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getAndResetSplitRetryThrowInternal(
  JNIEnv* env, jclass, jlong ptr, jlong task_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    return mr->get_and_reset_num_split_retry(task_id);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getAndResetBlockTimeInternal(JNIEnv* env,
                                                                                   jclass,
                                                                                   jlong ptr,
                                                                                   jlong task_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    return mr->get_and_reset_block_time(task_id);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getAndResetComputeTimeLostToRetry(
  JNIEnv* env, jclass, jlong ptr, jlong task_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    return mr->get_and_reset_lost_time(task_id);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_getAndResetGpuMaxMemoryAllocated(
  JNIEnv* env, jclass, jlong ptr, jlong task_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    return mr->get_and_reset_gpu_max_memory_allocated(task_id);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_startRetryBlock(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->start_retry_block(thread_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_endRetryBlock(
  JNIEnv* env, jclass, jlong ptr, jlong thread_id)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->end_retry_block(thread_id);
  }
  CATCH_STD(env, )
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_checkAndBreakDeadlocks(
  JNIEnv* env, jclass, jlong ptr)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->check_and_break_deadlocks();
  }
  CATCH_STD(env, )
}

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_preCpuAlloc(
  JNIEnv* env, jclass, jlong ptr, jlong amount, jboolean blocking)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    return mr->cpu_prealloc(amount, blocking);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL
Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_postCpuAllocSuccess(JNIEnv* env,
                                                                          jclass,
                                                                          jlong ptr,
                                                                          jlong addr,
                                                                          jlong amount,
                                                                          jboolean blocking,
                                                                          jboolean was_recursive)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->cpu_postalloc_success(reinterpret_cast<void*>(addr), amount, blocking, was_recursive);
  }
  CATCH_STD(env, )
}

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_postCpuAllocFailed(
  JNIEnv* env, jclass, jlong ptr, jboolean was_oom, jboolean blocking, jboolean was_recursive)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    return mr->cpu_postalloc_failed(was_oom, blocking, was_recursive);
  }
  CATCH_STD(env, 0)
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_SparkResourceAdaptor_cpuDeallocate(
  JNIEnv* env, jclass, jlong ptr, jlong addr, jlong amount)
{
  JNI_NULL_CHECK(env, ptr, "resource_adaptor is null", );
  try {
    cudf::jni::auto_set_device(env);
    auto mr = reinterpret_cast<spark_resource_adaptor*>(ptr);
    mr->cpu_dealloc(reinterpret_cast<void*>(addr), amount);
  }
  CATCH_STD(env, )
}
}
