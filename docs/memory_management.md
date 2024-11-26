## Memory management

The Spark-RAPIDS plugin manages device memory to effectively allocate the limited device memory resource among concurrent tasks.
For memory management, the plugin tracks every device memory allocation and de-allocation request during processing.
While there is enough memory available, the allocation request succeeds and the task continues processing.
However, when the allocation request cannot succeed due to lack of memory, the plugin pauses that thread. When all of the active tasks have at least one thread paused, the plugin starts to roll back some of those paused threads to points where all of their input data is spillable, and let the other threads try to complete. If every thread except one has been rolled back and the one remaining thread cannot still make progress, then pluging picks up one thread to split its input and try again.

### State machine for OOM handler

The Spark-RAPIDS plugin keeps track of the state of the individual threads. Note that one Spark task can use multiple threads during execution.

The thread can have one of these states at a time:

- `UNKNOWN`: the thread has not been registered with the tracking system.
- `THREAD_RUNNING`: the thread is running normally.
- `THREAD_ALLOC`: the thread has initiated a memory allocation.
- `THREAD_ALLOC_FREE`: the thread has requested a memory free before the allocation completes.
- `THREAD_BLOCKED`: the allocation is blocked due to lack of memory. The thread is waiting for enough memory to be available.
- `THREAD_BUFN_THROW`: a deadlock has been detected as all threads are blocked, and this thread has been selected to roll back to the point where all its data is spillable.
- `THREAD_BUFN_WAIT`: the thread has initiated the rollback.
- `THREAD_BUFN`: the thread has rolled back and is now blocked until further notice (BUFN). The task will be unblocked once high priority tasks release enough memory.
- `THREAD_SPLIT_THROW`: a deadlock has been detected as all threads are BUFN, and this thread has been selected to roll back, split its input, and retry.
- `THREAD_REMOVE_THROW`: the task has been unregistered while blocked.

The thread state can change based on the diagram below. Note that the thread state can transition from any state to `UNKNOWN`, but it is omitted in the diagram for brevity.

![alt text](img/memory_state_machine.png "Thread state machine")

### Thread priority

The Spark-RAPIDS plugin uses the thread priority when it needs to break ties between threads. See the [Deadlock busting](#deadlock-busting) section below for an example use case. The thread priority is currently decoupled with the query priority. That is, the threads processing a high priority query do not necessarily have the same high priority. Instead, each task thread is assigned a priority based on their `task_id` and `thread_id`. Shuffle threads have the highest priority, and thus are always prioritized over task threads. This is because other task threads may depend on shuffle indirectly, and this lets us avoid situations of priority inversion. In the future, we may consider taking the query priority into the thread priority.

### Deadlock busting

The deadlock can occur when every active task has at least one thread that is either directly blocked on a memory allocation or indirectly blocked by shuffle being blocked on a memory allocation. When this happens, the lowest priority thread (see the above [Thread priority](thread-priority) section for the thread priority) is selected to break the deadlock. There are two cases of the deadlock.

1) All threads are blocked and there is at least one thread in the `THREAD_BLOCKED` state. In this case, the lowest priority thread is selected among `THREAD_BLOCKED` threads to break the deadlock. The thread selected transitions its state to `THREAD_BUFN_THROW` and initiates the rollback-and-retry process. After the rollback, all input data of the thread will be spillable and the thread will block before allocating more GPU memory until enough memory is freed up for other threads.
2) If all threads are blocked and are in the `THREAD_BUFN` state, the lowest priority thread is selected to split its input first and then retry with a smaller input. The thread selected transitions its state to `THREAD_SPLIT_THROW` and initiates the rollback-split-and-retry process.

If the thread selected is a task thread and its priority is not the highest priority, the thread will transition its state into the `THREAD_BUFN_THROW` state. Any threads that was just marked as `THREAD_BUFN_THROW` will be awaken to start the rollback process and initiate the retry. After the rollback, all input data of the thread will be spillable and the thread will block before allocating more GPU memory until enough memory is freed up for other threads.
