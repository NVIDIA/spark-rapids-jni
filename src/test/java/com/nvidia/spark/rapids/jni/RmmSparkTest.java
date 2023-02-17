/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.RmmCudaMemoryResource;
import ai.rapids.cudf.RmmDeviceMemoryResource;
import ai.rapids.cudf.RmmEventHandler;
import ai.rapids.cudf.RmmLimitingResourceAdaptor;
import ai.rapids.cudf.RmmTrackingResourceAdaptor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

public class RmmSparkTest {
  @BeforeEach
  public void setup() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }

  @AfterEach
  public void teardown() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }

  public interface TaskThreadOp<T> {
    T doIt();
  }

  public static class TaskThread extends Thread {
    private final String name;
    private final boolean isShuffle;
    private long threadId = -1;
    private long taskId = 100;

    public TaskThread(String name, long taskId) {
      this(name, false);
      this.taskId = taskId;
    }

    public TaskThread(String name, boolean isShuffle) {
      super(name);
      this.name = name;
      this.isShuffle = isShuffle;
    }

    public synchronized long getThreadId() {
      return threadId;
    }

    private LinkedBlockingQueue<TaskThreadOp> queue = new LinkedBlockingQueue<>();

    public void initialize() throws ExecutionException, InterruptedException, TimeoutException {
      setDaemon(true);
      start();
      Future<Void> waitForStart = doIt(new TaskThreadOp<Void>() {
        @Override
        public Void doIt() {
          if (isShuffle) {
            RmmSpark.associateCurrentThreadWithShuffle();
          } else {
            RmmSpark.associateCurrentThreadWithTask(taskId);
          }
          return null;
        }

        @Override
        public String toString() {
          return "INIT TASK " + name + " " + (isShuffle ? "SHUFFLE" : ("TASK " + taskId));
        }
      });
      System.err.println("WAITING FOR STARTUP (" + name + ")");
      waitForStart.get(1000, TimeUnit.MILLISECONDS);
      System.err.println("THREAD IS READY TO GO (" + name + ")");
    }

    public void pollForState(RmmSparkThreadState state, long l, TimeUnit tu) throws TimeoutException, InterruptedException {
      long start = System.nanoTime();
      long timeoutAfter = start + tu.toNanos(l);
      RmmSparkThreadState currentState = null;
      while (System.nanoTime() <= timeoutAfter) {
        currentState = RmmSpark.getStateOf(threadId);
        if (currentState == state) {
          return;
        }
        // Yes we are essentially doing a busy wait...
        Thread.sleep(10);
      }
      throw new TimeoutException(name + " WAITING FOR STATE " + state + " BUT STATE IS " + currentState);
    }

    private static class TaskThreadDoneOp implements TaskThreadOp<Void>, Future<Object> {
      private TaskThread wrapped;

      TaskThreadDoneOp(TaskThread td) {
        wrapped = td;
      }

      @Override
      public String toString() {
        return "TASK DONE";
      }

      @Override
      public Void doIt() {
        return null;
      }

      @Override
      public boolean cancel(boolean b) {
        return false;
      }

      @Override
      public boolean isCancelled() {
        return false;
      }

      @Override
      public boolean isDone() {
        return !wrapped.isAlive();
      }

      @Override
      public Object get() throws InterruptedException, ExecutionException {
        throw new RuntimeException("FUTURE NEEDS A TIMEOUT. THIS IS A TEST!");
      }

      @Override
      public Object get(long l, TimeUnit timeUnit) throws InterruptedException, ExecutionException, TimeoutException {
        System.err.println("WAITING FOR THREAD DONE " + l + " " + timeUnit);
        wrapped.join(timeUnit.toMillis(l));
        return null;
      }
    }

    public Future done() {
      TaskThreadDoneOp op = new TaskThreadDoneOp(this);
      queue.offer(op);
      return op;
    }

    private static class TaskThreadTrackingOp<T> implements TaskThreadOp<T>, Future<T> {
      private final TaskThreadOp<T> wrapped;
      private boolean done = false;
      private Throwable t = null;
      private T ret = null;


      @Override
      public String toString() {
        return wrapped.toString();
      }

      TaskThreadTrackingOp(TaskThreadOp<T> td) {
        wrapped = td;
      }

      @Override
      public T doIt() {
        try {
          T tmp = wrapped.doIt();
          synchronized (this) {
            ret = tmp;
            return ret;
          }
        } catch (Throwable t) {
          synchronized (this) {
            this.t = t;
          }
          return null;
        } finally {
          synchronized (this) {
            done = true;
            this.notifyAll();
          }
        }
      }

      @Override
      public boolean cancel(boolean b) {
        return false;
      }

      @Override
      public boolean isCancelled() {
        return false;
      }

      @Override
      public synchronized boolean isDone() {
        return done;
      }

      @Override
      public synchronized T get() throws InterruptedException, ExecutionException {
        throw new RuntimeException("This is a test you should always have timeouts...");
      }

      @Override
      public synchronized T get(long l, TimeUnit timeUnit) throws InterruptedException, ExecutionException, TimeoutException {
        if (!done) {
          System.err.println("WAITING " + l + " " + timeUnit + " FOR '" + wrapped + "'");
          wait(timeUnit.toMillis(l));
          if (!done) {
            throw new TimeoutException();
          }
        }
        if (t != null) {
          throw new ExecutionException(t);
        }
        return ret;
      }
    }

    public <T> Future<T> doIt(TaskThreadOp<T> op) {
      if (!isAlive()) {
        throw new IllegalStateException("Thread is already done...");
      }
      TaskThreadTrackingOp<T> tracking = new TaskThreadTrackingOp<>(op);
      queue.offer(tracking);
      return tracking;
    }

    public Future<Void> blockUntilReady() {
      return doIt(new TaskThreadOp<Void>() {
        @Override
        public Void doIt() {
          RmmSpark.blockThreadUntilReady();
          return null;
        }

        @Override
        public String toString() {
          return "BLOCK UNTIL THREAD IS READY";
        }
      });
    }

    @Override
    public void run() {
      try {
        synchronized (this) {
          threadId = RmmSpark.getCurrentThreadId();
        }
        System.err.println("INSIDE THREAD RUNNING (" + name + ")");
        while (true) {
          TaskThreadOp op = queue.poll(1000, TimeUnit.MILLISECONDS);
          System.err.println("GOT '" + op + "' ON " + name);
          if (op instanceof TaskThreadDoneOp) {
            return;
          }
          // null is returned from the queue on a timeout
          if (op != null) {
            op.doIt();
            System.err.println("'" + op + "' FINISHED ON " + name);
          }
        }
      } catch (Throwable t) {
        System.err.println("THROWABLE CAUGHT IN " + name);
        t.printStackTrace(System.err);
      } finally {
        RmmSpark.removeCurrentThreadAssociation();
        System.err.println("THREAD EXITING " + name);
      }
    }
  }

  @Test
  public void testBasicInitAndTeardown() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
  }

  @Test
  public void testInsertOOMs() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
    long threadId = RmmSpark.getCurrentThreadId();
    long taskid = 0; // This is arbitrary
    assertEquals(RmmSparkThreadState.UNKNOWN, RmmSpark.getStateOf(threadId));
    RmmSpark.associateThreadWithTask(threadId, taskid);
    assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(threadId));
    try {
      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(threadId));

      // Force an exception
      RmmSpark.forceRetryOOM(threadId);
      // No change in the state after a force
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(threadId));
      assertThrows(RetryOOM.class, () -> Rmm.alloc(100).close());

      // Verify that injecting OOM does not cause the block to actually happen or
      // the state to change
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(threadId));
      RmmSpark.blockThreadUntilReady();

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(threadId));

      // Force another exception
      RmmSpark.forceSplitAndRetryOOM(threadId);
      // No change in state after force
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(threadId));
      assertThrows(SplitAndRetryOOM.class, () -> Rmm.alloc(100).close());
      // Verify that injecting OOM does not cause the block to actually happen
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(threadId));
      RmmSpark.blockThreadUntilReady();

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(threadId));
    } finally {
      RmmSpark.taskDone(taskid);
    }
    assertEquals(RmmSparkThreadState.UNKNOWN, RmmSpark.getStateOf(threadId));
  }

  @Test
  public void testReentrantAssociateThread() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
    long threadId = 100;
    long taskId = 1;
    try {
      RmmSpark.associateThreadWithTask(threadId, taskId);
      RmmSpark.associateThreadWithTask(threadId, taskId);
      RmmSpark.removeThreadAssociation(threadId);
      // Not removing twice because we don't have to match up the counts so it fits with how
      // the GPU semaphore is used.
      RmmSpark.associateThreadWithShuffle(threadId);
      RmmSpark.associateThreadWithShuffle(threadId);
      RmmSpark.removeThreadAssociation(threadId);
      RmmSpark.removeThreadAssociation(threadId);
    } finally {
      RmmSpark.taskDone(taskId);
    }
  }

  @Test
  public void testAssociateThread() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 512 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
    long threadIdOne = 200;
    long threadIdTwo = 300;
    long taskId = 2;
    long otherTaskId = 3;
    try {
      RmmSpark.associateThreadWithTask(threadIdOne, taskId);
      assertThrows(CudfException.class, () -> RmmSpark.associateThreadWithShuffle(threadIdOne));
      assertThrows(CudfException.class, () -> RmmSpark.associateThreadWithTask(threadIdOne, otherTaskId));

      RmmSpark.associateThreadWithShuffle(threadIdTwo);
      assertThrows(CudfException.class, () -> RmmSpark.associateThreadWithTask(threadIdTwo, otherTaskId));
      // Remove the association
      RmmSpark.removeThreadAssociation(threadIdTwo);
      // Add in a new association
      RmmSpark.associateThreadWithTask(threadIdTwo, taskId);
    } finally {
      RmmSpark.taskDone(taskId);
      RmmSpark.taskDone(otherTaskId);
    }
  }


  static class AllocOnAnotherThread implements AutoCloseable {
    final TaskThread thread;
    final long size;
    DeviceMemoryBuffer b = null;
    Future<Void> fb;
    Future<Void> fc = null;

    public AllocOnAnotherThread(TaskThread thread, long size) {
      this.thread = thread;
      this.size = size;
      fb = thread.doIt(new TaskThreadOp<Void>() {
        @Override
        public Void doIt() {
          doAlloc();
          return null;
        }

        @Override
        public String toString() {
          return "ALLOC(" + size + ")";
        }
      });
    }

    public void waitForAlloc() throws ExecutionException, InterruptedException, TimeoutException {
      fb.get(1000, TimeUnit.MILLISECONDS);
    }

    public void freeOnThread() throws ExecutionException, InterruptedException, TimeoutException {
      if (fc != null) {
        throw new IllegalStateException("free called multiple times");
      }

      fc = thread.doIt(new TaskThreadOp<Void>() {
        @Override
        public Void doIt() {
          close();
          return null;
        }

        @Override
        public String toString() {
          return "FREE(" + size + ")";
        }
      });
    }

    public void waitForFree() throws ExecutionException, InterruptedException, TimeoutException {
      if (fc == null) {
        freeOnThread();
      }
      fc.get(1000, TimeUnit.MILLISECONDS);
    }

    public void freeAndWait() throws ExecutionException, InterruptedException, TimeoutException {
      waitForFree();
    }

    private Void doAlloc() {
      DeviceMemoryBuffer tmp = Rmm.alloc(size);
      synchronized (this) {
        b = tmp;
      }
      return null;
    }

    @Override
    public synchronized void close() {
      if (b != null) {
        b.close();
        b = null;
      }
    }
  }

  void setupRmmForTestingWithLimits(long maxAllocSize) {
    // Rmm.initialize is not going to limit allocations without a pool, so we
    // need to set it up ourselves.
    RmmDeviceMemoryResource resource = null;
    boolean succeeded = false;
    try {
      resource = new RmmCudaMemoryResource();
      resource = new RmmLimitingResourceAdaptor<>(resource, maxAllocSize, 256);
      resource = new RmmTrackingResourceAdaptor<>(resource, 256);
      Rmm.setCurrentDeviceResource(resource, null, false);
      succeeded = true;
    } finally {
      if (!succeeded && resource != null) {
        resource.close();
      }
    }
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
  }

  @Test
  public void testBasicBlocking() throws ExecutionException, InterruptedException, TimeoutException {
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    TaskThread taskOne = new TaskThread("TEST THREAD ONE", 1);
    TaskThread taskTwo = new TaskThread("TEST THREAD TWO", 2);
    taskOne.initialize();
    taskTwo.initialize();
    try {
      long tOneId = taskOne.getThreadId();
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(tOneId));

      long tTwoId = taskTwo.getThreadId();
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(tTwoId));

      try (AllocOnAnotherThread firstOne = new AllocOnAnotherThread(taskOne, 5 * 1024 * 1024)) {
        firstOne.waitForAlloc();
        // This one should block
        try (AllocOnAnotherThread secondOne = new AllocOnAnotherThread(taskTwo, 6 * 1024 * 1024)) {
          taskTwo.pollForState(RmmSparkThreadState.TASK_BLOCKED, 1000, TimeUnit.MILLISECONDS);
          // Free the first allocation to wake up the second task...
          firstOne.freeAndWait();
          secondOne.waitForAlloc();
          secondOne.freeAndWait();
        }
      }

    } finally {
      taskOne.done();
      taskTwo.done();
    }
  }

  @Test
  public void testShuffleBlocking() throws ExecutionException, InterruptedException, TimeoutException {
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    TaskThread shuffleOne = new TaskThread("TEST THREAD SHUFFLE", true);
    TaskThread taskOne = new TaskThread("TEST THREAD ONE", 1);
    TaskThread taskTwo = new TaskThread("TEST THREAD TWO", 2);

    shuffleOne.initialize();
    taskOne.initialize();
    taskTwo.initialize();
    try {
      long sOneId = shuffleOne.getThreadId();
      assertEquals(RmmSparkThreadState.SHUFFLE_RUNNING, RmmSpark.getStateOf(sOneId));

      long tOneId = taskOne.getThreadId();
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(tOneId));

      long tTwoId = taskTwo.getThreadId();
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(tTwoId));

      try (AllocOnAnotherThread firstOne = new AllocOnAnotherThread(taskOne, 5 * 1024 * 1024)) {
        firstOne.waitForAlloc();
        // This one should block
        try (AllocOnAnotherThread secondOne = new AllocOnAnotherThread(taskTwo, 6 * 1024 * 1024)) {
          taskTwo.pollForState(RmmSparkThreadState.TASK_BLOCKED, 1000, TimeUnit.MILLISECONDS);

          // Make sure that shuffle has higher priority than do tasks...
          try (AllocOnAnotherThread thirdOne = new AllocOnAnotherThread(shuffleOne, 6 * 1024 * 1024)) {
            shuffleOne.pollForState(RmmSparkThreadState.SHUFFLE_BLOCKED, 1000, TimeUnit.MILLISECONDS);
            // Free the first allocation to wake up the shuffle thread, but not the second task yet...
            firstOne.freeAndWait();
            thirdOne.waitForAlloc();
            thirdOne.freeAndWait();
          }
          secondOne.waitForAlloc();
          secondOne.freeAndWait();
        }
      }
    } finally {
      shuffleOne.done();
      taskOne.done();
      taskTwo.done();
    }
  }

  @Test
  public void testBasicBUFN() throws ExecutionException, InterruptedException, TimeoutException {
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    // A task id of 3 is higher than a task id of 2, so it should be a lower
    // priority and become BUFN ahead of taskTwo.
    TaskThread taskThree = new TaskThread("TEST THREAD ONE", 3);
    TaskThread taskTwo = new TaskThread("TEST THREAD TWO", 2);
    taskThree.initialize();
    taskTwo.initialize();
    try {
      long tThreeId = taskThree.getThreadId();
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(tThreeId));

      long tTwoId = taskTwo.getThreadId();
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(tTwoId));

      try (AllocOnAnotherThread allocThreeOne = new AllocOnAnotherThread(taskThree, 5 * 1024 * 1024)) {
        allocThreeOne.waitForAlloc();
        try (AllocOnAnotherThread allocTwoOne = new AllocOnAnotherThread(taskTwo, 3 * 1024 * 1024)) {
          allocTwoOne.waitForAlloc();

          // This one should block
          try (AllocOnAnotherThread allocTwoTwo = new AllocOnAnotherThread(taskTwo, 3 * 1024 * 1024)) {
            taskTwo.pollForState(RmmSparkThreadState.TASK_BLOCKED, 1000, TimeUnit.MILLISECONDS);

            try (AllocOnAnotherThread allocThreeTwo = new AllocOnAnotherThread(taskThree, 4 * 1024 * 1024)) {
              // This one should be able to allocate because there is not enough memory, but
              // now all the threads would be blocked, so the lowest priority thread is going to
              // become BUFN
              taskThree.pollForState(RmmSparkThreadState.TASK_BUFN_WAIT, 1000, TimeUnit.MILLISECONDS);
              try {
                allocThreeTwo.waitForAlloc();
                fail("ALLOC AFTER BUFN SHOULD HAVE THROWN...");
              } catch (ExecutionException ee) {
                assert(ee.getCause() instanceof RetryOOM);
              }
              // allocOneTwo cannot be freed, nothing was allocated because it threw an exception.
              allocThreeOne.freeAndWait();
              Future<Void> f = taskThree.blockUntilReady();
              taskThree.pollForState(RmmSparkThreadState.TASK_BUFN, 1000, TimeUnit.MILLISECONDS);

              // taskOne should only wake up after we finish task 2
              // Task two is now able to alloc
              allocTwoTwo.freeAndWait();
              allocTwoOne.freeAndWait();
              // Task two has freed things, but is still not done, so task one will stay blocked...
              taskTwo.pollForState(RmmSparkThreadState.TASK_RUNNING, 1000, TimeUnit.MILLISECONDS);
              taskThree.pollForState(RmmSparkThreadState.TASK_BUFN, 1000, TimeUnit.MILLISECONDS);

              taskTwo.done().get(1000, TimeUnit.MILLISECONDS);
              // Now that task two is done see if task one is running again...
              taskThree.pollForState(RmmSparkThreadState.TASK_RUNNING, 1000, TimeUnit.MILLISECONDS);
              // Now we could finish trying our allocations, but this is good enough...
            }
          }
        }
      }
    } finally {
      taskThree.done();
      taskTwo.done();
    }
  }

  @Test
  public void testBUFNSplitAndRetrySingleThread() throws ExecutionException, InterruptedException, TimeoutException {
    // We are doing ths one single threaded.
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);

    TaskThread taskOne = new TaskThread("TEST THREAD ZERO", 0);
    taskOne.initialize();
    try {
      long threadId = taskOne.getThreadId();
      assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(threadId));
      try (AllocOnAnotherThread one = new AllocOnAnotherThread(taskOne, 5 * 1024 * 1024)) {
        one.waitForAlloc();
        try (AllocOnAnotherThread two = new AllocOnAnotherThread(taskOne, 6 * 1024 * 1024)) {
          two.waitForAlloc();
          fail("Expect that allocating more memory than is allowed would fail");
        } catch (ExecutionException oom) {
          assert oom.getCause() instanceof SplitAndRetryOOM : oom.toString();
        }
        // This should not block...
        taskOne.blockUntilReady();
        assertEquals(RmmSparkThreadState.TASK_RUNNING, RmmSpark.getStateOf(threadId));
        // Now we try to allocate with half the data.
        try (AllocOnAnotherThread secondTry = new AllocOnAnotherThread(taskOne, 3 * 1024 * 1024)) {
          secondTry.waitForAlloc();
        }
      }
    } finally {
      taskOne.done();
    }
  }

  @Test
  public void testInsertMultipleOOMs() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 10 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
    long threadId = RmmSpark.getCurrentThreadId();
    long taskid = 0; // This is arbitrary
    RmmSpark.associateThreadWithTask(threadId, taskid);
    try {
      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();

      // Force an exception
      int numRetryOOMs = 3;
      RmmSpark.forceRetryOOM(threadId, numRetryOOMs);
      for (int i = 0; i < numRetryOOMs; i++) {
        assertThrows(RetryOOM.class, () -> Rmm.alloc(100).close());
        // Verify that injecting OOM does not cause the block to actually happen
        RmmSpark.blockThreadUntilReady();
      }

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();

      // Force another exception
      int numSplitAndRetryOOMs = 5;
      RmmSpark.forceSplitAndRetryOOM(threadId, numSplitAndRetryOOMs);
      for (int i = 0; i < numSplitAndRetryOOMs; i++) {
        assertThrows(SplitAndRetryOOM.class, () -> Rmm.alloc(100).close());
        // Verify that injecting OOM does not cause the block to actually happen
        RmmSpark.blockThreadUntilReady();
      }

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
    } finally {
      RmmSpark.removeThreadAssociation(threadId);
    }
  }

  @Test
  public void testCudfException() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 10 * 1024 * 1024);
    RmmSpark.setEventHandler(new BaseRmmEventHandler(), "stderr");
    long threadId = RmmSpark.getCurrentThreadId();
    long taskid = 0; // This is arbitrary
    RmmSpark.associateThreadWithTask(threadId, taskid);
    try {
      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();

      // Force an exception
      int numCudfExceptions = 3;
      RmmSpark.forceCudfException(threadId, numCudfExceptions);
      for (int i = 0; i < numCudfExceptions; i++) {
        assertThrows(CudfException.class, () -> Rmm.alloc(100).close());
        // Verify that injecting OOM does not cause the block to actually happen
        RmmSpark.blockThreadUntilReady();
      }

      // Allocate something small and verify that it works...
      Rmm.alloc(100).close();
    } finally {
      RmmSpark.removeThreadAssociation(threadId);
    }
  }

  @Test
  public void retryWatchdog() {
    // 10 MiB
    setupRmmForTestingWithLimits(10 * 1024 * 1024);
    long threadId = RmmSpark.getCurrentThreadId();
    long taskid = 0; // This is arbitrary
    long numRetries = 0;
    RmmSpark.associateThreadWithTask(threadId, taskid);
    long startTime = System.nanoTime();
    try (DeviceMemoryBuffer filler = Rmm.alloc(9 * 1024 * 1024)) {
      while (numRetries < 10000) {
        try {
          Rmm.alloc(2 * 1024 * 1024).close();
          fail("overallocation should have failed");
        } catch (RetryOOM room) {
          fail("only a split and retry should be thrown...");
        } catch (SplitAndRetryOOM sroom) {
          // The block should be a noop, but this is really about measuring
          // overhead so include the callback that might, or might not happen
          // This does not include any GC that might happen.
          RmmSpark.blockThreadUntilReady();
          numRetries++;
        }
      }
      fail("retried too many times " + numRetries);
    } catch (OutOfMemoryError oom) {
      // The 500 is hard coded in the code below
      assertEquals(500, numRetries);
    } finally {
      RmmSpark.removeThreadAssociation(threadId);
    }
    long endTime = System.nanoTime();
    System.err.println("Took " + (endTime - startTime) + "ns to retry 500 times...");
  }

  private static class BaseRmmEventHandler implements RmmEventHandler {
    @Override
    public long[] getAllocThresholds() {
      return null;
    }

    @Override
    public long[] getDeallocThresholds() {
      return null;
    }

    @Override
    public void onAllocThreshold(long totalAllocSize) {
    }

    @Override
    public void onDeallocThreshold(long totalAllocSize) {
    }

    @Override
    public boolean onAllocFailure(long sizeRequested, int retryCount) {
      // This is just a test for now, no spilling...
      return false;
    }
  }
}
