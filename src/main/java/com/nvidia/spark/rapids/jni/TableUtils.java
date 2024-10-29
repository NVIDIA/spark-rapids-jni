package com.nvidia.spark.rapids.jni;

import java.util.Arrays;
import java.util.Iterator;
import java.util.function.Function;
import java.util.function.LongConsumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class TableUtils {

  public static void ensure(boolean condition, String message) {
    if (!condition) {
      throw new IllegalArgumentException(message);
    }
  }

  public static void ensure(boolean condition, Supplier<String> messageSupplier) {
    if (!condition) {
      throw new IllegalArgumentException(messageSupplier.get());
    }
  }

  /**
   * This method returns the length in bytes needed to represent X number of rows
   * e.g. getValidityLengthInBytes(5) => 1 byte
   * getValidityLengthInBytes(7) => 1 byte
   * getValidityLengthInBytes(14) => 2 bytes
   */
  public static long getValidityLengthInBytes(long rows) {
    return (rows + 7) / 8;
  }

  public static <R extends AutoCloseable, T> T closeIfException(R resource, Function<R, T> function) {
    try {
      return function.apply(resource);
    } catch (Exception e) {
      if (resource != null) {
        try {
          resource.close();
        } catch (Exception inner) {
          // ignore
        }
      }
      throw e;
    }
  }

  public static <R extends AutoCloseable> void closeQuietly(Iterator<R> resources) {
    while (resources.hasNext()) {
      try {
        resources.next().close();
      } catch (Exception e) {
        // ignore
      }
    }
  }

  public static <R extends AutoCloseable> void closeQuietly(R... resources) {
    closeQuietly(Arrays.stream(resources).collect(Collectors.toList()));
  }

  public static <R extends AutoCloseable> void closeQuietly(Iterable<R> resources) {
    closeQuietly(resources.iterator());
  }

  public static <T> T withTime(Supplier<T> task, LongConsumer timeConsumer) {
    long now = System.nanoTime();
    T ret = task.get();
    timeConsumer.accept(System.nanoTime() - now);
    return ret;
  }
}
