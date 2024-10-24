package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;

import java.util.Arrays;
import java.util.Iterator;
import java.util.function.Function;
import java.util.function.LongConsumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class TableUtils {
  public static Schema schemaOf(Table t) {
    Schema.Builder builder = Schema.builder();

    for (int i = 0; i < t.getNumberOfColumns(); i++) {
      ColumnVector cv = t.getColumn(i);
      addToSchema(cv, "col_" + i + "_", builder);
    }

    return builder.build();
  }

  public static void addToSchema(ColumnView cv, String namePrefix, Schema.Builder builder) {
    toSchemaInner(cv, 0, namePrefix, builder);
  }

  private static int toSchemaInner(ColumnView cv, int idx, String namePrefix,
      Schema.Builder builder) {
    String name = namePrefix + idx;

    Schema.Builder thisBuilder = builder.addColumn(cv.getType(), name);
    int lastIdx = idx;
    for (int i = 0; i < cv.getNumChildren(); i++) {
      lastIdx = toSchemaInner(cv.getChildColumnView(i), lastIdx + 1, namePrefix,
          thisBuilder);
    }

    return lastIdx;
  }

  public static void addToSchema(HostColumnVectorCore cv, String namePrefix, Schema.Builder builder) {
    toSchemaInner(cv, 0, namePrefix, builder);
  }

  private static int toSchemaInner(HostColumnVectorCore cv, int idx, String namePrefix,
      Schema.Builder builder) {
    String name = namePrefix + idx;

    Schema.Builder thisBuilder = builder.addColumn(cv.getType(), name);
    int lastIdx = idx;
    for (int i=0; i < cv.getNumChildren(); i++) {
      lastIdx = toSchemaInner(cv.getChildColumnView(i), lastIdx + 1, namePrefix, thisBuilder);
    }

    return lastIdx;
  }

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

  /**
   * This method returns the allocation size of the validity vector which is 64-byte aligned
   * e.g. getValidityAllocationSizeInBytes(5) => 64 bytes
   * getValidityAllocationSizeInBytes(14) => 64 bytes
   * getValidityAllocationSizeInBytes(65) => 128 bytes
   */
  static long getValidityAllocationSizeInBytes(long rows) {
    long numBytes = getValidityLengthInBytes(rows);
    return ((numBytes + 63) / 64) * 64;
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
