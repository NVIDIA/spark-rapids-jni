package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.*;

import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

public class RefUtils {
  private static Method MAKE_CUDF_COLUMN_VIEW;
  private static Method FROM_VIEW_WITH_CONTIGUOUS_ALLOCATION;
  private static Constructor<ContiguousTable> CONTIGUOUS_TABLE_CONSTRUCTOR;
  private static Method COPY_FROM_STREAM;

  static {
    try {
      MAKE_CUDF_COLUMN_VIEW = ColumnView.class.getDeclaredMethod("makeCudfColumnView",
            int.class, int.class, long.class, long.class, long.class, long.class, int.class,
          int.class, long[].class);
      MAKE_CUDF_COLUMN_VIEW.setAccessible(true);

      FROM_VIEW_WITH_CONTIGUOUS_ALLOCATION = ColumnVector.class.getDeclaredMethod(
          "fromViewWithContiguousAllocation",
          long.class, DeviceMemoryBuffer.class);
      FROM_VIEW_WITH_CONTIGUOUS_ALLOCATION.setAccessible(true);

      CONTIGUOUS_TABLE_CONSTRUCTOR = ContiguousTable.class.getDeclaredConstructor(Table.class,
          DeviceMemoryBuffer.class);
      CONTIGUOUS_TABLE_CONSTRUCTOR.setAccessible(true);

      COPY_FROM_STREAM = HostMemoryBuffer.class.getDeclaredMethod("copyFromStream",
          long.class, InputStream.class, long.class);
      COPY_FROM_STREAM.setAccessible(true);
    } catch (NoSuchMethodException e) {
      throw new RuntimeException(e);
    }
  }

  public static long makeCudfColumnView(int typeId, int scale, long dataAddress, long dataLen,
      long offsetsAddress, long validityAddress, int nullCount, int rowCount, long[] childrenView) {
    try {
      return (long) MAKE_CUDF_COLUMN_VIEW.invoke(null, typeId, scale, dataAddress, dataLen,
          offsetsAddress, validityAddress, nullCount, rowCount, childrenView);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public static ColumnVector fromViewWithContiguousAllocation(long colView, DeviceMemoryBuffer buffer) {
    try {
      return (ColumnVector) FROM_VIEW_WITH_CONTIGUOUS_ALLOCATION.invoke(null, colView, buffer);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public static ContiguousTable makeContiguousTable(Table table, DeviceMemoryBuffer buffer) {
    try {
      return CONTIGUOUS_TABLE_CONSTRUCTOR.newInstance(table, buffer);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public static void copyFromStream(HostMemoryBuffer buffer, long offset, InputStream in,
      long len) {
    try {
      COPY_FROM_STREAM.invoke(buffer, offset, in, len);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }
}
