package com.nvidia.spark.rapids.jni.kudo;

import static java.lang.Math.toIntExact;
import static java.util.Objects.requireNonNull;

import ai.rapids.cudf.HostMemoryBuffer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

public class ByteArrayOutputStreamWriter implements DataWriter {
  private static final Method ENSURE_CAPACITY;
  private static final Field BUF;
  private static final Field COUNT;

  static {
    try {
      ENSURE_CAPACITY = ByteArrayOutputStream.class.getDeclaredMethod("ensureCapacity", int.class);
      ENSURE_CAPACITY.setAccessible(true);

      BUF = ByteArrayOutputStream.class.getDeclaredField("buf");
      BUF.setAccessible(true);


      COUNT = ByteArrayOutputStream.class.getDeclaredField("count");
      COUNT.setAccessible(true);
    } catch (NoSuchMethodException | NoSuchFieldException e) {
      throw new RuntimeException("Failed to find ByteArrayOutputStream.ensureCapacity", e);
    }
  }

  private final ByteArrayOutputStream out;

  public ByteArrayOutputStreamWriter(ByteArrayOutputStream bout) {
    requireNonNull(bout, "Byte array output stream can't be null");
    this.out = bout;
  }

  @Override
  public void reserve(int size) throws IOException {
    try {
      ENSURE_CAPACITY.invoke(out, size);
    } catch (Exception e) {
      throw new RuntimeException("Failed to invoke ByteArrayOutputStream.ensureCapacity", e);
    }
  }

  @Override
  public void writeInt(int v) throws IOException {
    reserve(4 + out.size());
    out.write((v >>> 24) & 0xFF);
    out.write((v >>> 16) & 0xFF);
    out.write((v >>>  8) & 0xFF);
    out.write((v >>>  0) & 0xFF);
  }

  @Override
  public void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len) throws IOException {
    reserve(toIntExact(out.size() + len));

    try {
      byte[] buf = (byte[]) BUF.get(out);
      int count = out.size();

      src.getBytes(buf, count, srcOffset, len);
      COUNT.setInt(out, toIntExact(count + len));
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void flush() throws IOException {
  }

  @Override
  public void write(byte[] arr, int offset, int length) throws IOException {
    out.write(arr, offset, length);
  }
}
