/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.jni.kudo;

import ai.rapids.cudf.HostMemoryBuffer;

import java.io.ByteArrayOutputStream;
import java.util.Arrays;

import static java.util.Objects.requireNonNull;

/**
 * This class extends {@link ByteArrayOutputStream} to provide some internal methods to save copy.
 */
public class OpenByteArrayOutputStream extends ByteArrayOutputStream {

  /**
   * Creates a new byte array output stream. The buffer capacity is
   * initially 32 bytes, though its size increases if necessary.
   */
  public OpenByteArrayOutputStream() {
    this(32);
  }

  /**
   * Creates a new byte array output stream, with a buffer capacity of
   * the specified size, in bytes.
   *
   * @param   size   the initial size.
   * @exception  IllegalArgumentException if size is negative.
   */
  public OpenByteArrayOutputStream(int size) {
    super(size);
  }

  /**
   * Get underlying byte array.
   */
  public byte[] getBuf() {
    return buf;
  }

  /**
   * Get actual number of bytes that have been written to this output stream.
   * @return Number of bytes written to this output stream. Note that this maybe smaller than length of
   *      {@link OpenByteArrayOutputStream#getBuf()}.
   */
  public int getCount() {
    return count;
  }

  /**
   * Increases the capacity if necessary to ensure that it can hold
   * at least the number of elements specified by the minimum
   * capacity argument.
   *
   * <br/>
   *
   * This code is copied from jdk's implementation.
   *
   * @param minCapacity the desired minimum capacity
   * @throws OutOfMemoryError if {@code minCapacity < 0}.  This is
   * interpreted as a request for the unsatisfiably large capacity
   * {@code (long) Integer.MAX_VALUE + (minCapacity - Integer.MAX_VALUE)}.
   */
  public void reserve(int minCapacity) {
    // overflow-conscious code
    if (minCapacity - buf.length > 0)
      grow(minCapacity);
  }

  /**
   * The maximum size of array to allocate.
   * Some VMs reserve some header words in an array.
   * Attempts to allocate larger arrays may result in
   * OutOfMemoryError: Requested array size exceeds VM limit
   */
  private static final int MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;

  /**
   * Increases the capacity to ensure that it can hold at least the
   * number of elements specified by the minimum capacity argument.
   *
   * @param minCapacity the desired minimum capacity
   */
  private void grow(int minCapacity) {
    // overflow-conscious code
    int oldCapacity = buf.length;
    int newCapacity = oldCapacity << 1;
    if (newCapacity - minCapacity < 0)
      newCapacity = minCapacity;
    if (newCapacity - MAX_ARRAY_SIZE > 0)
      newCapacity = hugeCapacity(minCapacity);
    buf = Arrays.copyOf(buf, newCapacity);
  }

  private static int hugeCapacity(int minCapacity) {
    if (minCapacity < 0) // overflow
      throw new OutOfMemoryError();
    return (minCapacity > MAX_ARRAY_SIZE) ?
            Integer.MAX_VALUE :
            MAX_ARRAY_SIZE;
  }

  /**
   * Copy from {@link HostMemoryBuffer} to this output stream.
   * @param srcBuf {@link HostMemoryBuffer} to copy from.
   * @param offset Start position in source {@link HostMemoryBuffer}.
   * @param length Number of bytes to copy.
   */
  public void write(HostMemoryBuffer srcBuf, long offset, int length) {
    requireNonNull(srcBuf, "Source buf can't be null!");
    reserve(count + length);
    srcBuf.getBytes(buf, count, offset, length);
    count += length;
  }
}
