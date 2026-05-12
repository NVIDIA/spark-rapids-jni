/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.jni.fileio;

import ai.rapids.cudf.HostMemoryBuffer;
import org.junit.jupiter.api.Test;

import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class RapidsInputFileTest {
  private static final byte[] FILE_DATA = "abcdefghijklmnop".getBytes(StandardCharsets.UTF_8);

  @Test
  public void readVectoredUsesSeekableStreamFallback() throws IOException {
    RapidsInputFile inputFile = new TestRapidsInputFile(FILE_DATA);
    try (HostMemoryBuffer output = HostMemoryBuffer.allocate(5)) {
      inputFile.readVectored(output, Arrays.asList(
          new RapidsInputFile.CopyRange(2, 3, 0),
          new RapidsInputFile.CopyRange(10, 2, 3)));
      assertArrayEquals("cdekl".getBytes(StandardCharsets.UTF_8), readBytes(output));
    }
  }

  @Test
  public void readVectoredThrowsOnShortRead() throws IOException {
    RapidsInputFile inputFile = new TestRapidsInputFile(FILE_DATA);
    try (HostMemoryBuffer output = HostMemoryBuffer.allocate(3)) {
      assertThrows(EOFException.class,
          () -> inputFile.readVectored(output,
              Collections.singletonList(new RapidsInputFile.CopyRange(14, 3, 0))));
    }
  }

  @Test
  public void readVectoredRejectsNullRangesBeforeOpeningStream() throws IOException {
    TestRapidsInputFile inputFile = new TestRapidsInputFile(FILE_DATA);
    try (HostMemoryBuffer output = HostMemoryBuffer.allocate(3)) {
      assertThrows(NullPointerException.class,
          () -> inputFile.readVectored(output, Arrays.asList(
              new RapidsInputFile.CopyRange(0, 1, 0),
              null,
              new RapidsInputFile.CopyRange(1, 1, 1))));
    }
    assertEquals(0, inputFile.getOpenCount());
  }

  @Test
  public void readVectoredRejectsRangeExceedingOutputBeforeOpeningStream() throws IOException {
    TestRapidsInputFile inputFile = new TestRapidsInputFile(FILE_DATA);
    try (HostMemoryBuffer output = HostMemoryBuffer.allocate(3)) {
      assertThrows(IllegalArgumentException.class,
          () -> inputFile.readVectored(output,
              Collections.singletonList(new RapidsInputFile.CopyRange(0, 4, 0))));
      assertThrows(IllegalArgumentException.class,
          () -> inputFile.readVectored(output,
              Collections.singletonList(new RapidsInputFile.CopyRange(0, 2, 2))));
    }
    assertEquals(0, inputFile.getOpenCount());
  }

  @Test
  public void readTailUsesSeekableStreamFallback() throws IOException {
    RapidsInputFile inputFile = new TestRapidsInputFile(FILE_DATA);
    try (HostMemoryBuffer output = HostMemoryBuffer.allocate(4)) {
      inputFile.readTail(4, output);
      assertArrayEquals("mnop".getBytes(StandardCharsets.UTF_8), readBytes(output));
    }
  }

  @Test
  public void readTailThrowsWhenTailExceedsFileLength() throws IOException {
    RapidsInputFile inputFile = new TestRapidsInputFile(FILE_DATA);
    try (HostMemoryBuffer output = HostMemoryBuffer.allocate(FILE_DATA.length + 1L)) {
      assertThrows(EOFException.class, () -> inputFile.readTail(FILE_DATA.length + 1L, output));
    }
  }

  private static byte[] readBytes(HostMemoryBuffer buffer) {
    byte[] bytes = new byte[Math.toIntExact(buffer.getLength())];
    buffer.getBytes(bytes, 0, 0, bytes.length);
    return bytes;
  }

  private static final class TestRapidsInputFile implements RapidsInputFile {
    private final byte[] data;
    private int openCount;

    private TestRapidsInputFile(byte[] data) {
      this.data = data;
      this.openCount = 0;
    }

    @Override
    public long getLength() {
      return data.length;
    }

    private int getOpenCount() {
      return openCount;
    }

    @Override
    public SeekableInputStream open() {
      openCount++;
      return new ArraySeekableInputStream(data);
    }
  }

  private static final class ArraySeekableInputStream extends SeekableInputStream {
    private final byte[] data;
    private int position;

    private ArraySeekableInputStream(byte[] data) {
      this.data = data;
      this.position = 0;
    }

    @Override
    public long getPos() {
      return position;
    }

    @Override
    public void seek(long newPos) throws IOException {
      if (newPos < 0 || newPos > data.length) {
        throw new EOFException("Cannot seek to position " + newPos);
      }
      position = Math.toIntExact(newPos);
    }

    @Override
    public int read() {
      if (position >= data.length) {
        return -1;
      }
      return data[position++] & 0xFF;
    }

    @Override
    public int read(byte[] b, int off, int len) {
      if (b == null) {
        throw new NullPointerException("b can't be null");
      }
      if (off < 0 || len < 0 || len > b.length - off) {
        throw new IndexOutOfBoundsException("Invalid off/len for destination buffer");
      }
      if (len == 0) {
        return 0;
      }
      if (position >= data.length) {
        return -1;
      }

      int amountToRead = Math.min(len, data.length - position);
      System.arraycopy(data, position, b, off, amountToRead);
      position += amountToRead;
      return amountToRead;
    }
  }
}
