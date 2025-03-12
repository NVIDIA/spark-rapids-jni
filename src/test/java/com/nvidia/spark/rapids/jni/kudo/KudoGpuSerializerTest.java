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

import ai.rapids.cudf.*;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Optional;
import java.util.stream.Collectors;

import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializerTest.*;
import static org.junit.jupiter.api.Assertions.*;

public class KudoGpuSerializerTest {
  /**
   * Used in some performance tests.
   */
  private static class JustCountOutputStream extends OutputStream {
    long length = 0;

    @Override
    public void write(byte[] i) {
      this.length += i.length;
    }

    @Override
    public void write(byte[] i, int start, int length) {
      // Just assume that this is all correct for now...
      this.length += length;
    }

    @Override
    public void write(int i) {
      this.length += 1;
    }
  }

  private static int readIntHeader(int start, byte[] data) {
    int b1 = start < data.length ? data[start] & 0xFF : 0;
    int b2 = start + 1 < data.length ? data[start + 1] & 0xFF : 0;
    int b3 = start + 2 < data.length ? data[start + 2] & 0xFF : 0;
    int b4 = start + 3 < data.length ? data[start + 3] & 0xFF : 0;
    return (b1 << 24) + (b2 << 16) + (b3 << 8) + b4;
  }

  private static int readIntOffset(int start, byte[] data) {
    int b1 = start < data.length ? data[start] & 0xFF : 0;
    int b2 = start + 1 < data.length ? data[start + 1] & 0xFF : 0;
    int b3 = start + 2 < data.length ? data[start + 2] & 0xFF : 0;
    int b4 = start + 3 < data.length ? data[start + 3] & 0xFF : 0;
    return b1 + (b2 << 8) + (b3 << 16) + (b4 << 24);
  }

  private static String makeExtraEnd(String name, int start, byte[] hDataGPU, byte[] hDataCPU) {
    int gpuNum = readIntHeader(start, hDataGPU);
    int cpuNum = readIntHeader(start, hDataCPU);
    return " <-- " + name + " END (" + gpuNum + "/" + cpuNum + ")";
  }

  private static String makeOffsetEnd(int index, int start, byte[] hDataGPU, byte[] hDataCPU) {
    int gpuNum = readIntOffset(start, hDataGPU);
    int cpuNum = readIntOffset(start, hDataCPU);
    return " <-- OFFSET " + index + " (" + gpuNum + "/" + cpuNum + ")";
  }

  public static int logSinglePartitionComparison(String name, int startOffset, byte[] hDataGPU, byte[] hDataCPU) {
    System.err.println(name + " COMP GPU VS CPU AT START OFFSET "+ startOffset);
    int maxLength = Math.max(hDataGPU.length, hDataCPU.length) - startOffset;
    int hasValidStart = -1;
    int hasValidEnd = -1;
    int validityBuffersStart = -1;
    int validityBuffersEnd = -1;
    int offsetBuffersStart = -1;
    int offsetBuffersEnd = -1;
    int dataBuffersStart = -1;
    int dataBuffersEnd = -1;
    for (int i = 0; i < maxLength; i++) {
      Byte gpuByte = null;
      Byte cpuByte = null;
      String gpu = "N/A     ";
      if (i + startOffset < hDataGPU.length) {
        gpuByte = hDataGPU[i + startOffset];
        gpu = String.format("0x%02X %03d", hDataGPU[i + startOffset] & 0xFF, hDataGPU[i + startOffset]);
      }

      String cpu = "N/A     ";
      if (i + startOffset < hDataCPU.length) {
        cpuByte = hDataCPU[i + startOffset];
        cpu = String.format("0x%02X %03d", hDataCPU[i + startOffset] & 0xFF, hDataCPU[i + startOffset]);
      }

      String diff = "   ";
      if (cpuByte != gpuByte) {
        diff = "***";
      }
      String extra = "";
      if (i == 0) {
        extra = " <-- MAGIC START";
      } else if (i == 3) {
        extra = " <-- MAGIC END";
      } else if (i == 4) {
        extra = " <-- OFFSET START";
      } else if (i == 7) {
        extra = makeExtraEnd("OFFSET", 4 + startOffset, hDataGPU, hDataCPU);
      } else if (i == 8) {
        extra = " <-- NUM_ROWS START";
      } else if (i == 11) {
        extra = makeExtraEnd("NUM_ROWS", 8 + startOffset, hDataGPU, hDataCPU);
      } else if (i == 12) {
        extra = " <-- VALIDITY_BUF_LEN START";
      } else if (i == 15) {
        extra = makeExtraEnd("VALIDITY_BUF_LEN", 12 + startOffset, hDataGPU, hDataCPU);
      } else if (i == 16) {
        extra = " <-- OFFSET_BUF_LEN START";
      } else if (i == 19) {
        extra = makeExtraEnd("OFFSET_BUF_LEN", 16 + startOffset, hDataGPU, hDataCPU);
      } else if (i == 20) {
        extra = " <-- TOTAL_DATA_LEN START";
      } else if (i == 23) {
        extra = makeExtraEnd("TOTAL_DATA_LEN", 20 + startOffset, hDataGPU, hDataCPU);
      } else if (i == 24) {
        extra = " <-- NUM_COL START";
      } else if (i == 27) {
        extra = makeExtraEnd("NUM_COL", 24 + startOffset, hDataGPU, hDataCPU);
        int numCol = readIntHeader(24 + startOffset, hDataCPU);
        int hasValidLen = (numCol + 7) / 8;
        if (hasValidLen > 0) {
          hasValidStart = 28;
          hasValidEnd = hasValidStart + hasValidLen - 1;
        } else {
          hasValidEnd = 28;
        }
        int validityBuffsLen = readIntHeader(12 + startOffset, hDataCPU);
        if (validityBuffsLen > 0) {
          validityBuffersStart = hasValidEnd + 1;
          validityBuffersEnd = validityBuffersStart + validityBuffsLen - 1;
        } else {
          validityBuffersEnd = hasValidEnd;
        }
        int offsetBuffersLen = readIntHeader(16 + startOffset, hDataCPU);
        if (offsetBuffersLen > 0) {
          offsetBuffersStart = validityBuffersEnd + 1;
          offsetBuffersEnd = offsetBuffersStart + offsetBuffersLen - 1;
        } else {
          offsetBuffersEnd = validityBuffersEnd;
        }
        int totalDataLen = readIntHeader(20 + startOffset, hDataCPU);
        int dataLen = totalDataLen - offsetBuffersLen - validityBuffsLen;
        if (dataLen > 0) {
          dataBuffersStart = offsetBuffersEnd + 1;
          dataBuffersEnd = dataBuffersStart + dataLen - 1;
        } else {
          dataBuffersEnd = offsetBuffersEnd;
        }
      } else if (hasValidStart == hasValidEnd && i == hasValidStart) {
        extra = " <-- HAS_VALID_BUF";
      } else if (hasValidStart == i) {
        extra = " <-- HAS_VALID_BUF START";
      } else if (hasValidStart > 0 && hasValidEnd == i) {
        extra = " <-- HAS_VALID_BUF END";
      } else if (validityBuffersStart == validityBuffersEnd && i == validityBuffersStart) {
        extra = " <-- VALIDITY_BUFFERS";
      } else if (validityBuffersStart == i) {
        extra = " <-- VALIDITY_BUFFERS START";
      } else if (validityBuffersStart > 0 && validityBuffersEnd == i) {
        extra = " <-- VALIDITY_BUFFERS END";
      } else if (offsetBuffersStart == i) {
        extra = " <-- OFFSET_BUFFERS START";
      } else if (offsetBuffersStart > 0 && i > offsetBuffersStart && i <= offsetBuffersEnd && (i - offsetBuffersStart + 1) % 4 == 0) {
        extra = makeOffsetEnd(((i - offsetBuffersStart) / 4), i - 3 + startOffset, hDataGPU, hDataCPU);
      } else if (dataBuffersStart == dataBuffersEnd && i == dataBuffersStart) {
        extra = " <-- DATA_BUFFERS";
      } else if (dataBuffersStart == i) {
        extra = " <-- DATA_BUFFERS START";
      } else if (dataBuffersStart > 0 && dataBuffersEnd == i) {
        extra = " <-- DATA_BUFFERS END";
      }
      String i_str = String.format("%05d: ", i + startOffset);
      String aligned = "        ";
      if ((i + startOffset) % 4 == 0) {
        aligned = "ALIGNED ";
      }
      System.err.println(aligned + i_str + diff + " " + gpu + " VS " + cpu  + extra);
      if (i == dataBuffersEnd) {
        return i + startOffset + 1;
      }
    }
    return maxLength + startOffset + 1;
  }

  public static void logPartitionComparisons(String name, byte[] hDataGPU, byte[] hDataCPU) {
    int maxLength = Math.max(hDataGPU.length, hDataCPU.length);
    int offset = 0;
    while (offset < maxLength) {
      offset = logSinglePartitionComparison(name, offset, hDataGPU, hDataCPU);
    }
  }

  public static int[] calcEvenSlices(Table table, int numSlices) {
    assert table.getRowCount() >= numSlices : "slices must be <= number of rows " + table.getRowCount() + " < " + numSlices;
    assert numSlices > 0 : "slices must be > 0 " + numSlices;
    if (numSlices == 1) {
      return new int[0];
    } else {
      int rowsPerSlice = (int)table.getRowCount() / numSlices;
      int[] ret = new int[numSlices - 1];
      int offset = 0;
      for (int i = 0; i < ret.length; i++) {
        offset += rowsPerSlice;
        ret[i] = offset;
      }
      return ret;
    }
  }

  public static byte[] writeGPU(Table table, int[] slices) {
    DeviceMemoryBuffer[] buffers = KudoGpuSerializer.splitAndSerializeToDevice(table, slices);
    try {
      assertEquals(2, buffers.length);
      DeviceMemoryBuffer data = buffers[0];
      // Ignoring the offsets as we are not splitting it up...
      try (HostMemoryBuffer hostBuffer = HostMemoryBuffer.allocate(data.getLength())) {
        hostBuffer.copyFromDeviceBuffer(data);
        assert hostBuffer.getLength() <= Integer.MAX_VALUE : "Serialized buffer is too large for test " + hostBuffer.getLength();
        byte[] ret = new byte[(int)hostBuffer.getLength()];
        hostBuffer.getBytes(ret, 0, 0, ret.length);
        return ret;
      }
    } finally {
      for (DeviceMemoryBuffer b : buffers) {
        if (b != null) {
          b.close();
        }
      }
    }
  }

  public static void writeCPUCore(HostColumnVector[] columns, KudoSerializer serializer,
                           int [] slices, int numRows, OutputStream out) {
    int start = 0;
    int end;
    for (int i = 0; i < slices.length + 1; i++) {
      if (i >= slices.length) {
        end = numRows;
      } else {
        end = slices[i];
      }
      int len = end - start;
      serializer.writeToStreamWithMetrics(columns, out, start, len);
      start = end;
    }
  }

  public static byte[] writeCPU(Table table, int[] slices) {
    int numRows = (int)table.getRowCount();
    int numColumns = table.getNumberOfColumns();
    HostColumnVector[] columns = new HostColumnVector[numColumns];
    try {
      for (int c = 0; c < numColumns; c++) {
        columns[c] = table.getColumn(c).copyToHost();
      }
      Schema s = KudoSerializerTest.schemaOf(table);
      KudoSerializer serializer = new KudoSerializer(s);
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      writeCPUCore(columns, serializer, slices, numRows, baos);
      return baos.toByteArray();
    } finally {
      for (HostColumnVector hcv: columns) {
        if (hcv != null) {
          hcv.close();
        }
      }
    }
  }

  public void debugCompareWrite(String name, Table table, int[] slices) {
    byte[] gpuData = writeGPU(table, slices);
    byte[] cpuData = writeCPU(table, slices);
    if (!Arrays.equals(gpuData, cpuData)) {
      logPartitionComparisons(name, gpuData, cpuData);
    }
  }

  public byte[] testWrite(String name, Table table, int[] slices) {
    // First serialize it on the GPU...
    byte[] gpuData = writeGPU(table, slices);
    byte[] cpuData = writeCPU(table, slices);
    if (!Arrays.equals(gpuData, cpuData)) {
      logPartitionComparisons(name, gpuData, cpuData);
      fail("CPU and GPU serialized format is not the same");
    }
    return gpuData;
  }

  public Table readGPU(String name, Schema s, byte[] bytes) throws IOException {
    // We are going to parse the headers so we can set up the offsets properly.
    DataInputStream din = new DataInputStream(new ByteArrayInputStream(bytes));
    ArrayList<Long> offsets = new ArrayList<>();
    long currentOffset = 0;
    Optional<KudoTableHeader> header;
    do {
      header = KudoTableHeader.readFrom(din);
      if (header.isPresent()) {
        offsets.add(currentOffset);
        KudoTableHeader h = header.get();
        din.skipBytes(h.getTotalDataLen());
        currentOffset += h.getTotalDataLen() + h.getSerializedSize();
      }
    } while (header.isPresent());
    if (!offsets.isEmpty()) {
      offsets.add(currentOffset);
    }
    // Okay now we have offsets, lets put all of this onto the GPU.
    try (HostMemoryBuffer hostOffsets = HostMemoryBuffer.allocate(8L * offsets.size());
         HostMemoryBuffer hostPayload = HostMemoryBuffer.allocate(bytes.length)) {
      for (int i = 0; i < offsets.size(); i++) {
        hostOffsets.setLong(i * 8L, offsets.get(i));
      }
      hostPayload.setBytes(0, bytes, 0, bytes.length);
      try (DeviceMemoryBuffer devOffsets = DeviceMemoryBuffer.allocate(hostOffsets.getLength());
           DeviceMemoryBuffer devPayload = DeviceMemoryBuffer.allocate(hostPayload.getLength())) {
        devOffsets.copyFromHostBuffer(hostOffsets);
        devPayload.copyFromHostBuffer(hostPayload);
        return KudoGpuSerializer.assembleFromDeviceRaw(s, devPayload, devOffsets);
      }
    }
  }

  public Table readCPU(String name, Schema s, byte[] bytes) throws Exception {
    KudoSerializer serializer = new KudoSerializer(s);
    DataInputStream din = new DataInputStream(new ByteArrayInputStream(bytes));
    ArrayList<KudoTable> kudoTables = new ArrayList<>();
    Optional<KudoTable> kt;
    do {
      kt = KudoTable.from(din);
      kt.ifPresent(kudoTables::add);
    } while(kt.isPresent());
    return serializer.mergeToTable(kudoTables.toArray(new KudoTable[kudoTables.size()]));
  }


  public void testRead(String name, Schema s, byte[] bytes, Table expected) throws Exception {
    try (Table fromGpu = readGPU(name, s, bytes);
         Table fromCpu = readCPU(name, s, bytes)) {
      assertTablesAreEqual(expected, fromGpu);
      assertTablesAreEqual(expected, fromCpu);
    }
  }

  public void testRoundTrip(String name, Table table, int[] slices) throws Exception {
//    Schema s = KudoSerializerTest.schemaOf(table);
//    byte[] data = testWrite(name, table, slices);
//    testRead(name, s, data, table);

    debugCompareWrite(name, table, slices);
    testCPUOnlyRoundTrip(name + " CPU -> CPU", table, slices);
    testGPUOnlyRoundTrip(name + " GPU -> GPU", table, slices);
    testCPUWriteGPURead(name + " CPU -> GPU", table, slices);
    testGPUWriteCPURead(name + " GPU -> CPU", table, slices);
  }

  public void testCPUOnlyRoundTrip(String name, Table table, int[] slices) throws Exception {
    Schema s = KudoSerializerTest.schemaOf(table);
    byte[] data = writeCPU(table, slices);
    try (Table t = readCPU(name, s, data)) {
      assertTablesAreEqual(table, t);
    }
  }

  public void testGPUOnlyRoundTrip(String name, Table table, int[] slices) throws Exception {
    Schema s = KudoSerializerTest.schemaOf(table);
    byte[] data = writeGPU(table, slices);
    try (Table t = readGPU(name, s, data)) {
      System.err.println("SLICES " + Arrays.toString(slices));
      System.err.println();
      TableDebug.get().debug(name + " OUTPUT", t);
      System.err.println();
      TableDebug.get().debug(name + " EXPECTED", table);
      assertTablesAreEqual(table, t);
    }
  }

  public void testCPUWriteGPURead(String name, Table table, int[] slices) throws Exception {
    Schema s = KudoSerializerTest.schemaOf(table);
    byte[] data = writeCPU(table, slices);
    try (Table t = readGPU(name, s, data)) {
      assertTablesAreEqual(table, t);
    }
  }

  public void testGPUWriteCPURead(String name, Table table, int[] slices) throws Exception {
    Schema s = KudoSerializerTest.schemaOf(table);
    byte[] data = writeGPU(table, slices);
    try (Table t = readCPU(name, s, data)) {
      assertTablesAreEqual(table, t);
    }
  }

  static Table perfTable(int numRows, int numColumns) {
    try (Scalar s = Scalar.fromLong(1);
      ColumnVector cv = ColumnVector.sequence(s, numRows)) {
      ColumnVector[] columns = new ColumnVector[numColumns];
      Arrays.fill(columns, cv);
      return new Table(columns);
    }
  }

  public static void main(String [] args) throws Exception {
    Cuda.freeZero();
    Cuda.autoSetDevice();
    // 15 GiB device memory pool for now...
    Rmm.initialize(RmmAllocationMode.CUDA_ASYNC, null, 15L * 1024 * 1024 * 1024);
    // 5 GiB pinned pool for now...
    PinnedMemoryPool.initialize(5L * 1024 * 1024 * 1024);
      int[] sizes = new int[]{1024*2, 1024*64, 1024*1024, 10*1024*1024, 100*1024*1024, 1024*1024*1024};
//    for (int numColumns: new int[]{1, 10, 100, 200, 500, 1000, 2000}) {
//      runRoundTripPerTests(numColumns, sizes, 9);
//    }
    roundTripJustCopyTestPartiallyPinnedTest(sizes, 9);
    roundTripJustCopyTestFullyPinnedTest(sizes, 9);
  }

  static String toMedian(ArrayList<Long> numbers) {
    return "=MEDIAN(" + numbers.stream().map(String::valueOf).collect(Collectors.joining(",")) + ")";
  }

  public static void runRoundTripPerTests(int numColumns, int[] sizes, int numIters) throws Exception {
    int sizePerRow = numColumns * 8;
    int[] rowOptions = new int[sizes.length];
    int[] sliceOptions = {1, 10, 200, 1000, 10000, 100000};
    for (int i = 0; i < sizes.length; i++) {
      int value = Math.max(1, sizes[i] / sizePerRow);
      rowOptions[i] = value;
    }
    runRoundTripPerTests(numColumns, rowOptions, sliceOptions, numIters);
  }

  public static void runRoundTripPerTests(int numColumns, int[] rowOptions, int[] sliceOptions, int numIters) throws Exception {
    for (int numRows : rowOptions) {
      System.err.println();
      System.err.println("ROWS: "+ numRows);
      for (int numSlices : sliceOptions) {
        System.err.println("SLICES: " + numSlices);
        if (numSlices > numRows) {
          continue;
        }
        roundTripPerfTestCPU(numColumns, numRows, numSlices, numIters);
        roundTripPerfTestCPUJcudf(numColumns, numRows, numSlices, numIters);
        roundTripPerfTestGPU(numColumns, numRows, numSlices, numIters);
      }
    }
  }

  public static void roundTripJustCopyTestPartiallyPinnedTest(int[] sizes, int numIters) throws Exception {
    for (int size: sizes) {
      ArrayList<Long> copyToHostTimes = new ArrayList<>(numIters);
      ArrayList<Long> writeTimes = new ArrayList<>(numIters);
      ArrayList<Long> readTimes = new ArrayList<>(numIters);
      ArrayList<Long> mergeTimes = new ArrayList<>(numIters);
      ArrayList<Long> copyToDeviceTimes = new ArrayList<>(numIters);
      ArrayList<Long> endToEndTimes = new ArrayList<>(numIters);
      boolean dnf = false;
      try {
        try (DeviceMemoryBuffer t = DeviceMemoryBuffer.allocate(size)) {
          // the 1.5 is just extra space
          int estimatedSize = (int) Math.min((long) (size * 1.5), Integer.MAX_VALUE);

          // we will start off with the CPU test
          for (int iter = 0; iter < numIters; iter++) {
            ByteArrayOutputStream bao = new ByteArrayOutputStream(estimatedSize);
            long start;
            long copyEndWriteStart;
            long writeEndReadStart;
            long readEndMergeStart;
            long mergeEndTableStart;
            long copyToDeviceEnd;
            try (NvtxRange run = new NvtxRange(size + " PARTIAL PINNED", NvtxColor.BLUE)) {
              start = System.nanoTime();
              HostMemoryBuffer fromDev = HostMemoryBuffer.allocate(size, true);
              try (NvtxRange cp = new NvtxRange("PARTIAL PINNED COPY TO HOST", NvtxColor.RED)) {
                fromDev.copyFromDeviceBuffer(t);
              }
              copyEndWriteStart = System.nanoTime();
              ByteArrayOutputStreamWriter wrapped = new ByteArrayOutputStreamWriter(bao);
              try {
                try (NvtxRange r = new NvtxRange("WRITE PARTLY PINNED", NvtxColor.BLUE)) {
                  wrapped.copyDataFrom(fromDev, 0, size);
                  wrapped.flush();
                  writeEndReadStart = System.nanoTime();
                }
              } finally {
                fromDev.close();
              }
              byte[] arr = bao.toByteArray();
              size = arr.length;
              DataInputStream din = new DataInputStream(new ByteArrayInputStream(arr));
              try (NvtxRange r = new NvtxRange("READ PARTLY PINNED", NvtxColor.BLUE)) {
                HostMemoryBuffer merged = null;
                try {
                  try (HostMemoryBuffer readIn = HostMemoryBuffer.allocate(size, false)) {
                    readIn.copyFromStream(0, din, size);
                    readEndMergeStart = System.nanoTime();
                    merged = HostMemoryBuffer.allocate(size, true);
                    merged.copyFromHostBuffer(0, readIn, 0, size);
                  }
                  mergeEndTableStart = System.nanoTime();
                  try (DeviceMemoryBuffer dest = DeviceMemoryBuffer.allocate(size)) {
                    dest.copyFromHostBuffer(merged);
                    copyToDeviceEnd = System.nanoTime();
                  }
                } finally {
                  if (merged != null) {
                    merged.close();
                  }
                }
              }
            }
            // Add in the metrics
            copyToHostTimes.add(copyEndWriteStart - start);
            writeTimes.add(writeEndReadStart - copyEndWriteStart);
            readTimes.add(readEndMergeStart - writeEndReadStart);
            mergeTimes.add(mergeEndTableStart - readEndMergeStart);
            copyToDeviceTimes.add(copyToDeviceEnd - mergeEndTableStart);
            endToEndTimes.add(copyToDeviceEnd - start);
          }
        }
      } catch (Throwable e) {
        dnf = true;
        System.err.println(e);
        e.printStackTrace(System.err);
      }
      System.err.println("JUST COPY PARTLY PINNED:" + (dnf ? " DID NOT FINISH!!!" : "") +
          "\n\tSIZE: " + size +
          "\n\tEND2END:\n" + toMedian(endToEndTimes) +
          "\n\tCOPY_TO_HOST:\n" + toMedian(copyToHostTimes) +
          "\n\tWRITE:\n" + toMedian(writeTimes) +
          "\n\tREAD:\n" + toMedian(readTimes) +
          "\n\tMERGED:\n" + toMedian(mergeTimes) +
          "\n\tCOPY_TO_DEVICE:\n" + toMedian(copyToDeviceTimes)
      );
    }
  }

  public static void roundTripJustCopyTestFullyPinnedTest(int[] sizes, int numIters) throws Exception {
    for (int size: sizes) {
      ArrayList<Long> copyToHostTimes = new ArrayList<>(numIters);
      ArrayList<Long> writeTimes = new ArrayList<>(numIters);
      ArrayList<Long> readTimes = new ArrayList<>(numIters);
      ArrayList<Long> copyToDeviceTimes = new ArrayList<>(numIters);
      ArrayList<Long> endToEndTimes = new ArrayList<>(numIters);
      boolean dnf = false;
      try {
        try (DeviceMemoryBuffer t = DeviceMemoryBuffer.allocate(size)) {
          // the 1.5 is just extra space
          int estimatedSize = (int) Math.min((long) (size * 1.5), Integer.MAX_VALUE);

          // we will start off with the CPU test
          for (int iter = 0; iter < numIters; iter++) {
            ByteArrayOutputStream bao = new ByteArrayOutputStream(estimatedSize);
            long start;
            long copyEndWriteStart;
            long writeEndReadStart;
            long readEndToDeviceStart;
            long copyToDeviceEnd;
            try (NvtxRange run = new NvtxRange(size + " FULLY PINNED", NvtxColor.BLUE)) {
              start = System.nanoTime();
              HostMemoryBuffer fromDev = HostMemoryBuffer.allocate(size, true);
              try (NvtxRange cp = new NvtxRange("FULLY PINNED COPY TO HOST", NvtxColor.RED)) {
                fromDev.copyFromDeviceBuffer(t);
              }
              copyEndWriteStart = System.nanoTime();
              ByteArrayOutputStreamWriter wrapped = new ByteArrayOutputStreamWriter(bao);
              try {
                try (NvtxRange r = new NvtxRange("WRITE FULLY PINNED", NvtxColor.BLUE)) {
                  wrapped.copyDataFrom(fromDev, 0, size);
                  wrapped.flush();
                  writeEndReadStart = System.nanoTime();
                }
              } finally {
                fromDev.close();
              }
              byte[] arr = bao.toByteArray();
              size = arr.length;
              DataInputStream din = new DataInputStream(new ByteArrayInputStream(arr));
              try (NvtxRange r = new NvtxRange("READ FULLY PINNED", NvtxColor.BLUE)) {
                try (HostMemoryBuffer readIn = HostMemoryBuffer.allocate(size, true)) {
                    readIn.copyFromStream(0, din, size);
                    readEndToDeviceStart = System.nanoTime();
                  try (DeviceMemoryBuffer dest = DeviceMemoryBuffer.allocate(size)) {
                    dest.copyFromHostBuffer(readIn);
                    copyToDeviceEnd = System.nanoTime();
                  }
                }
              }
            }
            // Add in the metrics
            copyToHostTimes.add(copyEndWriteStart - start);
            writeTimes.add(writeEndReadStart - copyEndWriteStart);
            readTimes.add(readEndToDeviceStart - writeEndReadStart);
            copyToDeviceTimes.add(copyToDeviceEnd - readEndToDeviceStart);
            endToEndTimes.add(copyToDeviceEnd - start);
          }
        }
      } catch (Throwable e) {
        dnf = true;
        System.err.println(e);
        e.printStackTrace(System.err);
      }
      System.err.println("JUST COPY FULLY PINNED:" + (dnf ? " DID NOT FINISH!!!" : "") +
          "\n\tSIZE: " + size +
          "\n\tEND2END:\n" + toMedian(endToEndTimes) +
          "\n\tCOPY_TO_HOST:\n" + toMedian(copyToHostTimes) +
          "\n\tWRITE:\n" + toMedian(writeTimes) +
          "\n\tREAD:\n" + toMedian(readTimes) +
          "\n\tCOPY_TO_DEVICE:\n" + toMedian(copyToDeviceTimes)
      );
    }
  }

  public static void roundTripPerfTestCPU(int numColumns, int numRows, int numSlices, int numIters) throws Exception {
    if (numSlices > numRows) {
      return;
    }
    ArrayList<Long> copyToHostTimes = new ArrayList<>(numIters);
    ArrayList<Long> writeTimes = new ArrayList<>(numIters);
    ArrayList<Long> readTimes = new ArrayList<>(numIters);
    ArrayList<Long> mergeTimes = new ArrayList<>(numIters);
    ArrayList<Long> tableTimes = new ArrayList<>(numIters);
    ArrayList<Long> endToEndTimes = new ArrayList<>(numIters);
    long size = -1;
    boolean dnf = false;
    try {
      try (Table t = perfTable(numRows, numColumns)) {
        // the 1.5 is just extra space
        int estimatedSize = (int) Math.min((long) ((8L * numColumns * numRows) * 1.5), Integer.MAX_VALUE);
        int[] slices = calcEvenSlices(t, numSlices);
        Schema schema = schemaOf(t);

        // we will start off with the CPU test
        for (int iter = 0; iter < numIters; iter++) {
          ByteArrayOutputStream bao = new ByteArrayOutputStream(estimatedSize);
          long start;
          long copyEndWriteStart;
          long writeEndReadStart;
          long readEndMergeStart;
          long mergeEndTableStart;
          long tableEnd;
          try (NvtxRange run = new NvtxRange(numSlices + " " + numRows + " CPU", NvtxColor.BLUE)) {
            start = System.nanoTime();
            KudoSerializer ser = new KudoSerializer(schema);
            HostColumnVector[] columns = new HostColumnVector[numColumns];
            try (NvtxRange cp = new NvtxRange("CPU COPY TABLE TO HOST", NvtxColor.RED)) {
              for (int c = 0; c < numColumns; c++) {
                columns[c] = t.getColumn(c).copyToHost();
              }
            }
            copyEndWriteStart = System.nanoTime();
            try {
              try (NvtxRange r = new NvtxRange("WRITE CPU", NvtxColor.BLUE)) {
                writeCPUCore(columns, ser, slices, numRows, bao);
                writeEndReadStart = System.nanoTime();
              }
            } finally {
              for (int c = 0; c < numColumns; c++) {
                if (columns[c] != null) {
                  columns[c].close();
                }
              }
            }
            byte[] arr = bao.toByteArray();
            size = arr.length;
            DataInputStream din = new DataInputStream(new ByteArrayInputStream(arr));
            try (NvtxRange r = new NvtxRange("READ CPU", NvtxColor.BLUE)) {
              ArrayList<KudoTable> kudoTables = new ArrayList<>(numSlices);
              Optional<KudoTable> kt;
              do {
                kt = KudoTable.from(din);
                kt.ifPresent(kudoTables::add);
              } while (kt.isPresent());
              readEndMergeStart = System.nanoTime();
              try (KudoHostMergeResult merged = ser.mergeOnHost(kudoTables.toArray(new KudoTable[kudoTables.size()]))) {
                mergeEndTableStart = System.nanoTime();
                merged.toTable().close();
                tableEnd = System.nanoTime();
              }
            }
          }
          // Add in the metrics
          copyToHostTimes.add(copyEndWriteStart - start);
          writeTimes.add(writeEndReadStart - copyEndWriteStart);
          readTimes.add(readEndMergeStart - writeEndReadStart);
          mergeTimes.add(mergeEndTableStart - readEndMergeStart);
          tableTimes.add(tableEnd - mergeEndTableStart);
          endToEndTimes.add(tableEnd - start);
        }
      }
    } catch (Throwable e) {
      dnf = true;
      System.err.println(e);
      e.printStackTrace(System.err);
    }
    System.err.println("CPU:" + (dnf ? " DID NOT FINISH!!!" : "") +
        "\n\tROW: " + numRows +
        "\n\tSLICES: " + numSlices +
        "\n\tCOLS: " + numColumns +
        "\n\tSIZE: " + size +
        "\n\tEND2END:\n" + toMedian(endToEndTimes) +
        "\n\tCOPY_TO_HOST:\n" + toMedian(copyToHostTimes) +
        "\n\tWRITE:\n" + toMedian(writeTimes) +
        "\n\tREAD:\n" + toMedian(readTimes) +
        "\n\tMERGED:\n" + toMedian(mergeTimes) +
        "\n\tTO_TABLE:\n" + toMedian(tableTimes)
    );
  }

  public static void roundTripPerfTestCPUJcudf(int numColumns, int numRows, int numSlices, int numIters) throws Exception {
    if (numSlices > numRows) {
      return;
    }
    ArrayList<Long> copyToHostTimes = new ArrayList<>(numIters);
    ArrayList<Long> writeTimes = new ArrayList<>(numIters);
    ArrayList<Long> readTimes = new ArrayList<>(numIters);
    ArrayList<Long> mergeTimes = new ArrayList<>(numIters);
    ArrayList<Long> tableTimes = new ArrayList<>(numIters);
    ArrayList<Long> endToEndTimes = new ArrayList<>(numIters);
    long size = -1;
    boolean dnf = false;
    try {
      try (Table t = perfTable(numRows, numColumns)) {
        // the 1.5 is just extra space
        int estimatedSize = (int) Math.min((long) ((8L * numColumns * numRows) * 1.5), Integer.MAX_VALUE);
        int[] slices = calcEvenSlices(t, numSlices);

        // we will start off with the CPU test
        for (int iter = 0; iter < numIters; iter++) {
          ByteArrayOutputStream bao = new ByteArrayOutputStream(estimatedSize);
          long start;
          long copyEndWriteStart;
          long writeEndReadStart;
          long readEndMergeStart;
          long mergeEndTableStart;
          long tableEnd;
          try (NvtxRange run = new NvtxRange(numSlices + " " + numRows + " JCUDF", NvtxColor.DARK_GREEN)) {
            start = System.nanoTime();
            HostColumnVector[] columns = new HostColumnVector[numColumns];
            try (NvtxRange cp = new NvtxRange("CPU COPY TABLE TO HOST", NvtxColor.RED)) {
              for (int c = 0; c < numColumns; c++) {
                columns[c] = t.getColumn(c).copyToHost();
              }
            }
            copyEndWriteStart = System.nanoTime();
            try {
              try (NvtxRange r = new NvtxRange("WRITE JCUDF", NvtxColor.BLUE)) {
                int startOffset = 0;
                int endOffset;
                for (int i = 0; i < slices.length + 1; i++) {
                  if (i >= slices.length) {
                    endOffset = numRows;
                  } else {
                    endOffset = slices[i];
                  }
                  int len = endOffset - startOffset;
                  JCudfSerialization.writeToStream(columns, bao, startOffset, len);
                  startOffset = endOffset;
                }
                writeEndReadStart = System.nanoTime();
              }
            } finally {
              for (int c = 0; c < numColumns; c++) {
                if (columns[c] != null) {
                  columns[c].close();
                }
              }
            }
            byte[] arr = bao.toByteArray();
            size = arr.length;
            DataInputStream din = new DataInputStream(new ByteArrayInputStream(arr));
            try (NvtxRange r = new NvtxRange("READ JCUDF", NvtxColor.BLUE)) {
              JCudfSerialization.HostConcatResult merged;
              ArrayList<JCudfSerialization.SerializedTableHeader> headers = new ArrayList<>(numSlices);
              ArrayList<HostMemoryBuffer> payloads = new ArrayList<>(numSlices);
              try {
                boolean done = false;
                while (!done) {
                  JCudfSerialization.SerializedTableHeader header = new JCudfSerialization.SerializedTableHeader(din);
                  if (header.wasInitialized()) {
                    headers.add(header);
                    // We will concat on the CPU
                    HostMemoryBuffer tmp = HostMemoryBuffer.allocate(header.getDataLen(), false);
                    payloads.add(tmp);
                    JCudfSerialization.readTableIntoBuffer(din, header, tmp);
                  } else {
                    done = true;
                  }
                }
                readEndMergeStart = System.nanoTime();
                merged = JCudfSerialization.concatToHostBuffer(
                    headers.toArray(new JCudfSerialization.SerializedTableHeader[headers.size()]),
                    payloads.toArray(new HostMemoryBuffer[payloads.size()]));
                mergeEndTableStart = System.nanoTime();
              } finally {
                for (HostMemoryBuffer hmb : payloads) {
                  if (hmb != null) {
                    hmb.close();
                  }
                }
              }
              try {
                merged.toContiguousTable().close();
                tableEnd = System.nanoTime();
              } finally {
                merged.close();
              }
            }
          }
          // Add in the metrics
          copyToHostTimes.add(copyEndWriteStart - start);
          writeTimes.add(writeEndReadStart - copyEndWriteStart);
          readTimes.add(readEndMergeStart - writeEndReadStart);
          mergeTimes.add(mergeEndTableStart - readEndMergeStart);
          tableTimes.add(tableEnd - mergeEndTableStart);
          endToEndTimes.add(tableEnd - start);
        }
      }
    } catch (Throwable e) {
      dnf = true;
      System.err.println(e);
      e.printStackTrace(System.err);
    }
    System.err.println("JCUDF:" + (dnf ? " DID NOT FINISH!!!" : "") +
        "\n\tROW: " + numRows +
        "\n\tSLICES: " + numSlices +
        "\n\tCOLS: " + numColumns +
        "\n\tSIZE: " + size +
        "\n\tEND2END:\n" + toMedian(endToEndTimes) +
        "\n\tCOPY_TO_HOST:\n" + toMedian(copyToHostTimes) +
        "\n\tWRITE:\n" + toMedian(writeTimes) +
        "\n\tREAD:\n" + toMedian(readTimes) +
        "\n\tMERGED:\n" + toMedian(mergeTimes) +
        "\n\tTO_TABLE:\n" + toMedian(tableTimes)
    );
  }


  // TODO this needs to go into the KudoGpuSerializer in some form...
  public static long readTableIfFits(DataInputStream din, HostMemoryBuffer target, long offset) throws IOException {
    Optional<KudoTableHeader> header = KudoTableHeader.readFrom(din);
    if (header.isPresent()) {
      KudoTableHeader h = header.get();
      long totalLength = h.getTotalDataLen() + h.getSerializedSize();
      if (totalLength < target.getLength() - offset) {
        // We have room to write it
        long afterHeader = h.writeTo(target, offset);
        target.copyFromStream(afterHeader, din, h.getTotalDataLen());
        return afterHeader + h.getTotalDataLen();
      } else {
        // TODO this should offer a way to recover and try again later.
        throw new IllegalArgumentException("COULD NOT FIT TABLE IN...");
      }
    } else {
      return offset;
    }
  }

  public static void roundTripPerfTestGPU(int numColumns, int numRows, int numSlices, int numIters) throws Exception {
    if (numSlices > numRows) {
      return;
    }
    ArrayList<Long> encodeTimes = new ArrayList<>(numIters);
    ArrayList<Long> copyToHostTimes = new ArrayList<>(numIters);
    ArrayList<Long> writeTimes = new ArrayList<>(numIters);
    ArrayList<Long> readTimes = new ArrayList<>(numIters);
    ArrayList<Long> copyToDevTimes = new ArrayList<>(numIters);
    ArrayList<Long> decodeTimes = new ArrayList<>(numIters);
    ArrayList<Long> endToEndTimes = new ArrayList<>(numIters);
    long size = -1;
    boolean dnf = false;
    try {
      try (Table t = perfTable(numRows, numColumns)) {
        // the 1.5 is just extra space
        int estimatedSize = (int) Math.min((long) ((8L * numColumns * numRows) * 1.5), Integer.MAX_VALUE);
        int[] slices = calcEvenSlices(t, numSlices);
        Schema schema = schemaOf(t);

        // we will start off with the CPU test
        for (int iter = 0; iter < numIters; iter++) {
          ByteArrayOutputStream innerBao = new ByteArrayOutputStream(estimatedSize);
          ByteArrayOutputStreamWriter bao = new ByteArrayOutputStreamWriter(innerBao);
          long start;
          long encodeEndCopyToHostStart;
          long copyToHostEndWriteStart;
          long writeEndReadStart;
          long readEndCopyStart;
          long copyEndDecodeStart;
          long tableEnd;
          try (NvtxRange run = new NvtxRange(numSlices + " " + numRows + " GPU", NvtxColor.GREEN)) {
            start = System.nanoTime();
            HostMemoryBuffer hostData = null;
            HostMemoryBuffer hostOffsets = null;
            try {
              DeviceMemoryBuffer[] buffers = KudoGpuSerializer.splitAndSerializeToDevice(t, slices);
              try {
                size = buffers[0].getLength();
                encodeEndCopyToHostStart = System.nanoTime();
                hostData = HostMemoryBuffer.allocate(buffers[0].getLength(), true);
                hostData.copyFromDeviceBuffer(buffers[0]);
                hostOffsets = HostMemoryBuffer.allocate(buffers[1].getLength(), true);
                hostOffsets.copyFromDeviceBuffer(buffers[1]);
              } finally {
                buffers[0].close();
                buffers[1].close();
              }
              copyToHostEndWriteStart = System.nanoTime();
              try (NvtxRange r = new NvtxRange("WRITE GPU", NvtxColor.BLUE)) {
                // This is needed so we can slice things similar to what would happen in the real shuffle
                long startDataOffset = 0;
                for (int index = 8; index < hostOffsets.getLength(); index += 8) {
                  long endOffset = hostOffsets.getLong(index);
                  bao.copyDataFrom(hostData, startDataOffset, endOffset - startDataOffset);
                  startDataOffset = endOffset;
                }
              }
              writeEndReadStart = System.nanoTime();
            } finally {
              if (hostData != null) {
                hostData.close();
              }
              if (hostOffsets != null) {
                hostOffsets.close();
              }
            }

            try (NvtxRange r = new NvtxRange("READ GPU", NvtxColor.BLUE)) {
              byte[] arr = innerBao.toByteArray();
              DataInputStream din = new DataInputStream(new ByteArrayInputStream(arr));
              // Target batch size for allocation 2 GiB (need some things to deal with single batch too large, but...)
              DeviceMemoryBuffer devDataRead = null;
              DeviceMemoryBuffer devOffsetsRead = null;
              try {
                try (HostMemoryBuffer hostDataRead = HostMemoryBuffer.allocate(Integer.MAX_VALUE, true)) {
                  ArrayList<Long> offsets = new ArrayList<>();
                  long currentOffset = 0;
                  offsets.add(currentOffset);
                  boolean done = false;
                  while (!done) {
                    long newOffset = readTableIfFits(din, hostDataRead, currentOffset);
                    if (newOffset == currentOffset) {
                      done = true;
                    } else {
                      offsets.add(newOffset);
                      currentOffset = newOffset;
                    }
                  }
                  try (HostMemoryBuffer hostOffsetsRead = HostMemoryBuffer.allocate(8L * offsets.size(), true)) {
                    for (int i = 0; i < offsets.size(); i++) {
                      hostOffsetsRead.setLong(i * 8L, offsets.get(i));
                    }
                    readEndCopyStart = System.nanoTime();
                    devDataRead = DeviceMemoryBuffer.allocate(currentOffset);
                    devDataRead.copyFromHostBuffer(hostDataRead, 0, currentOffset);
                    devOffsetsRead = DeviceMemoryBuffer.allocate(hostOffsetsRead.getLength());
                    devOffsetsRead.copyFromHostBuffer(hostOffsetsRead);
                    copyEndDecodeStart = System.nanoTime();
                  }
                }
                KudoGpuSerializer.assembleFromDeviceRaw(schema, devDataRead, devOffsetsRead).close();
                tableEnd = System.nanoTime();
              } finally {
                if (devDataRead != null) {
                  devDataRead.close();
                }

                if (devOffsetsRead != null) {
                  devOffsetsRead.close();
                }
              }
            }
          }
          // ITER is done add in the metrics
          encodeTimes.add(encodeEndCopyToHostStart - start);
          copyToHostTimes.add(copyToHostEndWriteStart - encodeEndCopyToHostStart);
          writeTimes.add(writeEndReadStart - copyToHostEndWriteStart);
          readTimes.add(readEndCopyStart - writeEndReadStart);
          copyToDevTimes.add(copyEndDecodeStart - readEndCopyStart);
          decodeTimes.add(tableEnd - copyEndDecodeStart);
          endToEndTimes.add(tableEnd - start);
        }
      }
    } catch (Throwable e) {
      dnf = true;
      System.err.println(e);
      e.printStackTrace(System.err);
    }
    System.err.println("GPU:" + (dnf ? " DID NOT FINISH!!!" : "") +
        "\n\tROW: " + numRows +
        "\n\tSLICES: " + numSlices +
        "\n\tCOLS: " + numColumns +
        "\n\tSIZE: " + size +
        "\n\tEND2END:\n" + toMedian(endToEndTimes) +
        "\n\tENCODE:\n" + toMedian(encodeTimes) +
        "\n\tCOPY_TO_HOST:\n" + toMedian(copyToHostTimes) +
        "\n\tWRITE:\n" + toMedian(writeTimes) +
        "\n\tREAD:\n" + toMedian(readTimes) +
        "\n\tCOPY_TO_DEV:\n" + toMedian(copyToDevTimes) +
        "\n\tTO_TABLE:\n" + toMedian(decodeTimes)
    );
  }

  @Test
  public void testSimpleRoundTrip() throws Exception {
    try (Table table = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
        .build()) {
      for (int numSlices = 1; numSlices < table.getRowCount(); numSlices++) {
        System.err.println("TEST WITH "+ numSlices);
        int[] slices = calcEvenSlices(table, numSlices);
        testRoundTrip("simple", table, slices);
      }
    }
  }

  public static Table buildMediumTable() {
    HostColumnVector.StructType st = new HostColumnVector.StructType(
        true,
        new HostColumnVector.BasicType(true, DType.INT8),
        new HostColumnVector.BasicType(true, DType.INT64)
    );
    return new Table.TestBuilder()
        .column("1", null, "34", "45", "56", "67")
        .column(new Integer[]{null},
            new Integer[]{4},
            new Integer[]{7},
            null,
            new Integer[]{},
            new Integer[]{})
        .column(st, new HostColumnVector.StructData(null, 11L),
            new HostColumnVector.StructData((byte) 2, null),
            new HostColumnVector.StructData((byte) 3, 33L),
            new HostColumnVector.StructData((byte) 4, 44L),
            new HostColumnVector.StructData((byte) 5, 55L),
            null)
        .column(null, 2, 3, 4, 5, 6)
        .build();
  }

  @Test
  public void testMediumRoundTrip() throws Exception {
    try (Table table = buildMediumTable()) {
      for (int numSlices = 1; numSlices < table.getRowCount(); numSlices++) {
        System.err.println("TEST WITH "+ numSlices);
        int[] slices = calcEvenSlices(table, numSlices);
        testRoundTrip("medium", table, slices);
      }
    }
  }

  public static Table buildHalfEmptyStructTable() {
    HostColumnVector.StructType st = new HostColumnVector.StructType(
        true,
        new HostColumnVector.BasicType(true, DType.INT32)
    );
    HostColumnVector.ListType lt = new HostColumnVector.ListType(true, st);
    return new Table.TestBuilder()
        .column(lt, structs(struct(1), struct(2), null),
            structs(struct(4), struct(5), struct(6)),
            structs(struct(7), struct(8), struct(9)),
            null,
            structs(),
            structs())
        .build();
  }

  @Test
  public void testHalfEmptyStructRoundTrip() throws Exception {
    try (Table table = buildHalfEmptyStructTable()) {
      for (int numSlices = 1; numSlices < table.getRowCount(); numSlices++) {
        System.err.println("TEST WITH "+ numSlices);
        int[] slices = calcEvenSlices(table, numSlices);
        testRoundTrip("half empty struct", table, slices);
      }
    }
  }

  @Test
  public void testSimpleMultiColumnRoundTrip() throws Exception {
    try (Table table = new Table.TestBuilder()
        .column(null, (byte)0xF0, (byte)0x0F, (byte)0xAA, null)
        .column((short)0xFFFF, (short)0xF0F0, null, (short)0xAAAA, (short)0x5555)
        .column("A","B","C","D",null)
        .column("0xFF", null, "0x0F", "0xAA", "0x55")
        .build()) {
      for (int numSlices = 1; numSlices < table.getRowCount(); numSlices++) {
        System.err.println("TEST WITH "+ numSlices);
        int[] slices = calcEvenSlices(table, numSlices);
        testRoundTrip("simple multicolumn", table, slices);
      }
    }
  }

  @Test
  public void testComplexRoundTrip() throws Exception {
    try (Table table = KudoSerializerTest.buildTestTable()) {
      for (int numSlices = 1; numSlices < table.getRowCount(); numSlices++) {
        System.err.println("TEST WITH "+ numSlices);
        int[] slices = calcEvenSlices(table, numSlices);
        testRoundTrip("complex", table, slices);
      }
    }
  }

  static Table buildStringListTable() {
    return new Table.TestBuilder()
        .column(strings("*"), strings("*"), strings("****"),
            strings("", "*", null))
        .column(strings(null, null, null, null), strings(),
            strings(null, null, null), strings())
        .build();
  }

  @Test
  public void testStringListRoundTrip() throws Exception {
    try (Table table = buildStringListTable()) {
      for (int numSlices = 1; numSlices < table.getRowCount(); numSlices++) {
        System.err.println("TEST WITH "+ numSlices);
        int[] slices = calcEvenSlices(table, numSlices);
        testRoundTrip("string list table", table, slices);
      }
    }
  }
}
