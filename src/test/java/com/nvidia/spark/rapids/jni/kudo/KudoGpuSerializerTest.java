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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.stream.Collectors;

import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializerTest.schemaOf;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializerTest.strings;
import static org.junit.jupiter.api.Assertions.*;

public class KudoGpuSerializerTest {
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

  @Test
  public void testSimpleRoundTrip() {
    try (Table table = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
        .build()) {
      DeviceMemoryBuffer[] buffers = KudoGpuSerializer.splitAndSerializeToDevice(table, 5, 10, 15);
      assertEquals(2, buffers.length);
      try (DeviceMemoryBuffer data = buffers[0];
           DeviceMemoryBuffer offsets = buffers[1]) {
        Schema s = Schema.builder()
            .column(DType.INT32, "a")
            .build();
        try (Table combined = KudoGpuSerializer.assembleFromDeviceRaw(s, data, offsets)) {
          assertTablesAreEqual(table, combined);
        }
      }
    }
  }

  @Test
  public void testSinglePartition() {
    try (Table table = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
        .build()) {
      DeviceMemoryBuffer[] buffers = KudoGpuSerializer.splitAndSerializeToDevice(table);
      assertEquals(2, buffers.length);
      try (DeviceMemoryBuffer data = buffers[0];
           DeviceMemoryBuffer offsets = buffers[1]) {
        Schema s = Schema.builder()
            .column(DType.INT32, "a")
            .build();
        try (Table combined = KudoGpuSerializer.assembleFromDeviceRaw(s, data, offsets)) {
          assertTablesAreEqual(table, combined);
        }
      }
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

  private static String makeExtraEnd(String name, int start, byte[] hData) {
    int num = readIntHeader(start, hData);
    return " <-- " + name + " END (" + num + ")";
  }

  private static String makeOffsetEnd(int index, int start, byte[] hDataGPU, byte[] hDataCPU) {
    int gpuNum = readIntOffset(start, hDataGPU);
    int cpuNum = readIntOffset(start, hDataCPU);
    return " <-- OFFSET " + index + " (" + gpuNum + "/" + cpuNum + ")";
  }

  private static String makeOffsetEnd(int index, int start, byte[] hData) {
    int num = readIntOffset(start, hData);
    return " <-- OFFSET " + index + " (" + num + ")";
  }

  public static void logPartitionComparison(String name, byte[] hDataGPU, byte[] hDataCPU) {
    System.err.println(name + " COMP GPU(" + hDataGPU.length + ") VS CPU(" + hDataCPU.length + ")");
    int len = Math.max(hDataGPU.length, hDataCPU.length);
    int hasValidStart = -1;
    int hasValidEnd = -1;
    int validityBuffersStart = -1;
    int validityBuffersEnd = -1;
    int offsetBuffersStart = -1;
    int offsetBuffersEnd = -1;
    int dataBuffersStart = -1;
    int dataBuffersEnd = -1;
    for (int i = 0; i < len; i++) {
      Byte gpuByte = null;
      Byte cpuByte = null;
      String gpu = "N/A     ";
      if (i < hDataGPU.length) {
        gpuByte = hDataGPU[i];
        gpu = String.format("0x%02X %03d", hDataGPU[i] & 0xFF, hDataGPU[i]);
      }

      String cpu = "N/A     ";
      if (i < hDataCPU.length) {
        cpuByte = hDataCPU[i];
        cpu = String.format("0x%02X %03d", hDataCPU[i] & 0xFF, hDataCPU[i]);
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
        extra = makeExtraEnd("OFFSET", 4, hDataGPU, hDataCPU);
      } else if (i == 8) {
        extra = " <-- NUM_ROWS START";
      } else if (i == 11) {
        extra = makeExtraEnd("NUM_ROWS", 8, hDataGPU, hDataCPU);
      } else if (i == 12) {
        extra = " <-- VALIDITY_BUF_LEN START";
      } else if (i == 15) {
        extra = makeExtraEnd("VALIDITY_BUF_LEN", 12, hDataGPU, hDataCPU);
      } else if (i == 16) {
        extra = " <-- OFFSET_BUF_LEN START";
      } else if (i == 19) {
        extra = makeExtraEnd("OFFSET_BUF_LEN", 16, hDataGPU, hDataCPU);
      } else if (i == 20) {
        extra = " <-- TOTAL_DATA_LEN START";
      } else if (i == 23) {
        extra = makeExtraEnd("TOTAL_DATA_LEN", 20, hDataGPU, hDataCPU);
      } else if (i == 24) {
        extra = " <-- NUM_COL START";
      } else if (i == 27) {
        extra = makeExtraEnd("NUM_COL", 24, hDataGPU, hDataCPU);
        int numCol = readIntHeader(24, hDataCPU);
        int hasValidLen = (numCol + 7) / 8;
        if (hasValidLen > 0) {
          hasValidStart = 28;
          hasValidEnd = hasValidStart + hasValidLen - 1;
        } else {
          hasValidEnd = 28;
        }
        int validityBuffsLen = readIntHeader(12, hDataCPU);
        if (validityBuffsLen > 0) {
          validityBuffersStart = hasValidEnd + 1;
          validityBuffersEnd = validityBuffersStart + validityBuffsLen - 1;
        } else {
          validityBuffersEnd = hasValidEnd;
        }
        int offsetBuffersLen = readIntHeader(16, hDataCPU);
        if (offsetBuffersLen > 0) {
          offsetBuffersStart = validityBuffersEnd + 1;
          offsetBuffersEnd = offsetBuffersStart + offsetBuffersLen - 1;
        } else {
          offsetBuffersEnd = validityBuffersEnd;
        }
        int totalDataLen = readIntHeader(20, hDataCPU);
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
        extra = makeOffsetEnd(((i - offsetBuffersStart) / 4), i - 3, hDataGPU, hDataCPU);
      } else if (dataBuffersStart == dataBuffersEnd && i == dataBuffersStart) {
        extra = " <-- DATA_BUFFERS";
      } else if (dataBuffersStart == i) {
        extra = " <-- DATA_BUFFERS START";
      } else if (dataBuffersStart > 0 && dataBuffersEnd == i) {
        extra = " <-- DATA_BUFFERS END";
      }
      String i_str = String.format("%04d: ", i);
      String aligned = "        ";
      if (i % 4 == 0) {
        aligned = "ALIGNED ";
      }
      System.err.println(aligned + i_str + diff + " " + gpu + " VS " + cpu  + extra);
    }
  }

  public static void logPartition(String name, byte[] hData) {
    System.err.println(name + " PARTITION (" + hData.length + ")");
    int len = hData.length;
    int hasValidStart = -1;
    int hasValidEnd = -1;
    int validityBuffersStart = -1;
    int validityBuffersEnd = -1;
    int offsetBuffersStart = -1;
    int offsetBuffersEnd = -1;
    int dataBuffersStart = -1;
    int dataBuffersEnd = -1;
    for (int i = 0; i < len; i++) {
      String dataAsString = String.format("0x%02X %03d", hData[i] & 0xFF, hData[i]);

      String extra = "";
      if (i == 0) {
        extra = " <-- MAGIC START";
      } else if (i == 3) {
        extra = " <-- MAGIC END";
      } else if (i == 4) {
        extra = " <-- OFFSET START";
      } else if (i == 7) {
        extra = makeExtraEnd("OFFSET", 4, hData);
      } else if (i == 8) {
        extra = " <-- NUM_ROWS START";
      } else if (i == 11) {
        extra = makeExtraEnd("NUM_ROWS", 8, hData);
      } else if (i == 12) {
        extra = " <-- VALIDITY_BUF_LEN START";
      } else if (i == 15) {
        extra = makeExtraEnd("VALIDITY_BUF_LEN", 12, hData);
      } else if (i == 16) {
        extra = " <-- OFFSET_BUF_LEN START";
      } else if (i == 19) {
        extra = makeExtraEnd("OFFSET_BUF_LEN", 16, hData);
      } else if (i == 20) {
        extra = " <-- TOTAL_DATA_LEN START";
      } else if (i == 23) {
        extra = makeExtraEnd("TOTAL_DATA_LEN", 20, hData);
      } else if (i == 24) {
        extra = " <-- NUM_COL START";
      } else if (i == 27) {
        extra = makeExtraEnd("NUM_COL", 24, hData);
        int numCol = readIntHeader(24, hData);
        int hasValidLen = (numCol + 7) / 8;
        if (hasValidLen > 0) {
          hasValidStart = 28;
          hasValidEnd = hasValidStart + hasValidLen - 1;
        } else {
          hasValidEnd = 28;
        }
        int validityBuffsLen = readIntHeader(12, hData);
        if (validityBuffsLen > 0) {
          validityBuffersStart = hasValidEnd + 1;
          validityBuffersEnd = validityBuffersStart + validityBuffsLen - 1;
        } else {
          validityBuffersEnd = hasValidEnd;
        }
        int offsetBuffersLen = readIntHeader(16, hData);
        if (offsetBuffersLen > 0) {
          offsetBuffersStart = validityBuffersEnd + 1;
          offsetBuffersEnd = offsetBuffersStart + offsetBuffersLen - 1;
        } else {
          offsetBuffersEnd = validityBuffersEnd;
        }
        int totalDataLen = readIntHeader(20, hData);
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
        extra = makeOffsetEnd(((i - offsetBuffersStart) / 4), i - 3, hData);
      } else if (dataBuffersStart == dataBuffersEnd && i == dataBuffersStart) {
        extra = " <-- DATA_BUFFERS";
      } else if (dataBuffersStart == i) {
        extra = " <-- DATA_BUFFERS START";
      } else if (dataBuffersStart > 0 && dataBuffersEnd == i) {
        extra = " <-- DATA_BUFFERS END";
      }
      String i_str = String.format("%04d: ", i);
      String aligned = "        ";
      if (i % 4 == 0) {
        aligned = "ALIGNED ";
      }

      System.err.println(aligned + i_str + dataAsString  + extra);
    }
  }

  public void doSinglePartGPUWriteCPUReadTest(String name, Table table) throws Exception {
    DeviceMemoryBuffer[] buffers = KudoGpuSerializer.splitAndSerializeToDevice(table);
    assertEquals(2, buffers.length);
    try (DeviceMemoryBuffer data = buffers[0];
         DeviceMemoryBuffer offsets = buffers[1]) {
      // Ignoring the offsets for now because it should just be the start to the end of the buffer (one split)
      Schema s = KudoSerializerTest.schemaOf(table);
      KudoSerializer serializer = new KudoSerializer(s);
      ByteArrayOutputStream tmpOut = new ByteArrayOutputStream();
      serializer.writeToStreamWithMetrics(table, tmpOut, 0, (int)table.getRowCount());
      byte[] hDataCPU = tmpOut.toByteArray();
      byte[] hDataGPU = new byte[(int) data.getLength()]; // It will not be so large we need a long
      try (HostMemoryBuffer tmp = HostMemoryBuffer.allocate(data.getLength())) {
        tmp.copyFromDeviceBuffer(data);
        tmp.getBytes(hDataGPU, 0, 0, hDataGPU.length);
      }
      logPartitionComparison(name, hDataGPU, hDataCPU);

      // TODO verify that there is nothing more to read
      ByteArrayInputStream bin = new ByteArrayInputStream(hDataGPU);
      try (KudoTable kt = KudoTable.from(bin).get();
           Table combined = serializer.mergeToTable(Collections.singletonList(kt)).getLeft()) {
        TableDebug.get().debug("FROM CPU", combined);
        TableDebug.get().debug("expected", table);
        assertTablesAreEqual(table, combined);
      }
    }
  }

  public void doSinglePartGPUWriteGPUReadTest(String name, Table table) throws Exception {
    DeviceMemoryBuffer[] buffers = KudoGpuSerializer.splitAndSerializeToDevice(table);
    assertEquals(2, buffers.length);
    try (DeviceMemoryBuffer data = buffers[0];
         DeviceMemoryBuffer offsets = buffers[1]) {
      Schema s = KudoSerializerTest.schemaOf(table);
      KudoSerializer serializer = new KudoSerializer(s);
      ByteArrayOutputStream tmpOut = new ByteArrayOutputStream();
      serializer.writeToStreamWithMetrics(table, tmpOut, 0, (int)table.getRowCount());
      byte[] hDataCPU = tmpOut.toByteArray();
      byte[] hDataGPU = new byte[(int) data.getLength()]; // It will not be so large we need a long
      try (HostMemoryBuffer tmp = HostMemoryBuffer.allocate(data.getLength())) {
        tmp.copyFromDeviceBuffer(data);
        tmp.getBytes(hDataGPU, 0, 0, hDataGPU.length);
      }
      logPartitionComparison(name, hDataGPU, hDataCPU);
      TableDebug.get().debug("EXPECTED", table);

      try (Table combined = KudoGpuSerializer.assembleFromDeviceRaw(s, data, offsets)) {
        TableDebug.get().debug("FROM GPU", combined);
        assertTablesAreEqual(table, combined);
      }
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

  static Table buildSimpleTable() {
    HostColumnVector.StructType st = new HostColumnVector.StructType(
        true,
        new HostColumnVector.BasicType(true, DType.INT8),
        new HostColumnVector.BasicType(true, DType.INT64)
    );
    return new Table.TestBuilder()
//        .column("1", null, "34", "45", "56", "67")
//        .column(new Integer[]{1, 2, null},
//            new Integer[]{4, 5, 6},
//            new Integer[]{7, 8, 9},
//            null,
//            new Integer[]{},
//            new Integer[]{})
        .column(st, new HostColumnVector.StructData(null, 11L),
            new HostColumnVector.StructData((byte) 2, null),
            new HostColumnVector.StructData((byte) 3, 33L),
            new HostColumnVector.StructData((byte) 4, 44L),
            new HostColumnVector.StructData((byte) 5, 55L),
            null)
        .column(null, 2, 3, 4, 5, 6)
        .build();
  }

  static Table buildStringListTable() {
    return new Table.TestBuilder()
        .column(strings("*"), strings("*"), strings("****"),
            strings("", "*", null))
        .column(strings(null, null, null, null), strings(),
            strings(null, null, null), strings())
        .build();
  }

  public static void main(String [] args) {
    KudoGpuSerializerTest t = new KudoGpuSerializerTest();
    t.writePerfTest();
  }

  //@Test
  public void writePerfTest() {
    Rmm.initialize(RmmAllocationMode.CUDA_ASYNC, null, 5L * 1024 * 1024 * 1024);
    int numColumns = 10;
    int numIters = 9;
    int[] rowOptions = {10, 200, 1000, 10000, 100000, 1000000, 10000000};
    int[] sliceOptions = {10, 200, 1000, 10000, 100000};
    //int[] rowOptions = {10000000};
    //int[] sliceOptions = {200, 1000};
    ArrayList<Long> cpuTimes = new ArrayList<>();
    ArrayList<Long> gpuTimes = new ArrayList<>();
    for (int numSlices : sliceOptions) {
      System.err.println("\nSLICES: " + String.format("%,d", numSlices));
      for (int numRows : rowOptions) {
        System.err.println("ROWS: " + String.format("%,d", numRows));
        if (numSlices > numRows) {
          continue;
        }
        cpuTimes.clear();
        gpuTimes.clear();
        int rowsPerSlice = numRows / numSlices;
        long size = -1;
        int[] slices = new int[numSlices - 1];
        int offset = 0;
        //int strt;
        for (int i = 0; i < slices.length; i++) {
          //strt = offset;
          offset += rowsPerSlice;
          //System.err.println("EXPECTED " + i + " " + strt + " TO " + offset);
          slices[i] = offset;
        }
        //System.err.println("EXPECTED END " + offset + " TO " + numRows);
        try (Table t = perfTable(numRows, numColumns)) {
          // CPU TEST!!!
          HostColumnVector[] columns = new HostColumnVector[numColumns];
          try (NvtxRange r = new NvtxRange("PRE COPY TABLE TO HOST", NvtxColor.RED)) {
            for (int c = 0; c < numColumns; c++) {
              columns[c] = t.getColumn(c).copyToHost();
            }
          }
          try {
            for (int iter = 0; iter < numIters; iter++) {
              KudoSerializer ser = new KudoSerializer(schemaOf(t));
              JustCountOutputStream jc = new JustCountOutputStream();
              try (NvtxRange r = new NvtxRange(numSlices + " " + numRows + " CPU", NvtxColor.BLUE)) {
                long cpuStart = System.nanoTime();
                int start = 0;
                int end;
                for (int i = 0; i < numSlices; i++) {
                  if (i >= slices.length) {
                    end = numRows;
                  } else {
                    end = slices[i];
                  }
                  int len = end - start;
                  ser.writeToStreamWithMetrics(columns, jc, start, len);
                  start = end;
                }
                long cpuEnd = System.nanoTime();
                size = jc.length;
                cpuTimes.add((cpuEnd - cpuStart));
              }
            }
          } finally {
            for (int c = 0; c < numColumns; c++) {
              columns[c].close();
            }
          }
          System.err.println("CPU: " +
              " SIZE " + size +
              " WITH COPY\n=MEDIAN(" + cpuTimes.stream().map(String::valueOf).collect(Collectors.joining(",")) + ")");

          // GPU Test
          for (int iter = 0; iter < numIters; iter++) {
            try (NvtxRange r = new NvtxRange(numSlices + " " + numRows + " GPU", NvtxColor.GREEN)) {
              long gpuStart = System.nanoTime();
              DeviceMemoryBuffer[] buffers = KudoGpuSerializer.splitAndSerializeToDevice(t, slices);
              try {
                size = buffers[0].getLength();
                // We are not going to copy them back to the host here.
              } finally {
                buffers[0].close();
                buffers[1].close();
              }
              long gpuEnd = System.nanoTime();
              gpuTimes.add(gpuEnd - gpuStart);
            }
          }

          System.err.println("GPU: " +
              " SIZE " + size +
              " WITH COPY\n=MEDIAN(" + gpuTimes.stream().map(String::valueOf).collect(Collectors.joining(",")) + ")");
          System.err.println();
        }
      }
    }
  }

  @Test
  public void testSinglePartWriteCPURead() throws Exception {
    try (Table table = new Table.TestBuilder()
        .column(null, (byte)0xF0, (byte)0x0F, (byte)0xAA, null)
        .column((short)0xFFFF, (short)0xF0F0, null, (short)0xAAAA, (short)0x5555)
        .column("A","B","C","D",null)
        .column("0xFF", null, "0x0F", "0xAA", "0x55")
        .build()) {
     doSinglePartGPUWriteCPUReadTest("testSinglePartWriteCPURead", table);
    }
  }

  @Test
  public void testSimpleSinglePartWriteCPURead() throws Exception {
    try (Table table = buildSimpleTable()) {
      doSinglePartGPUWriteCPUReadTest("testSimpleSinglePartWriteCPURead", table);
    }
  }

  @Test
  public void testComplexSinglePartWriteCPURead() throws Exception {
    try (Table table = KudoSerializerTest.buildTestTable()) {
      doSinglePartGPUWriteCPUReadTest("testComplexSinglePartWriteCPURead", table);
    }
  }

  @Test
  public void testComplexSinglePartWriteGPURead() throws Exception {
    try (Table table = KudoSerializerTest.buildTestTable()) {
      doSinglePartGPUWriteGPUReadTest("testComplexSinglePartWriteGPURead", table);
    }
  }

  @Test
  public void testStringListCPURead() throws Exception {
    try (Table table = buildStringListTable()) {
      doSinglePartGPUWriteCPUReadTest("testStringListCPURead", table);
    }
  }

  @Test
  public void testStringListGPURead() throws Exception {
    try (Table table = buildStringListTable()) {
      doSinglePartGPUWriteGPUReadTest("testStringListGPURead", table);
    }
  }
}
