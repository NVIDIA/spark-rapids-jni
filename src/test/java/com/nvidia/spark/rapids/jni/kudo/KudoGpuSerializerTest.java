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
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static org.junit.jupiter.api.Assertions.*;

public class KudoGpuSerializerTest {
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
        extra = " <-- OFFSET END";
      } else if (i == 8) {
        extra = " <-- NUM_ROWS START";
      } else if (i == 11) {
        extra = " <-- NUM_ROWS END";
      } else if (i == 12) {
        extra = " <-- VALIDITY_BUF_LEN START";
      } else if (i == 15) {
        extra = " <-- VALIDITY_BUF_LEN END";
      } else if (i == 16) {
        extra = " <-- OFFSET_BUF_LEN START";
      } else if (i == 19) {
        extra = " <-- OFFSET_BUF_LEN END";
      } else if (i == 20) {
        extra = " <-- TOTAL_DATA_LEN START";
      } else if (i == 23) {
        extra = " <-- TOTAL_DATA_LEN END";
      } else if (i == 24) {
        extra = " <-- NUM_COL START";
      } else if (i == 27) {
        extra = " <-- NUM_COL END";
        int gpuNumCol = (hDataGPU[24] << 24) + (hDataGPU[25] << 16) + (hDataGPU[26] << 8) + hDataGPU[27];
        int hasValidLen = (gpuNumCol + 7) / 8;
        if (hasValidLen > 0) {
          hasValidStart = 28;
          hasValidEnd = hasValidStart + hasValidLen - 1;
        } else {
          hasValidEnd = 28;
        }
        int validityBuffsLen = (hDataGPU[12] << 24) + (hDataGPU[13] << 16) + (hDataGPU[14] << 8) + hDataGPU[15];
        if (validityBuffsLen > 0) {
          validityBuffersStart = hasValidEnd + 1;
          validityBuffersEnd = validityBuffersStart + validityBuffsLen - 1;
        } else {
          validityBuffersEnd = hasValidEnd;
        }
        int offsetBuffersLen = (hDataGPU[16] << 24) + (hDataGPU[17] << 16) + (hDataGPU[18] << 8) + hDataGPU[19];
        if (offsetBuffersLen > 0) {
          offsetBuffersStart = validityBuffersEnd + 1;
          offsetBuffersEnd = offsetBuffersStart + offsetBuffersLen;
        } else {
          offsetBuffersEnd = validityBuffersEnd;
        }
        int totalDataLen = (hDataGPU[20] << 24) + (hDataGPU[21] << 16) + (hDataGPU[22] << 8) + hDataGPU[23];
        int dataLen = totalDataLen - offsetBuffersLen - validityBuffsLen;
        if (dataLen > 0) {
          dataBuffersStart = offsetBuffersEnd + 1;
          dataBuffersEnd = dataBuffersStart + dataLen;
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
      } else if (offsetBuffersStart > 0 && offsetBuffersEnd == i) {
        extra = " <-- OFFSET_BUFFERS END";
      } else if (dataBuffersStart == dataBuffersEnd && i == dataBuffersStart) {
        extra = " <-- DATA_BUFFERS";
      } else if (dataBuffersStart == i) {
        extra = " <-- DATA_BUFFERS START";
      } else if (dataBuffersStart > 0 && dataBuffersEnd == i) {
        extra = " <-- DATA_BUFFERS END";
      }
      String i_str = String.format("%04d: ", i);
      System.err.println(i_str + diff + " " + gpu + " VS " + cpu  + extra);
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
        extra = " <-- OFFSET END";
      } else if (i == 8) {
        extra = " <-- NUM_ROWS START";
      } else if (i == 11) {
        extra = " <-- NUM_ROWS END";
      } else if (i == 12) {
        extra = " <-- VALIDITY_BUF_LEN START";
      } else if (i == 15) {
        extra = " <-- VALIDITY_BUF_LEN END";
      } else if (i == 16) {
        extra = " <-- OFFSET_BUF_LEN START";
      } else if (i == 19) {
        extra = " <-- OFFSET_BUF_LEN END";
      } else if (i == 20) {
        extra = " <-- TOTAL_DATA_LEN START";
      } else if (i == 23) {
        extra = " <-- TOTAL_DATA_LEN END";
      } else if (i == 24) {
        extra = " <-- NUM_COL START";
      } else if (i == 27) {
        extra = " <-- NUM_COL END";
        int gpuNumCol = (hData[24] << 24) + (hData[25] << 16) + (hData[26] << 8) + hData[27];
        int hasValidLen = (gpuNumCol + 7) / 8;
        if (hasValidLen > 0) {
          hasValidStart = 28;
          hasValidEnd = hasValidStart + hasValidLen - 1;
        } else {
          hasValidEnd = 28;
        }
        int validityBuffsLen = (hData[12] << 24) + (hData[13] << 16) + (hData[14] << 8) + hData[15];
        if (validityBuffsLen > 0) {
          validityBuffersStart = hasValidEnd + 1;
          validityBuffersEnd = validityBuffersStart + validityBuffsLen - 1;
        } else {
          validityBuffersEnd = hasValidEnd;
        }
        int offsetBuffersLen = (hData[16] << 24) + (hData[17] << 16) + (hData[18] << 8) + hData[19];
        if (offsetBuffersLen > 0) {
          offsetBuffersStart = validityBuffersEnd + 1;
          offsetBuffersEnd = offsetBuffersStart + offsetBuffersLen;
        } else {
          offsetBuffersEnd = validityBuffersEnd;
        }
        int totalDataLen = (hData[20] << 24) + (hData[21] << 16) + (hData[22] << 8) + hData[23];
        int dataLen = totalDataLen - offsetBuffersLen - validityBuffsLen;
        if (dataLen > 0) {
          dataBuffersStart = offsetBuffersEnd + 1;
          dataBuffersEnd = dataBuffersStart + dataLen;
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
      } else if (offsetBuffersStart > 0 && offsetBuffersEnd == i) {
        extra = " <-- OFFSET_BUFFERS END";
      } else if (dataBuffersStart == dataBuffersEnd && i == dataBuffersStart) {
        extra = " <-- DATA_BUFFERS";
      } else if (dataBuffersStart == i) {
        extra = " <-- DATA_BUFFERS START";
      } else if (dataBuffersStart > 0 && dataBuffersEnd == i) {
        extra = " <-- DATA_BUFFERS END";
      }
      String i_str = String.format("%04d: ", i);
      System.err.println(i_str + dataAsString  + extra);
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
      serializer.writeToStreamWithMetrics(table, tmpOut, 0, 5);
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
        assertTablesAreEqual(table, combined);
      }
    }
  }

  @Test
  public void testSinglePartWriteCPURead() throws Exception {
    try (Table table = new Table.TestBuilder()
        .column(null, (byte)0xF0, (byte)0x0F, (byte)0xAA, null)
//        .column((short)0xFFFF, (short)0xF0F0, null, (short)0xAAAA, (short)0x5555)
//        .column("0xFF", null, "0x0F", "0xAA", "0x55")
        .build()) {
     doSinglePartGPUWriteCPUReadTest("testSinglePartWriteCPURead", table);
    }
  }

//  @Test
//  public void testSimpleSinglePartWriteCPURead() throws Exception {
//    try (Table table = KudoSerializerTest.buildSimpleTable()) {
//      doSinglePartGPUWriteCPUReadTest("testSimpleSinglePartWriteCPURead", table);
//    }
//  }
}
