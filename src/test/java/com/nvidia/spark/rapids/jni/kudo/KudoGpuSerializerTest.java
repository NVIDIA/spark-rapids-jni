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

  public static void logPartitionComparison(byte[] hDataGPU, byte[] hDataCPU) {
    System.err.println("     COMP GPU(" + hDataGPU.length + ") VS CPU(" + hDataCPU.length + ")");
    int len = Math.max(hDataGPU.length, hDataCPU.length);
    int hasValidLen = -1;
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
        hasValidLen = (gpuNumCol + 7) / 8;
      } else if (hasValidLen == 1 && i == 28) {
        extra = " <-- HAS_VALID_BUF";
      } else if (hasValidLen > 1 && i == 28) {
        extra = " <-- HAS_VALID_BUF START";
      } else if (hasValidLen > 1 && i == (27 + hasValidLen)) {
        extra = " <-- HAS_VALID_BUF END";
      }
      String i_str = String.format("%04d: ", i);
      System.err.println(i_str + diff + " " + gpu + " VS " + cpu  + extra);
    }
  }

  public static void logPartition(byte[] hData) {
    System.err.println("PARTITION (" + hData.length + ")");
    int len = hData.length;
    int hasValidLen = -1;
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
        hasValidLen = (gpuNumCol + 7) / 8;
      } else if (hasValidLen == 1 && i == 28) {
        extra = " <-- HAS_VALID_BUF";
      } else if (hasValidLen > 1 && i == 28) {
        extra = " <-- HAS_VALID_BUF START";
      } else if (hasValidLen > 1 && i == (27 + hasValidLen)) {
        extra = " <-- HAS_VALID_BUF END";
      }
      String i_str = String.format("%04d: ", i);
      System.err.println(i_str + dataAsString  + extra);
    }
  }

  @Test
  public void testSinglePartWriteCPURead() throws Exception {
    try (Table table = new Table.TestBuilder()
        //.column(null, (byte)0xF0, (byte)0x0F, (byte)0xAA, null)
        //.column((short)0xFFFF, (short)0xF0F0, null, (short)0xAAAA, (short)0x5555)
        .column("0xFF", null, "0x0F", "0xAA", "0x55")
        .build()) {
      DeviceMemoryBuffer[] buffers = KudoGpuSerializer.splitAndSerializeToDevice(table);
      assertEquals(2, buffers.length);
      try (DeviceMemoryBuffer data = buffers[0];
           DeviceMemoryBuffer offsets = buffers[1]) {
        // Ignoring the offsets for now because it should just be the start to the end of the buffer (one split)
        Schema s = Schema.builder()
          //  .column(DType.INT8, "a")
           // .column(DType.INT16, "b")
            .column(DType.STRING, "c")
            .build();
        KudoSerializer serializer = new KudoSerializer(s);
        ByteArrayOutputStream tmpOut = new ByteArrayOutputStream();
        serializer.writeToStreamWithMetrics(table, tmpOut, 0, 5);
        byte[] hDataCPU = tmpOut.toByteArray();
        byte[] hDataGPU = new byte[(int) data.getLength()]; // It will not be so large we need a long
        try (HostMemoryBuffer tmp = HostMemoryBuffer.allocate(data.getLength())) {
          tmp.copyFromDeviceBuffer(data);
          tmp.getBytes(hDataGPU, 0, 0, hDataGPU.length);
        }
        logPartitionComparison(hDataGPU, hDataCPU);

        // TODO verify that there is nothing more to read
        ByteArrayInputStream bin = new ByteArrayInputStream(hDataGPU);
        try (KudoTable kt = KudoTable.from(bin).get();
             Table combined = serializer.mergeToTable(Collections.singletonList(kt)).getLeft()) {
          assertTablesAreEqual(table, combined);
        }
      }
    }
  }
}
