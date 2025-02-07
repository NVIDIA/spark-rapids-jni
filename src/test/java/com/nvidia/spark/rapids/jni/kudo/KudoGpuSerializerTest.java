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

  @Test
  public void testSinglePartWriteCPURead() throws Exception {
    try (Table table = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
        .build()) {
      DeviceMemoryBuffer[] buffers = KudoGpuSerializer.splitAndSerializeToDevice(table);
      assertEquals(2, buffers.length);
      try (DeviceMemoryBuffer data = buffers[0];
           DeviceMemoryBuffer offsets = buffers[1]) {
        // Ignoring the offsets for now because it should just be the start to the end of the buffer (one split)
        Schema s = Schema.builder()
            .column(DType.INT32, "a")
            .build();
        KudoSerializer serializer = new KudoSerializer(s);
        byte[] hData = new byte[(int) data.getLength()]; // It will not be so large we need a long
        try (HostMemoryBuffer tmp = HostMemoryBuffer.allocate(data.getLength())) {
          tmp.copyFromDeviceBuffer(data);
          tmp.getBytes(hData, 0, 0, hData.length);
        }
        // TODO verify that there is nothing more to read
        ByteArrayInputStream bin = new ByteArrayInputStream(hData);
        try (KudoTable kt = KudoTable.from(bin, serializer.getColumnCount()).get();
             Table combined = serializer.mergeToTable(Collections.singletonList(kt)).getLeft()) {
          assertTablesAreEqual(table, combined);
        }
      }
    }
  }
}
