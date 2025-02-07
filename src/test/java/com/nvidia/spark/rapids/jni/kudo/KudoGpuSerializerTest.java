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
      assert(buffers.length == 2);
      try (DeviceMemoryBuffer data = buffers[0];
           DeviceMemoryBuffer offsets = buffers[1]) {
        try (HostMemoryBuffer tmp = HostMemoryBuffer.allocate(offsets.getLength())) {
          tmp.copyFromDeviceBuffer(offsets);
          for (int i = 0; i < tmp.getLength()/8; i++) {
            System.err.println("OFFSET " + i + ": " + tmp.getLong(0) + "/" + data.getLength());
          }
        }
        Schema s = Schema.builder()
            .column(DType.INT32, "a")
            .build();
        try (Table combined = KudoGpuSerializer.assembleFromDeviceRaw(s, data, offsets)) {
          TableDebug.get().debug("EXPECTED", table);
          TableDebug.get().debug("GOT", combined);
          assertTablesAreEqual(table, combined);
        }
      }
    }
  }


}
