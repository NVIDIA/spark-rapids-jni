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

import ai.rapids.cudf.DType;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.Schema;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static org.junit.jupiter.api.Assertions.*;

public class KudoGpuSerializerTest {
  @Test
  public void testSimpleRoundTrip() {
    try (Table table = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        .column(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f)
        .build()) {
      DeviceMemoryBuffer[] buffers = KudoGpuSerializer.splitAndSerializeToDevice(table, 4);
      assert(buffers.length == 2);
      try (DeviceMemoryBuffer data = buffers[0];
           DeviceMemoryBuffer offsets = buffers[1]) {
        Schema s = Schema.builder()
            .column(DType.INT32, "a")
            .column(DType.FLOAT32, "b")
            .build();
        try (Table combined = KudoGpuSerializer.assembleFromDeviceRaw(s, data, offsets)) {
          assertTablesAreEqual(table, combined);
        }
      }
    }
  }


}
