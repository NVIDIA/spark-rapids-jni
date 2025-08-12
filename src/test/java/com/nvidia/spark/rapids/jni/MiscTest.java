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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

import java.util.HashSet;
import java.util.UUID;

public class MiscTest {

  @Test
  void testUUIDInvalidInput() {
    // row count must be positive
    assertThrows(CudfException.class, () -> {
      Misc.randomUUIDs(0);
    });

    // row count must be positive
    assertThrows(CudfException.class, () -> {
      Misc.randomUUIDs(-1);
    });
  }

  @Test
  void testUUID() {
    int rowCount = 128;
    try (
        ColumnVector round1 = Misc.randomUUIDs(rowCount);
        ColumnVector round2 = Misc.randomUUIDs(rowCount);
        HostColumnVector h1 = round1.copyToHost();
        HostColumnVector h2 = round2.copyToHost()) {
      HashSet<String> set = new HashSet<>();
      for (int i = 0; i < rowCount; i++) {
        String uuidStr1 = h1.getJavaString(i);
        String uuidStr2 = h2.getJavaString(i);
        UUID uuid1 = UUID.fromString(uuidStr1);
        UUID uuid2 = UUID.fromString(uuidStr2);
        assertEquals(uuid1.version(), 4);
        assertEquals(uuid2.version(), 4);
        assertEquals(uuid1.variant(), 2);
        assertEquals(uuid2.variant(), 2);
        set.add(uuidStr1);
        set.add(uuidStr2);
      }
      assertEquals(set.size(), rowCount * 2);
    }
  }
}
