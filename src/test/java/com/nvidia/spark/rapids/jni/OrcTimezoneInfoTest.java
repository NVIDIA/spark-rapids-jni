/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class OrcTimezoneInfoTest {

  @Test
  void testGetFixedOffsetZone() {
    // Fixed-offset zones must return a non-null OrcTimezoneInfo with the
    // offset derived from ZoneRules (not from TimeZone.getTimeZone, which
    // would silently map "+05:30" to GMT). +05:30 == 19_800_000 ms.
    OrcTimezoneInfo info = OrcTimezoneInfo.get("+05:30");
    assertNotNull(info);
    assertEquals(19_800_000, info.rawOffset);
    assertNull(info.transitions);
    assertNull(info.offsets);
  }

  @Test
  void testGetCachesByKey() {
    // computeIfAbsent must return the same instance on the second call so
    // that other threads sharing RUNTIME_TIMEZONE_INFOS see a stable object.
    OrcTimezoneInfo a = OrcTimezoneInfo.get("Asia/Kolkata");
    OrcTimezoneInfo b = OrcTimezoneInfo.get("Asia/Kolkata");
    assertSame(a, b);
  }

  @Test
  void testGetThrowsOnInvalidId() {
    // Documented contract: invalid IDs throw IllegalArgumentException.
    // There is no silent fallback to GMT.
    assertThrows(IllegalArgumentException.class,
        () -> OrcTimezoneInfo.get("Invalid/Zone"));
  }

  @Test
  void testGetHistoricalTransitionsZone() {
    // Asia/Shanghai is a non-DST named zone with real historical transitions.
    // Verify that the runtime build path populates both arrays consistently.
    OrcTimezoneInfo info = OrcTimezoneInfo.get("Asia/Shanghai");
    assertNotNull(info);
    assertNotNull(info.transitions, "Asia/Shanghai should have historical transitions");
    assertNotNull(info.offsets);
    assertEquals(info.transitions.length, info.offsets.length,
        "transitions and offsets must be the same length");
    // Transitions must be strictly increasing so the GPU binary search is well-defined.
    for (int i = 1; i < info.transitions.length; i++) {
      assertTrue(info.transitions[i] > info.transitions[i - 1],
          "transitions must be strictly increasing");
    }
  }
}
