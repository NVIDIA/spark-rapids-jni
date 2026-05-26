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

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
  void testGetFixedOffsetNamedZone() {
    // "UTC" is a named zone whose ZoneRules.isFixedOffset() is true. Cover
    // it explicitly so a regression that treats "UTC" as a historical zone
    // (non-null transitions) — or that silently maps it to GMT via
    // TimeZone.getTimeZone — is caught. rawOffset must be 0.
    OrcTimezoneInfo info = OrcTimezoneInfo.get("UTC");
    assertNotNull(info);
    assertEquals(0, info.rawOffset);
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
  void testGetAllTimezoneIdsContract() {
    List<String> ids = OrcTimezoneInfo.getAllTimezoneIds();

    assertFalse(ids.isEmpty(), "expected at least one supported timezone id");
    assertTrue(ids.contains("UTC"), "UTC must be present");
    assertTrue(ids.contains("Asia/Shanghai"), "Asia/Shanghai must be present");

    // Sorted ascending AND distinct (strict <, not <=, also catches duplicate ids).
    for (int i = 1; i < ids.size(); i++) {
      assertTrue(ids.get(i - 1).compareTo(ids.get(i)) < 0,
          "list must be sorted and distinct: " + ids.get(i - 1) + " >= " + ids.get(i));
    }

    // Every id must be one that OrcTimezoneInfo.get can build — i.e. the lister
    // and the loader agree.
    for (String id : ids) {
      assertTrue(GpuTimeZoneDB.isSupportedTimeZone(id),
          "getAllTimezoneIds returned an id that isSupportedTimeZone rejects: " + id);
    }
  }

  @Test
  void testGetHistoricalTransitionsZone() {
    // Asia/Shanghai is a non-DST named zone with real historical transitions.
    // Verify that the runtime build path populates both arrays consistently.
    //
    // Known coverage gap: zones whose ZoneRules.getTransitions() is empty
    // but whose historical offset changed cannot exercise the scan-only
    // path in collectTimeZoneTransitionsByScanning, because
    // buildHistoricalTransitions returns EMPTY early for empty transition
    // lists. Covering it would require a synthetic zone.
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
