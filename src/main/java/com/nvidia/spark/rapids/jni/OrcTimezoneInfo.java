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

import java.time.DateTimeException;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneRules;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TimeZone;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Holds ORC timezone metadata generated at runtime from public java.time/java.util APIs.
 * Historical transitions come from ZoneRules, while offsets before the first transition are
 * derived from java.util.TimeZone so ORC rebasing matches
 * SerializationUtils.convertBetweenTimezones semantics without relying on non-public ZoneInfo APIs.
 *
 * <p><b>Runtime dependency:</b> because the metadata is generated on the fly from
 * {@link java.util.TimeZone}/{@link java.time.zone.ZoneRules}, the exact transition table is
 * determined by the JVM's bundled IANA {@code tzdata}. Different JDK distributions or
 * {@code tzdata} versions may produce slightly different historical transitions for the same
 * zone id. This is strictly more correct than the previous frozen OpenJDK-8 snapshot, but users
 * debugging cross-environment differences should first check the JVM's {@code tzdata} version.
 */
class OrcTimezoneInfo {
  public OrcTimezoneInfo(int rawOffset, long[] transitions, int[] offsets) {
    this.rawOffset = rawOffset;
    this.transitions = transitions;
    this.offsets = offsets;
  }

  // in milliseconds
  final int rawOffset;

  // in milliseconds
  final long[] transitions;

  // in milliseconds
  final int[] offsets;

  // Lower bound of the range ORC supports (year 0001-01-01 UTC). Computed via
  // java.time.LocalDate, which uses the proleptic Gregorian calendar, whereas
  // java.util.TimeZone.getOffset(long) internally uses a hybrid Julian/Gregorian
  // calendar with the 1582 cutover for date-field interpretations. In practice
  // this difference does not affect offset lookup (which is purely instant-based
  // for ZoneInfo), so the two calendars agree on the offset at this instant.
  private static final long MIN_SUPPORTED_ORC_UTC_MILLIS = utcMillisForDate(1, 1, 1);
  private static final long HISTORICAL_TRANSITION_SCAN_STEP_MILLIS = 24L * 3600_000L;

  // year, month, and day are all 1-indexed, matching LocalDate.of conventions
  // (e.g. month=1 is January). This avoids the easy-to-misread mix of 0-based
  // month and 1-based day at the call site.
  private static long utcMillisForDate(int year, int month, int day) {
    return LocalDate.of(year, month, day).toEpochDay() * 24L * 3600_000L;
  }

  @Override
  public String toString() {
    return "OrcTimezoneInfo{" +
        "rawOffset=" + rawOffset +
        ", transitions=" + Arrays.toString(transitions) +
        ", offsets=" + Arrays.toString(offsets) +
        '}';
  }

  private static final ConcurrentMap<String, OrcTimezoneInfo> RUNTIME_TIMEZONE_INFOS =
      new ConcurrentHashMap<>();

  /**
   * Get timezone info for the specified timezone ID.
   * Historical transitions are generated at runtime from public JVM APIs and cached per ID.
   *
   * @param timezoneId timezone ID
   * @return timezone info
   * @throws IllegalArgumentException if {@code timezoneId} is not a valid zone ID accepted
   *     by {@link GpuTimeZoneDB#getZoneId(String)}. There is no silent fallback to GMT.
   */
  public static OrcTimezoneInfo get(String timezoneId) {
    return RUNTIME_TIMEZONE_INFOS.computeIfAbsent(
        timezoneId,
        OrcTimezoneInfo::buildRuntimeOrcTimezoneInfo);
  }

  /**
   * Build ORC timezone metadata from public java.time/java.util APIs. Invalid IDs use the same
   * validation as {@link GpuTimeZoneDB#getZoneId(String)} and fail with
   * {@link IllegalArgumentException} (no silent fallback to GMT).
   *
   * <p><b>Cost:</b> this is non-trivial — it scans every historical {@link ZoneOffsetTransition}
   * from year 1 onward. Results are cached in {@link #RUNTIME_TIMEZONE_INFOS} (see
   * {@link #get(String)}), so callers should always go through {@code get(...)} rather than
   * invoking this directly.
   */
  private static OrcTimezoneInfo buildRuntimeOrcTimezoneInfo(String timezoneId) {
    final ZoneId zoneId;
    try {
      zoneId = GpuTimeZoneDB.getZoneId(timezoneId);
    } catch (DateTimeException e) {
      throw new IllegalArgumentException("Timezone ID not found: " + timezoneId, e);
    }

    ZoneRules rules = zoneId.getRules();
    if (rules.isFixedOffset()) {
      // IDs like "+05:30" are valid ZoneIds but TimeZone.getTimeZone() silently
      // maps them to GMT (offset 0). Derive the offset from ZoneRules instead so
      // the GPU path doesn't treat them as UTC.
      int fixedOffsetMs = rules.getOffset(Instant.EPOCH).getTotalSeconds() * 1000;
      return new OrcTimezoneInfo(fixedOffsetMs, null, null);
    }
    TimeZone tz = TimeZone.getTimeZone(timezoneId);
    List<ZoneOffsetTransition> transitionList = rules.getTransitions();
    HistoricalTransitions historicalTransitions = buildHistoricalTransitions(tz, transitionList);
    if (historicalTransitions.transitions == null) {
      return new OrcTimezoneInfo(tz.getRawOffset(), null, null);
    }
    return new OrcTimezoneInfo(tz.getRawOffset(),
        historicalTransitions.transitions, historicalTransitions.offsets);
  }

  public static List<String> getAllTimezoneIds() {
    String[] ids = TimeZone.getAvailableIDs();
    Arrays.sort(ids);
    // Filter to ids that get(...) can actually build. TimeZone.getAvailableIDs()
    // on some JDK builds includes POSIX-style names (e.g. "EST5EDT", "SystemV/AST4")
    // that ZoneId.of(id, ZoneId.SHORT_IDS) rejects, which would make iterating this
    // list and feeding entries to OrcTimezoneInfo.get(...) throw IllegalArgumentException.
    List<String> result = new ArrayList<>(ids.length);
    for (String id : ids) {
      if (GpuTimeZoneDB.isSupportedTimeZone(id)) {
        result.add(id);
      }
    }
    return result;
  }

  private static int getInitialOffset(TimeZone tz) {
    // ORC only supports timestamps from year 0001 onward. For dates before the
    // first historical transition in that range, java.util.TimeZone can differ
    // from ZoneRules' earliest wall offset (for example, it may use the zone's
    // standard raw offset instead of an older LMT offset). Sample the beginning
    // of the supported range so the GPU matches TimeZone.getOffset().
    return tz.getOffset(MIN_SUPPORTED_ORC_UTC_MILLIS);
  }

  private static HistoricalTransitions buildHistoricalTransitions(
      TimeZone tz,
      List<ZoneOffsetTransition> transitionList) {
    if (transitionList.isEmpty()) {
      return HistoricalTransitions.EMPTY;
    }

    List<Long> transitions = new ArrayList<>();
    List<Integer> offsets = new ArrayList<>();
    long scanCursor = MIN_SUPPORTED_ORC_UTC_MILLIS;
    int currentOffset = getInitialOffset(tz);

    for (ZoneOffsetTransition transition : transitionList) {
      long transitionMs = transition.getInstant().toEpochMilli();
      if (transitionMs < MIN_SUPPORTED_ORC_UTC_MILLIS) {
        continue;
      }

      long beforeTransitionMs = transitionMs - 1;
      int offsetBeforeTransition = tz.getOffset(beforeTransitionMs);
      if (beforeTransitionMs >= scanCursor && offsetBeforeTransition != currentOffset) {
        currentOffset = collectTimeZoneTransitionsByScanning(
            tz, scanCursor, beforeTransitionMs, currentOffset, transitions, offsets);
      }

      int offsetAtTransition = tz.getOffset(transitionMs);
      if (offsetAtTransition != offsetBeforeTransition) {
        transitions.add(transitionMs);
        offsets.add(offsetAtTransition);
        currentOffset = offsetAtTransition;
      }
      scanCursor = transitionMs;
    }

    if (transitions.isEmpty()) {
      return HistoricalTransitions.EMPTY;
    }
    return new HistoricalTransitions(toLongArray(transitions), toIntArray(offsets));
  }

  private static int collectTimeZoneTransitionsByScanning(
      TimeZone tz,
      long scanStartMs,
      long scanEndMs,
      int startOffset,
      List<Long> transitions,
      List<Integer> offsets) {
    long cursor = scanStartMs;
    int currentOffset = startOffset;
    while (cursor < scanEndMs) {
      long probe = Math.min(cursor + HISTORICAL_TRANSITION_SCAN_STEP_MILLIS, scanEndMs);
      int probeOffset = tz.getOffset(probe);
      if (probeOffset == currentOffset) {
        cursor = probe;
        continue;
      }

      long exactTransition = binarySearchTransition(tz, cursor, probe);
      int offsetAfterTransition = tz.getOffset(exactTransition);
      transitions.add(exactTransition);
      offsets.add(offsetAfterTransition);
      currentOffset = offsetAfterTransition;
      cursor = exactTransition;
    }
    return currentOffset;
  }

  private static long binarySearchTransition(TimeZone tz, long lo, long hi) {
    int loOffset = tz.getOffset(lo);
    while (hi - lo > 1) {
      long mid = lo + (hi - lo) / 2;
      if (tz.getOffset(mid) == loOffset) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return hi;
  }

  private static long[] toLongArray(List<Long> values) {
    long[] result = new long[values.size()];
    for (int i = 0; i < values.size(); i++) {
      result[i] = values.get(i);
    }
    return result;
  }

  private static int[] toIntArray(List<Integer> values) {
    int[] result = new int[values.size()];
    for (int i = 0; i < values.size(); i++) {
      result[i] = values.get(i);
    }
    return result;
  }

  private static final class HistoricalTransitions {
    static final HistoricalTransitions EMPTY = new HistoricalTransitions(null, null);

    final long[] transitions;
    final int[] offsets;

    private HistoricalTransitions(long[] transitions, int[] offsets) {
      this.transitions = transitions;
      this.offsets = offsets;
    }
  }
}
