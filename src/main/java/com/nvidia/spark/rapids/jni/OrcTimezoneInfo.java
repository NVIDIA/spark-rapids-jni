/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneOffsetTransitionRule;
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
  // Base probe width used by collectTimeZoneTransitionsByScanning. The scanner
  // detects a transition by sampling tz.getOffset(probe) and comparing it to
  // the running offset; a pair of transitions A->B->A whose two endpoints fall
  // inside one probe step will net to zero and slip through. 6 hours is
  // smaller than the minimum spacing between any two real transitions in the
  // current IANA tzdata (the closest pairs are DST start/end, ~hours apart on
  // separate days), so paired transitions cannot hide in a single window.
  private static final long HISTORICAL_TRANSITION_SCAN_STEP_MILLIS = 6L * 3600_000L;

  // Reference years used to cross-check the extracted DST rule against
  // java.util.TimeZone.getOffset. A near-future anchor (2060) catches
  // divergence within the typical application lifetime; the far-future
  // anchors exercise the recurring-rule fallback well past any historical
  // transition entry in tzdata.
  private static final int[] DST_RULE_VALIDATION_YEARS = {2060, 2400, 9997};

  /**
   * Recurring DST rule for a single zone, encoded in the same shape that
   * {@link java.util.SimpleTimeZone} stores internally and that the GPU side
   * consumes. {@code month} is 0-based (Calendar.JANUARY=0), {@code dayOfWeek}
   * follows Calendar's 1=Sun..7=Sat convention, and {@code dstSavings} /
   * {@code time} are in milliseconds.
   *
   * <p>The {@code *Mode} fields encode how {@code *Day}/{@code *DayOfWeek}
   * combine:
   * <ul>
   *   <li>0 (DOM): exact day of month; {@code dayOfWeek} ignored</li>
   *   <li>1 (DOW_IN_MONTH): nth {@code dayOfWeek} in month
   *       ({@code day} negative = from end)</li>
   *   <li>2 (DOW_GE_DOM): first {@code dayOfWeek} on or after {@code day}</li>
   *   <li>3 (DOW_LE_DOM): last {@code dayOfWeek} on or before {@code day}</li>
   * </ul>
   *
   * <p>{@code *TimeMode}: 0=WALL, 1=STANDARD, 2=UTC.
   */
  static final class DstRule {
    // Day-rule modes for {start,end}Mode — matches SimpleTimeZone's internal
    // encoding so the GPU side can consume the values directly.
    static final int MODE_DOM          = 0;
    static final int MODE_DOW_IN_MONTH = 1;
    static final int MODE_DOW_GE_DOM   = 2;
    static final int MODE_DOW_LE_DOM   = 3;

    // Time-of-day basis for {start,end}TimeMode.
    static final int TIME_MODE_WALL     = 0;
    static final int TIME_MODE_STANDARD = 1;
    static final int TIME_MODE_UTC      = 2;

    int dstSavings;
    int startMonth;
    int startDay;
    int startDayOfWeek;
    int startTime;
    int startTimeMode;
    int startMode;
    int endMonth;
    int endDay;
    int endDayOfWeek;
    int endTime;
    int endTimeMode;
    int endMode;
  }

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
    // Use the canonical ID from the resolved ZoneId (e.g. "Asia/Kolkata" for
    // input "IST") so that TimeZone and ZoneRules always refer to the same
    // zone, regardless of how the JVM's legacy TimeZone database maps
    // 3-letter aliases. ZoneId.SHORT_IDS in getZoneId resolves "IST" to
    // "Asia/Kolkata"; TimeZone.getTimeZone("IST") may map to a different
    // zone on some JVM distributions, which would silently produce mixed
    // offset data with no exception.
    TimeZone tz = TimeZone.getTimeZone(zoneId.getId());
    List<ZoneOffsetTransition> transitionList = rules.getTransitions();
    HistoricalTransitions historicalTransitions = buildHistoricalTransitions(tz, transitionList);
    if (historicalTransitions.transitions == null) {
      return new OrcTimezoneInfo(tz.getRawOffset(), null, null);
    }
    return new OrcTimezoneInfo(tz.getRawOffset(),
        historicalTransitions.transitions, historicalTransitions.offsets);
  }

  /**
   * Returns the sorted list of timezone IDs that {@link #get(String)} can build —
   * the intersection of {@link TimeZone#getAvailableIDs()} and
   * {@link GpuTimeZoneDB#isSupportedTimeZone(String)}. POSIX-style entries (e.g.
   * {@code "EST5EDT"}, {@code "SystemV/AST4"}) that some JDK builds expose but
   * {@code ZoneId.of(id, ZoneId.SHORT_IDS)} rejects are filtered out.
   *
   * <p>The result is computed on every call; callers that need it repeatedly
   * should cache it themselves.
   *
   * @return sorted list of ORC-supported timezone IDs
   */
  public static List<String> getAllTimezoneIds() {
    String[] ids = TimeZone.getAvailableIDs();
    Arrays.sort(ids);
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
      // Invariant: between two consecutive entries returned by
      // ZoneRules.getTransitions(), the wall offset is constant — no hidden
      // paired round-trips (e.g. A->B->A) net to zero between entries. If
      // that ever breaks (DST zones, future tzdata revisions), the guard
      // below will not fire and both transitions in the pair will be
      // silently dropped. The DST guard in
      // GpuTimeZoneDB.convertOrcTimezones currently keeps this dormant;
      // any follow-up that relaxes it must revisit this code.
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
      // Exponentially expand the probe step while the offset stays equal to
      // currentOffset. This collapses long no-transition stretches (e.g. the
      // year-0001-to-first-historical-transition gap, ~1880 years for typical
      // IANA zones) from O(N) day probes to O(log N). Once the probe lands on
      // a different offset, the [lo, hi] bracket contains a transition and we
      // hand it to binarySearchTransition. The bracket may be wider than the
      // base 6h step, so this assumes at most one offset transition lives in
      // the expanded window — which holds for real IANA data; A->B->A pairs
      // narrower than the base step are addressed separately by the step size.
      long lo = cursor;
      long step = HISTORICAL_TRANSITION_SCAN_STEP_MILLIS;
      long hi = Math.min(lo + step, scanEndMs);
      int hiOffset = tz.getOffset(hi);
      while (hiOffset == currentOffset && hi < scanEndMs) {
        lo = hi;
        step = Math.min(step * 2L, scanEndMs - hi);
        hi = lo + step;
        hiOffset = tz.getOffset(hi);
      }
      if (hiOffset == currentOffset) {
        // Reached scanEndMs without seeing any transition.
        cursor = hi;
        continue;
      }

      long exactTransition = binarySearchTransition(tz, lo, hi);
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

  // ---------------------------------------------------------------------------
  // DST rule extraction (used by the GPU DST path; not wired into
  // buildRuntimeOrcTimezoneInfo yet — see Part 2 of the ORC-timezone work).
  // ---------------------------------------------------------------------------

  /**
   * Extract the recurring DST rule for a zone, or {@code null} if the zone has
   * no DST. The probing path is tried first because it captures what
   * {@link java.util.TimeZone#getOffset(long)} actually returns (which is the
   * source of truth the GPU side must match for ORC byte-compatibility);
   * {@link ZoneRules#getTransitionRules()} is used as a fallback for zones
   * whose recurring rule cannot be recovered from hourly probes.
   *
   * @throws IllegalStateException if the zone reports DST but neither
   *     extraction path produces a usable rule — for example, an unsupported
   *     {@link ZoneRules#getTransitionRules()} count (not 0 and not 2), a
   *     transition rule shape outside DOW_GE_DOM, mismatched start vs. end
   *     DST savings, zero-delta savings, or cross-year verification mismatch
   *     against {@code tz.getOffset} on the anchor years
   *     {@code 2060, 2400, 9997}.
   */
  static DstRule extractDstRule(String timezoneId, TimeZone tz, ZoneRules rules) {
    // Fixed-offset zones (e.g. "+05:30") have no DST. Guard explicitly because
    // TimeZone.getTimeZone(zoneId) silently returns GMT for such ids on most
    // JVMs, which would leave `tz` describing a different zone than `rules`.
    // Mirrors the guard in buildRuntimeOrcTimezoneInfo.
    if (rules.isFixedOffset() || !tz.useDaylightTime()) {
      return null;
    }
    DstRule rule = extractDstRuleByProbing(tz);
    if (rule != null) {
      return rule;
    }
    rule = extractDstRuleFromZoneRules(timezoneId, tz, rules);
    if (rule != null) {
      return rule;
    }
    throw new IllegalStateException("Failed to extract ORC DST rule for timezone: " + timezoneId);
  }

  // ---- Path A: from ZoneRules.getTransitionRules() ----

  private static DstRule extractDstRuleFromZoneRules(String timezoneId, TimeZone tz,
      ZoneRules rules) {
    List<ZoneOffsetTransitionRule> transitionRules = rules.getTransitionRules();
    if (transitionRules.isEmpty()) {
      return null;
    }
    if (transitionRules.size() != 2) {
      throw new IllegalStateException("Unsupported ORC DST rule count for timezone: " + timezoneId);
    }

    ZoneOffsetTransitionRule startTransitionRule = null;
    ZoneOffsetTransitionRule endTransitionRule = null;
    for (ZoneOffsetTransitionRule transitionRule : transitionRules) {
      int deltaMillis = (transitionRule.getOffsetAfter().getTotalSeconds()
          - transitionRule.getOffsetBefore().getTotalSeconds()) * 1000;
      if (deltaMillis > 0) {
        startTransitionRule = transitionRule;
      } else if (deltaMillis < 0) {
        endTransitionRule = transitionRule;
      } else {
        throw new IllegalStateException("Unsupported zero-delta ORC DST rule for timezone: "
            + timezoneId);
      }
    }
    if (startTransitionRule == null || endTransitionRule == null) {
      throw new IllegalStateException("Failed to identify ORC DST start/end rules for timezone: "
          + timezoneId);
    }

    int dstSavings = (startTransitionRule.getOffsetAfter().getTotalSeconds()
        - startTransitionRule.getOffsetBefore().getTotalSeconds()) * 1000;
    int endDeltaMillis = (endTransitionRule.getOffsetBefore().getTotalSeconds()
        - endTransitionRule.getOffsetAfter().getTotalSeconds()) * 1000;
    if (dstSavings != endDeltaMillis) {
      throw new IllegalStateException("Mismatched ORC DST savings for timezone: " + timezoneId);
    }

    DstRule rule = new DstRule();
    rule.dstSavings = dstSavings;
    fillDstRuleFromTransitionRule(timezoneId, rule, startTransitionRule, true);
    fillDstRuleFromTransitionRule(timezoneId, rule, endTransitionRule, false);

    if (!verifyDstRuleAcrossReferenceYears(tz, rule)) {
      throw new IllegalStateException("ZoneRules ORC DST rule verification failed for timezone: "
          + timezoneId);
    }
    return rule;
  }

  private static void fillDstRuleFromTransitionRule(String timezoneId, DstRule rule,
      ZoneOffsetTransitionRule transitionRule, boolean isStartRule) {
    // We only accept rules shaped as "first <dayOfWeek> on or after <dayOfMonthIndicator>",
    // i.e. ZoneRules' positive-day-indicator form. A negative indicator would mean
    // "last <dayOfWeek> on or before day" (DOW_LE_DOM_MODE, mode=3); we reject those
    // here so that downstream code can assume DOW_GE_DOM_MODE (mode=2) unconditionally.
    if (transitionRule.getDayOfWeek() == null
        || transitionRule.getDayOfMonthIndicator() <= 0) {
      throw new IllegalStateException("Unsupported ORC DST transition rule shape for timezone: "
          + timezoneId);
    }

    int month = transitionRule.getMonth().getValue() - 1;
    int day = transitionRule.getDayOfMonthIndicator();
    int dayOfWeek = toCalendarDayOfWeek(transitionRule.getDayOfWeek().getValue());
    int time = getTransitionRuleTimeMillis(transitionRule);
    int timeMode = getTransitionRuleTimeMode(transitionRule);
    // Guaranteed by the precondition above (DayOfMonthIndicator > 0).
    int mode = DstRule.MODE_DOW_GE_DOM;

    if (isStartRule) {
      rule.startMonth = month;
      rule.startDay = day;
      rule.startDayOfWeek = dayOfWeek;
      rule.startTime = time;
      rule.startTimeMode = timeMode;
      rule.startMode = mode;
    } else {
      rule.endMonth = month;
      rule.endDay = day;
      rule.endDayOfWeek = dayOfWeek;
      rule.endTime = time;
      rule.endTimeMode = timeMode;
      rule.endMode = mode;
    }
  }

  private static int getTransitionRuleTimeMillis(ZoneOffsetTransitionRule transitionRule) {
    int secondOfDay = transitionRule.isMidnightEndOfDay()
        ? 24 * 3600
        : transitionRule.getLocalTime().toSecondOfDay();
    return secondOfDay * 1000;
  }

  private static int getTransitionRuleTimeMode(ZoneOffsetTransitionRule transitionRule) {
    ZoneOffsetTransitionRule.TimeDefinition timeDef = transitionRule.getTimeDefinition();
    if (ZoneOffsetTransitionRule.TimeDefinition.UTC == timeDef) {
      return DstRule.TIME_MODE_UTC;
    } else if (ZoneOffsetTransitionRule.TimeDefinition.STANDARD == timeDef) {
      return DstRule.TIME_MODE_STANDARD;
    } else {
      return DstRule.TIME_MODE_WALL;
    }
  }

  private static int toCalendarDayOfWeek(int javaTimeDayOfWeek) {
    // java.time DayOfWeek: 1=Mon..7=Sun  ->  Calendar: 1=Sun..7=Sat
    return (javaTimeDayOfWeek % 7) + 1;
  }

  // ---- Path B: probe tz.getOffset() and recover the recurring rule ----

  private static DstRule extractDstRuleByProbing(TimeZone tz) {
    for (int refYear : DST_RULE_VALIDATION_YEARS) {
      DstRule rule = extractDstRuleByProbing(tz, refYear);
      if (rule != null && verifyDstRuleAcrossReferenceYears(tz, rule)) {
        return rule;
      }
    }
    return null;
  }

  private static DstRule extractDstRuleByProbing(TimeZone tz, int refYear) {
    long janFirst = utcMillisForDate(refYear, 1, 1);
    long nextJanFirst = utcMillisForDate(refYear + 1, 1, 1);

    long dstOnTransition = -1;
    long dstOffTransition = -1;
    int initialOffset = tz.getOffset(janFirst - 1);
    int prevOffset = initialOffset;
    long step = 3600_000L; // 1 hour

    for (long ms = janFirst; ms < nextJanFirst; ms += step) {
      int curOffset = tz.getOffset(ms);
      if (curOffset != prevOffset) {
        long exactMs = binarySearchTransition(tz, ms - step, ms);
        if (curOffset > prevOffset) {
          // More than one DST-on transition in the same year would mean this
          // year doesn't fit a SimpleTimeZone-style two-transition rule; let
          // the caller fall back to extractDstRuleFromZoneRules.
          if (dstOnTransition >= 0) return null;
          dstOnTransition = exactMs;
        } else {
          if (dstOffTransition >= 0) return null;
          dstOffTransition = exactMs;
        }
        prevOffset = curOffset;
      }
      // SimpleTimeZone-style zones have exactly two transitions per year.
      // Once both are recorded and the running offset is back to the year-
      // start standard offset, the remainder of the year is a no-op tail
      // (~1-3 months for typical northern-hemisphere zones). Any later
      // change would correctly trigger the "more than one DST-on/off"
      // early-return above, so the only thing this break can skip is
      // wasted probes -- ~720-2200 calls per year on common DST zones.
      if (dstOnTransition >= 0 && dstOffTransition >= 0 && prevOffset == initialOffset) {
        break;
      }
    }

    if (dstOnTransition < 0 || dstOffTransition < 0) {
      return null;
    }

    DstRule rule = new DstRule();
    rule.dstSavings = tz.getDSTSavings();

    int[] startFields = decodeTransition(dstOnTransition, tz.getRawOffset());
    rule.startMonth = startFields[0];
    rule.startDay = startFields[1];
    rule.startDayOfWeek = startFields[2];
    rule.startTime = startFields[3];
    // decodeTransition converts to standard local time.
    rule.startTimeMode = DstRule.TIME_MODE_STANDARD;
    rule.startMode = startFields[4];

    int[] endFields = decodeTransition(dstOffTransition, tz.getRawOffset());
    rule.endMonth = endFields[0];
    rule.endDay = endFields[1];
    rule.endDayOfWeek = endFields[2];
    rule.endTime = endFields[3];
    rule.endTimeMode = DstRule.TIME_MODE_STANDARD;
    rule.endMode = endFields[4];

    return rule;
  }

  /**
   * Decode a UTC transition instant into
   * {@code [month(0-11), baseDay, dayOfWeek(1-7), timeMs, mode]}.
   *
   * <p>Recurring weekday rules are encoded as DOW_GE_DOM (mode=2). The base
   * day is the earliest possible day of the matching occurrence in the month:
   * 1st=1, 2nd=8, 3rd=15, 4th=22, last={@code monthLength - 6}. This mirrors
   * encodings like "Sun >= 8" for the second Sunday in March and "Sun >= 25"
   * for the last Sunday in October.
   */
  private static int[] decodeTransition(long utcMs, int rawOffsetMs) {
    long localMs = utcMs + rawOffsetMs;
    LocalDateTime ldt = LocalDateTime.ofInstant(Instant.ofEpochMilli(localMs), ZoneOffset.UTC);

    int month = ldt.getMonthValue() - 1; // 0-based for Calendar compat
    int dayOfMonth = ldt.getDayOfMonth();
    int dayOfWeek = toCalendarDayOfWeek(ldt.getDayOfWeek().getValue());
    int timeInDay = ldt.getHour() * 3600_000
        + ldt.getMinute() * 60_000
        + ldt.getSecond() * 1000
        + ldt.getNano() / 1_000_000;

    int monthLength = ldt.toLocalDate().lengthOfMonth();
    int dayOfWeekInMonth = (dayOfMonth - 1) / 7 + 1;
    boolean isLastOccurrence = dayOfMonth + 7 > monthLength;
    int baseDayOfMonth = isLastOccurrence
        ? monthLength - 6
        : 1 + (dayOfWeekInMonth - 1) * 7;

    return new int[]{month, baseDayOfMonth, dayOfWeek, timeInDay, DstRule.MODE_DOW_GE_DOM};
  }

  // ---- Verification: ensure the extracted rule matches tz.getOffset ----

  private static boolean verifyDstRuleAcrossReferenceYears(TimeZone tz, DstRule rule) {
    for (int refYear : DST_RULE_VALIDATION_YEARS) {
      if (!verifyDstRule(tz, rule, refYear)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Verify the extracted rule matches {@code tz.getOffset} around transition
   * boundaries and at monthly sample points for {@code refYear ± 1} (3 years).
   * DST mismatches only manifest near transitions, so dense sampling at
   * boundaries plus monthly spot checks reduces a naive full-year scan from
   * ~52K probes to ~200 per reference year.
   */
  private static boolean verifyDstRule(TimeZone tz, DstRule rule, int refYear) {
    int rawOffsetMs = tz.getRawOffset();
    for (int y = refYear - 1; y <= refYear + 1; y++) {
      long dstStart = computeTransitionUtcMillis(y, rule.startMonth, rule.startDay,
          rule.startDayOfWeek, rule.startTime, rule.startTimeMode, rule.startMode,
          rawOffsetMs, rule.dstSavings, true);
      long dstEnd = computeTransitionUtcMillis(y, rule.endMonth, rule.endDay,
          rule.endDayOfWeek, rule.endTime, rule.endTimeMode, rule.endMode,
          rawOffsetMs, rule.dstSavings, false);

      long[] boundaries = {dstStart, dstEnd};
      for (long boundary : boundaries) {
        long from = boundary - 12 * 3600_000L;
        long to = boundary + 12 * 3600_000L;
        for (long ms = from; ms <= to; ms += 3600_000L) {
          if (tz.getOffset(ms) != computeDstOffset(ms, rawOffsetMs, rule)) {
            return false;
          }
        }
      }

      for (int m = 1; m <= 12; m++) {
        long ms = utcMillisForDate(y, m, 1) + 12 * 3600_000L;
        if (tz.getOffset(ms) != computeDstOffset(ms, rawOffsetMs, rule)) {
          return false;
        }
      }
    }
    return true;
  }

  private static int computeDstOffset(long utcMs, int rawOffsetMs, DstRule rule) {
    int year = LocalDate.ofEpochDay(Math.floorDiv(utcMs + rawOffsetMs, 86_400_000L)).getYear();
    long dstStart = computeTransitionUtcMillis(year, rule.startMonth, rule.startDay,
        rule.startDayOfWeek, rule.startTime, rule.startTimeMode, rule.startMode,
        rawOffsetMs, rule.dstSavings, true);
    long dstEnd = computeTransitionUtcMillis(year, rule.endMonth, rule.endDay,
        rule.endDayOfWeek, rule.endTime, rule.endTimeMode, rule.endMode,
        rawOffsetMs, rule.dstSavings, false);

    boolean inDst = dstStart < dstEnd
        ? (utcMs >= dstStart && utcMs < dstEnd)
        : (utcMs >= dstStart || utcMs < dstEnd);
    return inDst ? rawOffsetMs + rule.dstSavings : rawOffsetMs;
  }

  private static long computeTransitionUtcMillis(int year, int ruleMonth, int ruleDay,
      int ruleDayOfWeek, int ruleTime, int ruleTimeMode, int ruleMode, int rawOffsetMs,
      int dstSavingsMs, boolean isStartRule) {
    int actualDay = computeRuleDay(ruleMode, ruleDay, ruleDayOfWeek, year, ruleMonth);
    long utcMs = utcMillisForDate(year, ruleMonth + 1, actualDay) + ruleTime;
    if (ruleTimeMode == DstRule.TIME_MODE_WALL) {
      // WALL time: subtract raw offset and (for end transitions) also DST savings.
      utcMs -= rawOffsetMs;
      if (!isStartRule) {
        utcMs -= dstSavingsMs;
      }
    } else if (ruleTimeMode == DstRule.TIME_MODE_STANDARD) {
      utcMs -= rawOffsetMs;
    }
    // TIME_MODE_UTC is already in UTC.
    return utcMs;
  }

  private static int computeRuleDay(int ruleMode, int ruleDay, int ruleDayOfWeek, int year,
      int month) {
    LocalDate firstOfMonth = LocalDate.of(year, month + 1, 1);
    int monthLength = firstOfMonth.lengthOfMonth();
    int firstDayOfWeek = toCalendarDayOfWeek(firstOfMonth.getDayOfWeek().getValue());

    switch (ruleMode) {
      case DstRule.MODE_DOW_IN_MONTH: {
        if (ruleDay > 0) {
          int diff = ruleDayOfWeek - firstDayOfWeek;
          if (diff < 0) diff += 7;
          return 1 + diff + (ruleDay - 1) * 7;
        } else {
          int lastDayOfWeek = toCalendarDayOfWeek(
              LocalDate.of(year, month + 1, monthLength).getDayOfWeek().getValue());
          int diff = lastDayOfWeek - ruleDayOfWeek;
          if (diff < 0) diff += 7;
          return monthLength - diff + (ruleDay + 1) * 7;
        }
      }
      case DstRule.MODE_DOW_GE_DOM: {
        // Per ZoneOffsetTransitionRule.getDayOfMonthIndicator(), the indicator
        // may exceed monthLength (e.g. Feb 29 in a non-leap year, treated as
        // Mar 1). Clamp the anchor before LocalDate.of so it never throws, and
        // clamp the result so a DOW_GE_DOM rule whose computed day overflows
        // the month produces a valid in-month day rather than escaping with a
        // DateTimeException from utcMillisForDate. This mirrors SimpleTimeZone's
        // documented within-month clamp for DOW_GE_DOM.
        int anchorDay = Math.min(ruleDay, monthLength);
        int targetDayOfWeek = toCalendarDayOfWeek(
            LocalDate.of(year, month + 1, anchorDay).getDayOfWeek().getValue());
        int diff = ruleDayOfWeek - targetDayOfWeek;
        if (diff < 0) diff += 7;
        return Math.min(anchorDay + diff, monthLength);
      }
      case DstRule.MODE_DOW_LE_DOM: {
        int targetDayOfWeek = toCalendarDayOfWeek(
            LocalDate.of(year, month + 1, ruleDay).getDayOfWeek().getValue());
        int diff = targetDayOfWeek - ruleDayOfWeek;
        if (diff < 0) diff += 7;
        return ruleDay - diff;
      }
      case DstRule.MODE_DOM:
      default:
        return ruleDay;
    }
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
