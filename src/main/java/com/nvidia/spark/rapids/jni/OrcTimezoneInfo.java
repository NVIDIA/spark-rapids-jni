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
import java.time.LocalDate;
import java.time.ZoneId;
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
 * Historical transitions come from ZoneRules, while offsets before the first transition and
 * future recurring DST behavior are validated against java.util.TimeZone so ORC rebasing matches
 * SerializationUtils.convertBetweenTimezones semantics without relying on non-public ZoneInfo APIs.
 *
 * <p><b>Runtime dependency:</b> because the metadata is generated on the fly from
 * {@link java.util.TimeZone}/{@link java.time.zone.ZoneRules}, the exact transition table and
 * recurring DST rule are determined by the JVM's bundled IANA {@code tzdata}. Different JDK
 * distributions or {@code tzdata} versions may produce slightly different historical
 * transitions or future-year DST offsets for the same zone id. This is strictly more correct
 * than the previous frozen OpenJDK-8 snapshot, but users debugging cross-environment
 * differences should first check the JVM's {@code tzdata} version.
 */
class OrcTimezoneInfo {
  public OrcTimezoneInfo(
      int initialOffset,
      int rawOffset,
      long[] transitions,
      int[] offsets,
      DstRule dstRule) {
    this.initialOffset = initialOffset;
    this.rawOffset = rawOffset;
    this.transitions = transitions;
    this.offsets = offsets;
    this.dstRule = dstRule;
  }

  // in milliseconds
  int initialOffset;

  // in milliseconds. This is the standard/raw offset used for DST rule math.
  int rawOffset;

  // in milliseconds
  long[] transitions;

  // in milliseconds
  int[] offsets;

  /**
   * DST rule extracted from java.util.SimpleTimeZone for computing offsets
   * beyond the historical transition table. Null if the timezone has no DST.
   *
   * The CUDA kernel uses this to implement SimpleTimeZone.getOffset() on GPU,
   * eliminating the need for pre-generated transition files for future dates.
   */
  DstRule dstRule;

  /**
   * Holds the DST rule parameters needed by the GPU kernel.
   * These correspond to the fields of java.util.SimpleTimeZone.
   *
   * The "mode" fields encode how startDay/endDay are interpreted:
   *   0 = DOM_MODE: exact day of month
   *   1 = DOW_IN_MONTH_MODE: nth dayOfWeek in month (negative = from end)
   *   2 = DOW_GE_DOM_MODE: first dayOfWeek on or after day
   *   3 = DOW_LE_DOM_MODE: last dayOfWeek on or before day
   */
  static class DstRule {
    int dstSavings;      // DST offset in milliseconds (typically 3600000)
    int startMonth;      // 0-based (Calendar.JANUARY=0 .. Calendar.DECEMBER=11)
    int startDay;        // day-of-month or occurrence count depending on startMode
    int startDayOfWeek;  // Calendar day-of-week (1=Sun, ..., 7=Sat), 0 if DOM_MODE
    int startTime;       // milliseconds within day
    int startTimeMode;   // 0=WALL, 1=STANDARD, 2=UTC
    int startMode;       // 0=DOM, 1=DOW_IN_MONTH, 2=DOW_GE_DOM, 3=DOW_LE_DOM
    int endMonth;
    int endDay;
    int endDayOfWeek;
    int endTime;
    int endTimeMode;
    int endMode;
  }

  // Reference years used to cross-check CPU vs. GPU DST offset computation.
  // We include a near-future anchor (2060) to catch divergence within the
  // typical application lifetime, plus two far-future anchors to exercise the
  // recurring-rule fallback path well past any historical transition entry.
  private static final int[] DST_RULE_VALIDATION_YEARS = {2060, 2400, 9997};
  // Lower bound of the range ORC supports (year 0001-01-01 UTC). Computed via
  // java.time.LocalDate, which uses the proleptic Gregorian calendar, whereas
  // java.util.TimeZone.getOffset(long) internally uses a hybrid Julian/Gregorian
  // calendar with the 1582 cutover for date-field interpretations. In practice
  // this difference does not affect offset lookup (which is purely instant-based
  // for ZoneInfo), and zones with DST in year 0001 do not exist, so the two
  // calendars agree on the offset at this instant. Kept as a single anchor so
  // the GPU side matches whatever TimeZone.getOffset returns here.
  private static final long MIN_SUPPORTED_ORC_UTC_MILLIS = utcMillisForDate(1, 0, 1);
  private static final long HISTORICAL_TRANSITION_SCAN_STEP_MILLIS = 24L * 3600_000L;

  /**
   * Extract DST rule by probing getOffset() or from ZoneRules transition rules.
   * Returns null if the timezone has no DST.
   */
  static DstRule extractDstRule(String timezoneId, TimeZone tz, ZoneRules rules) {
    if (!tz.useDaylightTime()) {
      return null;
    }
    DstRule dstRule = extractDstRuleByProbing(tz);
    if (dstRule != null) {
      return dstRule;
    }

    dstRule = extractDstRuleFromZoneRules(timezoneId, tz, rules);
    if (dstRule != null) {
      return dstRule;
    }
    throw new IllegalStateException("Failed to extract ORC DST rule for timezone: " + timezoneId);
  }

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
      int deltaMillis = (transitionRule.getOffsetAfter().getTotalSeconds() -
          transitionRule.getOffsetBefore().getTotalSeconds()) * 1000;
      if (deltaMillis > 0) {
        startTransitionRule = transitionRule;
      } else if (deltaMillis < 0) {
        endTransitionRule = transitionRule;
      } else {
        throw new IllegalStateException("Unsupported zero-delta ORC DST rule for timezone: " +
            timezoneId);
      }
    }
    if (startTransitionRule == null || endTransitionRule == null) {
      throw new IllegalStateException("Failed to identify ORC DST start/end rules for timezone: " +
          timezoneId);
    }

    int dstSavings = (startTransitionRule.getOffsetAfter().getTotalSeconds() -
        startTransitionRule.getOffsetBefore().getTotalSeconds()) * 1000;
    int endDeltaMillis = (endTransitionRule.getOffsetBefore().getTotalSeconds() -
        endTransitionRule.getOffsetAfter().getTotalSeconds()) * 1000;
    if (dstSavings != endDeltaMillis) {
      throw new IllegalStateException("Mismatched ORC DST savings for timezone: " + timezoneId);
    }

    DstRule rule = new DstRule();
    rule.dstSavings = dstSavings;
    fillDstRuleFromTransitionRule(timezoneId, rule, startTransitionRule, true);
    fillDstRuleFromTransitionRule(timezoneId, rule, endTransitionRule, false);

    if (!verifyDstRuleAcrossReferenceYears(tz, rule)) {
      throw new IllegalStateException("ZoneRules ORC DST rule verification failed for timezone: " +
          timezoneId);
    }
    return rule;
  }

  private static void fillDstRuleFromTransitionRule(String timezoneId, DstRule rule,
      ZoneOffsetTransitionRule transitionRule, boolean isStartRule) {
    // We only accept rules shaped as "first <dayOfWeek> on or after <dayOfMonthIndicator>",
    // i.e. ZoneRules' positive-day-indicator form. A negative indicator would mean
    // "last <dayOfWeek> on or before day" (DOW_LE_DOM_MODE, mode=3); we reject those
    // here so that downstream code can assume DOW_GE_DOM_MODE (mode=2) unconditionally.
    if (transitionRule.getDayOfWeek() == null ||
        transitionRule.getDayOfMonthIndicator() <= 0) {
      throw new IllegalStateException("Unsupported ORC DST transition rule shape for timezone: " +
          timezoneId);
    }

    int month = transitionRule.getMonth().getValue() - 1;
    int day = transitionRule.getDayOfMonthIndicator();
    int dayOfWeek = toCalendarDayOfWeek(transitionRule.getDayOfWeek().getValue());
    int time = getTransitionRuleTimeMillis(transitionRule);
    int timeMode = getTransitionRuleTimeMode(transitionRule);
    // SimpleTimeZone mode constant: DOW_GE_DOM_MODE. Guaranteed by the precondition above.
    int mode = 2;

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

  private static int getTransitionRuleTimeMillis(
      ZoneOffsetTransitionRule transitionRule) {
    int secondOfDay = transitionRule.isMidnightEndOfDay() ?
        24 * 3600 :
        transitionRule.getLocalTime().toSecondOfDay();
    return secondOfDay * 1000;
  }

  private static int getTransitionRuleTimeMode(ZoneOffsetTransitionRule transitionRule) {
    ZoneOffsetTransitionRule.TimeDefinition timeDef = transitionRule.getTimeDefinition();
    if (ZoneOffsetTransitionRule.TimeDefinition.UTC == timeDef) {
      return 2;
    } else if (ZoneOffsetTransitionRule.TimeDefinition.STANDARD == timeDef) {
      return 1;
    } else {
      return 0;
    }
  }

  private static int toCalendarDayOfWeek(int javaTimeDayOfWeek) {
    return (javaTimeDayOfWeek % 7) + 1;
  }

  /**
   * Extract DST rule by probing getOffset() at hourly intervals in a reference year.
   * This works for any TimeZone implementation (ZoneInfo, SimpleTimeZone, etc.)
   * and captures the effective DST rule as the JVM sees it.
   *
   * We find the exact DST start and end transitions, then encode them in the
   * same format that SimpleTimeZone uses internally (month, day, dayOfWeek, time, mode).
   */
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
    long janFirst = utcMillisForDate(refYear, 0, 1);
    long nextJanFirst = utcMillisForDate(refYear + 1, 0, 1);

    // Find DST-on and DST-off transitions by scanning hourly
    long dstOnTransition = -1;
    long dstOffTransition = -1;
    int prevOffset = tz.getOffset(janFirst - 1);
    long step = 3600_000L; // 1 hour

    for (long ms = janFirst; ms < nextJanFirst; ms += step) {
      int curOffset = tz.getOffset(ms);
      if (curOffset != prevOffset) {
        // Found a transition; narrow down to exact millisecond with binary search
        long exactMs = binarySearchTransition(tz, ms - step, ms);
        if (curOffset > prevOffset) {
          dstOnTransition = exactMs;
        } else {
          dstOffTransition = exactMs;
        }
        prevOffset = curOffset;
      }
    }

    if (dstOnTransition < 0 || dstOffTransition < 0) {
      return null;
    }

    DstRule rule = new DstRule();
    rule.dstSavings = tz.getDSTSavings();

    // Decode the DST-on transition
    int[] startFields = decodeTransition(dstOnTransition, tz.getRawOffset());
    rule.startMonth = startFields[0];
    rule.startDay = startFields[1];
    rule.startDayOfWeek = startFields[2];
    rule.startTime = startFields[3];
    rule.startTimeMode = 1; // STANDARD_TIME - decodeTransition converts to standard local time
    rule.startMode = startFields[4];

    // Decode the DST-off transition
    int[] endFields = decodeTransition(dstOffTransition, tz.getRawOffset());
    rule.endMonth = endFields[0];
    rule.endDay = endFields[1];
    rule.endDayOfWeek = endFields[2];
    rule.endTime = endFields[3];
    rule.endTimeMode = 1; // STANDARD_TIME
    rule.endMode = endFields[4];

    return rule;
  }

  private static boolean verifyDstRuleAcrossReferenceYears(TimeZone tz, DstRule rule) {
    for (int refYear : DST_RULE_VALIDATION_YEARS) {
      if (!verifyDstRule(tz, rule, refYear)) {
        return false;
      }
    }
    return true;
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

  /**
   * Decode a UTC transition instant into (month, day, dayOfWeek, timeInDay, mode).
   * Returns [month(0-11), day, dayOfWeek(1-7), timeMs, mode(0-3)].
   *
   * We encode recurring weekday rules as DOW_GE_DOM_MODE (mode=2).
   * For nth-weekday rules, the base day is the earliest possible day of that
   * occurrence in the month:
   *   1st => 1, 2nd => 8, 3rd => 15, 4th => 22.
   * For last-weekday rules, the base day is the earliest day of the final week
   * in the month, i.e. {@code monthLength - 6}.
   *
   * This mirrors encodings such as "Sun >= 8" for the second Sunday in March
   * and "Sun >= 25" for the last Sunday in October.
   */
  private static int[] decodeTransition(long utcMs, int rawOffsetMs) {
    // Convert UTC ms to standard local time
    long localMs = utcMs + rawOffsetMs;
    java.time.Instant instant = java.time.Instant.ofEpochMilli(localMs);
    java.time.LocalDateTime ldt = java.time.LocalDateTime.ofInstant(
        instant, java.time.ZoneOffset.UTC);

    int month = ldt.getMonthValue() - 1; // 0-based for Calendar compat
    int dayOfMonth = ldt.getDayOfMonth();
    // Calendar: 1=Sun..7=Sat
    int dayOfWeek = toCalendarDayOfWeek(ldt.getDayOfWeek().getValue());
    int timeInDay = ldt.getHour() * 3600_000 + ldt.getMinute() * 60_000
        + ldt.getSecond() * 1000 + ldt.getNano() / 1_000_000;

    int monthLength = ldt.toLocalDate().lengthOfMonth();
    int dayOfWeekInMonth = (dayOfMonth - 1) / 7 + 1;
    boolean isLastOccurrence = dayOfMonth + 7 > monthLength;
    int baseDayOfMonth = isLastOccurrence ?
        monthLength - 6 :
        1 + (dayOfWeekInMonth - 1) * 7;

    // DOW_GE_DOM: first <dayOfWeek> on or after <baseDayOfMonth>
    return new int[]{month, baseDayOfMonth, dayOfWeek, timeInDay, 2};
  }

  /**
   * Verify the extracted DST rule matches JVM getOffset() around transition boundaries
   * and at monthly sample points for refYear +/- 1 (3 years).
   *
   * DST rule mismatches only manifest at transition boundaries, so we check
   * +/- 12 hours around each computed transition plus one sample per month.
   * This reduces ~52K getOffset() calls per verification to ~200.
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

      // Check +/- 12 hours around each transition at hourly granularity
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

      // Monthly mid-period spot checks (1st of each month at noon UTC)
      for (int m = 0; m < 12; m++) {
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

    boolean inDst = dstStart < dstEnd ?
        (utcMs >= dstStart && utcMs < dstEnd) :
        (utcMs >= dstStart || utcMs < dstEnd);
    return inDst ? rawOffsetMs + rule.dstSavings : rawOffsetMs;
  }

  private static long computeTransitionUtcMillis(int year, int ruleMonth, int ruleDay,
      int ruleDayOfWeek, int ruleTime, int ruleTimeMode, int ruleMode, int rawOffsetMs,
      int dstSavingsMs, boolean isStartRule) {
    int actualDay = computeRuleDay(ruleMode, ruleDay, ruleDayOfWeek, year, ruleMonth);
    long utcMs = utcMillisForDate(year, ruleMonth, actualDay) + ruleTime;
    if (ruleTimeMode == 0) {
      utcMs -= rawOffsetMs;
      if (!isStartRule) {
        utcMs -= dstSavingsMs;
      }
    } else if (ruleTimeMode == 1) {
      utcMs -= rawOffsetMs;
    }
    return utcMs;
  }

  private static int computeRuleDay(int ruleMode, int ruleDay, int ruleDayOfWeek, int year,
      int month) {
    LocalDate firstOfMonth = LocalDate.of(year, month + 1, 1);
    int monthLength = firstOfMonth.lengthOfMonth();
    int firstDayOfWeek = toCalendarDayOfWeek(firstOfMonth.getDayOfWeek().getValue());

    switch (ruleMode) {
      case 1: {
        if (ruleDay > 0) {
          int diff = ruleDayOfWeek - firstDayOfWeek;
          if (diff < 0) {
            diff += 7;
          }
          return 1 + diff + (ruleDay - 1) * 7;
        } else {
          int lastDayOfWeek = toCalendarDayOfWeek(
              LocalDate.of(year, month + 1, monthLength).getDayOfWeek().getValue());
          int diff = lastDayOfWeek - ruleDayOfWeek;
          if (diff < 0) {
            diff += 7;
          }
          return monthLength - diff + (ruleDay + 1) * 7;
        }
      }
      case 2: {
        int targetDayOfWeek = toCalendarDayOfWeek(
            LocalDate.of(year, month + 1, ruleDay).getDayOfWeek().getValue());
        int diff = ruleDayOfWeek - targetDayOfWeek;
        if (diff < 0) {
          diff += 7;
        }
        return ruleDay + diff;
      }
      case 3: {
        int targetDayOfWeek = toCalendarDayOfWeek(
            LocalDate.of(year, month + 1, ruleDay).getDayOfWeek().getValue());
        int diff = targetDayOfWeek - ruleDayOfWeek;
        if (diff < 0) {
          diff += 7;
        }
        return ruleDay - diff;
      }
      default:
        return ruleDay;
    }
  }

  private static long utcMillisForDate(int year, int month, int day) {
    return LocalDate.of(year, month + 1, day).toEpochDay() * 24L * 3600_000L;
  }

  @Override
  public String toString() {
    return "OrcTimezoneInfo{" +
        "initialOffset=" + initialOffset +
        ", rawOffset=" + rawOffset +
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
   * @param timezoneId timezone Id
   * @return timezone info with DST rules
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
   */
  private static OrcTimezoneInfo buildRuntimeOrcTimezoneInfo(String timezoneId) {
    final ZoneId zoneId;
    try {
      zoneId = GpuTimeZoneDB.getZoneId(timezoneId);
    } catch (DateTimeException e) {
      throw new IllegalArgumentException("Timezone ID not found: " + timezoneId, e);
    }

    TimeZone tz = TimeZone.getTimeZone(timezoneId);
    ZoneRules rules = zoneId.getRules();
    int initialOffset = getInitialOffset(tz);
    if (rules.isFixedOffset()) {
      return new OrcTimezoneInfo(initialOffset, tz.getRawOffset(), null, null, null);
    }
    DstRule dstRule = extractDstRule(timezoneId, tz, rules);

    List<ZoneOffsetTransition> transitionList = rules.getTransitions();
    HistoricalTransitions historicalTransitions = buildHistoricalTransitions(tz, transitionList);
    if (historicalTransitions.transitions == null) {
      return new OrcTimezoneInfo(initialOffset, tz.getRawOffset(), null, null, dstRule);
    }
    return new OrcTimezoneInfo(initialOffset, tz.getRawOffset(),
        historicalTransitions.transitions, historicalTransitions.offsets, dstRule);
  }

  public static List<String> getAllTimezoneIds() {

    String[] ids = TimeZone.getAvailableIDs();
    Arrays.sort(ids);
    return Arrays.asList(ids);
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
