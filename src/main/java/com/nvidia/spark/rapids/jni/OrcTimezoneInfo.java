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
  OrcTimezoneInfo(
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

  /**
   * SimpleTimeZone-compatible DST rule mode. The {@link #value} encoding must
   * stay in sync with {@code dst_rule_mode} in {@code timezones.cu}.
   */
  enum DstRuleMode {
    DOM_MODE(0),
    DOW_IN_MONTH_MODE(1),
    DOW_GE_DOM_MODE(2),
    DOW_LE_DOM_MODE(3);

    final int value;
    DstRuleMode(int value) { this.value = value; }
  }

  /**
   * SimpleTimeZone-compatible DST rule time mode. The {@link #value} encoding
   * must stay in sync with {@code dst_time_mode} in {@code timezones.cu}.
   */
  enum DstTimeMode {
    WALL_TIME(0),
    STANDARD_TIME(1),
    UTC_TIME(2);

    final int value;
    DstTimeMode(int value) { this.value = value; }
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
   * {@code startMode}/{@code endMode} are encoded as {@link DstRuleMode#value};
   * {@code startTimeMode}/{@code endTimeMode} as {@link DstTimeMode#value}.
   * Fields stay {@code int} because the rule is serialized to a JNI
   * {@code int[]} and the GPU kernel consumes the matching integer enum.
   */
  static class DstRule {
    int dstSavings;      // DST offset in milliseconds (typically 3600000)
    int startMonth;      // 0-based (Calendar.JANUARY=0 .. Calendar.DECEMBER=11)
    int startDay;        // day-of-month or occurrence count depending on startMode
    int startDayOfWeek;  // Calendar day-of-week (1=Sun, ..., 7=Sat), 0 if DOM_MODE
    int startTime;       // milliseconds within day
    int startTimeMode;   // see DstTimeMode
    int startMode;       // see DstRuleMode
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
    // "last <dayOfWeek> on or before day" (DOW_LE_DOM_MODE); we reject those here so that
    // downstream code can assume DOW_GE_DOM_MODE unconditionally.
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
    // Guaranteed by the precondition above.
    int mode = DstRuleMode.DOW_GE_DOM_MODE.value;

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
      return DstTimeMode.UTC_TIME.value;
    } else if (ZoneOffsetTransitionRule.TimeDefinition.STANDARD == timeDef) {
      return DstTimeMode.STANDARD_TIME.value;
    } else {
      return DstTimeMode.WALL_TIME.value;
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
          // More than one DST-on transition in the same year means this year
          // doesn't fit a SimpleTimeZone-style two-transition rule; let the
          // caller fall back to extractDstRuleFromZoneRules.
          if (dstOnTransition >= 0) return null;
          dstOnTransition = exactMs;
        } else {
          if (dstOffTransition >= 0) return null;
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

    // decodeTransition converts to standard local time, so the rule time mode is STANDARD_TIME.
    int[] startFields = decodeTransition(dstOnTransition, tz.getRawOffset());
    rule.startMonth = startFields[0];
    rule.startDay = startFields[1];
    rule.startDayOfWeek = startFields[2];
    rule.startTime = startFields[3];
    rule.startTimeMode = DstTimeMode.STANDARD_TIME.value;
    rule.startMode = startFields[4];

    int[] endFields = decodeTransition(dstOffTransition, tz.getRawOffset());
    rule.endMonth = endFields[0];
    rule.endDay = endFields[1];
    rule.endDayOfWeek = endFields[2];
    rule.endTime = endFields[3];
    rule.endTimeMode = DstTimeMode.STANDARD_TIME.value;
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
   * Returns [month(0-11), day, dayOfWeek(1-7), timeMs, {@link DstRuleMode#value}].
   *
   * We encode recurring weekday rules as {@link DstRuleMode#DOW_GE_DOM_MODE}.
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
    return new int[]{month, baseDayOfMonth, dayOfWeek, timeInDay, DstRuleMode.DOW_GE_DOM_MODE.value};
  }

}
