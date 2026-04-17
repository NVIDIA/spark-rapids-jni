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

import ai.rapids.cudf.*;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TimeZone;
import java.util.concurrent.TimeUnit;

public class GpuTimeZoneDBTest {

  private static final long microsPerMillis = TimeUnit.MILLISECONDS.toMicros(1);
  private static final String EXHAUSTIVE_HALF_HOUR_TEST_PROPERTY =
      "spark.rapids.tests.orc.exhaustiveHalfHour";
  private static final String EXHAUSTIVE_HALF_HOUR_BATCH_SIZE_PROPERTY =
      "spark.rapids.tests.orc.exhaustiveHalfHour.batchSize";
  private static final int DEFAULT_EXHAUSTIVE_HALF_HOUR_BATCH_SIZE = 16_384;

  /**
   * Java implementation of timezone conversion to compare against the GPU
   * results.
   * Refer to <a href="https://github.com/apache/orc/blob/rel/release-1.9.1/java/core/src/java/org/apache/orc/impl/SerializationUtils.java#L1440">ORC code link</a>
   *
   */
  private static ColumnVector convertOrcTimezonesOnCPU(
      long[] microseconds,
      String writeTzId,
      String readerTzId) {
    long[] results = new long[microseconds.length];
    TimeZone writeTz = TimeZone.getTimeZone(writeTzId);
    TimeZone readerTz = TimeZone.getTimeZone(readerTzId);
    for (int i = 0; i < microseconds.length; ++i) {
      long millis = microseconds[i] / microsPerMillis;
      long writerOffset = writeTz.getOffset(millis);
      long readerOffset = readerTz.getOffset(millis);
      long adjustedMillis = millis + writerOffset - readerOffset;
      long adjustedReader = readerTz.getOffset(adjustedMillis);
      long finalDiffs = writerOffset - adjustedReader;
      results[i] = (millis + finalDiffs) * microsPerMillis + (microseconds[i] % microsPerMillis);
    }
    return ColumnVector.timestampMicroSecondsFromLongs(results);
  }

  private static long microsUtc(LocalDateTime timestamp) {
    return timestamp.toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1);
  }

  private static long[] microsUtc(LocalDateTime... timestamps) {
    long[] result = new long[timestamps.length];
    for (int i = 0; i < timestamps.length; ++i) {
      result[i] = microsUtc(timestamps[i]);
    }
    return result;
  }

  private static void assertOrcConversionMatchesCpu(long[] microseconds, String[][] cases) {
    for (String[] timezones : cases) {
      try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
          ColumnVector expected = convertOrcTimezonesOnCPU(microseconds, timezones[0], timezones[1]);
          ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, 0L, timezones[0], timezones[1])) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  private static long[] collectTransitionBoundaryMicros(String timezoneId, int year) {
    TimeZone tz = TimeZone.getTimeZone(timezoneId);
    long start = microsUtc(LocalDateTime.of(year, 1, 1, 0, 0)) / microsPerMillis;
    long end = microsUtc(LocalDateTime.of(year + 1, 1, 1, 0, 0)) / microsPerMillis;
    long step = TimeUnit.HOURS.toMillis(1);
    int prevOffset = tz.getOffset(start - 1);
    List<Long> samples = new ArrayList<>();

    for (long millis = start; millis < end; millis += step) {
      int offset = tz.getOffset(millis);
      if (offset == prevOffset) {
        continue;
      }
      long transition = binarySearchTransition(tz, millis - step, millis);
      for (long deltaMillis : new long[]{-1L, 0L, 1L}) {
        samples.add((transition + deltaMillis) * microsPerMillis);
      }
      prevOffset = offset;
    }

    long[] result = new long[samples.size()];
    for (int i = 0; i < samples.size(); ++i) {
      result[i] = samples.get(i);
    }
    return result;
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

  private static long[] concat(long[]... arrays) {
    int total = 0;
    for (long[] array : arrays) {
      total += array.length;
    }
    long[] result = new long[total];
    int offset = 0;
    for (long[] array : arrays) {
      System.arraycopy(array, 0, result, offset, array.length);
      offset += array.length;
    }
    return result;
  }

  private static void assertOrcConversionMatchesCpuForOrderedPairs(
      long[] microseconds,
      String[] timezones) {
    for (String writerTz : timezones) {
      for (String readerTz : timezones) {
        if (writerTz.equals(readerTz)) {
          continue;
        }
        try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
            ColumnVector expected = convertOrcTimezonesOnCPU(microseconds, writerTz, readerTz);
            ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, 0L, writerTz, readerTz)) {
          assertColumnsAreEqual(expected, actual);
        }
      }
    }
  }

  @Test
  void testConvertOrcTimezones() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    // Full range: (0001-01-01 00:00:00, 9999-12-31 23:59:59)
    // DST rules from SimpleTimeZone handle dates beyond the transition table.
    long min = LocalDateTime.of(1, 1, 1, 0, 0, 0)
        .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1);
    long max = LocalDateTime.of(9999, 12, 31, 23, 59, 59)
        .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1);

    // use today as the random seed so we get different values each day
    Random rng = new Random(LocalDate.now().toEpochDay());

    List<String> timezones = Arrays.asList(
        "America/Los_Angeles",
        "Asia/Shanghai",
        "Antarctica/DumontDUrville",
        "Etc/GMT-12",
        "CNT",
        "Australia/Sydney",
        "Asia/Tokyo");

    for (String writerTz : timezones) {
      for (String readerTz : timezones) {
        // Use 1024 as a reasonable batch size for testing timezone conversions.
        long[] microseconds = new long[1024];
        for (int i = 0; i < microseconds.length; ++i) {
          // range is years from 0001 to 9999
          microseconds[i] = min + (long) (rng.nextDouble() * (max - min));
        }

        try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
            // Convert on CPU
            ColumnVector expected = convertOrcTimezonesOnCPU(microseconds, writerTz, readerTz);
            // Convert on GPU
            ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, 0L, writerTz, readerTz)) {
          assertColumnsAreEqual(expected, actual);
        }
      }
    }
  }

  @Test
  void testConvertOrcTimezonesBeforeFirstTransitionUsesHistoricalOffset() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    long[] microseconds = {
        LocalDateTime.of(1, 1, 15, 12, 0, 0)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1),
        LocalDateTime.of(1, 7, 15, 12, 0, 0)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1),
        LocalDateTime.of(1, 12, 15, 12, 0, 0)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1)
    };

    String[][] cases = {
        {"America/Los_Angeles", "UTC"},
        {"UTC", "America/Los_Angeles"},
        {"Australia/Sydney", "UTC"},
        {"UTC", "Australia/Sydney"}
    };

    for (String[] timezones : cases) {
      try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
          ColumnVector expected = convertOrcTimezonesOnCPU(microseconds, timezones[0], timezones[1]);
          ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, 0L, timezones[0], timezones[1])) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void testConvertOrcTimezonesHistoricalInitialOffsetMismatch() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    long[] microseconds = {
        LocalDateTime.of(1899, 12, 31, 23, 59, 59)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1),
        LocalDateTime.of(1900, 1, 1, 0, 0, 0)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1),
        LocalDateTime.of(1900, 1, 1, 12, 0, 0)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1)
    };

    String[][] cases = {
        {"Africa/Windhoek", "UTC"},
        {"UTC", "Africa/Windhoek"}
    };

    for (String[] timezones : cases) {
      try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
          ColumnVector expected = convertOrcTimezonesOnCPU(microseconds, timezones[0], timezones[1]);
          ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, 0L, timezones[0], timezones[1])) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void testConvertOrcTimezonesHistoricalTransitionBoundaries() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    long[] microseconds = microsUtc(
        LocalDateTime.of(1899, 12, 31, 23, 59, 59),
        LocalDateTime.of(1900, 1, 1, 0, 0, 0),
        LocalDateTime.of(1900, 1, 1, 0, 0, 1),
        LocalDateTime.of(1900, 12, 31, 15, 54, 16),
        LocalDateTime.of(1900, 12, 31, 15, 54, 17),
        LocalDateTime.of(1900, 12, 31, 15, 54, 18),
        LocalDateTime.of(1903, 2, 28, 22, 29, 59),
        LocalDateTime.of(1903, 2, 28, 22, 30, 0),
        LocalDateTime.of(1903, 2, 28, 22, 30, 1)
    );

    String[][] cases = {
        {"Asia/Shanghai", "UTC"},
        {"UTC", "Asia/Shanghai"},
        {"Africa/Windhoek", "UTC"},
        {"UTC", "Africa/Windhoek"},
        {"Asia/Shanghai", "Africa/Windhoek"},
        {"Africa/Windhoek", "Asia/Shanghai"}
    };

    assertOrcConversionMatchesCpu(microseconds, cases);
  }

  @Test
  void testConvertOrcTimezonesFutureDstRuleFallback() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    long[] microseconds = {
        LocalDateTime.of(9999, 1, 15, 12, 0, 0)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1),
        LocalDateTime.of(9999, 4, 15, 12, 0, 0)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1),
        LocalDateTime.of(9999, 7, 1, 12, 0, 0)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1),
        LocalDateTime.of(9999, 10, 15, 12, 0, 0)
            .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1)
    };

    // These zones previously failed the probing-only DST extraction path once
    // we moved beyond the static table. Some now rely on far-future probing of
    // java.util.TimeZone, while others require the ZoneRules fallback.
    String[][] cases = {
        {"Asia/Gaza", "UTC"},
        {"UTC", "Asia/Gaza"},
        {"Asia/Jerusalem", "UTC"},
        {"UTC", "Asia/Jerusalem"},
        {"America/Nuuk", "UTC"},
        {"UTC", "America/Nuuk"},
        {"America/Santiago", "UTC"},
        {"UTC", "America/Santiago"}
    };

    for (String[] timezones : cases) {
      try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
          ColumnVector expected = convertOrcTimezonesOnCPU(microseconds, timezones[0], timezones[1]);
          ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, 0L, timezones[0], timezones[1])) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void testConvertOrcTimezonesFutureDstTransitionBoundaries() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    long[] gazaSamples = collectTransitionBoundaryMicros("Asia/Gaza", 9998);
    long[] santiagoSamples = collectTransitionBoundaryMicros("America/Santiago", 9998);
    assertFalse(gazaSamples.length == 0, "precondition: expected future transitions for Asia/Gaza");
    assertFalse(santiagoSamples.length == 0,
        "precondition: expected future transitions for America/Santiago");

    long[] microseconds = concat(gazaSamples, santiagoSamples);

    String[][] cases = {
        {"Asia/Gaza", "UTC"},
        {"UTC", "Asia/Gaza"},
        {"America/Santiago", "UTC"},
        {"UTC", "America/Santiago"},
        {"Asia/Gaza", "America/Santiago"},
        {"America/Santiago", "Asia/Gaza"}
    };

    assertOrcConversionMatchesCpu(microseconds, cases);
  }

  /**
   * JVM-valid fixed-offset IDs (e.g. {@code +05:30}) are not returned by
   * {@link TimeZone#getAvailableIDs()}, so ORC conversion must synthesize them
   * via the runtime metadata path and still match java.util.TimeZone.
   */
  @Test
  void testConvertOrcTimezonesDynamicFixedOffset() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    Set<String> availableTimeZoneIds = new HashSet<>(GpuTimeZoneDB.getOrcSupportedTimezones());
    String[] customFixedOffsets = {"+05:30", "-02:30", "+14:00", "GMT+08:00"};
    for (String id : customFixedOffsets) {
      assertFalse(availableTimeZoneIds.contains(id),
          "precondition: ID should require runtime synthesis: " + id);
    }

    long min = LocalDateTime.of(1, 1, 1, 0, 0, 0)
        .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1);
    long max = LocalDateTime.of(9999, 12, 31, 23, 59, 59)
        .toEpochSecond(ZoneOffset.UTC) * TimeUnit.SECONDS.toMicros(1);
    Random rng = new Random(42);

    for (String writerTz : customFixedOffsets) {
      for (String readerTz : customFixedOffsets) {
        long[] microseconds = new long[256];
        for (int i = 0; i < microseconds.length; ++i) {
          microseconds[i] = min + (long) (rng.nextDouble() * (max - min));
        }
        try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(microseconds);
            ColumnVector expected = convertOrcTimezonesOnCPU(microseconds, writerTz, readerTz);
            ColumnVector actual = GpuTimeZoneDB.convertOrcTimezones(input, 0L, writerTz, readerTz)) {
          assertColumnsAreEqual(expected, actual);
        }
      }
    }
  }

  /**
   * Opt-in stress test for exhaustive half-hour ORC timezone rebasing across the
   * requested zones. This is intentionally disabled by default because it spans
   * every half-hour from year 0000 through year 9999 and validates all ordered
   * writer/reader pairs in chunks.
   *
   * Enable manually with:
   * {@code -Dspark.rapids.tests.orc.exhaustiveHalfHour=true}
   *
   * Optional chunk size override:
   * {@code -Dspark.rapids.tests.orc.exhaustiveHalfHour.batchSize=32768}
   */
  @Test
  void testConvertOrcTimezonesExhaustiveHalfHourNewYorkShanghaiLosAngelesUtc() {
    assumeTrue(Boolean.getBoolean(EXHAUSTIVE_HALF_HOUR_TEST_PROPERTY),
        "disabled by default; enable with -D" + EXHAUSTIVE_HALF_HOUR_TEST_PROPERTY + "=true");

    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();

    int batchSize = Integer.getInteger(
        EXHAUSTIVE_HALF_HOUR_BATCH_SIZE_PROPERTY,
        DEFAULT_EXHAUSTIVE_HALF_HOUR_BATCH_SIZE);
    assertFalse(batchSize <= 0, "batch size must be positive");

    String[] timezones = {
        "America/New_York",
        "Asia/Shanghai",
        "America/Los_Angeles",
        "UTC"
    };

    long stepMicros = TimeUnit.MINUTES.toMicros(30);
    long startMicros = microsUtc(LocalDateTime.of(0, 1, 1, 0, 0));
    long endMicros = microsUtc(LocalDateTime.of(9999, 12, 31, 23, 30));

    for (long batchStartMicros = startMicros; batchStartMicros <= endMicros; ) {
      long remaining = ((endMicros - batchStartMicros) / stepMicros) + 1;
      int currentBatchSize = (int) Math.min(batchSize, remaining);
      long[] microseconds = new long[currentBatchSize];
      for (int i = 0; i < currentBatchSize; ++i) {
        microseconds[i] = batchStartMicros + i * stepMicros;
      }

      assertOrcConversionMatchesCpuForOrderedPairs(microseconds, timezones);
      batchStartMicros += currentBatchSize * stepMicros;
    }
  }

  @Test
  void testConvertOrcTimezonesInvalidIdStillFails() {
    GpuTimeZoneDB.cacheDatabase();
    GpuTimeZoneDB.verifyDatabaseCached();
    long[] one = {0L};
    try (ColumnVector input = ColumnVector.timestampMicroSecondsFromLongs(one)) {
      assertThrows(IllegalArgumentException.class,
          () -> GpuTimeZoneDB.convertOrcTimezones(input, 0L, "Not/AValidZoneId", "UTC"));
    }
  }
}
