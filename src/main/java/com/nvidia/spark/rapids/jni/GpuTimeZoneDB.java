/*
* Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

import ai.rapids.cudf.BinaryOp;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.util.calendar.ZoneInfo;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneOffsetTransitionRule;
import java.time.zone.ZoneRules;
import java.time.zone.ZoneRulesException;
import java.util.*;
import java.util.concurrent.Executors;

/**
 * Used to save timezone info from `sun.util.calendar.ZoneInfo`
 */
class TzInfoInJavaUtilForORC implements AutoCloseable {
  // from `sun.util.calendar.ZoneInfo`
  private static final long OFFSET_MASK_IN_ZONE_INFO = 0x0FL;

  // from `sun.util.calendar.ZoneInfo`
  private static final int TRANSITION_NSHIFT_IN_ZONE_INFO = 12;

  // Uses 20 bits to store offset in seconds
  private static final int OFFSET_SHIFT = 20;

  // Offset bit mask: 20 one bits
  private static final long OFFSET_MASK = 0xFFFFFL;

  // Transition time in seconds from Gregorian January 1 1970, 00:00:00 GMT;
  // Each long is packed in form of:
  // - The least significant 20 bits are for offset in seconds.
  // - The most significant 44 bits are for transition time in seconds.
  long[] transitions;

  // from `sun.util.calendar.ZoneInfo`, but in seconds.
  int rawOffset;

  /**
   * Constructor for TimeZone info from `sun.util.calendar.ZoneInfo`.
   * Extract timezone info from `ZoneInfo` and convert to seconds.
   * The inputs are from `sun.util.calendar.ZoneInfo` via reflection.
   * 
   * @param transitions transitions in milliseconds from
   *                    `sun.util.calendar.ZoneInfo`
   * @param offsets     offsets in `sun.util.calendar.ZoneInfo` in milliseconds
   * @param rawOffset   raw offset in `sun.util.calendar.ZoneInfo` in milliseconds
   */
  TzInfoInJavaUtilForORC(long[] transitions, int[] offsets, int rawOffset) {
    if (transitions != null) {
      this.transitions = new long[transitions.length];
      for (int i = 0; i < transitions.length; i++) {
        long transitionMs = (transitions[i] >> TRANSITION_NSHIFT_IN_ZONE_INFO);
        if (transitionMs % 1000 != 0) {
          throw new IllegalArgumentException("transitions should be in seconds");
        }
        long offsetMs = offsets[(int) (transitions[i] & OFFSET_MASK_IN_ZONE_INFO)];
        if (offsetMs % 1000 != 0) {
          throw new IllegalArgumentException("offsets should be in seconds");
        }
        long offsetSeconds = offsetMs / 1000L;
        if (offsetSeconds < -18L * 3600L || offsetSeconds > 18L * 3600L) {
          throw new IllegalArgumentException("offsets should be in range [-18h, +18h]");
        }

        // pack transition(in seconds) and offset(in seconds) into a int64 value.
        // - The least significant 20 bits are for offset in seconds.
        // - The most significant 44 bits are for transition time in seconds.
        this.transitions[i] = ((transitionMs / 1000L) << OFFSET_SHIFT)
            | (OFFSET_MASK & offsetSeconds);
      }
    }
    if (rawOffset % 1000 != 0) {
      throw new IllegalArgumentException("rawOffset should be in seconds, but find milliseconds");
    }
    this.rawOffset = rawOffset / 1000;
  }

  @Override
  public void close() throws Exception {
    transitions = null;
  }
}

/**
 * Gpu timezone utility.
 * Provides two kinds of APIs
 *  - Timezone transitions cache APIs
 *      `cacheDatabaseAsync`, `cacheDatabase` and `shutdown` are synchronized.
 *      When cacheDatabaseAsync is running, the `shutdown` and `cacheDatabase` will wait;
 *      These APIs guarantee only one thread is loading transitions cache,
 *      And guarantee loading cache only occurs one time.
 *  - Rebase timezone APIs
 *    fromTimestampToUtcTimestamp, fromUtcTimestampToTimestamp ...
 */
public class GpuTimeZoneDB {
  private static final Logger log = LoggerFactory.getLogger(GpuTimeZoneDB.class);

  // For the timezone database, we store the transitions in a ColumnVector that is a list of 
  // structs. The type of this column vector is:
  //   LIST<STRUCT<utcInstant: int64, localInstant: int64, offset: int32>>
  private static java.util.Map<String, Integer> zoneIdToTable;

  // host column STRUCT<tz_name: string, index_to_transition_table: int, is_DST: int8>,
  // sorted by timezone, is used to query index to transition table and if tz is DST
  // Casting string with timezone to timestamp needs loading all timezone is successful.
  // If this is not null, it indicates loading is successful,
  // because it's the last variable to construct in `loadData` function.
  // use this reference to indicate if timezone cache is initialized successfully.
  // The tz_name column constains both normalized and non-normalized timezone names.
  private static volatile HostColumnVector timeZoneInfo;

  // This is used to map the index of the transition table to the timezone name
  private static ArrayList<String> sortedNormalizedTimeZones = new ArrayList<>();

  private static HostColumnVector transitions;
  // initial year set to 1900 because some transition rules start early
  private static final int initialTransitionYear = 1900;
  private static long maxTimestamp;
  private static int lastCachedYear;
  private static final ZoneId utcZoneId = ZoneId.of("UTC");

  // // Cache the timezone info for 6 timezones.

  /**
   * This should be called on startup of an executor.
   * Runs in a thread asynchronously.
   * If `shutdown` was called ever, then will not load the cache
   */
  public static void cacheDatabaseAsync(int maxYear) {
    // start a new thread to load
    Runnable runnable = () -> {
      try {
        cacheDatabaseImpl(maxYear);
      } catch (Exception e) {
        log.error("cache timezone transitions cache failed", e);
      }
    };
    Thread thread = Executors.defaultThreadFactory().newThread(runnable);
    thread.setName("gpu-timezone-database-0");
    thread.setDaemon(true);
    thread.start();
  }

  private static synchronized void verifyDatabaseCachedSync() {
    if (timeZoneInfo == null) {
      throw new IllegalStateException("Timezone DB is not loaded, or the loading was failed.");
    }
  }

  public static void verifyDatabaseCached() {
    if (timeZoneInfo != null) {
      // already loaded
      return;
    }
    // wait for the loading thread to finish
    verifyDatabaseCachedSync();
  }

  /**
   * Cache the database. This will take some time like several seconds.
   * If one `cacheDatabase` is running, other `cacheDatabase` will wait until caching is done.
   * If cache is exits, do not load cache again.
   */
  public static void cacheDatabase(int maxYear) {
    cacheDatabaseImpl(maxYear);
  }

  /**
   * close the cache, used when Plugin is closing
   */
  public static synchronized void shutdown() {
    closeResources();
  }

  private static synchronized void cacheDatabaseImpl(int maxYear) {
    if (transitions == null) {
      try {
        lastCachedYear = maxYear;
        loadData(maxYear);
      } catch (Exception e) {
        closeResources();
        throw e;
      }
    }
  }

  private static synchronized void closeResources()  {
    if (zoneIdToTable != null) {
      zoneIdToTable.clear();
      zoneIdToTable = null;
    }
    if (transitions != null) {
      transitions.close();
      transitions = null;
    }
    if (timeZoneInfo != null) {
      timeZoneInfo.close();
      timeZoneInfo = null;
    }
    sortedNormalizedTimeZones.clear();
  }

  private static long getScaleFactor(ColumnView input){
    DType inputType = input.getType();
    if (inputType == DType.TIMESTAMP_SECONDS){
      return 1;
    } else if (inputType == DType.TIMESTAMP_MILLISECONDS){
      return 1000;
    } else if (inputType == DType.TIMESTAMP_MICROSECONDS){
      return 1000*1000;
    } else if (inputType == DType.INT64) {
      // This is for seconds in long, this is for casting from string to timestamp
      return 1;
    }
    throw new UnsupportedOperationException("Unsupported data type: " + inputType);
  }

  public static boolean isSupportedTimeZone(String zoneId) {
    try {
      // check that zoneID is valid and supported by Java
      getZoneId(zoneId);
      return true;
    } catch (ZoneRulesException e) {
      return false;
    }
  }

  public static boolean exceedsMaxYearThresholdOfDST(ColumnView input) {
    return shouldFallbackToCpu(input, null, /* checkTimeZone */ false);
  }

  private static boolean shouldFallbackToCpu(
      ColumnView input, ZoneId zoneId) {
    return shouldFallbackToCpu(input, zoneId, /* checkTimeZone */ true);
  }

  private static Scalar getThresholdForDST(DType type, long scaleFactor) {
    if (type == DType.INT64) {
      return Scalar.fromLong(maxTimestamp * scaleFactor);
    } else {
      return Scalar.timestampFromLong(type, maxTimestamp * scaleFactor);
    }
  }

  // enforce that all timestamps, regardless of timezone, be less than the desired date
  private static boolean shouldFallbackToCpu(
      ColumnView input, ZoneId zoneId, boolean checkTimeZone){
    if (checkTimeZone && (zoneId.getRules().isFixedOffset() ||
        zoneId.getRules().getTransitionRules().isEmpty())) {
      return false;
    }
    boolean isValid = false;
    long scaleFactor = getScaleFactor(input);

    try (Scalar targetTimestamp = getThresholdForDST(input.getType(), scaleFactor);
         ColumnVector compareCv = input.binaryOp(BinaryOp.GREATER, targetTimestamp, DType.BOOL8);
         Scalar isGreater = compareCv.any()) {
      if (!isGreater.isValid()) {
        isValid = false;
      }
      else {
        isValid = isGreater.getBoolean();
      }
    } catch (Exception e) {
      log.error("Error validating input timestamps", e);
      // don't need to throw error, can try CPU processing
      return true;
    }
    return isValid;
  }

  private static ColumnVector cpuChangeTimestampTz(ColumnVector input, ZoneId currentTimeZone, ZoneId targetTimeZone) {
    log.warn("Performing timestamp conversion on the CPU. There is a timestamp with a year over " + lastCachedYear +
      ". You can modify the maxYear by setting spark.rapids.timezone.transitionCache.maxYear, or changing the inputs " +
      "to stay under the year " +  lastCachedYear + ".");
    ColumnVector resultCV = null;
    try (HostColumnVector hostCV = input.copyToHost()) {
      int rows = (int) hostCV.getRowCount();
      DType inputType = input.getType();
      long scaleFactor = getScaleFactor(input);
      
      try (HostColumnVector.Builder builder = HostColumnVector.builder(inputType, rows)) {
        for (int i = 0; i < rows; i++) {
          if (hostCV.isNull(i)) {
            builder.appendNull();
            continue;
          }
          
          long timestamp = hostCV.getLong(i);
          long unitOffset = timestamp % scaleFactor;
          timestamp /= scaleFactor;
          Instant instant = Instant.ofEpochSecond(timestamp);
          /*
           * .atZone(targetTimeZone) keeps same underlying timestamp, adds tzinfo
           * .toLocalDateTime() keeps only the local date time
           * .atZone(currentTimeZone) creates a ZonedDateTime, new underlying timestamp+tzinfo
           * .toInstant().getEpochSecond() grabs that underlying timestamp
           * 
           * Example: input = 0, currentTimeZone = UTC, targetTimeZone = LA
           * atZone(LA): 1969/12/31 16:00:00, tz=LA, timestamp=0
           * toLocalDateTime(): 1969/12/31 16:00:00
           * atZone(UTC): 1969/12/31 16:00:00, tz=UTC, timestamp=-28800
           * 
           * We reinterpret the underlying timestamp representation
           * Spark Code For Reference:
           * https://github.com/apache/spark/blob/ed702c0db71a2d185e9d56567375616170a1d6af/sql/api/src/main/scala/org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L175-L177
          */
          timestamp = instant.atZone(targetTimeZone).toLocalDateTime().atZone(currentTimeZone)
            .toInstant().getEpochSecond();
          timestamp = timestamp * scaleFactor + unitOffset;
          builder.append(timestamp);
        }
        
        resultCV = builder.buildAndPutOnDevice();
      }
    }
    return resultCV;
  }

  // From Spark, convert instant to microseconds with checking overflow
  private static final long MICROS_PER_SECOND = 1000 * 1000;
  private static final long MIN_SECONDS = Math.floorDiv(Long.MIN_VALUE, MICROS_PER_SECOND);

  private static long instantToMicros(Instant instant) {
    long secs = instant.getEpochSecond();
    if (secs == MIN_SECONDS) {
      long us = Math.multiplyExact(secs + 1, MICROS_PER_SECOND);
      return Math.addExact(us, instant.getNano() / 1000L - MICROS_PER_SECOND);
    } else {
      long us = Math.multiplyExact(secs, MICROS_PER_SECOND);
      return Math.addExact(us, instant.getNano() / 1000L);
    }
  }

  /**
   * Running on CPU to convert the intermediate result of casting string to
   * timestamp to timestamp.
   * This function is used for casting string with timezone to timestamp
   *
   * @param input_seconds      second part of UTC timestamp column
   * @param input_microseconds microseconds part of UTC timestamp column
   * @param invalid            if the parsing from string to timestamp is valid
   * @param tzType             if the timezone in string is fixed offset or not
   * @param tzOffset           the tz offset value, only applies to fixed type
   *                           timezone
   * @param tzIndex            the index to the timezone transition table
   * @return timestamp column in microseconds
   */
  public static ColumnVector cpuChangeTimestampTzWithTimezones(
      ColumnView invalid,
      ColumnView input_seconds,
      ColumnView input_microseconds,
      ColumnView tzType,
      ColumnView tzOffset,
      ColumnView tzIndex) {
    verifyDatabaseCached();
    ColumnVector resultCV = null;
    try (HostColumnVector hostInput = input_seconds.copyToHost();
        HostColumnVector hostMicroInput = input_microseconds.copyToHost();
        HostColumnVector hostInvalid = invalid.copyToHost();
        HostColumnVector hostTzType = tzType.copyToHost();
        HostColumnVector hostTzOffset = tzOffset.copyToHost();
        HostColumnVector hostTzIndex = tzIndex.copyToHost()) {
      int rows = (int) hostInput.getRowCount();

      try (HostColumnVector.Builder builder = HostColumnVector.builder(DType.TIMESTAMP_MICROSECONDS, rows)) {
        for (int i = 0; i < rows; i++) {
          if (hostInvalid.getByte(i) != 0) {
            // invalid parsing
            builder.appendNull();
            continue;
          }

          long seconds = hostInput.getLong(i);
          long microseconds = hostMicroInput.getInt(i);

          if (hostTzType.getByte(i) == 1) {
            // fixed offset in seconds
            int offset = hostTzOffset.getInt(i);
            try {
              seconds = Math.addExact(seconds, -offset);
              long microsecondsForSeconds = Math.multiplyExact(seconds, 1000000L);
              long result = Math.addExact(microsecondsForSeconds, microseconds);
              builder.append(result);
            } catch (ArithmeticException e) {
              // overflow
              builder.appendNull();
            }
            continue;
          }

          Instant instant = Instant.ofEpochSecond(seconds, microseconds * 1000L);

          // get the timezone index
          int tzIndexToTzInfo = hostTzIndex.getInt(i);
          String normalizedTz = sortedNormalizedTimeZones.get(tzIndexToTzInfo);

          // Please refer to the `cpuChangeTimestampTz` for more details
          Instant toInstance = instant.atZone(utcZoneId).toLocalDateTime()
              .atZone(ZoneId.of(normalizedTz)).toInstant();
          try {
            seconds = instantToMicros(toInstance);
            builder.append(seconds);
          } catch (ArithmeticException e) {
            // overflow
            builder.appendNull();
          }
        }

        resultCV = builder.buildAndPutOnDevice();
      }
    }
    return resultCV;
  }

  public static ColumnVector fromTimestampToUtcTimestamp(ColumnVector input, ZoneId currentTimeZone) {
    // there is technically a race condition on shutdown. Shutdown could be called after
    // the database is cached. This would result in a null pointer exception at some point
    // in the processing. This should be rare enough that it is not a big deal.
    if (shouldFallbackToCpu(input, currentTimeZone)) {
      return cpuChangeTimestampTz(input, currentTimeZone, utcZoneId);
    }
    Integer tzIndex = zoneIdToTable.get(currentTimeZone.normalized().toString());
    try (Table transitions = getTransitions()) {
      return new ColumnVector(convertTimestampColumnToUTC(input.getNativeView(),
          transitions.getNativeView(), tzIndex));
    }
  }
  
  public static ColumnVector fromUtcTimestampToTimestamp(ColumnVector input, ZoneId desiredTimeZone) {
    // there is technically a race condition on shutdown. Shutdown could be called after
    // the database is cached. This would result in a null pointer exception at some point
    // in the processing. This should be rare enough that it is not a big deal.
    if (shouldFallbackToCpu(input, desiredTimeZone)) {
      return cpuChangeTimestampTz(input, utcZoneId, desiredTimeZone);
    }
    Integer tzIndex = zoneIdToTable.get(desiredTimeZone.normalized().toString());
    try (Table transitions = getTransitions()) {
      return new ColumnVector(convertUTCTimestampColumnToTimeZone(input.getNativeView(),
          transitions.getNativeView(), tzIndex));
    }
  }


  // Ported from Spark. Used to format timezone ID string with (+|-)h:mm and (+|-)hh:m
  public static ZoneId getZoneId(String timeZoneId) {
    String formattedZoneId = timeZoneId
      // To support the (+|-)h:mm format because it was supported before Spark 3.0.
      .replaceFirst("(\\+|\\-)(\\d):", "$10$2:")
      // To support the (+|-)hh:m format because it was supported before Spark 3.0.
      .replaceFirst("(\\+|\\-)(\\d\\d):(\\d)$", "$1$2:0$3");
    return ZoneId.of(formattedZoneId, ZoneId.SHORT_IDS);
  }

  @SuppressWarnings("unchecked")
  private static synchronized void loadData(int finalTransitionYear) {
    try {
      // Spark uses timezones from TimeZone.getAvailableIDs
      // We use ZoneId.normalized to reduce the number of timezone names.
      // `transitions` saves transitions for normalized timezones.
      //
      // e.g.:
      //   "Etc/GMT" and "Etc/GMT+0" are from TimeZone.getAvailableIDs
      //   ZoneId.of("Etc/GMT").normalized.getId = Z;
      //   ZoneId.of("Etc/GMT+0").normalized.getId = Z
      // Both Etc/GMT and Etc/GMT+0 have normalized Z.
      // Use the normalized form will dedupe transition table size.
      //
      // For `fromTimestampToUtcTimestamp` and `fromUtcTimestampToTimestamp`, it will first
      // normalize the timezone, e.g.: Etc/GMT => Z, then the use Z to find the transition index.
      // But for cast string(with timezone) to timestamp, it may contain non-normalized tz.
      // E.g.: '2025-01-01 00:00:00 Etc/GMT', so should map "Etc/GMT", "Etc/GMT+0" and "Z" to
      // the same transition index. This means size of `zoneIdToTable` > size of `transitions`
      //

      // get and sort timezones
      String[] timeZones = TimeZone.getAvailableIDs();
      List<String> sortedTimeZones = new ArrayList<>(Arrays.asList(timeZones));
      // Note: Z is a special normalized timezone from UTC: ZoneId.of("UTC").normalized = Z
      // TimeZone.getAvailableIDs does not contain Z
      // Should add Z to `zoneIdToTable`
      sortedTimeZones.add("Z");
      Collections.sort(sortedTimeZones);

      List<List<HostColumnVector.StructData>> masterTransitions = new ArrayList<>();
      zoneIdToTable = new HashMap<>();
      for (String nonNormalizedTz : sortedTimeZones) {
        // we use the normalized form to dedupe
        ZoneId zoneId = ZoneId.of(nonNormalizedTz, ZoneId.SHORT_IDS).normalized();

        String normalizedTz = zoneId.getId();
        ZoneRules zoneRules = zoneId.getRules();
        if (!zoneIdToTable.containsKey(normalizedTz)) {
          List<ZoneOffsetTransition> zoneOffsetTransitions = new ArrayList<>(zoneRules.getTransitions());
          zoneOffsetTransitions.sort(Comparator.comparing(ZoneOffsetTransition::getInstant));
          // It is desired to get lastTransitionEpochSecond because some rules don't start until late (e.g. 2007)
          long lastTransitionEpochSecond = Long.MIN_VALUE;
          if (!zoneOffsetTransitions.isEmpty()) {
            long transitionInstant = zoneOffsetTransitions.get(zoneOffsetTransitions.size()-1).getInstant().getEpochSecond();
            lastTransitionEpochSecond = Math.max(transitionInstant, lastTransitionEpochSecond);
          }
          List<ZoneOffsetTransitionRule> transitionRules = zoneRules.getTransitionRules();
          for (ZoneOffsetTransitionRule transitionRule : transitionRules){
            for (int year = initialTransitionYear; year <= finalTransitionYear; year++){
              ZoneOffsetTransition transition = transitionRule.createTransition(year);
              if (transition.getInstant().getEpochSecond() > lastTransitionEpochSecond){
                zoneOffsetTransitions.add(transition);
              }
            }
          }
          // sort the transitions, multiple rules means transitions not added chronologically
          zoneOffsetTransitions.sort(Comparator.comparing(ZoneOffsetTransition::getInstant));
          int idx = masterTransitions.size();
          List<HostColumnVector.StructData> data = new ArrayList<>();
          if (zoneRules.isFixedOffset()) {
            data.add(
                new HostColumnVector.StructData(Long.MIN_VALUE, Long.MIN_VALUE,
                    zoneRules.getOffset(Instant.now()).getTotalSeconds())
            );
          } else {
            // Capture the first official offset (before any transition) using Long min
            ZoneOffsetTransition first = zoneOffsetTransitions.get(0);
            data.add(
                new HostColumnVector.StructData(Long.MIN_VALUE, Long.MIN_VALUE,
                    first.getOffsetBefore().getTotalSeconds())
            );
            zoneOffsetTransitions.forEach(t -> {
              // Whether transition is an overlap vs gap.
              // In Spark:
              // if it's a gap, then we use the offset after *on* the instant
              // If it's an overlap, then there are 2 sets of valid timestamps in that are overlapping
              // So, for the transition to UTC, you need to compare to instant + {offset before}
              // The time math still uses {offset after}
              if (t.isGap()) {
                data.add(
                    new HostColumnVector.StructData(
                        t.getInstant().getEpochSecond(),
                        t.getInstant().getEpochSecond() + t.getOffsetAfter().getTotalSeconds(),
                        t.getOffsetAfter().getTotalSeconds())
                );
              } else {
                data.add(
                    new HostColumnVector.StructData(
                        t.getInstant().getEpochSecond(),
                        t.getInstant().getEpochSecond() + t.getOffsetBefore().getTotalSeconds(),
                        t.getOffsetAfter().getTotalSeconds())
                );
              }
            });
          }
          masterTransitions.add(data);
          // add index for normalized timezone
          zoneIdToTable.put(normalizedTz, idx);
          sortedNormalizedTimeZones.add(normalizedTz);
        } // end of: if (!zoneIdToTable.containsKey(normalizedTz)) {

        // set max year:
        maxTimestamp = LocalDateTime.of(finalTransitionYear + 1, 1, 2, 0, 0, 0)
            .atZone(utcZoneId).toEpochSecond();

        // Add index for non-normalized timezones
        // e.g.:
        //   normalize "Etc/GMT" = Z
        //   normalize "Etc/GMT+0" = Z
        // use the index of Z for Etc/GMT and Etc/GMT+0
        zoneIdToTable.put(nonNormalizedTz, zoneIdToTable.get(normalizedTz));
      } // end of for

      HostColumnVector.DataType childType = new HostColumnVector.StructType(false,
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT32));
      HostColumnVector.DataType resultType =
          new HostColumnVector.ListType(false, childType);
      transitions = HostColumnVector.fromLists(resultType,
          masterTransitions.toArray(new List[0]));
      timeZoneInfo = getTimeZoneInfo(sortedTimeZones, zoneIdToTable);
    } catch (Exception e) {
      throw new IllegalStateException("load timezone DB cache failed!", e);
    }
  }

  public static synchronized Table getTransitions() {
    verifyDatabaseCached();
    try (ColumnVector fixedTransitions = transitions.copyToDevice()) {
      return new Table(fixedTransitions);
    }
  }

  /**
   * FOR TESTING PURPOSES ONLY, DO NOT USE IN PRODUCTION
   *
   * This method retrieves the raw list of struct data that forms the list of 
   * fixed transitions for a particular zoneId. 
   *
   * It has default visibility so the test can access it.
   * @param zoneId
   * @return list of fixed transitions
   */
  static synchronized List getHostTransitions(String zoneId) {
    verifyDatabaseCached();
    zoneId = ZoneId.of(zoneId).normalized().toString(); // we use the normalized form to dedupe
    Integer idx = zoneIdToTable.get(zoneId);
    if (idx == null) {
      return null;
    }
    return transitions.getList(idx);
  }

  /**
   * Generate a struct column to record timezone information
   * STRUCT<tz_name: string, index_to_transition_table: int, is_DST: int8>
   * The struct column is sorted by tz_name, it is used to query the index to the
   * transition table, to query if tz is Daylight Saving timezone.
   *
   * @param sortedTimezones is sorted and supported timezones
   * @param zoneIdToTable   is a map from non-normalized timezone to index in transition table
   */
  private static HostColumnVector getTimeZoneInfo(List<String> sortedTimezones,
      java.util.Map<String, Integer> zoneIdToTable) {
    HostColumnVector.DataType type = new HostColumnVector.StructType(false,
        new HostColumnVector.BasicType(false, DType.STRING),
        new HostColumnVector.BasicType(false, DType.INT32),
        new HostColumnVector.BasicType(false, DType.BOOL8));
    ArrayList<HostColumnVector.StructData> data = new ArrayList<>();

    for (String tz : sortedTimezones) {
      ZoneId zoneId = ZoneId.of(tz, ZoneId.SHORT_IDS);
      boolean isDST = !zoneId.getRules().getTransitionRules().isEmpty();
      Integer indexToTable = zoneIdToTable.get(tz);
      if (indexToTable != null) {
        data.add(new HostColumnVector.StructData(tz, indexToTable, isDST));
      } else {
        throw new IllegalStateException("Could not find timezone " + tz);
      }
    }
    return HostColumnVector.fromStructs(type, data);
  }

  /**
   * Return a struct column which contains timezone information
   * STRUCT<tz_name: string, index_to_transition_table: int, is_DST: int8>
   * The struct column is sorted by tz_name, it is used to query the index to the
   * transition table, to query if tz is Daylight Saving timezone.
   * The caller is responsible to close the returned column vector.
   */
  public static synchronized ColumnVector getTimeZoneInfo() {
    verifyDatabaseCached();
    return timeZoneInfo.copyToDevice();
  }

  public static Integer getIndexToTransitionTable(String timezone) {
    verifyDatabaseCached();
    return zoneIdToTable.get(timezone);
  }

  /**
   * Running on GPU to convert the intermediate result of casting string to
   * timestamp to timestamp.
   * This function is used for casting string with timezone to timestamp.
   * MUST make sure input does not exceed max year threshold and has no DST
   *
   * @param input_seconds      second part of UTC timestamp column
   * @param input_microseconds microseconds part of UTC timestamp column
   * @param invalid            if the parsing from string to timestamp is valid
   * @param tzType             if the timezone in string is fixed offset or not
   * @param tzOffset           the tz offset value, only applies to fixed type
   *                           timezone
   * @param tzIndex            the index to the timezone transition/timeZoneInfo
   *                           table
   * @return timestamp column in microseconds
   */
  public static ColumnVector fromTimestampToUtcTimestampWithTzCv(
      ColumnView invalid,
      ColumnView input_seconds,
      ColumnView input_microseconds,
      ColumnView tzType,
      ColumnView tzOffset,
      ColumnView tzIndex) {
    try (Table transitions = getTransitions()) {
      return new ColumnVector(convertTimestampColumnToUTCWithTzCv(
          input_seconds.getNativeView(),
          input_microseconds.getNativeView(),
          invalid.getNativeView(),
          tzType.getNativeView(),
          tzOffset.getNativeView(),
          transitions.getNativeView(),
          tzIndex.getNativeView()));
    }
  }

  public static boolean isDST(String timezone) {
    ZoneId zoneId = ZoneId.of(timezone, ZoneId.SHORT_IDS);
    return !zoneId.getRules().getTransitionRules().isEmpty();
  }

  /**
   * Read from `sun.util.calendar.ZoneInfo` via reflection to get timezone info.
   * 
   * @param tzId timezone id
   * @return timezone info
   */
  private static TzInfoInJavaUtilForORC getInfoForUtilTZ(String tzId) {
    TimeZone tz = TimeZone.getTimeZone(tzId);
    if (!(tz instanceof ZoneInfo)) {
      throw new UnsupportedOperationException("Unsupported timezone: " + tzId);
    }
    ZoneInfo zoneInfo = (ZoneInfo) tz;

    // The constructor of TimeZoneInfoInJavaUtilPackage will extract and repack
    // transitions info.
    return new TzInfoInJavaUtilForORC(
        (long[]) FieldUtils.readField(zoneInfo, "transitions"),
        (int[]) FieldUtils.readField(zoneInfo, "offsets"),
        (int) FieldUtils.readField(zoneInfo, "rawOffset"));
  }

  private static ColumnVector getTransitionsForUtilTZ(TzInfoInJavaUtilForORC info) {
    try (HostColumnVector hcv = HostColumnVector.fromLongs(info.transitions)) {
      return hcv.copyToDevice();
    }
  }

  /**
   * Get timezone info from `java.util.TimeZone`.
   * 
   * @param info timezone info
   * @return a table on GPU containing timezone info from `java.util.TimeZone`
   */
  private static Table getTableForUtilTZ(TzInfoInJavaUtilForORC info) {
    if (info.transitions == null) {
      // fixed offset timezone
      return null;
    }
    try (ColumnVector trans = getTransitionsForUtilTZ(info)) {
      return new Table(trans);
    } catch (Exception e) {
      throw new IllegalStateException("get timezone info from java.util.TimeZone failed!", e);
    }
  }

  /**
   * Does the given timezone have Daylight Saving Time(DST) rules.
   */
  private static boolean hasDaylightSavingTime(String timezoneId) {
    ZoneId zoneId = ZoneId.of(timezoneId, ZoneId.SHORT_IDS);
    return !zoneId.getRules().getTransitionRules().isEmpty();
  }

  /**
   * Convert timestamps between writer/reader timezones for ORC reading.
   * Similar to `org.apache.orc.impl.SerializationUtils.convertBetweenTimezones`.
   * `SerializationUtils.convertBetweenTimezones` gets offset between timezones.
   * This function does the same thing and then apply the offset to get the
   * final timestamps.
   * For more details, refer to link:
   * https://github.com/apache/orc/blob/rel/release-1.9.1/java/core/src/
   * java/org/apache/orc/impl/SerializationUtils.java#L1440
   * 
   * @param input          input timestamp column in microseconds.
   * @param writerTimezone writer timezone, it's from ORC stripe metadata.
   * @param readerTimezone reader timezone, it's from current JVM default
   *                       timezone.
   * @return timestamp column in microseconds after converting between timezones
   */
  public static ColumnVector convertBetweenTimezones(
      ColumnVector input,
      String writerTimezone,
      String readerTimezone) {
    // Does not support DST timezone now, just throw exception.
    if (hasDaylightSavingTime(writerTimezone) ||
        hasDaylightSavingTime(readerTimezone)) {
      throw new UnsupportedOperationException("Daylight Saving Time is not supported now.");
    }

    // get timezone info from `java.util.TimeZone`
    try (TzInfoInJavaUtilForORC writerTzInfo = getInfoForUtilTZ(writerTimezone);
        TzInfoInJavaUtilForORC readerTzInfo = getInfoForUtilTZ(readerTimezone);
        Table writerTzInfoTable = getTableForUtilTZ(writerTzInfo);
        Table readerTzInfoTable = getTableForUtilTZ(readerTzInfo)) {

      // convert between timezones
      return new ColumnVector(convertBetweenTimezones(
          input.getNativeView(),
          writerTzInfoTable != null ? writerTzInfoTable.getNativeView() : 0L,
          writerTzInfo.rawOffset,
          readerTzInfoTable != null ? readerTzInfoTable.getNativeView() : 0L,
          readerTzInfo.rawOffset));
    } catch (Exception e) {
      throw new IllegalStateException("convert between timezones failed!", e);
    }
  }

  private static native long convertTimestampColumnToUTC(long input, long transitions, int tzIndex);

  private static native long convertUTCTimestampColumnToTimeZone(long input, long transitions, int tzIndex);

  private static native long convertTimestampColumnToUTCWithTzCv(
      long input_seconds, long input_microseconds, long invalid, long tzType,
      long tzOffset, long transitions, long tzIndex);

  private static native long convertBetweenTimezones(
      long input,
      long writerTzInfoTable,
      int writerTzRawOffset,
      long readerTzInfoTable,
      int readerTzRawOffset);
}
