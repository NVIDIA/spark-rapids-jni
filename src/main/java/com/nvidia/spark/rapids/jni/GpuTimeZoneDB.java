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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.DayOfWeek;
import java.time.Instant;
import java.time.ZoneId;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneOffsetTransitionRule;
import java.time.zone.ZoneRules;
import java.time.zone.ZoneRulesException;
import java.util.*;
import java.util.concurrent.Executors;

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

  // Timezone transition info:
  // LIST<STRUCT<utcInstant: int64, localInstant: int64, offset: int32>>
  private static HostColumnVector transitions;

  // Timezone DST rules: LIST<LIST<INT>>
  // Each sub list has constant 16 integers, each 8 integers is a DST rule.
  //    index 0: month:int,            // from 1 (January) to 12 (December)
  //    index 1: dayOfMonth: int,      // from -28 to 31 excluding 0
  //    index 2: dayOfWeek: int,       // from 0 (Monday) to 6 (Sunday)
  //    index 3: secondsOfDay: int,    // transition time in seconds in a day
  //    index 4: timeMode: int,        // the mode of `secondsOfDay`: 0 UTC, 1 WALL, 2 STANDARD
  //    index 5: standardOffset: int,  // standard offset
  //    index 6: offsetBefore: int,    // the offset before the cutover
  //    index 7: offsetAfter: int      // The offset after the cutover
  //    index 8: the 2nd rule begin 
  //    ...
  //    index 15: the 2nd rule end
  private static HostColumnVector dstRules;

  // Map from timezone name to the index in the `transitions`
  private static java.util.Map<String, Integer> zoneIdToTable;

  // host column STRUCT<tz_name: string, index_to_transition_table: int, is_DST: int8>,
  // sorted by timezone, is used to query index to transition table and if tz is DST
  // Casting string with timezone to timestamp needs loading all timezone is successful.
  // If this is not null, it indicates loading is successful,
  // because it's the last variable to construct in `loadData` function.
  // use this reference to indicate if timezone cache is initialized successfully.
  // The tz_name column contains both normalized and non-normalized timezone names.
  private static volatile HostColumnVector timeZoneInfo;

  /**
   * This is deprecated, will be removed.
   */
  public static void cacheDatabaseAsync(int maxYear) {
    // start a new thread to load
    Runnable runnable = () -> {
      try {
        cacheDatabaseImpl();
      } catch (Exception e) {
        log.error("cache timezone transitions cache failed", e);
      }
    };
    Thread thread = Executors.defaultThreadFactory().newThread(runnable);
    thread.setName("gpu-timezone-database-0");
    thread.setDaemon(true);
    thread.start();
  }

  /**
   * This is the replacement of the above function.
   * This should be called on startup of an executor.
   * Runs in a thread asynchronously.
   * If `shutdown` was called ever, then will not load the cache
   */
  public static void cacheDatabaseAsync() {
    // start a new thread to load
    Runnable runnable = () -> {
      try {
        cacheDatabaseImpl();
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
   * This is deprecated, will be removed.
   */
  public static void cacheDatabase(int maxYear) {
    cacheDatabaseImpl();
  }

  /**
   * This is the replacement of the above function
   * Cache the database. This will take some time like several seconds.
   * If one `cacheDatabase` is running, other `cacheDatabase` will wait until caching is done.
   * If cache is exits, do not load cache again.
   */
  public static void cacheDatabase() {
    cacheDatabaseImpl();
  }

  /**
   * close the cache, used when Plugin is closing
   */
  public static synchronized void shutdown() {
    closeResources();
  }

  private static synchronized void cacheDatabaseImpl() {
    if (transitions == null) {
      try {
        loadData();
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
    if (dstRules != null) {
      dstRules.close();
      dstRules = null;
    }
    if (timeZoneInfo != null) {
      timeZoneInfo.close();
      timeZoneInfo = null;
    }
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

  public static ColumnVector fromTimestampToUtcTimestamp(ColumnVector input, ZoneId currentTimeZone) {
    // there is technically a race condition on shutdown. Shutdown could be called after
    // the database is cached. This would result in a null pointer exception at some point
    // in the processing. This should be rare enough that it is not a big deal.
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
  private static synchronized void loadData() {
    try {
      // Spark uses timezones from TimeZone.getAvailableIDs
      // We use ZoneId.normalized to reduce the number of timezone names.
      // `transitions` saves transitions for normalized timezones.
      //
      // e.g.:
      // "Etc/GMT" and "Etc/GMT+0" are from TimeZone.getAvailableIDs
      // ZoneId.of("Etc/GMT").normalized.getId = Z;
      // ZoneId.of("Etc/GMT+0").normalized.getId = Z
      // Both Etc/GMT and Etc/GMT+0 have normalized Z.
      // Use the normalized form will dedupe transition table size.
      //
      // For `fromTimestampToUtcTimestamp` and `fromUtcTimestampToTimestamp`, it will
      // first
      // normalize the timezone, e.g.: Etc/GMT => Z, then the use Z to find the
      // transition index.
      // But for cast string(with timezone) to timestamp, it may contain
      // non-normalized tz.
      // E.g.: '2025-01-01 00:00:00 Etc/GMT', so should map "Etc/GMT", "Etc/GMT+0" and
      // "Z" to
      // the same transition index. This means size of `zoneIdToTable` > size of
      // `transitions`
      //

      // get and sort timezones
      String[] timeZones = TimeZone.getAvailableIDs();
      List<String> sortedTimeZones = new ArrayList<>(Arrays.asList(timeZones));
      // Note: Z is a special normalized timezone from UTC:
      // ZoneId.of("UTC").normalized = Z
      // TimeZone.getAvailableIDs does not contain Z
      // Should add Z to `zoneIdToTable`
      sortedTimeZones.add("Z");
      Collections.sort(sortedTimeZones);

      List<List<HostColumnVector.StructData>> masterTransitions = new ArrayList<>();
      List<List<Integer>> masterDsts = new ArrayList<>();

      zoneIdToTable = new HashMap<>();
      for (String nonNormalizedTz : sortedTimeZones) {
        // we use the normalized form to dedupe
        ZoneId zoneId = ZoneId.of(nonNormalizedTz, ZoneId.SHORT_IDS).normalized();

        String normalizedTz = zoneId.getId();
        ZoneRules zoneRules = zoneId.getRules();
        if (!zoneIdToTable.containsKey(normalizedTz)) {
          List<ZoneOffsetTransition> zoneOffsetTransitions = new ArrayList<>(zoneRules.getTransitions());
          zoneOffsetTransitions.sort(Comparator.comparing(ZoneOffsetTransition::getInstant));
          List<ZoneOffsetTransitionRule> dstTransitionRules = zoneRules.getTransitionRules();
          int idx = masterTransitions.size();
          List<HostColumnVector.StructData> data = new ArrayList<>();
          List<Integer> dstData = new ArrayList<>();
          if (zoneRules.isFixedOffset()) {
            data.add(new HostColumnVector.StructData(Long.MIN_VALUE, Long.MIN_VALUE,
                zoneRules.getOffset(Instant.now()).getTotalSeconds()));
          } else {
            // Capture the first official offset (before any transition) using Long min
            ZoneOffsetTransition first = zoneOffsetTransitions.get(0);
            data.add(new HostColumnVector.StructData(Long.MIN_VALUE, Long.MIN_VALUE,
                first.getOffsetBefore().getTotalSeconds()));
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
                        t.getOffsetAfter().getTotalSeconds()));
              } else {
                data.add(
                    new HostColumnVector.StructData(
                        t.getInstant().getEpochSecond(),
                        t.getInstant().getEpochSecond() + t.getOffsetBefore().getTotalSeconds(),
                        t.getOffsetAfter().getTotalSeconds()));
              }
            });

            // collect DST rules
            if (dstTransitionRules.size() != 0 && dstTransitionRules.size() != 2) {
              // Checked all the timezones, the size of DST rules for a timezone is 2.
              throw new IllegalStateException("DST rules size is not 2.");
            }

            dstTransitionRules.forEach(dstRule -> {
              if (dstRule.isMidnightEndOfDay()) {
                // Checked all the timezones, there is no midnight end of day for DST rules.
                // This is a protection in case JVM adds new timezones in the future.
                throw new IllegalStateException("Unsupported midnight end of day for DST rules.");
              }

              DayOfWeek dow = dstRule.getDayOfWeek();
              int dayOfWeek = dow != null ? dow.getValue() - 1 : -1;
              dstData.add(dstRule.getMonth().getValue()); // from 1 (January) to 12 (December)
              dstData.add(dstRule.getDayOfMonthIndicator()); // from -28 to 31 excluding 0
              dstData.add(dayOfWeek); // from 0 (Monday) to 6 (Sunday)
              dstData.add(dstRule.getLocalTime().toSecondOfDay()); // transition time in seconds in a day
              dstData.add(dstRule.getTimeDefinition().ordinal()); // 0 UTC, 1 WALL, 2 STANDARD
              dstData.add(dstRule.getStandardOffset().getTotalSeconds()); // standard offset
              dstData.add(dstRule.getOffsetBefore().getTotalSeconds()); // the offset before the cutover
              dstData.add(dstRule.getOffsetAfter().getTotalSeconds()); // the offset after the cutover
            });
          }
          masterTransitions.add(data);
          masterDsts.add(dstData);
          // add index for normalized timezone
          zoneIdToTable.put(normalizedTz, idx);
        } // end of: if (!zoneIdToTable.containsKey(normalizedTz)) {

        // Add index for non-normalized timezones
        // e.g.:
        // normalize "Etc/GMT" = Z
        // normalize "Etc/GMT+0" = Z
        // use the index of Z for Etc/GMT and Etc/GMT+0
        zoneIdToTable.put(nonNormalizedTz, zoneIdToTable.get(normalizedTz));
      } // end of for

      HostColumnVector.DataType childType = new HostColumnVector.StructType(false,
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT32));
      HostColumnVector.DataType transitionType = new HostColumnVector.ListType(false, childType);
      transitions = HostColumnVector.fromLists(transitionType,
          masterTransitions.toArray(new List[0]));
      dstRules = HostColumnVector.fromLists(getDstDataType(), masterDsts.toArray(new List[0]));
      timeZoneInfo = getTimeZoneInfo(sortedTimeZones, zoneIdToTable);
    } catch (Exception e) {
      throw new IllegalStateException("load timezone DB cache failed!", e);
    }
  }

  private static HostColumnVector.DataType getDstDataType() {
      return new HostColumnVector.ListType(false,
          new HostColumnVector.BasicType(false, DType.INT32));
  }

  public static synchronized Table getTransitions() {
    verifyDatabaseCached();
    try (ColumnVector fixedTransitions = transitions.copyToDevice();
         ColumnVector dsts = dstRules.copyToDevice()) {
      return new Table(fixedTransitions, dsts);
    }
  }

  /**
   * FOR TESTING PURPOSES ONLY, DO NOT USE IN PRODUCTION
   * This method retrieves the raw list of struct data that forms the list of
   * fixed transitions for a particular zoneId. 
   * It has default visibility so the test can access it.
   * @param zoneId timezone id
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
   * This is deprecated, will be removed.
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
   * This is deprecated, will be removed.
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
   * Running on GPU to convert the intermediate result of casting string to timestamp.
   * This function is used for casting string with timezone to timestamp.
   * TODO: Handle the case exceed max year threshold and has no DST
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

  private static native long convertTimestampColumnToUTC(long input, long transitions, int tzIndex);

  private static native long convertUTCTimestampColumnToTimeZone(long input, long transitions, int tzIndex);

  private static native long convertTimestampColumnToUTCWithTzCv(
      long input_seconds, long input_microseconds, long invalid, long tzType,
      long tzOffset, long transitions, long tzIndex);

}
