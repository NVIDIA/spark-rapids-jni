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
 *
 * Provides the following APIs
 * - Timezone rebasing APIs: `fromTimestampToUtcTimestamp`, etc.
 * - Utilities for casting string with timezone to timestamp APIs
 * - Loading, shutdown, and checking APIs, etc.
 *
 * Note: `cacheDatabase` and `verifyDatabaseCachedSync` are synchronized.
 * When cacheDatabaseAsync is running, the `verifyDatabaseCachedSync` will
 * wait; These APIs guarantee only one thread is loading timezone info cache
 */
public class GpuTimeZoneDB {
  private static final Logger log = LoggerFactory.getLogger(GpuTimeZoneDB.class);

  /**
   * Timezone fixed transitions column, column type is:
   * LIST<STRUCT<utcInstant: int64, localInstant: int64, offset: int32>>
   * This is from `ZoneRules.getTransitions()`
   */
  private static HostColumnVector fixedTransitions;

  /**
   * Timezone DST rules column, column type is: LIST<INT32>
   * This is from `ZoneRules.getTransitionRules()`
   * `fixedTransitions` and `dstRules` compose the full timezone database.
   * If a timezone has no DST, then the list is empty.
   * If a timezone has DST, then the list has 12 integers, which contains 2
   * rules(start rule and end rule)
   * The integers in a list are:
   *
   * index 0: month:int, // from 1 (January) to 12 (December)
   * index 1: dayOfMonth: int, // from -28 to 31 excluding 0
   * index 2: dayOfWeek: int, // from 0 (Monday) to 6 (Sunday), -1 means ignore
   * index 3: timeDiffToMidnight: int, // transition time in seconds compared to
   * midnight
   * index 4: offsetBefore: int, // the offset before the cutover
   * index 5: offsetAfter: int // The offset after the cutover
   * index 6: the 2nd rule begin
   * ...
   * index 11: the 2nd rule end
   *
   */
  private static HostColumnVector dstRules;

  // Map from timezone name to the index in the timezone info table
  private static java.util.Map<String, Integer> zoneIdToTable;

  /**
   * Used by Casting string with timezone to timestamp.
   * Host column STRUCT<tz_name: string, index_to_tz_info_table: int>,
   * sorted by timezone names.
   * Casting string with timezone to timestamp needs loading all timezone.
   * If this is not null, it indicates loading is successful, because it's the
   * last variable to construct in `loadData` function.
   * The tz_name column contains both normalized and non-normalized tz names.
   */
  private static volatile HostColumnVector tzNameToIndexMap;

  /**
   * This is deprecated, will be removed.
   */
  public static void cacheDatabaseAsync(int maxYear) {
    // start a new thread to load
    Runnable runnable = () -> {
      try {
        cacheDatabaseImpl();
      } catch (Exception e) {
        log.error("cache timezone info cache failed", e);
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
        log.error("cache timezone info cache failed", e);
      }
    };
    Thread thread = Executors.defaultThreadFactory().newThread(runnable);
    thread.setName("gpu-timezone-database-0");
    thread.setDaemon(true);
    thread.start();
  }

  private static synchronized void verifyDatabaseCachedSync() {
    if (tzNameToIndexMap == null) {
      throw new IllegalStateException("Timezone DB is not loaded, or the loading was failed.");
    }
  }

  public static void verifyDatabaseCached() {
    if (tzNameToIndexMap != null) {
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
   * If one `cacheDatabase` is running, other `cacheDatabase` will wait until
   * caching is done.
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
    if (fixedTransitions == null) {
      try {
        loadData();
      } catch (Exception e) {
        closeResources();
        throw e;
      }
    }
  }

  private static synchronized void closeResources() {
    if (zoneIdToTable != null) {
      zoneIdToTable.clear();
      zoneIdToTable = null;
    }
    if (fixedTransitions != null) {
      fixedTransitions.close();
      fixedTransitions = null;
    }
    if (dstRules != null) {
      dstRules.close();
      dstRules = null;
    }
    if (tzNameToIndexMap != null) {
      tzNameToIndexMap.close();
      tzNameToIndexMap = null;
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
    // there is technically a race condition on shutdown. Shutdown could be called
    // after
    // the database is cached. This would result in a null pointer exception at some
    // point
    // in the processing. This should be rare enough that it is not a big deal.
    Integer tzIndex = zoneIdToTable.get(currentTimeZone.normalized().toString());
    try (Table timezoneInfo = getTimezoneInfo()) {
      return new ColumnVector(convertTimestampColumnToUTC(input.getNativeView(),
          timezoneInfo.getNativeView(), tzIndex));
    }
  }

  public static ColumnVector fromUtcTimestampToTimestamp(ColumnVector input, ZoneId desiredTimeZone) {
    // there is technically a race condition on shutdown. Shutdown could be called
    // after
    // the database is cached. This would result in a null pointer exception at some
    // point
    // in the processing. This should be rare enough that it is not a big deal.
    Integer tzIndex = zoneIdToTable.get(desiredTimeZone.normalized().toString());
    try (Table timezoneInfo = getTimezoneInfo()) {
      return new ColumnVector(convertUTCTimestampColumnToTimeZone(input.getNativeView(),
          timezoneInfo.getNativeView(), tzIndex));
    }
  }

  // Ported from Spark. Used to format timezone ID string with (+|-)h:mm and
  // (+|-)hh:m
  public static ZoneId getZoneId(String timeZoneId) {
    String formattedZoneId = timeZoneId
        // To support the (+|-)h:mm format because it was supported before Spark 3.0.
        .replaceFirst("(\\+|\\-)(\\d):", "$10$2:")
        // To support the (+|-)hh:m format because it was supported before Spark 3.0.
        .replaceFirst("(\\+|\\-)(\\d\\d):(\\d)$", "$1$2:0$3");
    return ZoneId.of(formattedZoneId, ZoneId.SHORT_IDS);
  }

  /**
   * Get the time difference in seconds compared to the midnight for a transition
   * rule.
   * Note: The returned time is based on Time 00:00:00, may be negative.
   * E.g.: Give transition date "2000-01-02", and transition time diff in seconds
   * "-3600",
   * then the actual transition datetime is "2000-01-01 23:00:00"
   *
   * @param rule transition rule
   * @return the time diff in seconds compared to the midnight
   */
  private static int getTransitionRuleTimeDiffComparedToMidnight(ZoneOffsetTransitionRule rule) {
    int localTimeInSeconds = rule.getLocalTime().toSecondOfDay();
    ZoneOffsetTransitionRule.TimeDefinition timeDef = rule.getTimeDefinition();
    if (ZoneOffsetTransitionRule.TimeDefinition.UTC == timeDef) {
      // UTC mode
      return localTimeInSeconds + rule.getOffsetBefore().getTotalSeconds();
    } else if (ZoneOffsetTransitionRule.TimeDefinition.STANDARD == timeDef) {
      // STANDARD mode
      return localTimeInSeconds + rule.getOffsetBefore().getTotalSeconds()
          - rule.getStandardOffset().getTotalSeconds();
    } else {
      // WALL mode
      return localTimeInSeconds;
    }
  }

  @SuppressWarnings("unchecked")
  private static synchronized void loadData() {
    try {
      // Spark uses timezones from TimeZone.getAvailableIDs
      // We use ZoneId.normalized to reduce the number of timezone names.
      // `fixedTransitions` and `dstRules` only save info for normalized timezones,
      // while `zoneIdToTable` contains both normalized and non-normalized timezones.
      //
      // e.g.:
      // "Etc/GMT" and "Etc/GMT+0" are from TimeZone.getAvailableIDs
      // ZoneId.of("Etc/GMT").normalized.getId = Z;
      // ZoneId.of("Etc/GMT+0").normalized.getId = Z
      // Both Etc/GMT and Etc/GMT+0 have normalized Z.
      // Use the normalized form will dedupe timezone info table size.
      //
      // For `fromTimestampToUtcTimestamp` and `fromUtcTimestampToTimestamp`, it will
      // first normalize the timezone, e.g.: Etc/GMT => Z, then the use Z to find the
      // transition index. But for cast string(with timezone) to timestamp, it may
      // contain non-normalized tz. E.g.: '2025-01-01 00:00:00 Etc/GMT', so should
      // map "Etc/GMT", "Etc/GMT+0" and "Z" to the same transition index.
      // This means size of `zoneIdToTable` > `fixedTransitions` and `dstRules` size
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
              // If it's an overlap, then there are 2 sets of valid timestamps in that are
              // overlapping
              // So, for the transition to UTC, you need to compare to instant + {offset
              // before}
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
              dstData.add(dayOfWeek); // from 0 (Monday) to 6 (Sunday), -1 means not specified
              dstData.add(getTransitionRuleTimeDiffComparedToMidnight(dstRule)); // transition time
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
      fixedTransitions = HostColumnVector.fromLists(transitionType,
          masterTransitions.toArray(new List[0]));
      dstRules = HostColumnVector.fromLists(getDstDataType(), masterDsts.toArray(new List[0]));
      tzNameToIndexMap = getTzNameToIndexMap(sortedTimeZones, zoneIdToTable);
    } catch (Exception e) {
      throw new IllegalStateException("load timezone DB cache failed!", e);
    }
  }

  private static HostColumnVector.DataType getDstDataType() {
    return new HostColumnVector.ListType(false,
        new HostColumnVector.BasicType(false, DType.INT32));
  }

  /**
   * This is deprecated, will be removed.
   * Renamed to `getTimezoneInfo`.
   */
  public static synchronized Table getTransitions() {
    verifyDatabaseCached();
    try (ColumnVector fixedInfo = fixedTransitions.copyToDevice();
        ColumnVector dstInfo = dstRules.copyToDevice()) {
      return new Table(fixedInfo, dstInfo);
    }
  }

   /**
   * Get the timezone info table, which contains two columns:
   * - fixed transitions: LIST<STRUCT<utcInstant: int64, localInstant: int64,
   * offset: int32>>
   * - dst rules: LIST<INT32>
   * The caller is responsible to close the returned table.
   * 
   * @return timezone info table
   */
  public static Table getTimezoneInfo() {
    verifyDatabaseCached();
    try (ColumnVector fixedInfo = fixedTransitions.copyToDevice();
        ColumnVector dstInfo = dstRules.copyToDevice()) {
      return new Table(fixedInfo, dstInfo);
    }
  }

  /**
   * FOR TESTING PURPOSES ONLY, DO NOT USE IN PRODUCTION
   * This method retrieves the raw list of struct data that forms the list of
   * fixed transitions for a particular zoneId.
   * It has default visibility so the test can access it.
   * 
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
    return fixedTransitions.getList(idx);
  }

  /**
   * Generate a map from timezone name to index of transition table.
   * return a column of STRUCT<tz_name: string, index_to_tz_info_table: int>
   * The struct column is sorted by tz_name, it is used to query the index to the
   * transition table.
   *
   * @param sortedTimezones is sorted and supported timezones
   * @param zoneIdToTable   is a map from non-normalized timezone to index in
   *                        transition table
   */
  private static HostColumnVector getTzNameToIndexMap(List<String> sortedTimezones,
      java.util.Map<String, Integer> zoneIdToTable) {
    HostColumnVector.DataType type = new HostColumnVector.StructType(false,
        new HostColumnVector.BasicType(false, DType.STRING),
        new HostColumnVector.BasicType(false, DType.INT32));
    ArrayList<HostColumnVector.StructData> data = new ArrayList<>();

    for (String tz : sortedTimezones) {
      Integer indexToTable = zoneIdToTable.get(tz);
      if (indexToTable != null) {
        data.add(new HostColumnVector.StructData(tz, indexToTable));
      } else {
        throw new IllegalStateException("Could not find timezone " + tz);
      }
    }
    return HostColumnVector.fromStructs(type, data);
  }

  /**
   * Return a struct column which contains timezone information
   * STRUCT<tz_name: string, index_to_tz_info_table: int>
   * The struct column is sorted by tz_name, it is used to query the index to the
   * timezone information table from timezone name.
   * The caller is responsible to close the returned column vector.
   */
  public static synchronized ColumnVector getTzNameToIndexMap() {
    verifyDatabaseCached();
    return tzNameToIndexMap.copyToDevice();
  }

  public static Integer getIndexToTransitionTable(String timezone) {
    verifyDatabaseCached();
    return zoneIdToTable.get(timezone);
  }

  /**
   * Convert the intermediate result of casting string to timestamp.
   * This is used for casting string with timezone to timestamp.
   *
   * @param invalidCv if the parsing from string to timestamp is valid
   * @param tsCv      long column with UTC microseconds parsed from string
   * @param tzIndexCv the index to the timezone transition table
   * @return timestamp column in microseconds
   */
  public static ColumnVector fromTimestampToUtcTimestampWithTzCv(
      ColumnView invalid,
      ColumnView input_seconds,
      ColumnView input_microseconds,
      ColumnView tzType,
      ColumnView tzOffset,
      ColumnView tzIndex) {
    try (Table timezoneInfo = getTimezoneInfo()) {
      return new ColumnVector(convertTimestampColumnToUTCWithTzCv(
          input_seconds.getNativeView(),
          input_microseconds.getNativeView(),
          invalid.getNativeView(),
          tzType.getNativeView(),
          tzOffset.getNativeView(),
          timezoneInfo.getNativeView(),
          tzIndex.getNativeView()));
    }
  }

  public static boolean isDST(String timezone) {
    ZoneId zoneId = ZoneId.of(timezone, ZoneId.SHORT_IDS);
    return !zoneId.getRules().getTransitionRules().isEmpty();
  }

  private static native long convertTimestampColumnToUTC(long input, long timezoneInfo, int tzIndex);

  private static native long convertUTCTimestampColumnToTimeZone(long input, long timezoneInfo, int tzIndex);

  private static native long convertTimestampColumnToUTCWithTzCv(
      long input_seconds, long input_microseconds, long invalid, long tzType,
      long tzOffset, long timezoneInfo, long tzIndex);

}
