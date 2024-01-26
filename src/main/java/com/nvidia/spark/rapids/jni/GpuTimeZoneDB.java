/*
* Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneRules;
import java.time.zone.ZoneRulesException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TimeZone;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Function;

/**
 * Gpu time zone utility.
 * Provides two kinds of APIs
 *  - Time zone transitions cache APIs
 *      `cacheDatabaseAsync`, `cacheDatabase` and `shutdown` are synchronized.
 *      When cacheDatabaseAsync is running, the `shutdown` and `cacheDatabase` will wait;
 *      These APIs guarantee only one thread is loading transitions cache,
 *      And guarantee loading cache only occurs one time.
 *  - Rebase time zone APIs
 *    fromTimestampToUtcTimestamp, fromUtcTimestampToTimestamp ...
 */
public class GpuTimeZoneDB {
  private static final Logger log = LoggerFactory.getLogger(GpuTimeZoneDB.class);

  // For the timezone database, we store the transitions in a ColumnVector that is a list of
  // structs. The type of this column vector is:
  //   LIST<STRUCT<utcInstant: int64, localInstant: int64, offset: int32, looseInstant: int64>>
  // use this reference to indicate if time zone cache is initialized.
  // `fixedTransitions` saves transitions for deduplicated time zones, diferent time zones
  // may map to one normalized time zone.
  private HostColumnVector fixedTransitions;

  // time zone to index in `fixedTransitions`
  // The key of `zoneIdToTable` is the time zone names before dedup.
  private Map<String, Integer> zoneIdToTable;

  // host column vector<String, Integer> for `zoneIdToTable`, sorted by time zone strings
  private HostColumnVector zoneIdToTableVec;

  // Guarantee singleton instance
  private GpuTimeZoneDB() {
  }

  // singleton instance
  private static final GpuTimeZoneDB instance = new GpuTimeZoneDB();

  // This method is default visibility for testing purposes only.
  // The instance will be never be exposed publicly for this class.
  static GpuTimeZoneDB getInstance() {
    return instance;
  }

  static class LoadingLock {
    Boolean isLoading = false;

    // record whether a shutdown is called ever.
    // if `isCloseCalledEver` is true, then the following loading should be skipped.
    Boolean isShutdownCalledEver = false;
  }

  private static final LoadingLock lock = new LoadingLock();

  /**
   * This should be called on startup of an executor.
   * Runs in a thread asynchronously.
   * If `shutdown` was called ever, then will not load the cache
   */
  public static void cacheDatabaseAsync() {
    synchronized (lock) {
      if (lock.isShutdownCalledEver) {
        // shutdown was called ever, will never load cache again.
        return;
      }

      if (lock.isLoading) {
        // another thread is loading(), return
        return;
      } else {
        lock.isLoading = true;
      }
    }

    // start a new thread to load
    Runnable runnable = () -> {
      try {
        instance.cacheDatabaseImpl();
      } catch (Exception e) {
        log.error("cache time zone transitions cache failed", e);
      } finally {
        synchronized (lock) {
          // now loading is done
          lock.isLoading = false;
          // `cacheDatabase` and `shutdown` may wait loading is done.
          lock.notify();
        }
      }
    };
    Thread thread = Executors.defaultThreadFactory().newThread(runnable);
    thread.setName("gpu-timezone-database-0");
    thread.setDaemon(true);
    thread.start();
  }

  /**
   * Cache the database. This will take some time like several seconds.
   * If one `cacheDatabase` is running, other `cacheDatabase` will wait until caching is done.
   * If cache is exits, do not load cache again.
   */
  public static void cacheDatabase() {
    synchronized (lock) {
      if (lock.isLoading) {
        // another thread is loading(), wait loading is done
        while (lock.isLoading) {
          try {
            lock.wait();
          } catch (InterruptedException e) {
            throw new IllegalStateException("cache time zone transitions cache failed", e);
          }
        }
        return;
      } else {
        lock.isLoading = true;
      }
    }

    try {
      instance.cacheDatabaseImpl();
    } finally {
      // loading is done.
      synchronized (lock) {
        lock.isLoading = false;
        // `cacheDatabase` and/or `shutdown` may wait loading is done.
        lock.notify();
      }
    }
  }

  /**
   * close the cache, used when Plugin is closing
   */
  public static void shutdown() {
    synchronized (lock) {
      lock.isShutdownCalledEver = true;
      while (lock.isLoading) {
        // wait until loading is done
        try {
          lock.wait();
        } catch (InterruptedException e) {
          throw new IllegalStateException("shutdown time zone transitions cache failed", e);
        }
      }
      instance.shutdownImpl();
      // `cacheDatabase` and/or `shutdown` may wait loading is done.
      lock.notify();
    }
  }

  private void cacheDatabaseImpl() {
    if (fixedTransitions == null) {
      try {
        loadData();
      } catch (Exception e) {
        closeResources();
        throw e;
      }
    }
  }

  private void shutdownImpl() {
    closeResources();
  }

  private void closeResources() {
    if (zoneIdToTable != null) {
      zoneIdToTable.clear();
      zoneIdToTable = null;
    }
    if (fixedTransitions != null) {
      fixedTransitions.close();
      fixedTransitions = null;
    }
    if (zoneIdToTableVec != null) {
      zoneIdToTableVec.close();
      zoneIdToTableVec = null;
    }
  }

  public static ColumnVector fromTimestampToUtcTimestamp(ColumnVector input, ZoneId currentTimeZone) {
    // TODO: Remove this check when all timezones are supported
    // (See https://github.com/NVIDIA/spark-rapids/issues/6840)
    if (!isSupportedTimeZone(currentTimeZone)) {
      throw new IllegalArgumentException(String.format("Unsupported timezone: %s",
          currentTimeZone.toString()));
    }
    cacheDatabase();
    Integer tzIndex = instance.getZoneIDMap().get(currentTimeZone.normalized().toString());
    try (Table transitions = instance.getTransitions()) {
      return new ColumnVector(convertTimestampColumnToUTC(input.getNativeView(),
          transitions.getNativeView(), tzIndex));
    }
  }
  
  public static ColumnVector fromUtcTimestampToTimestamp(ColumnVector input, ZoneId desiredTimeZone) {
    // TODO: Remove this check when all timezones are supported
    // (See https://github.com/NVIDIA/spark-rapids/issues/6840)
    if (!isSupportedTimeZone(desiredTimeZone)) {
      throw new IllegalArgumentException(String.format("Unsupported timezone: %s",
          desiredTimeZone.toString()));
    }
    cacheDatabase();
    Integer tzIndex = instance.getZoneIDMap().get(desiredTimeZone.normalized().toString());
    try (Table transitions = instance.getTransitions()) {
      return new ColumnVector(convertUTCTimestampColumnToTimeZone(input.getNativeView(),
          transitions.getNativeView(), tzIndex));
    }
  }
  
  // TODO: Deprecate this API when we support all timezones 
  // (See https://github.com/NVIDIA/spark-rapids/issues/6840)
  public static boolean isSupportedTimeZone(ZoneId desiredTimeZone) {
    return desiredTimeZone != null &&
      (desiredTimeZone.getRules().isFixedOffset() ||
      desiredTimeZone.getRules().getTransitionRules().isEmpty());
  }

  public static boolean isSupportedTimeZone(String zoneId) {
    try {
      return isSupportedTimeZone(getZoneId(zoneId));
    } catch (ZoneRulesException e) {
      return false;
    }
  }

  // Ported from Spark. Used to format time zone ID string with (+|-)h:mm and (+|-)hh:m
  public static ZoneId getZoneId(String timeZoneId) {
    String formattedZoneId = timeZoneId
      // To support the (+|-)h:mm format because it was supported before Spark 3.0.
      .replaceFirst("(\\+|\\-)(\\d):", "$10$2:")
      // To support the (+|-)hh:m format because it was supported before Spark 3.0.
      .replaceFirst("(\\+|\\-)(\\d\\d):(\\d)$", "$1$2:0$3");
    return ZoneId.of(formattedZoneId, ZoneId.SHORT_IDS);
  }

  @SuppressWarnings("unchecked")
  private void loadData() {
    try {
      // Note: ZoneId.normalized will transform fixed offset time zone to standard fixed offset
      // e.g.: ZoneId.of("Etc/GMT").normalized.getId = Z; ZoneId.of("Etc/GMT+0").normalized.getId = Z
      // Both Etc/GMT and Etc/GMT+0 have normalized Z.
      // We use the normalized form to dedupe,
      // but should record map from TimeZone.getAvailableIDs() Set to normalized Set.
      // `fixedTransitions` saves transitions for normalized time zones.
      // Spark uses time zones from TimeZone.getAvailableIDs()
      // So we have a Map<String, Int> from TimeZone.getAvailableIDs() to index of `fixedTransitions`.

      // get and sort time zones
      String[] timeZones = TimeZone.getAvailableIDs();
      List<String> sortedTimeZones = new ArrayList<>(Arrays.asList(timeZones));
      // Note: Z is a special normalized time zone from UTC: ZoneId.of("UTC").normalized = Z
      // TimeZone.getAvailableIDs does not contains Z and ZoneId.SHORT_IDS also does not contain Z
      // Should add Z to `zoneIdToTable`
      sortedTimeZones.add("Z");
      Collections.sort(sortedTimeZones);

      // Note: Spark uses ZoneId.SHORT_IDS
      // `TimeZone.getAvailableIDs` contains all keys in `ZoneId.SHORT_IDS`
      // So do not need extra work for ZoneId.SHORT_IDS, here just check this assumption
      for (String tz : ZoneId.SHORT_IDS.keySet()) {
        if (!sortedTimeZones.contains(tz)) {
          throw new IllegalStateException(
              String.format("Can not find short Id %s in time zones %s", tz, sortedTimeZones));
        }
      }

      // A simple approach to transform LocalDateTime to a value which is proportional to
      // the exact EpochSecond. After caching these values along with EpochSeconds, we
      // can easily search out which time zone transition rule we should apply according
      // to LocalDateTime structs. The searching procedure is same as the binary search with
      // exact EpochSeconds(convert_timestamp_tz_functor), except using "loose instant"
      // as search index instead of exact EpochSeconds.
      Function<LocalDateTime, Long> localToLooseEpochSecond = lt ->
              86400L * (lt.getYear() * 400L + (lt.getMonthValue() - 1) * 31L +
                      lt.getDayOfMonth() - 1) +
                      3600L * lt.getHour() + 60L * lt.getMinute() + lt.getSecond();

      List<List<HostColumnVector.StructData>> masterTransitions = new ArrayList<>();

      // map: normalizedTimeZone -> index in fixedTransitions
      Map<String, Integer> mapForNormalizedTimeZone = new HashMap<>();
      // go though all time zones and save by normalized time zone
      List<String> sortedSupportedTimeZones = new ArrayList<>();
      for (String timeZone : sortedTimeZones) {
        ZoneId normalizedZoneId = ZoneId.of(timeZone, ZoneId.SHORT_IDS).normalized();
        String normalizedTimeZone = normalizedZoneId.getId();
        ZoneRules zoneRules = normalizedZoneId.getRules();
        // Filter by non-repeating rules
        if (!zoneRules.isFixedOffset() && !zoneRules.getTransitionRules().isEmpty()) {
          continue;
        }
        sortedSupportedTimeZones.add(timeZone);
        if (!mapForNormalizedTimeZone.containsKey(normalizedTimeZone)) { // dedup
          List<HostColumnVector.StructData> data = getTransitionData(localToLooseEpochSecond, zoneRules);
          // add transition data for time zone
          int idx = masterTransitions.size();
          mapForNormalizedTimeZone.put(normalizedTimeZone, idx);
          masterTransitions.add(data);
        }
      }

      HostColumnVector.DataType childType = new HostColumnVector.StructType(false,
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT32),
          new HostColumnVector.BasicType(false, DType.INT64));
      HostColumnVector.DataType resultType =
          new HostColumnVector.ListType(false, childType);

      // generate all transitions for all time zones
      fixedTransitions = HostColumnVector.fromLists(resultType, masterTransitions.toArray(new List[0]));

      // generate `zoneIdToTable`, key should be time zone not normalized time zone
      zoneIdToTable = new HashMap<>();
      for (String timeZone : sortedSupportedTimeZones) {
        // map from time zone to normalized
        String normalized = ZoneId.of(timeZone, ZoneId.SHORT_IDS).normalized().getId();
        Integer index = mapForNormalizedTimeZone.get(normalized);
        if (index != null) {
          zoneIdToTable.put(timeZone, index);
        } else {
          throw new IllegalStateException("Could not find index for normalized time zone " + normalized);
        }
      }
      // generate host vector
      zoneIdToTableVec = generateZoneIdToTableVec(sortedSupportedTimeZones, zoneIdToTable);
    } catch (IllegalStateException e) {
      throw e;
    } catch (Exception e) {
      throw new IllegalStateException("load time zone DB cache failed!", e);
    }
  }

  // generate transition data for a time zone
  private List<HostColumnVector.StructData> getTransitionData(Function<LocalDateTime, Long> localToLooseEpochSecond,
      ZoneRules zoneRules) {
    List<ZoneOffsetTransition> transitions = zoneRules.getTransitions();
    List<HostColumnVector.StructData> data = new ArrayList<>();
    if (zoneRules.isFixedOffset()) {
      data.add(
          new HostColumnVector.StructData(Long.MIN_VALUE, Long.MIN_VALUE,
              zoneRules.getOffset(Instant.now()).getTotalSeconds(), Long.MIN_VALUE)
      );
    } else {
      // Capture the first official offset (before any transition) using Long min
      ZoneOffsetTransition first = transitions.get(0);
      data.add(
          new HostColumnVector.StructData(Long.MIN_VALUE, Long.MIN_VALUE,
              first.getOffsetBefore().getTotalSeconds(), Long.MIN_VALUE)
      );
      transitions.forEach(t -> {
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
                  t.getOffsetAfter().getTotalSeconds(),
                  localToLooseEpochSecond.apply(t.getDateTimeAfter()) // this column is for rebase local date time
              )
          );
        } else {
          data.add(
              new HostColumnVector.StructData(
                  t.getInstant().getEpochSecond(),
                  t.getInstant().getEpochSecond() + t.getOffsetBefore().getTotalSeconds(),
                  t.getOffsetAfter().getTotalSeconds(),
                  localToLooseEpochSecond.apply(t.getDateTimeBefore()) // this column is for rebase local date time
              )
          );
        }
      });
    }
    return data;
  }

  /**
   * Generate map from time zone to index in transition table.
   * regular time zone map to normalized time zone, then get from 
   * @param sortedSupportedTimeZones is sorted and supported time zones
   * @param zoneIdToTableMap is a map from non-normalized time zone to index in transition table
   */
  private static HostColumnVector generateZoneIdToTableVec(List<String> sortedSupportedTimeZones, Map<String, Integer> zoneIdToTableMap) {
    HostColumnVector.DataType type = new HostColumnVector.StructType(false,
    new HostColumnVector.BasicType(false, DType.STRING),
    new HostColumnVector.BasicType(false, DType.INT32));
    ArrayList<HostColumnVector.StructData> data = new ArrayList<>();

    for (String timeZone : sortedSupportedTimeZones) {
      Integer mapTo = zoneIdToTableMap.get(timeZone);
      if (mapTo != null) {
        data.add(new HostColumnVector.StructData(timeZone, mapTo));
      } else {
        throw new IllegalStateException("Could not find index for time zone " + timeZone);
      }
    }
    return HostColumnVector.fromStructs(type, data);
  }

  /**
   * get map from time zone to time zone index in transition table. 
   * @return map from time zone to time zone index in transition table. 
   */
  public static Map<String, Integer> getZoneIDMap() {
    cacheDatabase();
    return instance.zoneIdToTable;
  }

  /**
   * Get vector from time zone to index in transition table
   * @return
   */
  public static ColumnVector getZoneIDVector() {
    cacheDatabase();
    return instance.zoneIdToTableVec.copyToDevice();
  }

  /**
   * Transition table
   * @return
   */
  public static Table getTransitions() {
    cacheDatabase();
    try (ColumnVector fixedTransitions = instance.getFixedTransitions()) {
      return new Table(fixedTransitions);
    }
  }

  private ColumnVector getFixedTransitions() {
    return fixedTransitions.copyToDevice();
  }

  /**
   * FOR TESTING PURPOSES ONLY, DO NOT USE IN PRODUCTION
   *
   * This method retrieves the raw list of struct data that forms the list of 
   * fixed transitions for a particular zoneId. 
   *
   * It has default visibility so the test can access it.
   * @param zoneId the time zones from TimeZone.getAvailableIDs without `ZoneId.normalized`
   * @return list of fixed transitions
   */
  List getHostFixedTransitions(String zoneId) {
    Integer idx = getZoneIDMap().get(zoneId);
    if (idx == null) {
      return null;
    }
    return fixedTransitions.getList(idx);
  }

  private static native long convertTimestampColumnToUTC(long input, long transitions, int tzIndex);

  private static native long convertUTCTimestampColumnToTimeZone(long input, long transitions, int tzIndex);
}
