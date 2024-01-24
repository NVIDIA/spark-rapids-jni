/*
* Copyright (c) 2023-2024-2024, NVIDIA CORPORATION.
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
import java.util.List;
import java.util.Map;
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
  private HostColumnVector fixedTransitions;

  // zone id to index in `fixedTransitions`
  private Map<String, Integer> zoneIdToTable;

  // Used to store Java ZoneId.SHORT_IDS Map, e.g.: For PST -> America/Los_Angeles
  // Save: PST -> index of America/Los_Angeles in transition table.
  private HostColumnVector shortIDs;

  // zone id list
  private HostColumnVector zoneIdVector;

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
    if (shortIDs != null) {
      shortIDs.close();
      shortIDs = null;
    }
    if (zoneIdVector != null) {
      zoneIdVector.close();
      zoneIdVector = null;
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

  /**
   * load ZoneId.SHORT_IDS and map to time zone index in transition table.
   * Note: ignored EST: -05:00; HST: -10:00; MST: -07:00
   */
  private void loadTimeZoneShortIDs(Map<String, Integer> zoneIdToTable) {
    HostColumnVector.DataType type = new HostColumnVector.StructType(false,
    new HostColumnVector.BasicType(false, DType.STRING),
    new HostColumnVector.BasicType(false, DType.INT32));
    ArrayList<HostColumnVector.StructData> data = new ArrayList<>();
    // copy short IDs
    List<String> idList = new ArrayList<>(ZoneId.SHORT_IDS.keySet());
    // sort short IDs
    Collections.sort(idList);
    for (String id : idList) {
      assert(id.length() == 3); // short ID lenght is always 3
      String mapTo = ZoneId.SHORT_IDS.get(id);
      if (mapTo.startsWith("+") || mapTo.startsWith("-")) {
        // skip: EST: -05:00; HST: -10:00; MST: -07:00
        // kernel will handle EST, HST, MST
        // ZoneId.SHORT_IDS is deprecated, so it will not probably change
      } else {
        Integer index = zoneIdToTable.get(mapTo);
        // some short IDs are DST, skip unsupported
        if (index != null) {
          data.add(new HostColumnVector.StructData(id, index));
        } else {
          // TODO: index should not be null after DST is supported.
        }
      }
    }
    shortIDs = HostColumnVector.fromStructs(type, data);
  }

  @SuppressWarnings("unchecked")
  private void loadData() {
    try {
      zoneIdToTable = new HashMap<>();
      List<List<HostColumnVector.StructData>> masterTransitions = new ArrayList<>();
      // Build a timezone ID index for the rendering of timezone IDs which may be included in datetime-like strings.
      // For instance: "2023-11-5T03:04:55.1 Asia/Shanghai" -> This index helps to find the
      // offset of "Asia/Shanghai" in timezoneDB.
      //
      // Currently, we do NOT support all timezone IDs. For unsupported time zones, like invalid ones,
      // we replace them with NULL value when ANSI mode is off when parsing string to timestamp.
      // This list only contains supported time zones.
      List<String> zondIdList = new ArrayList<>();

      // collect zone id and sort
      List<ZoneId> ids = new ArrayList<>();
      for (String tzId : TimeZone.getAvailableIDs()) {
        ZoneId zoneId;
        try {
          zoneId = ZoneId.of(tzId, ZoneId.SHORT_IDS).normalized(); // we use the normalized form to dedupe
          ids.add(zoneId);
        } catch (ZoneRulesException e) {
          // Sometimes the list of getAvailableIDs() is one of the 3-letter abbreviations, however,
          // this use is deprecated due to ambiguity reasons (same abbrevation can be used for
          // multiple time zones). These are not supported by ZoneId.of(...) directly here.
          continue;
        }
      }
      Collections.sort(ids, new Comparator<ZoneId>() {
        @Override
        public int compare(ZoneId o1, ZoneId o2) {
          // sort by `getId`
          return o1.getId().compareTo(o2.getId());
        }
      });

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

      for (ZoneId zoneId : ids) {
        ZoneRules zoneRules = zoneId.getRules();
        // Filter by non-repeating rules
        if (!zoneRules.isFixedOffset() && !zoneRules.getTransitionRules().isEmpty()) {
          continue;
        }
        if (!zoneIdToTable.containsKey(zoneId.getId())) {
          List<ZoneOffsetTransition> transitions = zoneRules.getTransitions();
          int idx = masterTransitions.size();
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
          masterTransitions.add(data);
          zoneIdToTable.put(zoneId.getId(), idx);
          // Collect the IDs of all supported timezones in the order of masterTransitions
          zondIdList.add(zoneId.getId());
        }
      }

      // load ZoneId.SHORT_IDS
      loadTimeZoneShortIDs(zoneIdToTable);

      HostColumnVector.DataType childType = new HostColumnVector.StructType(false,
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT32),
          new HostColumnVector.BasicType(false, DType.INT64));
      HostColumnVector.DataType resultType =
          new HostColumnVector.ListType(false, childType);

      fixedTransitions = HostColumnVector.fromLists(resultType, masterTransitions.toArray(new List[0]));
      zoneIdVector = HostColumnVector.fromStrings(zondIdList.toArray(new String[0]));
    } catch (Exception e) {
      throw new IllegalStateException("load time zone DB cache failed!", e);
    }

  }

  /**
   * get map from time zone to time zone index in transition table. 
   * @return map from time zone to time zone index in transition table. 
   */
  public Map<String, Integer> getZoneIDMap() {
    return zoneIdToTable;
  }

  /**
   * Get a map from short ID to time zone index in transitions for the short ID mapped time zone
   * @return
   */
  public ColumnVector getTimeZoneShortIDs() {
    return shortIDs.copyToDevice();
  }

  /**
   * Get a time zone list which is corresponding to the transitions
   * @return
   */
  public ColumnVector getZoneIDVector() {
    return zoneIdVector.copyToDevice();
  }

  /**
   * Transition table
   * @return
   */
  public Table getTransitions() {
    try (ColumnVector fixedTransitions = getFixedTransitions()) {
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
   * @param zoneId
   * @return list of fixed transitions
   */
  List getHostFixedTransitions(String zoneId) {
    zoneId = ZoneId.of(zoneId).normalized().toString(); // we use the normalized form to dedupe
    Integer idx = getZoneIDMap().get(zoneId);
    if (idx == null) {
      return null;
    }
    return fixedTransitions.getList(idx);
  }

  private static native long convertTimestampColumnToUTC(long input, long transitions, int tzIndex);

  private static native long convertUTCTimestampColumnToTimeZone(long input, long transitions, int tzIndex);
}
