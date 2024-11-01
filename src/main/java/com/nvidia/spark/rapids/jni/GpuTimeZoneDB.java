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
import java.time.ZoneId;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneRules;
import java.time.zone.ZoneRulesException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TimeZone;
import java.util.concurrent.Executors;

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
  //   LIST<STRUCT<utcInstant: int64, localInstant: int64, offset: int32>>
  private static Map<String, Integer> zoneIdToTable;

  // use this reference to indicate if time zone cache is initialized.
  private static HostColumnVector fixedTransitions;

  private static boolean isShutdownCalledEver = false;

  /**
   * This should be called on startup of an executor.
   * Runs in a thread asynchronously.
   * If `shutdown` was called ever, then will not load the cache
   */
  public static void cacheDatabaseAsync() {
    // This has a race in that we could still launch a thread after
    // shutting down. This is just to prevent the thread from launching
    // in some cases.
    synchronized (GpuTimeZoneDB.class) {
      if (isShutdownCalledEver) {
        log.error("cache async called after DB already loaded");
        return;
      }
    }
    // start a new thread to load
    Runnable runnable = () -> {
      try {
        cacheDatabaseImpl();
      } catch (Exception e) {
        log.error("cache time zone transitions cache failed", e);
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
    cacheDatabaseImpl();
  }

  /**
   * close the cache, used when Plugin is closing
   */
  public static synchronized void shutdown() {
    isShutdownCalledEver = true;
    closeResources();
  }

  private static synchronized void cacheDatabaseImpl() {
    if (isShutdownCalledEver) {
      throw new IllegalStateException("GpuTimeZoneDB has already been shut down");
    }
    if (fixedTransitions == null) {
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
    if (fixedTransitions != null) {
      fixedTransitions.close();
      fixedTransitions = null;
    }
  }

  public static ColumnVector fromTimestampToUtcTimestamp(ColumnVector input, ZoneId currentTimeZone) {
    // TODO: Remove this check when all timezones are supported
    // (See https://github.com/NVIDIA/spark-rapids/issues/6840)
    if (!isSupportedTimeZone(currentTimeZone)) {
      throw new IllegalArgumentException(String.format("Unsupported timezone: %s",
          currentTimeZone.toString()));
    }
    // there is technically a race condition on shutdown. Shutdown could be called after
    // the database is cached. This would result in a null pointer exception at some point
    // in the processing. This should be rare enough that it is not a big deal.
    cacheDatabase();
    Integer tzIndex = zoneIdToTable.get(currentTimeZone.normalized().toString());
    try (Table transitions = getTransitions()) {
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
    // there is technically a race condition on shutdown. Shutdown could be called after
    // the database is cached. This would result in a null pointer exception at some point
    // in the processing. This should be rare enough that it is not a big deal.
    cacheDatabase();
    Integer tzIndex = zoneIdToTable.get(desiredTimeZone.normalized().toString());
    try (Table transitions = getTransitions()) {
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
  private static synchronized void loadData() {
    try {
      List<List<HostColumnVector.StructData>> masterTransitions = new ArrayList<>();
      zoneIdToTable = new HashMap<>();
      for (String tzId : TimeZone.getAvailableIDs()) {
        ZoneId zoneId;
        try {
          zoneId = ZoneId.of(tzId).normalized(); // we use the normalized form to dedupe
        } catch (ZoneRulesException e) {
          // Sometimes the list of getAvailableIDs() is one of the 3-letter abbreviations, however,
          // this use is deprecated due to ambiguity reasons (same abbrevation can be used for
          // multiple time zones). These are not supported by ZoneId.of(...) directly here.
          continue;
        }
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
                    zoneRules.getOffset(Instant.now()).getTotalSeconds())
            );
          } else {
            // Capture the first official offset (before any transition) using Long min
            ZoneOffsetTransition first = transitions.get(0);
            data.add(
                new HostColumnVector.StructData(Long.MIN_VALUE, Long.MIN_VALUE,
                    first.getOffsetBefore().getTotalSeconds())
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
          zoneIdToTable.put(zoneId.getId(), idx);
        }
      }
      HostColumnVector.DataType childType = new HostColumnVector.StructType(false,
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT32));
      HostColumnVector.DataType resultType =
          new HostColumnVector.ListType(false, childType);
      fixedTransitions = HostColumnVector.fromLists(resultType,
          masterTransitions.toArray(new List[0]));
    } catch (Exception e) {
      throw new IllegalStateException("load time zone DB cache failed!", e);
    }
  }

  private static synchronized Table getTransitions() {
    try (ColumnVector fixedTransitions = getFixedTransitions()) {
      return new Table(fixedTransitions);
    }
  }

  private static synchronized ColumnVector getFixedTransitions() {
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
  static synchronized List getHostFixedTransitions(String zoneId) {
    zoneId = ZoneId.of(zoneId).normalized().toString(); // we use the normalized form to dedupe
    Integer idx = zoneIdToTable.get(zoneId);
    if (idx == null) {
      return null;
    }
    return fixedTransitions.getList(idx);
  }

  private static native long convertTimestampColumnToUTC(long input, long transitions, int tzIndex);

  private static native long convertUTCTimestampColumnToTimeZone(long input, long transitions, int tzIndex);
}
