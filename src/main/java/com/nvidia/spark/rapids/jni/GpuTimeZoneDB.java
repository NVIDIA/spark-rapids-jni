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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneOffsetTransitionRule;
import java.time.zone.ZoneRules;
import java.time.zone.ZoneRulesException;
import java.util.ArrayList;
import java.util.Comparator;
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
  private static HostColumnVector transitions;
  // initial year set to 1900 because some transition rules start early
  private static final int initialTransitionYear = 1900;
  private static long maxTimestamp;
  private static final ZoneId utcZoneId = ZoneId.of("UTC");

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
        maxTimestamp = LocalDateTime.of(maxYear+1, 1, 1, 0, 0, 0)
          .atZone(utcZoneId).toEpochSecond();
      } catch (Exception e) {
        log.error("cache time zone transitions cache failed", e);
      }
    };
    Thread thread = Executors.defaultThreadFactory().newThread(runnable);
    thread.setName("gpu-timezone-database-0");
    thread.setDaemon(true);
    thread.start();
  }

  public static synchronized void verifyDatabaseCached() {
    if (transitions == null) throw new IllegalStateException("Time Zone DB not loaded!");
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
  }

  private static long getScaleFactor(ColumnVector input){
    DType inputType = input.getType();
    if (inputType == DType.TIMESTAMP_SECONDS){
      return 1;
    } else if (inputType == DType.TIMESTAMP_MILLISECONDS){
      return 1000;
    } else if (inputType == DType.TIMESTAMP_MICROSECONDS){
      return 1000*1000;
    }
    throw new UnsupportedOperationException("Unsupported data type: " + inputType);
  }

  public static boolean isSupportedTimeZone(String zoneId) {
    try {
      getZoneId(zoneId); // check that zoneId is a valid zone
      return true;
    } catch (ZoneRulesException e) {
      return false;
    }
  }

  // enforce that all timestamps, regardless of timezone, be less than the desired date
  private static boolean shouldFallbackToCpu(ColumnVector input, ZoneId zoneId){
    if (zoneId.getRules().isFixedOffset()){
      return true;
    }
    boolean isValid = false;
    long scaleFactor = getScaleFactor(input);
    try (Scalar targetTimestamp = Scalar.timestampFromLong(input.getType(), maxTimestamp*scaleFactor);
         ColumnVector compareCv = input.binaryOp(BinaryOp.GREATER, targetTimestamp, DType.BOOL8);
         Scalar isGreater = compareCv.any() ) {
      if (!isGreater.isValid()) isValid = true;
      else isValid = !isGreater.getBoolean();
    } catch (Exception e) {
      log.error("Error validating input timestamps", e);
      // don't need to throw error, can try CPU processing
      return false;
    }
    return isValid;
  }

  private static ColumnVector cpuChangeTimestampTz(ColumnVector input, ZoneId currentTimeZone, ZoneId targetTimeZone) {
    ColumnVector resultCV = null;
    try (HostColumnVector hostCV = input.copyToHost()) {
      // assuming we don't have more than 2^31-1 rows
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
          /***
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
          ***/
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

  public static ColumnVector fromTimestampToUtcTimestamp(ColumnVector input, ZoneId currentTimeZone) {
    // there is technically a race condition on shutdown. Shutdown could be called after
    // the database is cached. This would result in a null pointer exception at some point
    // in the processing. This should be rare enough that it is not a big deal.
    if (!shouldFallbackToCpu(input, currentTimeZone)) {
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
    if (!shouldFallbackToCpu(input, desiredTimeZone)) {
      return cpuChangeTimestampTz(input, utcZoneId, desiredTimeZone);
    }
    Integer tzIndex = zoneIdToTable.get(desiredTimeZone.normalized().toString());
    try (Table transitions = getTransitions()) {
      return new ColumnVector(convertUTCTimestampColumnToTimeZone(input.getNativeView(),
          transitions.getNativeView(), tzIndex));
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
  private static synchronized void loadData(int finalTransitionYear) {
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
        if (!zoneIdToTable.containsKey(zoneId.getId())) {
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
          zoneIdToTable.put(zoneId.getId(), idx);
        }
      }
      HostColumnVector.DataType childType = new HostColumnVector.StructType(false,
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT64),
          new HostColumnVector.BasicType(false, DType.INT32));
      HostColumnVector.DataType resultType =
          new HostColumnVector.ListType(false, childType);
      transitions = HostColumnVector.fromLists(resultType,
          masterTransitions.toArray(new List[0]));
    } catch (Exception e) {
      throw new IllegalStateException("load time zone DB cache failed!", e);
    }
  }

  private static synchronized Table getTransitions() {
    verifyDatabaseCached();
    try (ColumnVector fixedTransitions = transitions.copyToDevice();) {
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

  private static native long convertTimestampColumnToUTC(long input, long transitions, int tzIndex);

  private static native long convertUTCTimestampColumnToTimeZone(long input, long transitions, int tzIndex);
}
