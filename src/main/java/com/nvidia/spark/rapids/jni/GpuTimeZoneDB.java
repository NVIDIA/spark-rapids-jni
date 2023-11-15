/*
* Copyright (c) 2023, NVIDIA CORPORATION.
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
import java.util.concurrent.*;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;

public class GpuTimeZoneDB {
  
  private CompletableFuture<Map<String, Integer>> zoneIdToTableFuture;
  private CompletableFuture<HostColumnVector> fixedTransitionsFuture;

  GpuTimeZoneDB() {
    zoneIdToTableFuture = new CompletableFuture<>();
    fixedTransitionsFuture = new CompletableFuture<>();
  }
  
  private static GpuTimeZoneDB instance;
  static GpuTimeZoneDB getInstance() {
    return instance;
  }
  
  /**
   * Start to cache the database. This should be called on startup of an executor. It should start
   * to cache the data on the CPU in a background thread. It should return immediately and allow the
   * other APIs to be called. Depending on what we want to * do we can have the other APIs block
   * until this is done caching, or we can have private APIs that would let us load and use specific
   * parts of the database. I prefer the former solution at least until we see a performance hit
   * where we are waiting on the database to finish loading.
   */
  public static void cacheDatabase() {
    if (instance == null) {
      instance = new GpuTimeZoneDB();
    }
    Executor executor = Executors.newSingleThreadExecutor(
      new ThreadFactory() {
        private ThreadFactory defaultFactory = Executors.defaultThreadFactory();

        @Override
        public Thread newThread(Runnable r) {
          Thread thread = defaultFactory.newThread(r);
          thread.setName("gpu-timezone-database-0");
          thread.setDaemon(true);
          return thread;
        }
      });
    instance.loadData(executor);
  }

  public static void shutdown() {
    instance.getHostFixedTransitions().close();
  }

  public static ColumnVector fromTimestampToUtcTimestamp(ColumnVector input, ZoneId currentTimeZone) {
    // TODO: Remove this check when all timezones are supported
    // (See https://github.com/NVIDIA/spark-rapids/issues/6840)
    if (!isSupportedTimeZone(currentTimeZone)) {
      throw new IllegalArgumentException(String.format("Unsupported timezone: %s",
          currentTimeZone.toString()));
    }
    Integer tzIndex = instance.getZoneIDMap().get(currentTimeZone.normalized().toString());
    Table transitions = instance.getTransitions();
    ColumnVector result = new ColumnVector(convertTimestampColumnToUTC(input.getNativeView(),
        transitions.getNativeView(), tzIndex));
    transitions.close();
    return result;
  }
  
  public static ColumnVector fromUtcTimestampToTimestamp(ColumnVector input, ZoneId desiredTimeZone) {
    // TODO: Remove this check when all timezones are supported
    // (See https://github.com/NVIDIA/spark-rapids/issues/6840)
    if (!isSupportedTimeZone(desiredTimeZone)) {
      throw new IllegalArgumentException(String.format("Unsupported timezone: %s",
          desiredTimeZone.toString()));
    }
    Integer tzIndex = instance.getZoneIDMap().get(desiredTimeZone.normalized().toString());
    Table transitions = instance.getTransitions();
    ColumnVector result = new ColumnVector(convertUTCTimestampColumnToTimeZone(input.getNativeView(),
        transitions.getNativeView(), tzIndex));
    transitions.close();
    return result;
  }
  
  // TODO: Deprecate this API when we support all timezones 
  // (See https://github.com/NVIDIA/spark-rapids/issues/6840)
  public static boolean isSupportedTimeZone(ZoneId desiredTimeZone) {
    String id = desiredTimeZone.normalized().getId();
    return instance.getZoneIDMap().containsKey(id);
  }
  
  private void loadData(Executor executor) throws IllegalStateException {
    // Start loading the data in separate thread and return
    try {
      executor.execute(this::doLoadData);
    } catch (RejectedExecutionException e) {
      throw new IllegalStateException(e);
    }
  }

  @SuppressWarnings("unchecked")
  private void doLoadData() {
    synchronized (this) {
      try {
        Map<String, Integer> zoneIdToTable = new HashMap<>();
        List<List<HostColumnVector.StructData>> masterTransitions = new ArrayList<>();
        for (String tzId : TimeZone.getAvailableIDs()) {
          ZoneId zoneId;
          try {
            zoneId = ZoneId.of(tzId).normalized(); // we use the normalized form to dedupe
          } catch (ZoneRulesException e) {
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
                          t.getInstant().getEpochSecond() + t.getOffsetBefore().getTotalSeconds() - 1,
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
        HostColumnVector fixedTransitions = HostColumnVector.fromLists(resultType,
            masterTransitions.toArray(new List[0]));
        zoneIdToTableFuture.complete(zoneIdToTable);
        fixedTransitionsFuture.complete(fixedTransitions);
      } catch (Exception e) {
        zoneIdToTableFuture.completeExceptionally(e);
        fixedTransitionsFuture.completeExceptionally(e);
        throw e;
      }
    }
  }

  HostColumnVector getHostFixedTransitions() {
    try {
      return fixedTransitionsFuture.get();
    } catch (InterruptedException | ExecutionException e) {
      throw new RuntimeException(e);
    }
  }

  Map<String, Integer> getZoneIDMap() {
    try {
      return zoneIdToTableFuture.get();
    } catch (InterruptedException | ExecutionException e) {
      throw new RuntimeException(e);
    }
  }

  Table getTransitions() {
    ColumnVector fixedTransitions = getFixedTransitions();
    Table transitions = new Table(fixedTransitions);
    fixedTransitions.close();
    return transitions;
  }

  ColumnVector getFixedTransitions() {
    HostColumnVector hostTransitions = getHostFixedTransitions();
    return hostTransitions.copyToDevice();
  }

  List getHostFixedTransitions(String zoneId) {
    zoneId = ZoneId.of(zoneId).normalized().toString(); // we use the normalized form to dedupe
    Integer idx = getZoneIDMap().get(zoneId);
    if (idx == null) {
      return null;
    }
    HostColumnVector transitions = getHostFixedTransitions();
    return transitions.getList(idx);
  }


  private static native long convertTimestampColumnToUTC(long input, long transitions, int tzIndex);

  private static native long convertUTCTimestampColumnToTimeZone(long input, long transitions, int tzIndex);
}
