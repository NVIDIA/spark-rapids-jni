/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
import ai.rapids.cudf.NativeDepsLoader;

/**
 * Utility APIs for map column operations that require Spark-specific semantics
 * not available in the standard cuDF Java bindings.
 */
public class MapUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Converts a LIST(STRUCT(KEY, VALUE)) column to a map column following Spark semantics.
   *
   * <p>Spark semantics for {@code map_from_entries}:
   * <ul>
   *   <li>If a row's array contains any null struct entry (the whole struct is null), the output
   *       row is null — even if another entry in that same row has a null key inside a valid
   *       struct.</li>
   *   <li>If a row's array contains no null struct entry but a valid struct's key is null,
   *       behavior depends on {@code throwOnNullKey}:
   *       <ul>
   *         <li>{@code true}  — throws a {@link RuntimeException}.</li>
   *         <li>{@code false} — returns the row as-is (caller handles dedup policy).</li>
   *       </ul>
   *   </li>
   * </ul>
   *
   * <p><b>Fast-path contract:</b> returns {@code null} when every row would be returned
   * unchanged.  A Spark {@code MAP<K,V>} and a cuDF {@code LIST<STRUCT<K,V>>} share the same
   * physical layout, so the caller should treat {@code null} as a signal to reinterpret the
   * input as the result (typically via {@code input.incRefCount()}) — no copy, no kernel past
   * the state-collection phase.  An empty input also returns {@code null}.
   *
   * <p><b>Sliced input is not supported.</b>  Materialize the slice (e.g. via {@code copyToHost
   * / fromColumnViews}) before calling.
   *
   * <p>Duplicate-key deduplication is intentionally left to the caller so that the EXCEPTION
   * and LAST_WIN policies can be applied after this function returns.
   *
   * @param input          Input LIST(STRUCT(KEY, VALUE)) column.  Must not be sliced.
   * @param throwOnNullKey When {@code true}, throw if any valid-struct entry has a null key.
   * @return A new column with rows containing null struct entries replaced with null outer
   *         rows; or {@code null} when no row needs masking (fast path).
   * @throws RuntimeException if {@code throwOnNullKey} is true and any row (with no null struct
   *         entry) contains a null key inside a valid struct.
   * @throws RuntimeException if {@code input} or its struct child is sliced.
   */
  public static ColumnVector mapFromEntries(ColumnView input, boolean throwOnNullKey) {
    long handle = mapFromEntries(input.getNativeView(), throwOnNullKey);
    return handle == 0L ? null : new ColumnVector(handle);
  }

  private static native long mapFromEntries(long inputHandle, boolean throwOnNullKey);
}
