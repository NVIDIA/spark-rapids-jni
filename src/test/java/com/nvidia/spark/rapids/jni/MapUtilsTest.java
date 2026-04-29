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
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests for MapUtils.mapFromEntries — null struct entry and null key handling.
 *
 * Column schema: LIST(STRUCT(INT32 key, INT32 value))
 *
 * Fast-path contract: when every row would be returned unchanged, the function returns
 * {@code null} so the caller can reinterpret the input as the result (zero copies).
 */
public class MapUtilsTest {

  // Common schema: LIST(STRUCT<key:INT32, value:INT32>)
  private static final HostColumnVector.StructType STRUCT_TYPE =
      new HostColumnVector.StructType(true,
          Arrays.asList(
              new HostColumnVector.BasicType(true, DType.INT32),   // key (nullable)
              new HostColumnVector.BasicType(true, DType.INT32))); // value (nullable)
  private static final HostColumnVector.ListType LIST_TYPE =
      new HostColumnVector.ListType(true, STRUCT_TYPE);

  // Helpers
  private static HostColumnVector.StructData entry(Integer key, Integer value) {
    return new HostColumnVector.StructData(Arrays.asList(key, value));
  }

  // --------------------------------------------------------------------------
  // Fast-path tests — function returns null; input IS the output (caller incRefCounts).
  // --------------------------------------------------------------------------

  @Test
  void noNullsFastPath() {
    // [{1,10}, {2,20}], [{3,30}]  →  every row STATE_VALID  →  null returned
    List<HostColumnVector.StructData> row0 =
        Arrays.asList(entry(1, 10), entry(2, 20));
    List<HostColumnVector.StructData> row1 =
        Arrays.asList(entry(3, 30));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0, row1)) {
      ColumnVector result = MapUtils.mapFromEntries(input, true);
      assertNull(result, "all-valid input must take the fast path (null result)");
    }
  }

  @Test
  void nullKeyInValidStructThrows() {
    // [{null_key, 10}]  →  must throw because the struct is valid but key is null
    List<HostColumnVector.StructData> row0 = Arrays.asList(entry(null, 10));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0)) {
      assertThrows(CudfException.class, () -> MapUtils.mapFromEntries(input, true));
    }
  }

  @Test
  void nullKeyInValidStructFastPathWhenPolicyAllows() {
    // [{null_key, 10}] with throwOnNullKey=false  →  STATE_VALID  →  null returned
    List<HostColumnVector.StructData> row0 = Arrays.asList(entry(null, 10));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0)) {
      ColumnVector result = MapUtils.mapFromEntries(input, false);
      assertNull(result, "throw policy off: rows with null keys take the fast path");
    }
  }

  @Test
  void emptyListRowFastPath() {
    // Row 0: []          →  STATE_VALID (no entries → no null struct, no null key, size 0)
    // Row 1: [{1, 10}]   →  STATE_VALID
    List<HostColumnVector.StructData> row0 = Arrays.asList();
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(1, 10));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0, row1)) {
      ColumnVector result = MapUtils.mapFromEntries(input, true);
      assertNull(result, "empty list rows + valid rows ⇒ all STATE_VALID ⇒ fast path");
    }
  }

  @Test
  void zeroRowInputFastPath() {
    // Zero-row column: the function returns null so the caller reinterprets the empty input
    // as the empty result — no allocation, no kernel.
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE)) {
      ColumnVector result = MapUtils.mapFromEntries(input, true);
      assertNull(result, "zero-row input must take the fast path");
    }
  }

  // --------------------------------------------------------------------------
  // Slow-path tests (at least one row needs masking).
  // --------------------------------------------------------------------------

  @Test
  void nullStructEntryMasksRow() {
    // Row 0: [null_struct, {2,20}]  →  STATE_NULL  (CPU short-circuits on null struct)
    // Row 1: [{3,30}]               →  STATE_VALID
    List<HostColumnVector.StructData> row0 = Arrays.asList(null, entry(2, 20));
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(3, 30));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result = MapUtils.mapFromEntries(input, true)) {
      assertNotNull(result, "slow path expected: row 0 has a null struct entry");
      try (ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, row1)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void allNullStructEntriesAllRowsNull() {
    // Row 0: [null_struct]  →  STATE_NULL
    // Row 1: [null_struct, null_struct]  →  STATE_NULL
    List<HostColumnVector.StructData> row0 = Arrays.asList((HostColumnVector.StructData) null);
    List<HostColumnVector.StructData> row1 =
        Arrays.asList((HostColumnVector.StructData) null, (HostColumnVector.StructData) null);
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null)) {
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void mixedNullStructAndNullKeyReturnsNullNotThrow() {
    // Row 0: [null_struct, {null_key, 20}]
    // CPU: short-circuits on the null struct entry at index 0 → returns null without
    //      inspecting the null key at index 1.  Kernel matches: STATE_NULL wins.
    List<HostColumnVector.StructData> row0 = Arrays.asList(null, entry(null, 20));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0);
         ColumnVector result = MapUtils.mapFromEntries(input, true)) {
      assertNotNull(result);
      try (ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, (List<?>) null)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void nullStructRowDoesNotSuppressThrowInOtherRow() {
    // Row 0: [null_struct]      →  STATE_NULL (masked)
    // Row 1: [{null_key, 20}]   →  STATE_NULL_KEY  →  must throw
    List<HostColumnVector.StructData> row0 =
        Arrays.asList((HostColumnVector.StructData) null);
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(null, 20));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0, row1)) {
      assertThrows(CudfException.class, () -> MapUtils.mapFromEntries(input, true));
    }
  }

  @Test
  void outerNullRowPreservedAsNull() {
    // Row 0: null outer row  →  STATE_NULL
    // Row 1: [{1,10}]         →  STATE_VALID
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(1, 10));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, null, row1);
         ColumnVector result = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, row1)) {
      assertNotNull(result, "slow path expected: row 0 is outer-null");
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void allOuterNullRowsRemainNull() {
    // All rows are outer-null → STATE_NULL.  Slow path with total_entries=0.
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, null, null);
         ColumnVector result = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null)) {
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void allOuterNullRowsRemainNullNoThrowPolicy() {
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, null, null);
         ColumnVector result = MapUtils.mapFromEntries(input, false);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null)) {
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void singleOuterNullRowRemainNull() {
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, (List<?>) null);
         ColumnVector result = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, (List<?>) null)) {
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void outerNullRowAndNullStructEntryRowBothNull() {
    // Row 0: null outer row                →  STATE_NULL
    // Row 1: [null_struct, {2,20}]         →  STATE_NULL
    // Row 2: [{3,30}]                      →  STATE_VALID
    List<HostColumnVector.StructData> row1 = Arrays.asList(null, entry(2, 20));
    List<HostColumnVector.StructData> row2 = Arrays.asList(entry(3, 30));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, null, row1, row2);
         ColumnVector result = MapUtils.mapFromEntries(input, true)) {
      assertNotNull(result);
      try (ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null, row2)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  // --------------------------------------------------------------------------
  // Sliced input is rejected.
  // --------------------------------------------------------------------------

  @Test
  void slicedInputThrows() {
    // The function rejects sliced input outright — callers must materialize first.
    List<HostColumnVector.StructData> row0 = Arrays.asList(entry(1, 10));
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(2, 20));
    List<HostColumnVector.StructData> row2 = Arrays.asList(entry(3, 30));
    try (ColumnVector full = ColumnVector.fromLists(LIST_TYPE, row0, row1, row2)) {
      ColumnView[] views = full.splitAsViews(1);  // [0..1), [1..3) — second has offset != 0
      try {
        ColumnView sliced = views[1];
        assertThrows(CudfException.class, () -> MapUtils.mapFromEntries(sliced, true));
      } finally {
        for (ColumnView v : views) {
          v.close();
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // Slow path opt-out coverage and bitmask boundary stress.
  // --------------------------------------------------------------------------

  @Test
  void slowPathNullKeyNoThrowWhenPolicyAllows() {
    // Slow path (row 0 has a null struct entry) plus a null-key entry in row 1.
    // With throwOnNullKey=false, row 1 takes STATE_VALID (NULL_KEY only triggers when the
    // throw policy is on), so the slow path returns row 1 unchanged.  Row 0 is masked to
    // null by the null-struct-entry rule.  Inverting the throw_on_null_key guard would
    // silently flip this to a throw — this case pins the false branch.
    List<HostColumnVector.StructData> row0 = Arrays.asList(null, entry(1, 10));
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(null, 20));
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result   = MapUtils.mapFromEntries(input, false);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, row1)) {
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  @SuppressWarnings("unchecked")
  void slowPathAcrossMultipleBitmaskWords() {
    // 70 rows: rows 0, 33, 65 contain a null struct entry — each must become null in the
    // output.  Crosses the 32-row warp boundary and the 64-row bitmask-word boundary to
    // guard against bit-alignment regressions in bools_to_mask + the gather-map kernel.
    final int numRows = 70;
    List<HostColumnVector.StructData>[] rows         = new List[numRows];
    List<HostColumnVector.StructData>[] expectedRows = new List[numRows];
    for (int i = 0; i < numRows; i++) {
      if (i == 0 || i == 33 || i == 65) {
        rows[i]         = Arrays.asList(null, entry(i, i * 10));
        expectedRows[i] = null;
      } else {
        rows[i]         = Arrays.asList(entry(i, i * 10));
        expectedRows[i] = rows[i];
      }
    }
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, rows);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, expectedRows)) {
      assertNotNull(result);
      assertColumnsAreEqual(expected, result);
    }
  }
}
