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
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Tests for MapUtils.mapFromEntries — null struct entry and null key handling.
 *
 * Column schema: LIST(STRUCT(INT32 key, INT32 value))
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
  // Fast-path tests (no null struct entries)
  // --------------------------------------------------------------------------

  @Test
  void noNullsReturnedUnchanged() {
    // [{1,10}, {2,20}], [{3,30}]  →  same rows, no masking
    List<HostColumnVector.StructData> row0 =
        Arrays.asList(entry(1, 10), entry(2, 20));
    List<HostColumnVector.StructData> row1 =
        Arrays.asList(entry(3, 30));
    try (ColumnVector input  = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, row0, row1)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void nullKeyInValidStructThrows() {
    // [{null_key, 10}]  →  must throw because the struct is valid but key is null
    List<HostColumnVector.StructData> row0 = Arrays.asList(entry(null, 10));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0)) {
      assertThrows(CudfException.class, () -> MapUtils.mapFromEntries(input, true).close());
    }
  }

  @Test
  void nullKeyInValidStructNoThrowWhenPolicyAllows() {
    // [{null_key, 10}] with throwOnNullKey=false  →  row returned as-is
    List<HostColumnVector.StructData> row0 = Arrays.asList(entry(null, 10));
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, row0);
         ColumnVector result   = MapUtils.mapFromEntries(input, false);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, row0)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  // --------------------------------------------------------------------------
  // Slow-path tests (at least one row has a null struct entry)
  // --------------------------------------------------------------------------

  @Test
  void nullStructEntryMasksRow() {
    // Row 0: [null_struct, {2,20}]  →  output row 0 = null (CPU short-circuits on null struct)
    // Row 1: [{3,30}]               →  output row 1 unchanged
    List<HostColumnVector.StructData> row0 = Arrays.asList(null, entry(2, 20));
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(3, 30));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result = MapUtils.mapFromEntries(input, true)) {
      // row 0 must be null, row 1 must equal [{3,30}]
      try (ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, row1)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void allNullStructEntriesAllRowsNull() {
    // Row 0: [null_struct]  →  null
    // Row 1: [null_struct, null_struct]  →  null
    List<HostColumnVector.StructData> row0 = Arrays.asList((HostColumnVector.StructData) null);
    List<HostColumnVector.StructData> row1 =
        Arrays.asList((HostColumnVector.StructData) null, (HostColumnVector.StructData) null);
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void mixedNullStructAndNullKeyReturnsNullNotThrow() {
    // Row 0: [null_struct, {null_key, 20}]
    // CPU: short-circuits on the null struct entry at index 0 → returns null without
    //      inspecting the null key at index 1.  GPU must match (null, not throw).
    List<HostColumnVector.StructData> row0 =
        Arrays.asList(null, entry(null, 20));
    try (ColumnVector input  = ColumnVector.fromLists(LIST_TYPE, row0);
         ColumnVector result = MapUtils.mapFromEntries(input, true)) {
      try (ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, (List<?>) null)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }

  @Test
  void nullStructRowDoesNotSuppressThrowInOtherRow() {
    // Row 0: [null_struct]      →  null (masked)
    // Row 1: [{null_key, 20}]   →  must throw (valid struct, null key, no null struct entry)
    List<HostColumnVector.StructData> row0 =
        Arrays.asList((HostColumnVector.StructData) null);
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(null, 20));
    try (ColumnVector input = ColumnVector.fromLists(LIST_TYPE, row0, row1)) {
      assertThrows(CudfException.class, () -> MapUtils.mapFromEntries(input, true).close());
    }
  }

  @Test
  void emptyListRowHandledCorrectly() {
    // Row 0: []          →  empty map, no error
    // Row 1: [{1, 10}]   →  normal
    List<HostColumnVector.StructData> row0 = Arrays.asList();
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(1, 10));
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, row0, row1);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, row0, row1)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void outerNullRowPreservedAsNull() {
    // Row 0: null outer row (the list itself is null)  →  stays null
    // Row 1: [{1,10}]                                 →  unchanged
    List<HostColumnVector.StructData> row1 = Arrays.asList(entry(1, 10));
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, null, row1);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, row1)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void allOuterNullRowsRemainNull() {
    // All rows are outer-null — contains_nulls returns null for each, reduce(any) returns
    // an invalid scalar.  The is_valid guard must prevent reading the invalid scalar's value.
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, null, null);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void allOuterNullRowsRemainNullNoThrowPolicy() {
    // Same as allOuterNullRowsRemainNull but with throwOnNullKey=false,
    // independently verifying the is_valid guard in bool_scalar_value (map_utils.cu:44).
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, null, null);
         ColumnVector result   = MapUtils.mapFromEntries(input, false);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void singleOuterNullRowRemainNull() {
    // Single-row all-outer-null: boundary check for the is_valid guard on the reduce scalar.
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE, (List<?>) null);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, (List<?>) null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void emptyColumnReturnsEmptyColumn() {
    // Zero-row column: exercises the `if (input.size() == 0)` early-return path.
    // Result must be an empty LIST column of the same type — no error thrown.
    try (ColumnVector input    = ColumnVector.fromLists(LIST_TYPE);
         ColumnVector result   = MapUtils.mapFromEntries(input, true);
         ColumnVector expected = ColumnVector.fromLists(LIST_TYPE)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void outerNullRowAndNullStructEntryRowBothNull() {
    // Exercises the bitmask_and path: input already has a null row (outer null) AND
    // another row with a null struct entry.  Both must appear null in the output.
    //
    // Row 0: null outer row                →  stays null  (existing null mask)
    // Row 1: [null_struct, {2,20}]         →  becomes null (null struct entry)
    // Row 2: [{3,30}]                      →  unchanged
    List<HostColumnVector.StructData> row1 = Arrays.asList(null, entry(2, 20));
    List<HostColumnVector.StructData> row2 = Arrays.asList(entry(3, 30));
    try (ColumnVector input  = ColumnVector.fromLists(LIST_TYPE, null, row1, row2);
         ColumnVector result = MapUtils.mapFromEntries(input, true)) {
      try (ColumnVector expected = ColumnVector.fromLists(LIST_TYPE, null, null, row2)) {
        assertColumnsAreEqual(expected, result);
      }
    }
  }
}
