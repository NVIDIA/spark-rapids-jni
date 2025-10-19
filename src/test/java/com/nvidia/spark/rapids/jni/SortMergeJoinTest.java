/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

import ai.rapids.cudf.AssertUtils;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class SortMergeJoinTest {

  @Test
  void testLeftJoinBasic() {
    // Test basic left join
    // Left:  {0, 1, 2}
    // Right: {1, 2, 3}
    // Expected: all left rows, with rows 1 and 2 matching, row 0 unmatched

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2, 3);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      GatherMap[] gatherMaps = SortMergeJoin.leftJoin(
          leftTable, rightTable, false, false, true);

      try {
        // Expected: all 3 left rows
        assertEquals(3, gatherMaps[0].getRowCount());
        assertEquals(3, gatherMaps[1].getRowCount());

        // Copy results to host to verify content (order not guaranteed)
        try (HostColumnVector leftHost = gatherMaps[0].toColumnView(0, (int)gatherMaps[0].getRowCount()).copyToHost();
             HostColumnVector rightHost = gatherMaps[1].toColumnView(0, (int)gatherMaps[1].getRowCount()).copyToHost()) {
          
          // Track which left rows we've seen and verify their right indices
          boolean foundLeft0 = false;
          boolean foundLeft1 = false;
          boolean foundLeft2 = false;
          
          for (int i = 0; i < leftHost.getRowCount(); i++) {
            int leftIdx = leftHost.getInt(i);
            int rightIdx = rightHost.getInt(i);
            
            if (leftIdx == 0) {
              // Left 0 (value 0) has no match, should have OOB sentinel
              assertEquals(3, rightIdx, "Left row 0 should have OOB right index");
              foundLeft0 = true;
            } else if (leftIdx == 1) {
              // Left 1 (value 1) matches right 0 (value 1)
              assertEquals(0, rightIdx, "Left row 1 should match right row 0");
              foundLeft1 = true;
            } else if (leftIdx == 2) {
              // Left 2 (value 2) matches right 1 (value 2)
              assertEquals(1, rightIdx, "Left row 2 should match right row 1");
              foundLeft2 = true;
            }
          }
          
          assertTrue(foundLeft0, "Should have left row 0");
          assertTrue(foundLeft1, "Should have left row 1");
          assertTrue(foundLeft2, "Should have left row 2");
        }
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testLeftJoinEmptyRight() {
    // Test left join with empty right table

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts();
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      GatherMap[] gatherMaps = SortMergeJoin.leftJoin(
          leftTable, rightTable, false, false, true);

      try {
        // Expected: all left rows with OOB right indices
        assertEquals(3, gatherMaps[0].getRowCount());
        assertEquals(3, gatherMaps[1].getRowCount());

        try (ColumnVector expectedLeft = ColumnVector.fromInts(0, 1, 2);
             ColumnVector expectedRight = ColumnVector.fromInts(0, 0, 0)) {  // All OOB
          AssertUtils.assertColumnsAreEqual(expectedLeft, gatherMaps[0].toColumnView(0, (int)gatherMaps[0].getRowCount()));
          AssertUtils.assertColumnsAreEqual(expectedRight, gatherMaps[1].toColumnView(0, (int)gatherMaps[1].getRowCount()));
        }
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testLeftJoinNullKeys() {
    // Test left join with nulls in keys (nulls equal)

    try (ColumnVector leftKeys = ColumnVector.fromBoxedInts(1, null);
         ColumnVector rightKeys = ColumnVector.fromBoxedInts(null, 2);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      GatherMap[] gatherMaps = SortMergeJoin.leftJoin(
          leftTable, rightTable, false, false, true);

      try {
        // Expected: both left rows should match
        assertEquals(2, gatherMaps[0].getRowCount());
        assertEquals(2, gatherMaps[1].getRowCount());
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testLeftSemiJoinBasic() {
    // Test basic left semi join
    // Left:  {0, 1, 2}
    // Right: {1, 2, 3}
    // Expected: {1, 2} from left (rows with matches)

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2, 3);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys);
         GatherMap gatherMap = SortMergeJoin.leftSemiJoin(
             leftTable, rightTable, false, false, true)) {

      // Expected: indices 1 and 2 from left
      assertEquals(2, gatherMap.getRowCount());

      try (ColumnVector expected = ColumnVector.fromInts(1, 2)) {
        AssertUtils.assertColumnsAreEqual(expected, gatherMap.toColumnView(0, (int)gatherMap.getRowCount()));
      }
    }
  }

  @Test
  void testLeftSemiJoinEmptyRight() {
    // Test left semi join with empty right table - no matches

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts();
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys);
         GatherMap gatherMap = SortMergeJoin.leftSemiJoin(
             leftTable, rightTable, false, false, true)) {

      assertEquals(0, gatherMap.getRowCount());
    }
  }

  @Test
  void testLeftSemiJoinDuplicateKeys() {
    // Test left semi join with duplicate keys - each left row appears once
    // Left:  {1, 1, 2}
    // Right: {1, 2}
    // Expected: {0, 1, 2} (all left rows match)

    try (ColumnVector leftKeys = ColumnVector.fromInts(1, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys);
         GatherMap gatherMap = SortMergeJoin.leftSemiJoin(
             leftTable, rightTable, false, false, true)) {

      // Expected: 3 left rows all match (indices 0, 1, 2)
      assertEquals(3, gatherMap.getRowCount());
    }
  }

  @Test
  void testLeftAntiJoinBasic() {
    // Test basic left anti join
    // Left:  {0, 1, 2}
    // Right: {1, 2, 3}
    // Expected: {0} from left (rows without matches)

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2, 3);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys);
         GatherMap gatherMap = SortMergeJoin.leftAntiJoin(
             leftTable, rightTable, false, false, true)) {

      // Expected: index 0 from left (value 0 has no match)
      assertEquals(1, gatherMap.getRowCount());

      try (ColumnVector expected = ColumnVector.fromInts(0)) {
        AssertUtils.assertColumnsAreEqual(expected, gatherMap.toColumnView(0, (int)gatherMap.getRowCount()));
      }
    }
  }

  @Test
  void testLeftAntiJoinEmptyRight() {
    // Test left anti join with empty right table - all left rows returned

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightKeys = ColumnVector.fromInts();
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys);
         GatherMap gatherMap = SortMergeJoin.leftAntiJoin(
             leftTable, rightTable, false, false, true)) {

      // Expected: all left rows
      assertEquals(3, gatherMap.getRowCount());

      try (ColumnVector expected = ColumnVector.fromInts(0, 1, 2)) {
        AssertUtils.assertColumnsAreEqual(expected, gatherMap.toColumnView(0, (int)gatherMap.getRowCount()));
      }
    }
  }

  @Test
  void testLeftAntiJoinAllMatch() {
    // Test left anti join where all rows match - empty result

    try (ColumnVector leftKeys = ColumnVector.fromInts(1, 2, 3);
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2, 3);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys);
         GatherMap gatherMap = SortMergeJoin.leftAntiJoin(
             leftTable, rightTable, false, false, true)) {

      assertEquals(0, gatherMap.getRowCount());
    }
  }

  @Test
  void testPreSortedTables() {
    // Test with pre-sorted tables (performance optimization)

    try (ColumnVector leftKeys = ColumnVector.fromInts(0, 1, 2);  // sorted
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2, 3);  // sorted
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys);
         GatherMap gatherMap = SortMergeJoin.leftSemiJoin(
             leftTable, rightTable, true, true, true)) {  // both sorted

      // Expected: indices 1 and 2 from left
      assertEquals(2, gatherMap.getRowCount());

      try (ColumnVector expected = ColumnVector.fromInts(1, 2)) {
        AssertUtils.assertColumnsAreEqual(expected, gatherMap.toColumnView(0, (int)gatherMap.getRowCount()));
      }
    }
  }

  @Test
  void testMultipleKeyColumns() {
    // Test with multiple key columns

    try (ColumnVector leftCol1 = ColumnVector.fromInts(1, 1, 2, 2);
         ColumnVector leftCol2 = ColumnVector.fromInts(1, 2, 1, 2);
         ColumnVector rightCol1 = ColumnVector.fromInts(1, 2);
         ColumnVector rightCol2 = ColumnVector.fromInts(2, 1);
         Table leftTable = new Table(leftCol1, leftCol2);
         Table rightTable = new Table(rightCol1, rightCol2);
         GatherMap gatherMap = SortMergeJoin.leftSemiJoin(
             leftTable, rightTable, false, false, true)) {

      // Expected: (1,2) matches at index 1, (2,1) matches at index 2
      assertEquals(2, gatherMap.getRowCount());
    }
  }

  @Test
  void testLeftJoinEmptyLeftTable() {
    // Test left join with empty left table

    try (ColumnVector leftKeys = ColumnVector.fromInts();
         ColumnVector rightKeys = ColumnVector.fromInts(1, 2, 3);
         Table leftTable = new Table(leftKeys);
         Table rightTable = new Table(rightKeys)) {

      GatherMap[] gatherMaps = SortMergeJoin.leftJoin(
          leftTable, rightTable, false, false, true);

      try {
        // No left rows to preserve
        assertEquals(0, gatherMaps[0].getRowCount());
        assertEquals(0, gatherMaps[1].getRowCount());
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }
}

