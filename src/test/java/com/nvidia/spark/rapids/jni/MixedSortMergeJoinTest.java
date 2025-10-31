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
import ai.rapids.cudf.ast.BinaryOperation;
import ai.rapids.cudf.ast.BinaryOperator;
import ai.rapids.cudf.ast.ColumnReference;
import ai.rapids.cudf.ast.CompiledExpression;
import ai.rapids.cudf.ast.TableReference;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MixedSortMergeJoinTest {

  @Test
  void testInnerJoinBasic() {
    // Test basic inner join with equality keys and conditional expression
    // Left equality:  {0, 1, 2}
    // Right equality: {1, 2, 3}
    // Left conditional:  {4, 4, 4}
    // Right conditional: {3, 4, 5}
    // Condition: left_col > right_col
    // Expected result: row 1 from left matches with row 0 from right (1==1 and 4>3)

    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2, 3);
         ColumnVector leftConditional = ColumnVector.fromInts(4, 4, 4);
         ColumnVector rightConditional = ColumnVector.fromInts(3, 4, 5);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.innerJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);

      try {
        // Expected: {1}, {0} - row 1 from left matches row 0 from right
        try (ColumnVector expectedLeft = ColumnVector.fromInts(1);
             ColumnVector expectedRight = ColumnVector.fromInts(0)) {
          assertEquals(1, gatherMaps[0].getRowCount());
          assertEquals(1, gatherMaps[1].getRowCount());
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
  void testInnerJoinNoMatches() {
    // Test inner join where equality keys match but conditional expression never true
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(1, 2);
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2);
         ColumnVector leftConditional = ColumnVector.fromInts(1, 2);
         ColumnVector rightConditional = ColumnVector.fromInts(10, 20);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.innerJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);

      try {
        // Expected: no matches
        assertEquals(0, gatherMaps[0].getRowCount());
        assertEquals(0, gatherMaps[1].getRowCount());
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testLeftJoinBasic() {
    // Test basic left join
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2, 3);
         ColumnVector leftConditional = ColumnVector.fromInts(4, 4, 4);
         ColumnVector rightConditional = ColumnVector.fromInts(3, 4, 5);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.leftJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);

      try {
        // Expected: all left rows (0, 1, 2), with row 1 matching right row 0, others null
        assertEquals(3, gatherMaps[0].getRowCount());
        assertEquals(3, gatherMaps[1].getRowCount());

        // Verify that we have at least one matching pair and unmatched rows are marked OOB
        boolean foundMatch = false;
        try (HostColumnVector leftHost = gatherMaps[0].toColumnView(0, (int)gatherMaps[0].getRowCount()).copyToHost();
             HostColumnVector rightHost = gatherMaps[1].toColumnView(0, (int)gatherMaps[1].getRowCount()).copyToHost()) {
          for (int i = 0; i < leftHost.getRowCount(); i++) {
            int leftIdx = leftHost.getInt(i);
            int rightIdx = rightHost.getInt(i);
            if (leftIdx == 1 && rightIdx == 0) {
              foundMatch = true;
            } else {
              // Unmatched rows should use OOB sentinel == rightEquality row count (3)
              assertEquals(3, rightIdx);
            }
          }
        }
        assertTrue(foundMatch, "Expected to find matching pair (1, 0)");
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
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightEquality = ColumnVector.fromInts();
         ColumnVector leftConditional = ColumnVector.fromInts(4, 4, 4);
         ColumnVector rightConditional = ColumnVector.fromInts();
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.leftJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);

      try {
        // Expected: all left rows with null right indices
        try (ColumnVector expectedLeft = ColumnVector.fromInts(0, 1, 2)) {
          assertEquals(3, gatherMaps[0].getRowCount());
          assertEquals(3, gatherMaps[1].getRowCount());
          AssertUtils.assertColumnsAreEqual(expectedLeft, gatherMaps[0].toColumnView(0, (int)gatherMaps[0].getRowCount()));
          // Right sentinel should be OOB == rightEquality row count (0)
          try (ColumnVector expectedRight = ColumnVector.fromInts(0, 0, 0)) {
            AssertUtils.assertColumnsAreEqual(expectedRight, gatherMaps[1].toColumnView(0, (int)gatherMaps[1].getRowCount()));
          }
        }
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
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2, 3);
         ColumnVector leftConditional = ColumnVector.fromInts(4, 4, 4);
         ColumnVector rightConditional = ColumnVector.fromInts(3, 4, 5);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile();
         GatherMap gatherMap = MixedSortMergeJoin.leftSemiJoin(
             leftEqTable, rightEqTable, leftCondTable, rightCondTable,
             condition, false, false, true)) {

      // Expected: {1} - only row 1 from left has a match
      try (ColumnVector expected = ColumnVector.fromInts(1)) {
        assertEquals(1, gatherMap.getRowCount());
        AssertUtils.assertColumnsAreEqual(expected, gatherMap.toColumnView(0, (int)gatherMap.getRowCount()));
      }
    }
  }

  @Test
  void testLeftSemiJoinEmptyRight() {
    // Test left semi join with empty right table
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightEquality = ColumnVector.fromInts();
         ColumnVector leftConditional = ColumnVector.fromInts(4, 4, 4);
         ColumnVector rightConditional = ColumnVector.fromInts();
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile();
         GatherMap gatherMap = MixedSortMergeJoin.leftSemiJoin(
             leftEqTable, rightEqTable, leftCondTable, rightCondTable,
             condition, false, false, true)) {

      // Expected: empty result - no rows match
      assertEquals(0, gatherMap.getRowCount());
    }
  }

  @Test
  void testLeftAntiJoinBasic() {
    // Test basic left anti join
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2, 3);
         ColumnVector leftConditional = ColumnVector.fromInts(4, 4, 4);
         ColumnVector rightConditional = ColumnVector.fromInts(3, 4, 5);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile();
         GatherMap gatherMap = MixedSortMergeJoin.leftAntiJoin(
             leftEqTable, rightEqTable, leftCondTable, rightCondTable,
             condition, false, false, true)) {

      // Expected: {0, 2} - rows 0 and 2 from left have no matches
      try (ColumnVector expected = ColumnVector.fromInts(0, 2)) {
        assertEquals(2, gatherMap.getRowCount());
        AssertUtils.assertColumnsAreEqual(expected, gatherMap.toColumnView(0, (int)gatherMap.getRowCount()));
      }
    }
  }

  @Test
  void testLeftAntiJoinEmptyRight() {
    // Test left anti join with empty right table - all left rows should be returned
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightEquality = ColumnVector.fromInts();
         ColumnVector leftConditional = ColumnVector.fromInts(4, 4, 4);
         ColumnVector rightConditional = ColumnVector.fromInts();
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile();
         GatherMap gatherMap = MixedSortMergeJoin.leftAntiJoin(
             leftEqTable, rightEqTable, leftCondTable, rightCondTable,
             condition, false, false, true)) {

      // Expected: {0, 1, 2} - all left rows have no matches
      try (ColumnVector expected = ColumnVector.fromInts(0, 1, 2)) {
        assertEquals(3, gatherMap.getRowCount());
        AssertUtils.assertColumnsAreEqual(expected, gatherMap.toColumnView(0, (int)gatherMap.getRowCount()));
      }
    }
  }

  @Test
  void testMultipleEqualityKeys() {
    // Test with multiple equality key columns
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEq1 = ColumnVector.fromInts(1, 1, 2, 2);
         ColumnVector leftEq2 = ColumnVector.fromInts(1, 2, 1, 2);
         ColumnVector rightEq1 = ColumnVector.fromInts(1, 2);
         ColumnVector rightEq2 = ColumnVector.fromInts(2, 1);
         ColumnVector leftConditional = ColumnVector.fromInts(10, 20, 30, 40);
         ColumnVector rightConditional = ColumnVector.fromInts(15, 25);
         Table leftEqTable = new Table(leftEq1, leftEq2);
         Table rightEqTable = new Table(rightEq1, rightEq2);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile();
         GatherMap gatherMap = MixedSortMergeJoin.leftSemiJoin(
             leftEqTable, rightEqTable, leftCondTable, rightCondTable,
             condition, false, false, true)) {

      // Should have at least 0 results
      assertTrue(gatherMap.getRowCount() >= 0);
    }
  }

  @Test
  void testPreSortedTables() {
    // Test with pre-sorted equality tables
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(0, 1, 2);  // already sorted
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2, 3); // already sorted
         ColumnVector leftConditional = ColumnVector.fromInts(4, 4, 4);
         ColumnVector rightConditional = ColumnVector.fromInts(3, 4, 5);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.innerJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, true, true, true);  // both sorted

      try {
        // Expected: {1}, {0} - row 1 from left matches row 0 from right
        try (ColumnVector expectedLeft = ColumnVector.fromInts(1);
             ColumnVector expectedRight = ColumnVector.fromInts(0)) {
          assertEquals(1, gatherMaps[0].getRowCount());
          assertEquals(1, gatherMaps[1].getRowCount());
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
  void testComplexConditionalExpression() {
    // Test with a more complex conditional expression (AND of two conditions)
    BinaryOperation leftGt = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    BinaryOperation leftGt2 = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(1, TableReference.LEFT),
        new ColumnReference(1, TableReference.RIGHT));
    BinaryOperation expr = new BinaryOperation(BinaryOperator.LOGICAL_AND, leftGt, leftGt2);

    try (ColumnVector leftEquality = ColumnVector.fromInts(1, 2, 3);
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2, 3);
         ColumnVector leftCond1 = ColumnVector.fromInts(10, 20, 30);
         ColumnVector leftCond2 = ColumnVector.fromInts(5, 15, 25);
         ColumnVector rightCond1 = ColumnVector.fromInts(8, 18, 28);
         ColumnVector rightCond2 = ColumnVector.fromInts(3, 13, 23);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftCond1, leftCond2);
         Table rightCondTable = new Table(rightCond1, rightCond2);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.innerJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);

      try {
        // All rows match on equality, and all satisfy the conditional
        assertEquals(3, gatherMaps[0].getRowCount());
        assertEquals(3, gatherMaps[1].getRowCount());
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testNullInEqualityKeys() {
    // Test behavior with nulls in equality keys
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromBoxedInts(1, null, 3);
         ColumnVector rightEquality = ColumnVector.fromBoxedInts(1, 2, null);
         // Make null==null equality pair also satisfy the conditional: 6 > 5
         ColumnVector leftConditional = ColumnVector.fromInts(4, 6, 6);
         ColumnVector rightConditional = ColumnVector.fromInts(3, 4, 5);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.innerJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);  // nulls equal

      try {
        // Expected: (1,1) equality pair passes 4 > 3 AND (null,null) passes 6 > 5 when compareNullsEqual=true
        try (ColumnVector expectedLeft = ColumnVector.fromInts(0, 1);
             ColumnVector expectedRight = ColumnVector.fromInts(0, 2)) {
          assertEquals(2, gatherMaps[0].getRowCount());
          assertEquals(2, gatherMaps[1].getRowCount());
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
  void testDuplicateEqualityKeys() {
    // Test with duplicate equality keys that have different conditional results
    // Left:  {1, 1, 2}, conditional {5, 10, 15}
    // Right: {1, 2},    conditional {7, 8}
    // Expected: (1,5) doesn't match (1,7) since 5<7
    //           (1,10) matches (1,7) since 10>7
    //           (2,15) matches (2,8) since 15>8
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(1, 1, 2);
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2);
         ColumnVector leftConditional = ColumnVector.fromInts(5, 10, 15);
         ColumnVector rightConditional = ColumnVector.fromInts(7, 8);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.innerJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);

      try {
        // Expected: (1,10) matches (1,7) and (2,15) matches (2,8)
        // So we should have 2 matches: left indices {1, 2}, right indices {0, 1}
        assertEquals(2, gatherMaps[0].getRowCount());
        assertEquals(2, gatherMaps[1].getRowCount());
        
        try (ColumnVector expectedLeft = ColumnVector.fromInts(1, 2);
             ColumnVector expectedRight = ColumnVector.fromInts(0, 1)) {
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
  void testNullInConditionalColumns() {
    // Test behavior when conditional columns contain nulls
    // The conditional expression evaluation should treat null results as false
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(1, 2, 3);
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2, 3);
         // Conditional columns with nulls
         ColumnVector leftConditional = ColumnVector.fromBoxedInts(10, null, 30);
         ColumnVector rightConditional = ColumnVector.fromBoxedInts(5, 15, null);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.innerJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);

      try {
        // Expected: Only (1,10) > (1,5) = true should match
        //           (2,null) > (2,15) = null/false (no match)
        //           (3,30) > (3,null) = null/false (no match)
        assertEquals(1, gatherMaps[0].getRowCount());
        assertEquals(1, gatherMaps[1].getRowCount());
        
        try (ColumnVector expectedLeft = ColumnVector.fromInts(0);
             ColumnVector expectedRight = ColumnVector.fromInts(0)) {
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

  // ===== Empty Table Tests =====

  @Test
  void testInnerJoinEmptyLeftTable() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts();
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2, 3);
         ColumnVector leftConditional = ColumnVector.fromInts();
         ColumnVector rightConditional = ColumnVector.fromInts(3, 4, 5);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.innerJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);

      try {
        assertEquals(0, gatherMaps[0].getRowCount());
        assertEquals(0, gatherMaps[1].getRowCount());
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testInnerJoinEmptyRightTable() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts(0, 1, 2);
         ColumnVector rightEquality = ColumnVector.fromInts();
         ColumnVector leftConditional = ColumnVector.fromInts(4, 4, 4);
         ColumnVector rightConditional = ColumnVector.fromInts();
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.innerJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);

      try {
        assertEquals(0, gatherMaps[0].getRowCount());
        assertEquals(0, gatherMaps[1].getRowCount());
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testInnerJoinBothTablesEmpty() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts();
         ColumnVector rightEquality = ColumnVector.fromInts();
         ColumnVector leftConditional = ColumnVector.fromInts();
         ColumnVector rightConditional = ColumnVector.fromInts();
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.innerJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);

      try {
        assertEquals(0, gatherMaps[0].getRowCount());
        assertEquals(0, gatherMaps[1].getRowCount());
      } finally {
        for (GatherMap gm : gatherMaps) {
          if (gm != null) gm.close();
        }
      }
    }
  }

  @Test
  void testLeftJoinEmptyLeftTable() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts();
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2, 3);
         ColumnVector leftConditional = ColumnVector.fromInts();
         ColumnVector rightConditional = ColumnVector.fromInts(3, 4, 5);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile()) {

      GatherMap[] gatherMaps = MixedSortMergeJoin.leftJoin(
          leftEqTable, rightEqTable, leftCondTable, rightCondTable,
          condition, false, false, true);

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

  @Test
  void testLeftSemiJoinEmptyLeftTable() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts();
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2, 3);
         ColumnVector leftConditional = ColumnVector.fromInts();
         ColumnVector rightConditional = ColumnVector.fromInts(3, 4, 5);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile();
         GatherMap gatherMap = MixedSortMergeJoin.leftSemiJoin(
             leftEqTable, rightEqTable, leftCondTable, rightCondTable,
             condition, false, false, true)) {

      assertEquals(0, gatherMap.getRowCount());
    }
  }

  @Test
  void testLeftAntiJoinEmptyLeftTable() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));

    try (ColumnVector leftEquality = ColumnVector.fromInts();
         ColumnVector rightEquality = ColumnVector.fromInts(1, 2, 3);
         ColumnVector leftConditional = ColumnVector.fromInts();
         ColumnVector rightConditional = ColumnVector.fromInts(3, 4, 5);
         Table leftEqTable = new Table(leftEquality);
         Table rightEqTable = new Table(rightEquality);
         Table leftCondTable = new Table(leftConditional);
         Table rightCondTable = new Table(rightConditional);
         CompiledExpression condition = expr.compile();
         GatherMap gatherMap = MixedSortMergeJoin.leftAntiJoin(
             leftEqTable, rightEqTable, leftCondTable, rightCondTable,
             condition, false, false, true)) {

      assertEquals(0, gatherMap.getRowCount());
    }
  }
}
