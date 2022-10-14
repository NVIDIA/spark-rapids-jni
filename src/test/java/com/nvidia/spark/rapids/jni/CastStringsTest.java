/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Table;
import com.nvidia.spark.rapids.jni.CastException;
import org.junit.jupiter.api.Test;

import java.math.RoundingMode;
import java.util.stream.IntStream;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class CastStringsTest {
  @Test
  void castToIntegerTest() {
    Table.TestBuilder tb = new Table.TestBuilder();
    tb.column(3l, 9l, 4l, 2l, 20l, null, null);
    tb.column(5, 1, 0, 2, 7, null, null);
    tb.column(new Byte[]{2, 3, 4, 5, 9, null, null});
    Table expected = tb.build();

    Table.TestBuilder tb2 = new Table.TestBuilder();
    tb2.column("3", "9", "4", "2", "20", null, "7.6asd");
    tb2.column("5", "1", "0", "2", "7", null, "asdf");
    tb2.column("2", "3", "4", "5", "9", null, "7.8.3");

    List<ColumnVector> result = new ArrayList<>();

    try (Table origTable = tb2.build()) {
      for (int i = 0; i < origTable.getNumberOfColumns(); i++) {
        ColumnVector string_col = origTable.getColumn(i);
        result.add(CastStrings.toInteger(string_col, false, 
                   expected.getColumn(i).getType()));
      }
      Table result_tbl = new Table(
        result.toArray(new ColumnVector[result.size()]));
      AssertUtils.assertTablesAreEqual(expected, result_tbl);
    }
  }

  @Test
  void castToIntegerAnsiTest() {
    Table.TestBuilder tb = new Table.TestBuilder();
    tb.column(3l, 9l, 4l, 2l, 20l);
    tb.column(5, 1, 0, 2, 7);
    tb.column(new Byte[]{2, 3, 4, 5, 9});
    Table expected = tb.build();

    Table.TestBuilder tb2 = new Table.TestBuilder();
    tb2.column("3", "9", "4", "2", "20");
    tb2.column("5", "1", "0", "2", "7");
    tb2.column("2", "3", "4", "5", "9");

    List<ColumnVector> result = new ArrayList<>();

    try (Table origTable = tb2.build()) {
      for (int i = 0; i < origTable.getNumberOfColumns(); i++) {
        ColumnVector string_col = origTable.getColumn(i);
        result.add(CastStrings.toInteger(string_col, true, 
                   expected.getColumn(i).getType()));
      }
      Table result_tbl = new Table(
        result.toArray(new ColumnVector[result.size()]));
      AssertUtils.assertTablesAreEqual(expected, result_tbl);
    }

    Table.TestBuilder fail = new Table.TestBuilder();
    fail.column("asdf", "9.0.2", "- 4e", "b2", "20-fe");

    try {
        Table failTable = fail.build();
        CastStrings.toInteger(failTable.getColumn(0), true,
                              expected.getColumn(0).getType());
        fail("Should have thrown");
      } catch (CastException e) {
        assertEquals("asdf", e.getStringWithError());
        assertEquals(0, e.getRowWithError());
    }
  }
}
