/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

import ai.rapids.cudf.*;


/**
 * Exedute SQL `case when` semantic.
 * If there are multiple branches and each branch uses scalar to generator value,
 * then it's fast to use this class because it does not generate temp string columns.
 *
 * E.g.:
 *   SQL is:
 *     select
 *        case
 *          when bool_1_expr then "value_1"
 *          when bool_2_expr then "value_2"
 *          when bool_3_expr then "value_3"
 *          else "value_else"
 *        end
 *      from tab
 *
 * Execution steps:
 *   Execute bool exprs to get bool columns, e.g., gets:
 *     bool column 1: [true,  false, false, false]  // bool_1_expr result
 *     bool column 2: [false, true,  false, flase]  // bool_2_expr result
 *     bool column 3: [false, false, true,  flase]  // bool_3_expr result
 *   Execute `selectFirstTrueIndex` to get the column index for the first true in bool columns.
 *   Generate a column to store salars: "value_1", "value_1", "value_1", "value_else"
 *   Execute `selectFromIndex` to generate the final output column
 * For more details, refer to the functions in this class:
 *   `selectFirstTrueIndex`
 *   `selectFromIndex`
 *
 */
public class CaseWhen {

  /**
   *
   * Select the column index for the first true in bool columns.
   * For the row does not contain true, use end index(number of columns).
   *
   * e.g.:
   *   column 0: true,  false, false, false
   *   column 1: false, true,  false, false
   *   column 2: false, false, true, false
   *
   *   1st row is: true, flase, false; first true index is 0
   *   2nd row is: false, true, false; first true index is 1
   *   3rd row is: false, flase, false; first true index is 2
   *   4th row is: false, false, false; do not find true, set index to the end index 3
   *
   *   output column: 0, 1, 2, 3
   *   In the `case when` context, here 3 index means using NULL value.
   *
  */
  public static ColumnVector selectFirstTrueIndex(ColumnVector[] boolColumns) {
    for (ColumnVector cv : boolColumns) {
      assert(cv.getType().equals(DType.BOOL8)) : "Columns must be bools";
    }
    
    long[] boolHandles = new long[boolColumns.length];
    for (int i = 0; i < boolColumns.length; ++i) {
      boolHandles[i] = boolColumns[i].getNativeView();
    }

    return new ColumnVector(selectFirstTrueIndex(boolHandles));
  }

  /**
   *
   * Select strings in scalar column according to index column.
   * If index is out of bound, use NULL value
   * e.g.:
   *   scalar column: s0, s1, s2
   *   index  column: 0,  1,  2,  2,  1,  0,  3
   *   output column: s0, s1, s2, s2, s1, s0, NULL
   *
  */
  public static ColumnVector selectFromIndex(ColumnVector scalarCol, ColumnVector indexCol) {
    assert(scalarCol.getType().equals(DType.STRING)) : "Scalar column must be a String";
    assert(indexCol.getType().equals(DType.INT32)) : "Index column must be a INT32";

    return new ColumnVector(selectFromIndex(scalarCol.getNativeView(), indexCol.getNativeView()));
  }

  private static native long selectFirstTrueIndex(long[] boolHandles);

  private static native long selectFromIndex(long scalarHandle, long indexHandle);
}
