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
import org.junit.jupiter.api.Test;

import java.math.RoundingMode;

public class RowConversionTest {
  @Test
  void fixedWidthRowsRoundTrip() {
    try (Table t = new Table.TestBuilder()
        .column(3L, 9L, 4L, 2L, 20L, null)
        .column(5.0d, 9.5d, 0.9d, 7.23d, 2.8d, null)
        .column(5, 1, 0, 2, 7, null)
        .column(true, false, false, true, false, null)
        .column(1.0f, 3.5f, 5.9f, 7.1f, 9.8f, null)
        .column(new Byte[]{2, 3, 4, 5, 9, null})
        .decimal32Column(-3, RoundingMode.UNNECESSARY, 5.0d, 9.5d, 0.9d, 7.23d, 2.8d, null)
        .decimal64Column(-8, 3L, 9L, 4L, 2L, 20L, null)
        .build()) {
      ColumnVector[] rows = RowConversion.convertToRows(t);
      try {
        // We didn't overflow
        assert rows.length == 1;
        ColumnVector cv = rows[0];
        assert cv.getRowCount() == t.getRowCount();
        DType[] types = new DType[t.getNumberOfColumns()];
        for (int i = 0; i < t.getNumberOfColumns(); i++) {
          types[i] = t.getColumn(i).getType();
        }
        try (Table backAgain = RowConversion.convertFromRows(cv, types)) {
          AssertUtils.assertTablesAreEqual(t, backAgain);
        }
      } finally {
        for (ColumnVector cv : rows) {
          cv.close();
        }
      }
    }
  }
}
