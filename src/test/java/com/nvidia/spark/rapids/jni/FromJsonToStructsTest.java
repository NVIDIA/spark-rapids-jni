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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.JSONOptions;
import ai.rapids.cudf.Schema;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FromJsonToStructsTest {
  private static JSONOptions getOptions() {
    return JSONOptions.builder()
        .withNormalizeSingleQuotes(true)
        .withLeadingZeros(true)
        .withNonNumericNumbers(true)
        .withUnquotedControlChars(true)
        .build();
  }

  /** Schema: {@code struct< a: struct< b: int > >}. */
  private static Schema nestedSchema() {
    Schema.Builder root = Schema.builder();
    Schema.Builder a = root.addColumn(DType.STRUCT, "a");
    a.addColumn(DType.INT32, "b");
    return root.build();
  }

  @Test
  void testFromJsonReportsSchemaMismatch() {
    // Row 1 matches struct<a:struct<b:int>>; row 2 has `a` as a scalar where a struct is expected
    // (a depth-1 nested schema-category mismatch, the SPARK-33134 shape). spark-rapids consumes
    // this flag to fall back the whole batch to CPU JsonToStructs.
    try (ColumnVector input =
             ColumnVector.fromStrings("{\"a\": {\"b\": 1}}", "{\"a\": 5}")) {
      JSONUtils.FromJSONResult result =
          JSONUtils.fromJSONToStructs(input, nestedSchema(), getOptions(), true);
      try (ColumnVector data = result.getData()) {
        assertTrue(result.hasSchemaMismatch(),
            "a depth-1 scalar where a struct is expected must flag a schema mismatch");
      }
    }
  }

  @Test
  void testFromJsonNoSchemaMismatch() {
    try (ColumnVector input =
             ColumnVector.fromStrings("{\"a\": {\"b\": 1}}", "{\"a\": {\"b\": 2}}")) {
      JSONUtils.FromJSONResult result =
          JSONUtils.fromJSONToStructs(input, nestedSchema(), getOptions(), true);
      try (ColumnVector data = result.getData()) {
        assertFalse(result.hasSchemaMismatch(),
            "well-formed nested rows must not flag a schema mismatch");
      }
    }
  }
}
