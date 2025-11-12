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
import ai.rapids.cudf.HostColumnVector.*;

import ai.rapids.cudf.*;
import ai.rapids.cudf.HostColumnVector.BasicType;

import org.junit.jupiter.api.Test;

import com.nvidia.spark.rapids.jni.iceberg.IcebergTruncate;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

import java.util.Arrays;
import java.util.Collections;

public class IcebergTruncateTest {

  @Test
  void testTruncateInt() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(null, 0, 1, 5, 9, 10, 11, -1, -5, -10, -11, null);
        ColumnVector expected = ColumnVector.fromBoxedInts(null, 0, 0, 0, 0, 10, 10, -10, -10, -10, -20, null);
        ColumnVector result = IcebergTruncate.truncate(input, 10)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testTruncateLong() {
    try (
        ColumnVector input = ColumnVector.fromBoxedLongs(null, 0L, 1L, 5L, 9L, 10L, 11L, -1L, -5L, -10L, -11L, null);
        ColumnVector expected = ColumnVector.fromBoxedLongs(null, 0L, 0L, 0L, 0L, 10L, 10L, -10L, -10L, -10L, -20L,
            null);
        ColumnVector result = IcebergTruncate.truncate(input, 10)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testTruncateString() {
    try (
        ColumnVector input = ColumnVector.fromStrings(null, "🚀23四😁678", "中华人民共和国", null);
        ColumnVector expected = ColumnVector.fromStrings(null, "🚀23四😁", "中华人民共", null);
        ColumnVector result = IcebergTruncate.truncate(input, 5)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testTruncateBinary() {
    try (ColumnVector input = ColumnVector.fromLists(
        new ListType(true, new BasicType(true, DType.UINT8)),
          Arrays.asList(1, 2, 3), // Normal case
          Arrays.asList(11, null, 33), // Contains null
          null, // Entire array is null
          Collections.emptyList(), // Empty list
          Collections.singletonList(null), // Single null
          Arrays.asList(null, null, null), // All nulls
          Arrays.asList(111, 222, 333));
        ColumnVector expected = ColumnVector.fromLists(
          new ListType(true, new BasicType(true, DType.UINT8)),
          Arrays.asList(1, 2), // Normal case
          Arrays.asList(11, null), // Contains null
          null, // Entire array is null
          Collections.emptyList(), // Empty list
          Collections.singletonList(null), // Single null
          Arrays.asList(null, null), // All nulls
          Arrays.asList(111, 222));

        ColumnVector result = IcebergTruncate.truncate(input, 2)) {
      assertColumnsAreEqual(expected, result);
    }
  }

}
