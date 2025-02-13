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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector.*;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static ai.rapids.cudf.AssertUtils.*;

public class GpuSliceUtilsTest {
  @Test
  void testSliceStartIntLengthInt() {
    try (ColumnVector intListCV = ColumnVector.fromLists(
        new ListType(true, new BasicType(true, DType.INT32)),
        Arrays.asList(0, 1),
        Arrays.asList(2, 3, 7, 8),
        Arrays.asList(4, 5));
        ColumnVector result = GpuSliceUtils.slice(intListCV, 1, 2);
        ColumnVector expected = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.INT32)),
            Arrays.asList(0, 1),
            Arrays.asList(2, 3),
            Arrays.asList(4, 5))) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSliceStartIntLengthCol() {
    try (ColumnVector intListCV = ColumnVector.fromLists(
        new ListType(true, new BasicType(true, DType.INT32)),
        Arrays.asList(0, 1),
        Arrays.asList(2, 3, 7, 8),
        Arrays.asList(4, 5));
        ColumnVector lengthCV = ColumnVector.fromInts(0, 1, 2);
        ColumnVector result = GpuSliceUtils.slice(intListCV, 1, lengthCV);
        ColumnVector expected = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.INT32)),
            Collections.emptyList(),
            Collections.singletonList(2),
            Arrays.asList(4, 5))) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSliceStartColLengthInt() {
    try (ColumnVector intListCV = ColumnVector.fromLists(
        new ListType(true, new BasicType(true, DType.INT32)),
        Arrays.asList(0, 1),
        Arrays.asList(2, 3, 7, 8),
        Arrays.asList(4, 5));
        ColumnVector startCV = ColumnVector.fromInts(1, 2, 2);
        ColumnVector result = GpuSliceUtils.slice(intListCV, startCV, 2);
        ColumnVector expected = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.INT32)),
            Arrays.asList(0, 1),
            Arrays.asList(3, 7),
            Collections.singletonList(5))) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSliceStartColLengthCol() {
    try (ColumnVector intListCV = ColumnVector.fromLists(
        new ListType(true, new BasicType(true, DType.INT32)),
        Arrays.asList(0, 1),
        Arrays.asList(2, 3, 7, 8),
        Arrays.asList(4, 5));
        ColumnVector startCV = ColumnVector.fromInts(1, 2, 1);
        ColumnVector lengthCV = ColumnVector.fromInts(0, 1, 2);
        ColumnVector result = GpuSliceUtils.slice(intListCV, startCV, lengthCV);
        ColumnVector expected = ColumnVector.fromLists(
            new ListType(true, new BasicType(true, DType.INT32)),
            Collections.emptyList(),
            Collections.singletonList(3),
            Arrays.asList(4, 5))) {
      assertColumnsAreEqual(expected, result);
    }
  }
}
