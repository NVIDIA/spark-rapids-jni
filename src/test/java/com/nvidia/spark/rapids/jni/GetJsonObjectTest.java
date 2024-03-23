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

import ai.rapids.cudf.ColumnVector;
import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class GetJsonObjectTest {
  /**
   * Test: query is $.k
   */
  @Test
  void getJsonObjectTest() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[2];
    query[0] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[1] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k", -1);
    try (ColumnVector jsonCv = ColumnVector.fromStrings(
        "{\"k\": \"v\"}");
         ColumnVector expected = ColumnVector.fromStrings(
             "v");
         ColumnVector actual = JSONUtils.getJsonObject(jsonCv, 2, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test: query is $.k1
   */
  @Test
  void getJsonObjectTest2() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[2];
    query[0] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[1] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k1_111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", -1);

    String JSON = "{\"k1_111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111\"" +
        ":\"v1_111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111\"}";
    String expectedStr = "v1_111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON, JSON, JSON, JSON, JSON, JSON, JSON);
        ColumnVector expected = ColumnVector.fromStrings(
            expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, 2, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Test: query is $.k1.k2
   */
  @Test
  void getJsonObjectTest3() {
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[4];
    query[0] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[1] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k1", -1);
    query[2] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[3] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k2", -1);
    String JSON = "{\"k1\":{\"k2\":\"v2\"}}";
    String expectedStr = "v2";
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON, JSON, JSON, JSON, JSON, JSON, JSON);
        ColumnVector expected = ColumnVector.fromStrings(
            expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, 4, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }
}
