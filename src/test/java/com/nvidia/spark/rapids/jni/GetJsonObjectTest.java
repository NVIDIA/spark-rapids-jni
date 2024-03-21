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
import ai.rapids.cudf.HostColumnVector;
import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class GetJsonObjectTest {
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
      HostColumnVector hostCv = actual.copyToHost();
      System.out.println("rows: " + hostCv.getRowCount());
      if (hostCv.isNull(0)) {
        System.out.println("v: null");  
      } else {
        String v = hostCv.getJavaString(0);
        System.out.println(v.length());
        int vv = v.charAt(0);
        System.out.println("int v: " + vv);
      }
      assertColumnsAreEqual(expected, actual);
    }
  }
}
