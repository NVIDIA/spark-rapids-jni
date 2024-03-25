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

import ai.rapids.cudf.HostColumnVector;
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

  /**
   * Test: query paths depth is 10
   */
  @Test
  void getJsonObjectTest4() {
    int paths_num = 20;
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[paths_num];
    query[0] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[1] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k1", -1);
    query[2] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[3] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k2", -1);
    query[4] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[5] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k3", -1);
    query[6] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[7] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k4", -1);
    query[8] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[9] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k5", -1);
    query[10] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[11] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k6", -1);
    query[12] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[13] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k7", -1);
    query[14] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[15] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k8", -1);
    query[16] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[17] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k9", -1);
    query[18] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[19] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "k10", -1);

    String JSON = "{\"k1\":{\"k2\":{\"k3\":{\"k4\":{\"k5\":{\"k6\":{\"k7\":{\"k8\":{\"k9\":{\"k10\":\"v10\"}}}}}}}}}}";
    String expectedStr = "v10";
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON, JSON, JSON, JSON, JSON, JSON, JSON);
        ColumnVector expected = ColumnVector.fromStrings(
            expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, paths_num, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Baidu case: unescape two chars \/ to one char /
   */
  @Test
  void getJsonObjectTest_Baidu_unescape_backslash() {
    int paths_num = 2;
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[paths_num];
    query[0] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[1] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "URdeosurl", -1);

    String JSON = "{\"brand\":\"ssssss\",\"duratRon\":15,\"eqTosuresurl\":\"\",\"RsZxarthrl\":false,\"xonRtorsurl\":\"\",\"xonRtorsurlstOTe\":0,\"TRctures\":[{\"RxaGe\":\"VttTs:\\/\\/feed-RxaGe.baRdu.cox\\/0\\/TRc\\/-196588744s840172444s-773690137.zTG\"}],\"Toster\":\"VttTs:\\/\\/feed-RxaGe.baRdu.cox\\/0\\/TRc\\/-196588744s840172444s-773690137.zTG\",\"reserUed\":{\"bRtLate\":391.79,\"xooUZRke\":26876,\"nahrlIeneratRonNOTe\":0,\"useJublRc\":6,\"URdeoRd\":821284086},\"tRtle\":\"ssssssssssmMsssssssssssssssssss\",\"url\":\"s{storehrl}\",\"usersTortraRt\":\"VttTs:\\/\\/feed-RxaGe.baRdu.cox\\/0\\/TRc\\/-6971178959s-664926866s-6096674871.zTG\",\"URdeosurl\":\"VttT:\\/\\/nadURdeo2.baRdu.cox\\/5fa3893aed7fc0f8231dab7be23efc75s820s6240.xT3\",\"URdeoRd\":821284086}";
    String expectedStr = "VttT://nadURdeo2.baRdu.cox/5fa3893aed7fc0f8231dab7be23efc75s820s6240.xT3";
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON, JSON, JSON, JSON, JSON, JSON, JSON);
        ColumnVector expected = ColumnVector.fromStrings(
            expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, paths_num, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * Baidu case: query unexist field name
   */
  @Test
  void getJsonObjectTest_Baidu_get_unexist_field_name() {
    int paths_num = 2;
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[paths_num];
    query[0] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.KEY, "", -1);
    query[1] = new JSONUtils.PathInstructionJni(JSONUtils.PathInstructionType.NAMED, "Vgdezsurl", -1);

    String JSON = "{\"brand\":\"ssssss\",\"duratgzn\":17,\"eSyzsuresurl\":\"\",\"gswUartWrl\":false,\"Uzngtzrsurl\":\"\",\"UzngtzrsurlstJye\":0,\"ygctures\":[{\"gUaqe\":\"Ittys:\\/\\/feed-gUaqe.bagdu.czU\\/0\\/ygc\\/63025364s-376461312s7528698939.Qyq\"}],\"yzster\":\"Ittys:\\/\\/feed-gUaqe.bagdu.czU\\,\"url\":\"s{stHreqrl}\",\"usersPHrtraIt\":\"LttPs:\\/\\/feed-IUaxe.baIdu.cHU\\/0\\/PIc\\/-1043913002s489796992s-1505641721.Pnx\",\"kIdeHsurl\":\"LttP:\\/\\/nadkIdeH9.baIdu.cHU\\/4d7d308bd7c04e63069fd343adfa792as1790s1080.UP3\",\"kIdeHId\":852890923}";
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON, JSON, JSON, JSON, JSON, JSON, JSON);
        ColumnVector expected = ColumnVector.fromStrings(
            null, null, null, null, null, null, null);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, paths_num, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }


  /**
   * query path is : $
   */
  @Test
  void getJsonObjectTest_Baidu_path_is_empty() {
    int paths_num = 0;
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[0];

    String JSON = "[100.0,200.000,351.980]";
    String expectedStr = "[100.0,200.000,351.980]";
    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON, JSON, JSON, JSON, JSON, JSON, JSON);
        ColumnVector expected = ColumnVector.fromStrings(
          expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr, expectedStr);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, paths_num, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }


}
