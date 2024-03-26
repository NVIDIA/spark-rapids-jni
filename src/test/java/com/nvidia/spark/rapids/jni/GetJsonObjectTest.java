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
   * test escape chars: " in ' pair; ' in " pair
   */
  @Test
  void getJsonObjectTest_Escape() {
    int paths_num = 0;
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[0];

    String JSON1 = "{ \"a\": \"A\" }";
    String JSON2 = "{'a':'A\"'}";
    String JSON3 = "{'a':\"B'\"}";
    String JSON4 = "['a','b','\"C\"']";
    // \\u4e2d\\u56FD is 中国
    String JSON5 = "'\\u4e2d\\u56FD\\\"\\'\\\\\\/\\b\\f\\n\\r\\t\\b'";

    String expectedStr1 = "{\"a\":\"A\"}";
    String expectedStr2 = "{\"a\":\"A\\\"\"}";
    String expectedStr3 = "{\"a\":\"B'\"}";
    String expectedStr4 = "[\"a\",\"b\",\"\\\"C\\\"\"]";
    String expectedStr5 = "中国\"'\\/\b\f\n\r\t\b";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON1, JSON2, JSON3, JSON4, JSON5);
        ColumnVector expected = ColumnVector.fromStrings(
          expectedStr1, expectedStr2, expectedStr3, expectedStr4, expectedStr5);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, paths_num, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * test number normalizations
   */
  @Test
  void getJsonObjectTest_Number_Normalization() {
    int paths_num = 0;
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[0];

    String JSON1 = "[100.0,200.000,351.980]";
    String JSON2 = "[12345678900000000000.0]";
    String JSON3 = "[0.0]";
    String JSON4 = "[-0.0]";
    String JSON5 = "[-0]";
    String JSON6 = "[12345678999999999999999999]";
    String JSON7 = "[1E308]";
    String JSON8 = "[1.0E309,-1E309,1E5000]";

    String expectedStr1 = "[100.0,200.0,351.98]";
    String expectedStr2 = "[1.23456789E19]";
    String expectedStr3 = "[0.0]";
    String expectedStr4 = "[-0.0]";
    String expectedStr5 = "[0]";
    String expectedStr6 = "[12345678999999999999999999]";
    String expectedStr7 = "[1.0E308]";
    String expectedStr8 = "[\"Infinity\",\"-Infinity\",\"Infinity\"]";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(
            JSON1, JSON2, JSON3, JSON4, JSON5, JSON6, JSON7, JSON8);
        ColumnVector expected = ColumnVector.fromStrings(
          expectedStr1, expectedStr2, expectedStr3, expectedStr4, expectedStr5, expectedStr6, expectedStr7, expectedStr8);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, paths_num, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  /**
   * case (VALUE_STRING, Nil) if style == RawStyle
   */
  @Test
  void getJsonObjectTest_Test_case_path1() {
    int paths_num = 0;
    JSONUtils.PathInstructionJni[] query = new JSONUtils.PathInstructionJni[0];

    String JSON1 = "'abc'";
    String expectedStr1 = "abc";

    try (
        ColumnVector jsonCv = ColumnVector.fromStrings(JSON1);
        ColumnVector expected = ColumnVector.fromStrings(expectedStr1);
        ColumnVector actual = JSONUtils.getJsonObject(jsonCv, paths_num, query)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

}
