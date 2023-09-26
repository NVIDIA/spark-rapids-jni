/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
import ai.rapids.cudf.BinaryOp;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class MapUtilsTest {

  @Test
  void testFromJsonSimpleInput() {
    String jsonString1 = "{\"Zipcode\" : 704 , \"ZipCodeType\" : \"STANDARD\" , \"City\" : \"PARC" +
        " PARQUE\" , \"State\" : \"PR\"}";
    String jsonString2 = "{}";
    String jsonString3 = "{\"category\": \"reference\", \"index\": [4,{},null,{\"a\":[{ }, {}] } " +
        "], \"author\": \"Nigel Rees\", \"title\": \"{}[], <=semantic-symbols-string\", " +
        "\"price\": 8.95}";

    try (ColumnVector input =
             ColumnVector.fromStrings(jsonString1, jsonString2, null, jsonString3);
         ColumnVector outputMap = MapUtils.extractRawMapFromJsonString(input);

         ColumnVector expectedKeys = ColumnVector.fromStrings("Zipcode", "ZipCodeType", "City",
             "State", "category", "index", "author", "title", "price");
         ColumnVector expectedValues = ColumnVector.fromStrings("704", "STANDARD", "PARC PARQUE",
             "PR", "reference", "[4,{},null,{\"a\":[{ }, {}] } ]", "Nigel Rees", "{}[], " +
                 "<=semantic-symbols-string", "8.95");
         ColumnVector expectedStructs = ColumnVector.makeStruct(expectedKeys, expectedValues);
         ColumnVector expectedOffsets = ColumnVector.fromInts(0, 4, 4, 4, 9);
         ColumnVector tmpMap = expectedStructs.makeListFromOffsets(4, expectedOffsets);
         ColumnVector templateBitmask = ColumnVector.fromBoxedInts(1, 1, null, 1);
         ColumnVector expectedMap = tmpMap.mergeAndSetValidity(BinaryOp.BITWISE_AND,
             templateBitmask);
    ) {
      assertColumnsAreEqual(expectedMap, outputMap);
    }
  }

  @Test
  void testFromJsonWithUTF8() {
    String jsonString1 = "{\"Zipc\u00f3de\" : 704 , \"Z\u00edpCodeTyp\u00e9\" : \"STANDARD\" ," +
        " \"City\" : \"PARC PARQUE\" , \"St\u00e2te\" : \"PR\"}";
    String jsonString2 = "{}";
    String jsonString3 = "{\"Zipc\u00f3de\" : 704 , \"Z\u00edpCodeTyp\u00e9\" : " +
        "\"\uD867\uDE3D\" , " + "\"City\" : \"\uD83C\uDFF3\" , \"St\u00e2te\" : " +
        "\"\uD83C\uDFF3\"}";

    try (ColumnVector input =
             ColumnVector.fromStrings(jsonString1, jsonString2, null, jsonString3);
         ColumnVector outputMap = MapUtils.extractRawMapFromJsonString(input);

         ColumnVector expectedKeys = ColumnVector.fromStrings("Zipc\u00f3de", "Z\u00edpCodeTyp" +
                 "\u00e9", "City", "St\u00e2te", "Zipc\u00f3de", "Z\u00edpCodeTyp\u00e9",
             "City", "St\u00e2te");
         ColumnVector expectedValues = ColumnVector.fromStrings("704", "STANDARD", "PARC PARQUE",
             "PR", "704", "\uD867\uDE3D", "\uD83C\uDFF3", "\uD83C\uDFF3");
         ColumnVector expectedStructs = ColumnVector.makeStruct(expectedKeys, expectedValues);
         ColumnVector expectedOffsets = ColumnVector.fromInts(0, 4, 4, 4, 8);
         ColumnVector tmpMap = expectedStructs.makeListFromOffsets(4, expectedOffsets);
         ColumnVector templateBitmask = ColumnVector.fromBoxedInts(1, 1, null, 1);
         ColumnVector expectedMap = tmpMap.mergeAndSetValidity(BinaryOp.BITWISE_AND,
             templateBitmask);
    ) {
      assertColumnsAreEqual(expectedMap, outputMap);
    }
  }

  @Test
  void testFromJsonInvalidRows() {
    String jsonString1 = "{\"Zipcode\" : 704 , \"ZipCodeType\" : \"STANDARD\" , \"City\" : \"PARC" +
            " PARQUE\" , \"State\" : \"PR\"}";
    String jsonString2 = "{ \"Zipcode\": 90210"; // intentionally incomplete json
    String jsonString3 = "{\"category\": \"reference\", \"index\": [4,{},null,{\"a\":[{ }, {}] } " +
            "], \"author\": \"Nigel Rees\", \"title\": \"{}[], <=semantic-symbols-string\", " +
            "\"price\": 8.95}";

    try (ColumnVector input =
                 ColumnVector.fromStrings(jsonString1, jsonString2, null, jsonString3);
         ColumnVector outputMap = MapUtils.extractRawMapFromJsonString(input);

         ColumnVector expectedKeys = ColumnVector.fromStrings("Zipcode", "ZipCodeType", "City",
                 "State", "category", "index", "author", "title", "price");
         ColumnVector expectedValues = ColumnVector.fromStrings("704", "STANDARD", "PARC PARQUE",
                 "PR", "reference", "[4,{},null,{\"a\":[{ }, {}] } ]", "Nigel Rees", "{}[], " +
                         "<=semantic-symbols-string", "8.95");
         ColumnVector expectedStructs = ColumnVector.makeStruct(expectedKeys, expectedValues);
         ColumnVector expectedOffsets = ColumnVector.fromInts(0, 4, 4, 4, 9);
         ColumnVector tmpMap = expectedStructs.makeListFromOffsets(4, expectedOffsets);
         ColumnVector templateBitmask = ColumnVector.fromBoxedInts(1, 1, null, 1);
         ColumnVector expectedMap = tmpMap.mergeAndSetValidity(BinaryOp.BITWISE_AND,
                 templateBitmask);
    ) {
      assertColumnsAreEqual(expectedMap, outputMap);
    }
  }

}
