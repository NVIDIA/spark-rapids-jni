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

import ai.rapids.cudf.ColumnVector;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class MapUtilsTest {

  @Test
  void testFromJsonSimpleInput() {
    String jsonString1 = "{\"Zipcode\" : 704 , \"ZipCodeType\" : \"STANDARD\" , \"City\" : \"PARC" +
        " PARQUE\" , \"State\" : \"PR\"}";
    String jsonString2 = "{\"Integer\":12345,\"String\":\"ABCXYZ\",\"Double\":1.1245}";
    String jsonString3 = "{\"category\": \"reference\", \"index\": [4,{},null,{\"a\":[{ }, {}] } " +
        "], \"author\": \"Nigel Rees\", \"title\": \"{}[], <=semantic-symbols-string\", " +
        "\"price\": 8.95}";

    try (ColumnVector input =
             ColumnVector.fromStrings(jsonString1, jsonString2, jsonString3);
         ColumnVector outputMap = MapUtils.extractRawMapFromJsonString(input);

         ColumnVector expectedKeys = ColumnVector.fromStrings("Zipcode", "ZipCodeType", "City",
             "State", "Integer", "String", "Double", "category", "index", "author", "title",
             "price");
         ColumnVector expectedValues = ColumnVector.fromStrings("704", "STANDARD", "PARC PARQUE",
             "PR", "12345", "ABCXYZ", "1.1245", "reference", "[4,{},null,{\"a\":[{ }, {}] } ]",
             "Nigel Rees", "{}[], <=semantic-symbols-string", "8.95");
         ColumnVector expectedStructs = ColumnVector.makeStruct(expectedKeys, expectedValues);
         ColumnVector expectedOffsets = ColumnVector.fromInts(0, 4, 7, 12);
         ColumnVector expectedMap = expectedStructs.makeListFromOffsets(3, expectedOffsets)
    ) {
      assertColumnsAreEqual(expectedMap, outputMap);
    }
  }

}
