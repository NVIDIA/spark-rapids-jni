/*
* Copyright (c)  2024, NVIDIA CORPORATION.
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

import ai.rapids.cudf.GroupByAggregation;
import ai.rapids.cudf.Table;

import org.junit.jupiter.api.Test;


public class HLLPPTest {

  @Test
  void testGroupByHLL() {
    // A trivial test:
    try (Table input = new Table.TestBuilder().column(1, 2, 3, 1, 2, 2, 1, 3, 3, 2)
             .column(0, 1, -2, 3, -4, -5, -6, 7, -8, 9)
             .build()){
        input.groupBy(0).aggregate(GroupByAggregation.HLLPP(0)
               .onColumn(1));
    }
  }
}
