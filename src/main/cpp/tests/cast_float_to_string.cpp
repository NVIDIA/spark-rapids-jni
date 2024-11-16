/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <cudf/io/json.hpp>

#include <rmm/device_uvector.hpp>

#include <json_utils.hpp>

using namespace cudf;

struct FloatToStringTests : public cudf::test::BaseFixture {};

TEST_F(FloatToStringTests, FromFloats32)
{
  cudf::test::strings_column_wrapper input{R"({"c2": [19]})"};

  {
    /*
     * names: (size = 4)
"c2", "element", "c3", "c4",
num children:
1, 2, 0, 0,
types:
24, 28, 4, 23,
list, struct, int64, string
scales:
0, 0, 0, 0,
     */
    std::vector<std::string> col_names{"c2", "element", "c3", "c4"};
    std::vector<int> num_children{1, 2, 0, 0};
    std::vector<int> types{24, 28, 4, 23};
    std::vector<int> scales{0, 0, 0, 0};
    std::vector<int> precisions{-1, -1, -1, -1};

    auto out = spark_rapids_jni::from_json_to_structs(cudf::strings_column_view{input},
                                                      col_names,
                                                      num_children,
                                                      types,
                                                      scales,
                                                      precisions,
                                                      true,
                                                      true,
                                                      true,
                                                      true,
                                                      true);
    cudf::test::print(*out);
  }
}
