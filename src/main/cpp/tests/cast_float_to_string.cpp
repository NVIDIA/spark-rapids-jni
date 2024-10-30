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
  std::string json_string = R"({"student": [{"name": "abc", "class": "junior"}]})";

  {
    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{json_string.data(), json_string.size()})
        .prune_columns(true)
        .mixed_types_as_string(true)
        .lines(true);

    cudf::io::schema_element dtype_schema{cudf::data_type{cudf::type_id::STRUCT},
                                          {
                                            {"student",
                                             {data_type{cudf::type_id::LIST},
                                              {{"element",
                                                {data_type{cudf::type_id::STRUCT},
                                                 {
                                                   {"name", {data_type{cudf::type_id::STRING}}},
                                                   {"abc", {data_type{cudf::type_id::STRING}}},
                                                   {"class", {data_type{cudf::type_id::STRING}}},
                                                 },
                                                 {{"name", "abc", "class"}}}}}}},
                                          },
                                          {{"student"}}};
    in_options.set_dtypes(dtype_schema);

    auto const parsed_table_with_meta = cudf::io::read_json(in_options);
    // auto const& parsed_meta           = parsed_table_with_meta.metadata;
    auto parsed_columns = parsed_table_with_meta.tbl->release();
    for (auto& col : parsed_columns) {
      cudf::test::print(*col);
    }
  }

  {
    /*
     * colname:
student,
element,
name,
abc,
class,
num child:
1,
3,
0,
0,
0,
num child:
1,
3,
0,
0,
0,
types:
24,
28,
23,
23,
23,

     */

    std::vector<std::string> col_names{"student", "element", "name", "abc", "class"};
    std::vector<int> num_children{1, 3, 0, 0, 0};
    std::vector<int> types{24, 28, 23, 23, 23};
    std::vector<int> scales{0, 0, 0, 0, 0};
    std::vector<int> precisions{-1, -1, -1, -1, -1};

    auto const input = cudf::test::strings_column_wrapper{json_string};
    auto out         = spark_rapids_jni::from_json_to_structs(cudf::strings_column_view{input},
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
