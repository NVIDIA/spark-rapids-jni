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

#include "from_json.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <rmm/exec_policy.hpp>

class FromJsonTest : public cudf::test::BaseFixture {};

TEST_F(FromJsonTest, T1)
{
  // The last row is invalid (has an extra quote).
  auto const json_string =
    cudf::test::strings_column_wrapper{"{'a': [{'b': 1, 'c': 2}, {'b': 3, 'c': 4}]}"};

  spark_rapids_jni::json_schema_element a{cudf::data_type{cudf::type_id::LIST}, {}};
  a.child_types.emplace_back(
    "struct", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::STRUCT}, {}});
  a.child_types.front().second.child_types.emplace_back(
    "b", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::INT32}, {}});
  a.child_types.front().second.child_types.emplace_back(
    "c", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::INT32}, {}});

  std::vector<std::pair<std::string, spark_rapids_jni::json_schema_element>> schema;
  schema.emplace_back("a", std::move(a));

  auto const output = spark_rapids_jni::from_json_to_structs(
    cudf::strings_column_view{json_string}, schema, false, false);

  printf("\n\ninput: \n");
  cudf::test::print(json_string);

  printf("\n\noutput: \n");
  for (auto const& col : output) {
    cudf::test::print(col->view());
  }
}

TEST_F(FromJsonTest, T2)
{
  // The last row is invalid (has an extra quote).
  auto const json_string =
    cudf::test::strings_column_wrapper{"{'a': [{'b': \"1\", 'c': 2}, {'b': \"3\", 'c': 4}]}"};

  spark_rapids_jni::json_schema_element a{cudf::data_type{cudf::type_id::LIST}, {}};
  a.child_types.emplace_back(
    "struct", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::STRUCT}, {}});
  a.child_types.front().second.child_types.emplace_back(
    "b", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::INT32}, {}});
  a.child_types.front().second.child_types.emplace_back(
    "c", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::INT32}, {}});

  std::vector<std::pair<std::string, spark_rapids_jni::json_schema_element>> schema;
  schema.emplace_back("a", std::move(a));

  auto const output = spark_rapids_jni::from_json_to_structs(
    cudf::strings_column_view{json_string}, schema, false, false);

  printf("\n\ninput: \n");
  cudf::test::print(json_string);

  printf("\n\noutput: \n");
  for (auto const& col : output) {
    cudf::test::print(col->view());
  }
}
