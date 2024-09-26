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

TEST_F(FromJsonTest, T3)
{
  // The last row is invalid (has an extra quote).
  auto const json_string = cudf::test::strings_column_wrapper{"{'data': [1,0]}"};

  spark_rapids_jni::json_schema_element a{cudf::data_type{cudf::type_id::STRUCT}, {}};
  a.child_types.emplace_back(
    "b", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::INT32}, {}});
  a.child_types.emplace_back(
    "c", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::INT32}, {}});

  std::vector<std::pair<std::string, spark_rapids_jni::json_schema_element>> schema;
  schema.emplace_back("data", std::move(a));

  auto const output = spark_rapids_jni::from_json_to_structs(
    cudf::strings_column_view{json_string}, schema, false, false);

  printf("\n\ninput: \n");
  cudf::test::print(json_string);

  printf("\n\noutput: \n");
  for (auto const& col : output) {
    cudf::test::print(col->view());
  }
}

TEST_F(FromJsonTest, T4)
{
  // The last row is invalid (has an extra quote).
  auto const json_string = cudf::test::strings_column_wrapper{"{'data': ['1', '2']}"};

  spark_rapids_jni::json_schema_element a{cudf::data_type{cudf::type_id::LIST}, {}};
  a.child_types.emplace_back(
    "string", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::STRING}, {}});

  std::vector<std::pair<std::string, spark_rapids_jni::json_schema_element>> schema;
  schema.emplace_back("data", std::move(a));

  auto const output = spark_rapids_jni::from_json_to_structs(
    cudf::strings_column_view{json_string}, schema, false, false);

  printf("\n\ninput: \n");
  cudf::test::print(json_string);

  printf("\n\noutput: \n");
  for (auto const& col : output) {
    cudf::test::print(col->view());
  }
}

#include <fstream>
#include <streambuf>
#include <string>

TEST_F(FromJsonTest, T5)
{
  // clang-format off
// {"BEACAHEBBO":{"GPECEDGF":"Az[M`Q.'mn`","MFCEINDHFNPJE":"FsZ!/*!){O5>M","OCIKAF":"FsZ!/*!)","GPIHMJ":"|i2l\\J)u8I*Z|TBG$Ho%t","JHG":"B]0r@jN&\"pvP=X}/##H8sRZCc?G [u\".T(FuW@bq2#AgS,S& gqy.emb3?!MfP8Vb.1*eW.WyK)7DF8b.\"","BJKAPMIHEGA":"Az[M","OFEIBPMAEIBALDDD":"FsZ!/*!"},"CGEGPD":[{"JD":"\">z\"'","GMFDD":"y:Mb`Efozq2","NHKPJLNJBJ":"Az[M`","BCCOEEALBP":"2Jn.","CJKIKCGA":"j8(9Sf)7wetOhXt{N%=y-Xu!k ijVfcNKQ+RX)Y}!!ezc)#6i!GX?Z~LvIpI.h/DBt`7y`mu*W6v6*K#8Aw\\.`\\(4G4","OPHLHN":"FsZ!/"}]}
  // clang-format on
  std::ifstream t("/home/nghiat/Devel/data/from_json_array_issue.json");
  std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
  std::cout << "input:\n" << str << std::endl;

  // The last row is invalid (has an extra quote).
  auto const json_string =
    cudf::test::strings_column_wrapper{std::initializer_list<std::string>{str}};
  spark_rapids_jni::json_schema_element a{cudf::data_type{cudf::type_id::LIST}, {}};

  a.child_types.emplace_back(
    "struct", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::STRUCT}, {}});
  a.child_types.front().second.child_types.emplace_back(
    "KMEJHDA", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::STRING}, {}});
  a.child_types.front().second.child_types.emplace_back(
    "CJKIKCGA", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::STRING}, {}});

  std::vector<std::pair<std::string, spark_rapids_jni::json_schema_element>> schema;
  schema.emplace_back("CGEGPD", std::move(a));

  auto const output = spark_rapids_jni::from_json_to_structs(
    cudf::strings_column_view{json_string}, schema, false, false);

  printf("\n\ninput: \n");
  cudf::test::print(json_string);

  printf("\n\noutput: \n");
  for (auto const& col : output) {
    cudf::test::print(col->view());
  }
}

TEST_F(FromJsonTest, T6)
{
  auto const json_string =
    cudf::test::strings_column_wrapper{"{'data':[{'a':1}, {'a':2, 'b':3}, {'b':4}]}"};
  spark_rapids_jni::json_schema_element a{cudf::data_type{cudf::type_id::LIST}, {}};

  a.child_types.emplace_back(
    "struct", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::STRUCT}, {}});
  a.child_types.front().second.child_types.emplace_back(
    "a", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::INT32}, {}});
  a.child_types.front().second.child_types.emplace_back(
    "b", spark_rapids_jni::json_schema_element{cudf::data_type{cudf::type_id::INT32}, {}});

  std::vector<std::pair<std::string, spark_rapids_jni::json_schema_element>> schema;
  schema.emplace_back("data", std::move(a));

  auto const output = spark_rapids_jni::from_json_to_structs(
    cudf::strings_column_view{json_string}, schema, false, false);

  printf("\n\ninput: \n");
  cudf::test::print(json_string);

  printf("\n\noutput: \n");
  for (auto const& col : output) {
    cudf::test::print(col->view());
  }
}
