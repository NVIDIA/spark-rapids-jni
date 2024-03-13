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

#include <cudf_test/base_fixture.hpp>

#include <get_json_object.hpp>

struct JsonParserTests : public cudf::test::BaseFixture {};
using spark_rapids_jni::json_parser;
using spark_rapids_jni::json_parser_options;
using spark_rapids_jni::json_token;
using spark_rapids_jni::detail::json_generator;

template <int max_json_depth = 128>
std::string getJsonString(json_generator<max_json_depth> g)
{
  if (g.get_output_start_position() != nullptr) {
    return std::string(g.get_output_start_position(), g.get_output_len());
  } else {
    return "";
  }
}

TEST_F(JsonParserTests, TestGenerator1)
{
  json_parser_options options;
  options.set_allow_single_quotes(true);
  options.set_allow_unescaped_control_chars(true);
  options.set_allow_tailing_sub_string(true);
  options.set_max_string_len(20000000);
  options.set_max_num_len(1000);
  std::string json_str = R"({"key" : 100, "key2" : "value", "key3" : true, "key4" : 1.2333 })";
  json_parser<128> parser(options, json_str.data(), json_str.size());
  int max_output_len = 1000000;
  char* const output = new char[max_output_len];
  json_generator<128> generator(output, 0);
  generator.copy_current_structure(parser);

  std::cout << "FINAL Output:" << std::endl;
  std::cout << getJsonString(generator) << std::endl;
}

TEST_F(JsonParserTests, TestGenerator2)
{
  json_parser_options options;
  options.set_allow_single_quotes(true);
  options.set_allow_unescaped_control_chars(true);
  options.set_allow_tailing_sub_string(true);
  options.set_max_string_len(20000000);
  options.set_max_num_len(1000);
  std::string json_str = R"({
        "person": {
            "name": "John Doe",
            "age": 30,
            "address": {
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zip_code": "10001"
            },
            "phone_numbers": [
                {
                    "type": "home",
                    "number": "555-1234"
                },
                {
                    "type": "work",
                    "number": "555-5678"
                }
            ],
            "friends": [],
            "tags": ["friend", "colleague", "classmate"],
            "notes": ""
        },
        "empty_object": {},
        "empty_array": []
    })";
  json_parser<128> parser(options, json_str.data(), json_str.size());
  int max_output_len = 1000000;
  char* const output = new char[max_output_len];
  json_generator<128> generator(output, 0);
  generator.copy_current_structure(parser);

  std::cout << "FINAL Output:" << std::endl;
  std::cout << getJsonString(generator) << std::endl;
}