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

#include <json_parser.hpp>

struct JsonParserTests : public cudf::test::BaseFixture {};
using spark_rapids_jni::json_parser;
using spark_rapids_jni::json_parser_options;
using spark_rapids_jni::json_token;

template <int max_json_depth = 128>
std::vector<json_token> parse(std::string json_str,
                              bool single_quote,
                              bool control_char,
                              bool allow_tailing = true,
                              int max_string_len = 20000000,
                              int max_num_len    = 1000)
{
  json_parser_options options;
  options.set_allow_single_quotes(single_quote);
  options.set_allow_unescaped_control_chars(control_char);
  options.set_allow_tailing_sub_string(allow_tailing);
  options.set_max_string_len(max_string_len);
  options.set_max_num_len(max_num_len);
  json_parser<max_json_depth> parser(options, json_str.data(), json_str.size());
  std::vector<json_token> tokens;
  json_token token = parser.next_token();
  tokens.push_back(token);
  while (token != json_token::ERROR && token != json_token::SUCCESS) {
    token = parser.next_token();
    tokens.push_back(token);
  }
  return tokens;
}

void test_basic(bool allow_single_quote, bool allow_control_char)
{
  std::vector<std::pair<std::string, std::vector<json_token>>> cases = {
    std::make_pair(
      // test terminal number
      std::string{"  \r\n\t  \r\n\t  1   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_NUMBER_INT, json_token::SUCCESS}),
    std::make_pair(
      // test terminal float
      std::string{"  \r\n\t  \r\n\t  1.5   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_NUMBER_FLOAT, json_token::SUCCESS}),
    std::make_pair(
      // test terminal string
      std::string{"  \r\n\t  \r\n\t  \"abc\"   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_STRING, json_token::SUCCESS}),
    std::make_pair(
      // test terminal true
      std::string{"  \r\n\t  \r\n\t  true   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_TRUE, json_token::SUCCESS}),
    std::make_pair(
      // test terminal false
      std::string{"  \r\n\t  \r\n\t  false   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_FALSE, json_token::SUCCESS}),
    std::make_pair(
      // test terminal null
      std::string{"  \r\n\t  \r\n\t  null   \r\n\t  \r\n\t "},
      std::vector{json_token::VALUE_NULL, json_token::SUCCESS}),

    std::make_pair(
      // test numbers
      std::string{R"(
            [
              0, 102, -0, -102, 0.3, -0.3000, 1e-050, -1e-5, 1.0e-5, -1.0010e-050, 1E+5, 1e0, 1E0, 1.3e5, -1e01, 1e00000
            ]
          )"},
      std::vector{json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::END_ARRAY,
                  json_token::SUCCESS}),
    std::make_pair(
      // test string
      std::string{"\"美国,中国\\u12f3\\u113E---abc---\\\", \\/, \\\\, \\b, "
                  "\\f, \\n, \\r, \\t\""},
      std::vector{json_token::VALUE_STRING, json_token::SUCCESS}),
    std::make_pair(
      // test empty object
      std::string{"  {   }   "},
      std::vector{json_token::START_OBJECT, json_token::END_OBJECT, json_token::SUCCESS}),
    std::make_pair(
      // test empty array
      std::string{"   [   ]   "},
      std::vector{json_token::START_ARRAY, json_token::END_ARRAY, json_token::SUCCESS}),
    std::make_pair(
      // test nesting arrays
      std::string{R"(
            [
              1 ,
              [
                2 ,
                [
                  3 ,
                  [
                    41 , 42 , 43
                  ]
                ]
              ]
            ]
          )"},
      std::vector{json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::END_ARRAY,
                  json_token::END_ARRAY,
                  json_token::END_ARRAY,
                  json_token::END_ARRAY,
                  json_token::SUCCESS}),
    std::make_pair(
      // test nesting objects
      std::string{R"(
            {
              "k1" : "v1" ,
              "k2" : {
                "k3" : {
                  "k4" : {
                    "k51" : "v51" ,
                    "k52" : "v52"
                  }
                }
              }
            }
          )"},
      std::vector{json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_STRING,
                  json_token::FIELD_NAME,
                  json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_STRING,
                  json_token::FIELD_NAME,
                  json_token::VALUE_STRING,
                  json_token::END_OBJECT,
                  json_token::END_OBJECT,
                  json_token::END_OBJECT,
                  json_token::END_OBJECT,
                  json_token::SUCCESS}),
    std::make_pair(
      // test nesting objects and arrays
      std::string{R"(
            {
              "k1" : "v1",
              "k2" : [
                1, {
                  "k21" : "v21",
                  "k22" : [1 , 2 , -1.5]
                }
              ]
            }
          )"},
      std::vector{json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_STRING,
                  json_token::FIELD_NAME,
                  json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_STRING,
                  json_token::FIELD_NAME,
                  json_token::START_ARRAY,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_INT,
                  json_token::VALUE_NUMBER_FLOAT,
                  json_token::END_ARRAY,
                  json_token::END_OBJECT,
                  json_token::END_ARRAY,
                  json_token::END_OBJECT,
                  json_token::SUCCESS}),

    std::make_pair(
      // test invalid string: should have 4 HEX
      std::string{"\"  \\uFFF  \""},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid string: invalid HEX 'T'
      std::string{"\"  \\uTFFF  \""},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid string: unclosed string
      std::string{"  \"abc   "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid string: have no char after escape char '\'
      std::string{"\"\\"},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid string:  \X is not allowed
      std::string{"\" \\X   \""},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid num
      std::string{" +5 "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid num
      std::string{" 1.  "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid num
      std::string{" 1e  "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid num
      std::string{" 1e-  "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid num
      std::string{" infinity  "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{" {"},
      std::vector{json_token::START_OBJECT, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{" ["},
      std::vector{json_token::START_ARRAY, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{" {1} "},
      std::vector{json_token::START_OBJECT, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{R"(
        {"k",}
      )"},
      std::vector{json_token::START_OBJECT, json_token::FIELD_NAME, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{R"(
        {"k": }
      )"},
      std::vector{json_token::START_OBJECT, json_token::FIELD_NAME, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{R"(
        {"k": 1 :}
      )"},
      std::vector{json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_NUMBER_INT,
                  json_token::ERROR}),

    std::make_pair(
      // test invalid structure
      std::string{R"(
        {"k": 1 , }
      )"},
      std::vector{json_token::START_OBJECT,
                  json_token::FIELD_NAME,
                  json_token::VALUE_NUMBER_INT,
                  json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{R"(
        [ 1 :
      )"},
      std::vector{json_token::START_ARRAY, json_token::VALUE_NUMBER_INT, json_token::ERROR}),
    std::make_pair(
      // test invalid structure
      std::string{R"(
        [ 1,
      )"},
      std::vector{json_token::START_ARRAY, json_token::VALUE_NUMBER_INT, json_token::ERROR}),
    std::make_pair(
      // test invalid null
      std::string{" nul "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid false
      std::string{" fals "},
      std::vector{json_token::ERROR}),
    std::make_pair(
      // test invalid true
      std::string{" tru "},
      std::vector{json_token::ERROR}),

  };
  for (std::size_t i = 0; i < cases.size(); ++i) {
    std::string json_str                    = cases[i].first;
    std::vector<json_token> expected_tokens = cases[i].second;
    std::vector<json_token> actual_tokens = parse(json_str, allow_single_quote, allow_control_char);
    ASSERT_EQ(actual_tokens, expected_tokens);
  }
}

void test_len_limitation()
{
  std::vector<std::string> v;
  v.push_back("  '123456'        ");
  v.push_back("  'k\n\\'\\\"56'  ");  // do not count escape char '\', actual
                                      // has 6 chars: k \n ' " 5 6
  v.push_back("  123456          ");
  v.push_back("  -1.23e-456      ");

  auto error_token = std::vector<json_token>{json_token::ERROR};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,  //  bool single_quote,
                                                  true,  // control_char
                                                  true,  // allow_tailing
                                                  5,     // max_string_len
                                                  5);    // max_num_len
    // exceed num/str length limits
    ASSERT_EQ(actual_tokens, error_token);
  }

  v.clear();
  v.push_back("   '12345'           ");
  v.push_back("   'k\n\\'\\\"5'     ");  // do not count escape char '\',
                                         // has 5 chars: k \n ' " 5
  auto expect_str_ret = std::vector<json_token>{json_token::VALUE_STRING, json_token::SUCCESS};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,  // bool single_quote,
                                                  true,  // control_char
                                                  true,  // allow_tailing
                                                  5,     // max_string_len
                                                  5);    // max_num_len
    ASSERT_EQ(actual_tokens, expect_str_ret);
  }

  v.clear();
  v.push_back("    12345            ");
  v.push_back("    -1.23e-45        ");
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,   // bool single_quote,
                                                  false,  // control_char
                                                  true,   // allow_tailing
                                                  5,      // max_string_len
                                                  5);     // max_num_len
    ASSERT_EQ(actual_tokens[1], json_token::SUCCESS);
  }
}

void test_single_double_quote()
{
  std::vector<std::string> v;
  // allow \'  \" " in single quote
  v.push_back("'    \\\'     \\\"      \"          '");
  // allow \'  \"  ' in double quote
  v.push_back("\"   \\\' \\\"   '    \'      \"");  // C++ allow \' to represent
                                                    // ' in string
  auto expect_ret = std::vector<json_token>{json_token::VALUE_STRING, json_token::SUCCESS};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,  //  bool single_quote,
                                                  false  // control_char
    );
    ASSERT_EQ(actual_tokens, expect_ret);
  }

  v.clear();
  v.push_back("\"     \\'      \"");  // not allow \' when single_quote is disabled
  expect_ret = std::vector<json_token>{json_token::ERROR};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  false,  //  bool single_quote,
                                                  true    // control_char
    );

    ASSERT_EQ(actual_tokens, expect_ret);
  }

  v.clear();
  v.push_back("\"     '   \\\"      \"");  // allow ' \" in double quote
  expect_ret = std::vector<json_token>{json_token::VALUE_STRING, json_token::SUCCESS};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  false,  //  bool single_quote,
                                                  true    // control_char
    );
    ASSERT_EQ(actual_tokens, expect_ret);
  }

  v.clear();
  v.push_back("      'str'      ");  // ' is not allowed to quote string
  expect_ret = std::vector<json_token>{json_token::ERROR};
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  false,  //  bool single_quote,
                                                  true    // control_char
    );
    ASSERT_EQ(actual_tokens, expect_ret);
  }
}

void test_max_nested_len()
{
  std::vector<std::string> v;
  v.push_back("[[[[[]]]]]");
  v.push_back("{'k1':{'k2':{'k3':{'k4':{'k5': 5}}}}}");
  for (std::size_t i = 0; i < v.size(); ++i) {
    // set max nested len template value as 5
    std::vector<json_token> actual_tokens = parse<5>(v[i],
                                                     true,  //  bool single_quote,
                                                     true   // control_char
    );
    ASSERT_EQ(actual_tokens[actual_tokens.size() - 1], json_token::SUCCESS);
  }

  v.clear();
  v.push_back("[[[[[[]]]]]]");
  v.push_back("{'k1':{'k2':{'k3':{'k4':{'k5': {'k6': 6}}}}}}");
  for (std::size_t i = 0; i < v.size(); ++i) {
    // set max nested len template value as 5
    std::vector<json_token> actual_tokens = parse<5>(v[i],
                                                     true,  //  bool single_quote,
                                                     false  // control_char
    );
    ASSERT_EQ(actual_tokens[actual_tokens.size() - 1], json_token::ERROR);
  }
}

void test_control_char()
{
  std::vector<std::string> v;
  v.push_back("'   \t   \n   \b '");  // \t \n \b are control chars
  for (std::size_t i = 0; i < v.size(); ++i) {
    // set max nested len template value as 5
    std::vector<json_token> actual_tokens = parse<5>(v[i],
                                                     true,  //  bool single_quote,
                                                     true   // control_char
    );
    ASSERT_EQ(actual_tokens[actual_tokens.size() - 1], json_token::SUCCESS);
  }

  for (std::size_t i = 0; i < v.size(); ++i) {
    // set max nested len template value as 5
    std::vector<json_token> actual_tokens = parse<5>(v[i],
                                                     true,  //  bool single_quote,
                                                     false  // control_char
    );
    ASSERT_EQ(actual_tokens[actual_tokens.size() - 1], json_token::ERROR);
  }
}

void test_allow_tailing_useless_chars()
{
  std::vector<std::string> v;
  v.push_back("  0xxxx        ");  // 0 is valid JSON, tailing xxxx is ignored
                                   // when allow tailing
  v.push_back("  {}xxxx  ");       // tailing xxxx is ignored
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,  //  bool single_quote,
                                                  true,  // control_char
                                                  true   // allow_tailing is true
    );
    ASSERT_TRUE(actual_tokens.size() > 0);
    ASSERT_EQ(actual_tokens[actual_tokens.size() - 1], json_token::SUCCESS);
  }
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,  //  bool single_quote,
                                                  true,  // control_char
                                                  false  // allow_tailing is false
    );
    ASSERT_TRUE(actual_tokens.size() > 0);
    ASSERT_EQ(actual_tokens[actual_tokens.size() - 1], json_token::ERROR);
  }

  v.clear();
  v.push_back("    12345xxxxxx            ");
  v.push_back("    -1.23e-45xxxxx   ");
  for (std::size_t i = 0; i < v.size(); ++i) {
    std::vector<json_token> actual_tokens = parse(v[i],
                                                  true,   //  bool single_quote,
                                                  false,  // control_char
                                                  true,   // allow_tailing
                                                  5,      // max_string_len
                                                  5);     // max_num_len
    ASSERT_TRUE(actual_tokens.size() > 0);
    ASSERT_EQ(actual_tokens[actual_tokens.size() - 1], json_token::SUCCESS);
  }
}

void test_is_valid()
{
  std::string json_str = " {    \"k\"   :     [1,2,3]}   ";
  json_parser_options options;
  json_parser<10> parser1(options, json_str.data(), json_str.size());
  ASSERT_TRUE(parser1.is_valid());

  json_str = " {[1,2,    ";
  json_parser<10> parser2(options, json_str.data(), json_str.size());
  ASSERT_FALSE(parser2.is_valid());
}

TEST_F(JsonParserTests, NormalTest)
{
  test_basic(/*single_quote*/ true, /*control_char*/ true);
  test_basic(/*single_quote*/ true, /*control_char*/ false);
  test_basic(/*single_quote*/ false, /*control_char*/ true);
  test_basic(/*single_quote*/ false, /*control_char*/ false);
  test_len_limitation();
  test_single_double_quote();
  test_max_nested_len();
  test_control_char();
  test_allow_tailing_useless_chars();
  test_is_valid();
}

template <int max_json_depth = 128>
std::unique_ptr<json_parser<max_json_depth>> get_parser(std::string const& json_str,
                                                        bool single_quote,
                                                        bool control_char,
                                                        bool allow_tailing = true,
                                                        int max_string_len = 20000000,
                                                        int max_num_len    = 1000)
{
  json_parser_options options;
  options.set_allow_single_quotes(single_quote);
  options.set_allow_unescaped_control_chars(control_char);
  options.set_allow_tailing_sub_string(allow_tailing);
  options.set_max_string_len(max_string_len);
  options.set_max_num_len(max_num_len);
  return std::make_unique<json_parser<max_json_depth>>(options, json_str.data(), json_str.size());
}

TEST_F(JsonParserTests, SkipChildrenForObject)
{
  // test skip for the first {
  std::string json = " { 'k1' : 'v1' , 'k2' : { 'k3' : { 'k4' : 'v5' }  }  } ";
  auto parser      = *get_parser(json, /*single_quote*/ true, /*control_char*/ true);
  // can not skip for INIT token
  ASSERT_FALSE(parser.try_skip_children());
  ASSERT_EQ(json_token::START_OBJECT, parser.next_token());
  // test skip for tokens: {
  ASSERT_TRUE(parser.try_skip_children());
  ASSERT_EQ(json_token::END_OBJECT, parser.get_current_token());
  ASSERT_EQ(json_token::SUCCESS, parser.next_token());
  // can not skip for SUCCESS token
  ASSERT_FALSE(parser.try_skip_children());

  // test skip for tokens: not [ {
  parser.reset();
  ASSERT_EQ(json_token::START_OBJECT, parser.next_token());
  ASSERT_EQ(json_token::FIELD_NAME, parser.next_token());
  ASSERT_TRUE(parser.try_skip_children());
  ASSERT_EQ(json_token::FIELD_NAME, parser.get_current_token());
}

TEST_F(JsonParserTests, SkipChildrenForArray)
{
  // skip for [
  std::string json = " [ [ [ [ 1, 2, 3 ] ] ] ] ";
  auto parser      = *get_parser(json, /*single_quote*/ true, /*control_char*/ true);
  ASSERT_FALSE(parser.try_skip_children());
  ASSERT_EQ(json_token::START_ARRAY, parser.next_token());
  ASSERT_EQ(json_token::START_ARRAY, parser.next_token());
  ASSERT_TRUE(parser.try_skip_children());
  ASSERT_EQ(json_token::END_ARRAY, parser.get_current_token());
  ASSERT_EQ(json_token::END_ARRAY, parser.next_token());
  ASSERT_EQ(json_token::SUCCESS, parser.next_token());
  // can not skip for SUCCESS token
  ASSERT_FALSE(parser.try_skip_children());

  parser.reset();
  // can not skip for INIT token
  ASSERT_FALSE(parser.try_skip_children());
}

TEST_F(JsonParserTests, SkipChildrenInvalid)
{
  std::string json = " invalid ";
  auto parser      = *get_parser(json, /*single_quote*/ true, /*control_char*/ true);
  parser.next_token();
  ASSERT_EQ(json_token::ERROR, parser.get_current_token());
  // can not skip for ERROR token
  ASSERT_FALSE(parser.try_skip_children());
}

void clear_buff(char buf[], std::size_t size) { memset(buf, 0, size); }

void assert_start_with(char* buf, std::size_t buf_size, const std::string& prefix)
{
  std::string str(buf, buf_size);
  ASSERT_EQ(0, str.find(prefix));
  for (std::size_t i = prefix.size(); i < str.size(); i++) {
    ASSERT_EQ('\0', str[i]);
  }
}

TEST_F(JsonParserTests, CopyRawStringText)
{
  constexpr std::size_t buf_size = 256;
  char buf[buf_size];

  std::string json = " {  'key123'  :  'value123' } ";
  auto parser      = *get_parser(json, /*single_quote*/ true, /*control_char*/ true);

  ASSERT_EQ(json_token::START_OBJECT, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(1, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "{");

  ASSERT_EQ(json_token::FIELD_NAME, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(6, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "key123");

  ASSERT_EQ(json_token::VALUE_STRING, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(8, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "value123");

  ASSERT_EQ(json_token::END_OBJECT, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(1, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "}");
}

TEST_F(JsonParserTests, CopyRawNumberText)
{
  constexpr std::size_t buf_size = 256;
  char buf[buf_size];

  std::string json = " [  -12345 ,  -1.23e-000123 , true , false , null  ] ";
  auto parser      = *get_parser(json, /*single_quote*/ true, /*control_char*/ true);

  ASSERT_EQ(json_token::START_ARRAY, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(1, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "[");

  ASSERT_EQ(json_token::VALUE_NUMBER_INT, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(6, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "-12345");

  ASSERT_EQ(json_token::VALUE_NUMBER_FLOAT, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(13, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "-1.23e-000123");

  ASSERT_EQ(json_token::VALUE_TRUE, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(4, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "true");

  ASSERT_EQ(json_token::VALUE_FALSE, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(5, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "false");

  ASSERT_EQ(json_token::VALUE_NULL, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(4, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "null");

  ASSERT_EQ(json_token::END_ARRAY, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(1, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "]");

  ASSERT_EQ(json_token::SUCCESS, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(0, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "");
}

TEST_F(JsonParserTests, CopyRawTextInvalid)
{
  constexpr std::size_t buf_size = 256;
  char buf[buf_size];

  std::string json = " invalid ";
  auto parser      = *get_parser(json, /*single_quote*/ true, /*control_char*/ true);

  ASSERT_EQ(json_token::INIT, parser.get_current_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(0, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "");

  ASSERT_EQ(json_token::ERROR, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(0, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "");
}

TEST_F(JsonParserTests, CopyRawTextEscape)
{
  constexpr std::size_t buf_size = 256;
  char buf[buf_size];
  // test escape: \", \', \\, \/, \b, \f, \n, \r, \t
  std::string json = "   '\\\"\\'\\\\\\/\\b\\f\\n\\r\\t\\b'   ";
  auto parser      = *get_parser(json, /*single_quote*/ true, /*control_char*/ true);
  // avoid unused-but-set-variable compile warnning
  parser.reset();

  ASSERT_EQ(json_token::VALUE_STRING, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(10, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "\"\'\\/\b\f\n\r\t\b");
}

TEST_F(JsonParserTests, CopyRawTextUnicode)
{
  // "中国".getBytes(StandardCharsets.UTF_8) is:
  // Array(-28, -72, -83, -27, -101, -67)
  constexpr std::size_t buf_size = 256;
  char buf[buf_size];
  auto json   = "   '\\u4e2d\\u56FD'   ";  // Represents 中国
  auto parser = *get_parser(json, /*single_quote*/ true, /*control_char*/ true);

  ASSERT_EQ(json_token::VALUE_STRING, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(6, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "中国");
}

TEST_F(JsonParserTests, CopyRawTextOther)
{
  constexpr std::size_t buf_size = 256;
  char buf[buf_size];
  auto json   = "   '中国'   ";
  auto parser = *get_parser(json, /*single_quote*/ true, /*control_char*/ true);

  ASSERT_EQ(json_token::VALUE_STRING, parser.next_token());
  clear_buff(buf, buf_size);
  ASSERT_EQ(6, parser.copy_raw_text(buf));
  assert_start_with(buf, buf_size, "中国");
}

void assert_ptr_len(char const* actaul_ptr,
                    cudf::size_type actual_len,
                    char* expected_ptr,
                    cudf::size_type expected_len)
{
  ASSERT_EQ(expected_ptr, actaul_ptr);
  ASSERT_EQ(expected_len, actual_len);
}

TEST_F(JsonParserTests, GetNumberText)
{
  std::string json = "[-12.45e056,123456789]  ";
  auto parser      = *get_parser(json, /*single_quote*/ true, /*control_char*/ true);

  ASSERT_EQ(json_token::INIT, parser.get_current_token());
  ASSERT_EQ(json_token::START_ARRAY, parser.next_token());

  ASSERT_EQ(json_token::VALUE_NUMBER_FLOAT, parser.next_token());
  auto [ptr1, len1] = parser.get_current_number_text();
  assert_ptr_len(ptr1, len1, json.data() + 1, 10);

  ASSERT_EQ(json_token::VALUE_NUMBER_INT, parser.next_token());
  auto [ptr2, len2] = parser.get_current_number_text();
  assert_ptr_len(ptr2, len2, json.data() + 12, 9);
}

void assert_float_parts(bool float_sign,
                        char const* float_integer_pos,
                        int float_integer_len,
                        char const* float_fraction_pos,
                        int float_fraction_len,
                        char const* float_exp_pos,
                        int float_exp_len,
                        bool actual_float_sign,
                        char const* actual_float_integer_pos,
                        int actual_float_integer_len,
                        char const* actual_float_fraction_pos,
                        int actual_float_fraction_len,
                        char const* actual_float_exp_pos,
                        int actual_float_exp_len)
{
  ASSERT_EQ(float_sign, actual_float_sign);
  ASSERT_EQ(float_integer_pos, actual_float_integer_pos);
  ASSERT_EQ(float_integer_len, actual_float_integer_len);
  ASSERT_EQ(float_fraction_pos, actual_float_fraction_pos);
  ASSERT_EQ(float_fraction_len, actual_float_fraction_len);
  ASSERT_EQ(float_exp_pos, actual_float_exp_pos);
  ASSERT_EQ(float_exp_len, actual_float_exp_len);
}

TEST_F(JsonParserTests, GetFloatParts)
{
  // int part is 123, fraction part is 0345, exp part is -05678
  std::string json = "[-123.0345e-05678]  ";
  auto parser      = *get_parser(json, /*single_quote*/ true, /*control_char*/ true);

  ASSERT_EQ(json_token::INIT, parser.get_current_token());
  ASSERT_EQ(json_token::START_ARRAY, parser.next_token());

  ASSERT_EQ(json_token::VALUE_NUMBER_FLOAT, parser.next_token());
  auto parts = parser.get_current_float_parts();
  assert_float_parts(false,
                     json.data() + 2,
                     3,
                     json.data() + 6,
                     4,
                     json.data() + 11,
                     6,
                     thrust::get<0>(parts),
                     thrust::get<1>(parts),
                     thrust::get<2>(parts),
                     thrust::get<3>(parts),
                     thrust::get<4>(parts),
                     thrust::get<5>(parts),
                     thrust::get<6>(parts));
}
