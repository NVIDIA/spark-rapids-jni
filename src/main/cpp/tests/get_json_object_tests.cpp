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

// defined in json_parser_tests.cpp
void clear_buff(char buf[], std::size_t size);
void assert_start_with(char* buf, std::size_t buf_size, const std::string& prefix);

struct GetJsonObjectTests : public cudf::test::BaseFixture {};
using spark_rapids_jni::json_parser;
using spark_rapids_jni::path_instruction_type;
using spark_rapids_jni::detail::path_instruction;

spark_rapids_jni::json_parser<> get_parser(std::string const& json_str)
{
  return json_parser<>(json_str.data(), json_str.size());
}

spark_rapids_jni::detail::json_generator<> get_generator(char* buf)
{
  return spark_rapids_jni::detail::json_generator<>(buf);
}

spark_rapids_jni::detail::json_generator<> get_nullptr_generator()
{
  return spark_rapids_jni::detail::json_generator<>(nullptr);
}

bool eval_path(spark_rapids_jni::json_parser<>& p,
               spark_rapids_jni::detail::json_generator<>& g,
               spark_rapids_jni::detail::path_instruction const* path_ptr,
               int path_size)
{
  return spark_rapids_jni::detail::path_evaluator::evaluate_path(
    p, g, spark_rapids_jni::detail::write_style::raw_style, path_ptr, path_size);
}

path_instruction get_subscript_path() { return path_instruction(path_instruction_type::subscript); }

path_instruction get_wildcard_path() { return path_instruction(path_instruction_type::wildcard); }

path_instruction get_key_path() { return path_instruction(path_instruction_type::key); }

path_instruction get_index_path(int index)
{
  auto p  = path_instruction(path_instruction_type::index);
  p.index = index;
  return p;
}

path_instruction get_named_path(std::string name)
{
  auto p = path_instruction(path_instruction_type::named);
  p.name = cudf::string_view(name.data(), name.size());
  return p;
}

void test_get_json_object(std::string json,
                          std::vector<path_instruction> paths,
                          std::string expected)
{
  size_t buf_len = 100 * 1024;
  char buf[buf_len];
  clear_buff(buf, buf_len);

  auto p = get_parser(json);
  auto g = get_generator(buf);
  p.next_token();

  ASSERT_TRUE(eval_path(p, g, paths.data(), paths.size()));
  assert_start_with(buf, buf_len, expected);

  // the following checks generator output size without writes bytes
  clear_buff(buf, buf_len);
  auto p2 = get_parser(json);
  auto g2 = get_nullptr_generator();
  p2.next_token();

  ASSERT_TRUE(eval_path(p2, g2, paths.data(), paths.size()));
  ASSERT_EQ(g2.get_output_len(), expected.size());
}

void test_get_json_object_fail(std::string json, std::vector<path_instruction> paths)
{
  size_t buf_len = 100 * 1024;
  char buf[buf_len];
  clear_buff(buf, buf_len);

  auto p = get_parser(json);
  auto g = get_generator(buf);
  p.next_token();

  ASSERT_FALSE(eval_path(p, g, paths.data(), paths.size()));
}

void test_get_json_object(std::string json, std::string expected)
{
  size_t buf_len = 100 * 1024;
  char buf[buf_len];
  clear_buff(buf, buf_len);

  auto p = get_parser(json);
  auto g = get_generator(buf);
  p.next_token();

  ASSERT_TRUE(eval_path(p, g, nullptr, 0));
  assert_start_with(buf, buf_len, expected);
}

static const std::string json_for_test = R"(
{"store":{"fruit":[{"weight":8,"type":"apple"},{"weight":9,"type":"pear"}],
"basket":[[1,2,{"b":"y","a":"x"}],[3,4],[5,6]],"book":[{"author":"Nigel Rees",
"title":"Sayings of the Century","category":"reference","price":8.95},
{"author":"Herman Melville","title":"Moby Dick","category":"fiction","price":8.99,
"isbn":"0-553-21311-3"},{"author":"J. R. R. Tolkien","title":"The Lord of the Rings",
"category":"fiction","reader":[{"age":25,"name":"bob"},{"age":26,"name":"jack"}],
"price":22.99,"isbn":"0-395-19395-8"}],"bicycle":{"price":19.95,"color":"red"}},
"email":"amy@only_for_json_udf_test.net","owner":"amy","zip code":"94025",
"fb:testid":"1234"}
)";

/**
 * Tests from Spark JsonExpressionsSuite
 */
TEST_F(GetJsonObjectTests, NormalTest)
{
  test_get_json_object(" {  'k'  :  [1, [21, 22, 23], 3]   }  ",
                       std::vector<path_instruction>{get_key_path(), get_named_path("k")},
                       "[1,[21,22,23],3]");
  test_get_json_object(" {  'k'  :  [1, [21, 22, 23], 3]   }  ", R"({"k":[1,[21,22,23],3]})");
  test_get_json_object(
    " {  'k'  :  [1, [21, 22, 23], 3]   }  ",
    std::vector<path_instruction>{
      get_key_path(), get_named_path("k"), get_subscript_path(), get_wildcard_path()},
    R"([1,[21,22,23],3])");
  test_get_json_object(" {  'k'  :  [1, [21, 22, 23], 3]   }  ",
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("k"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_subscript_path(),
                                                     get_wildcard_path()},
                       R"([1,21,22,23,3])");
  test_get_json_object(" {  'k'  :  [1, [21, 22, 23], 3]   }  ",
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("k"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_subscript_path(),
                                                     get_index_path(0)},
                       R"(21)");
  test_get_json_object(
    " [[11,12,13], [21, 22, 23], [31, 32, 33]]  ",
    std::vector<path_instruction>{
      get_subscript_path(), get_wildcard_path(), get_subscript_path(), get_index_path(0)},
    R"([11,21,31])");
  test_get_json_object(
    " [[11,12,13]]  ",
    std::vector<path_instruction>{
      get_subscript_path(), get_wildcard_path(), get_subscript_path(), get_index_path(0)},
    R"(11)");

  test_get_json_object(
    " [[11,12,13]]  ",
    std::vector<path_instruction>{
      get_subscript_path(), get_wildcard_path(), get_subscript_path(), get_index_path(0)},
    R"(11)");

  // tests from Spark unit test cases
  test_get_json_object(
    json_for_test,
    std::vector<path_instruction>{
      get_key_path(), get_named_path("store"), get_key_path(), get_named_path("bicycle")},
    R"({"price":19.95,"color":"red"})");

  test_get_json_object(
    R"({ "key with spaces": "it works" })",
    std::vector<path_instruction>{get_key_path(), get_named_path("key with spaces")},
    R"(it works)");

  std::string e1 =
    R"([{"author":"Nigel Rees","title":"Sayings of the Century","category":"reference",)";
  e1 += R"("price":8.95},{"author":"Herman Melville","title":"Moby Dick","category":"fiction",)";
  e1 += R"("price":8.99,"isbn":"0-553-21311-3"},{"author":"J. R. R. Tolkien","title":)";
  e1 += R"("The Lord of the Rings","category":"fiction","reader":[{"age":25,"name":"bob"},)";
  e1 += R"({"age":26,"name":"jack"}],"price":22.99,"isbn":"0-395-19395-8"}])";

  test_get_json_object(
    json_for_test,
    std::vector<path_instruction>{
      get_key_path(), get_named_path("store"), get_key_path(), get_named_path("book")},
    e1);

  std::string e2 = R"({"author":"Nigel Rees","title":"Sayings of the Century",)";
  e2 += R"("category":"reference","price":8.95})";
  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_index_path(0)},
                       e2);

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_wildcard_path()},
                       e1);

  auto e3 = json_for_test;
  e3.erase(std::remove(e3.begin(), e3.end(), '\n'), e3.end());
  test_get_json_object(json_for_test, e3);

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_index_path(0),
                                                     get_key_path(),
                                                     get_named_path("category")},
                       "reference");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_key_path(),
                                                     get_named_path("category")},
                       R"(["reference","fiction","fiction"])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_key_path(),
                                                     get_named_path("isbn")},
                       R"(["0-553-21311-3","0-395-19395-8"])");

  // Fix https://github.com/NVIDIA/spark-rapids/issues/10216
  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_key_path(),
                                                     get_named_path("reader")},
                       R"([{"age":25,"name":"bob"},{"age":26,"name":"jack"}])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_index_path(0),
                                                     get_subscript_path(),
                                                     get_index_path(1)},
                       R"(2)");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_wildcard_path()},
                       R"([[1,2,{"b":"y","a":"x"}],[3,4],[5,6]])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_subscript_path(),
                                                     get_index_path(0)},
                       R"([1,3,5])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_index_path(0),
                                                     get_subscript_path(),
                                                     get_wildcard_path()},
                       R"([1,2,{"b":"y","a":"x"}])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_subscript_path(),
                                                     get_wildcard_path()},
                       R"([1,2,{"b":"y","a":"x"},3,4,5,6])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_index_path(0),
                                                     get_subscript_path(),
                                                     get_index_path(2),
                                                     get_key_path(),
                                                     get_named_path("b")},
                       R"(y)");

  // Fix https://github.com/NVIDIA/spark-rapids/issues/10217
  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_index_path(0),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_key_path(),
                                                     get_named_path("b")},
                       R"(["y"])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(), get_named_path("zip code")},
                       R"(94025)");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(), get_named_path("fb:testid")},
                       R"(1234)");

  test_get_json_object(
    R"({"a":"b\nc"})", std::vector<path_instruction>{get_key_path(), get_named_path("a")}, "b\nc");

  test_get_json_object(
    R"({"a":"b\"c"})", std::vector<path_instruction>{get_key_path(), get_named_path("a")}, "b\"c");

  test_get_json_object_fail(
    json_for_test, std::vector<path_instruction>{get_key_path(), get_named_path("non_exist_key")});

  test_get_json_object_fail(json_for_test,
                            std::vector<path_instruction>{get_key_path(),
                                                          get_named_path("store"),
                                                          get_key_path(),
                                                          get_named_path("book"),
                                                          get_subscript_path(),
                                                          get_index_path(10)});

  test_get_json_object_fail(json_for_test,
                            std::vector<path_instruction>{get_key_path(),
                                                          get_named_path("store"),
                                                          get_key_path(),
                                                          get_named_path("book"),
                                                          get_subscript_path(),
                                                          get_index_path(0),
                                                          get_key_path(),
                                                          get_named_path("non_exist_key")});

  test_get_json_object_fail(json_for_test,
                            std::vector<path_instruction>{get_key_path(),
                                                          get_named_path("store"),
                                                          get_key_path(),
                                                          get_named_path("basket"),
                                                          get_subscript_path(),
                                                          get_wildcard_path(),
                                                          get_key_path(),
                                                          get_named_path("non_exist_key")});

  std::string bad_json = "\u0000\u0000\u0000A\u0001AAA";
  test_get_json_object_fail(bad_json,
                            std::vector<path_instruction>{get_key_path(), get_named_path("a")});
}

/**
 * https://github.com/NVIDIA/spark-rapids/issues/10537
 */
TEST_F(GetJsonObjectTests, TestIssue_10537)
{
  test_get_json_object(
    R"({"'a":"v"})", std::vector<path_instruction>{get_key_path(), get_named_path("'a")}, "v");
}

/**
 * https://github.com/NVIDIA/spark-rapids/issues/10218
 */
TEST_F(GetJsonObjectTests, TestIssue_10218)
{
  test_get_json_object(R"({"a" : "A"})", R"({"a":"A"})");
  test_get_json_object(
    R"({'a' : 'A"'})", std::vector<path_instruction>{get_key_path(), get_named_path("a")}, "A\"");
  test_get_json_object(R"({'a' : 'A"'})", R"({"a":"A\""})");
  test_get_json_object(R"({"a" : "B\'"})", R"({"a":"B'"})");
  test_get_json_object(R"({"a" : "B'"})", R"({"a":"B'"})");
}

/**
 * https://github.com/NVIDIA/spark-rapids/issues/10196
 * one char '\t' and 2 chars '\\', 't' in field name both are one char '\t' after unescape.
 */
TEST_F(GetJsonObjectTests, TestIssue_10196)
{
  // filed name is 2 chars: 't', '\t'; path is 2 chars: 't', '\t'
  // because of allowing control char, '\t' char can be in the string directly without escape
  test_get_json_object("               { \"t\t\"  :   \"t\" }         ",
                       std::vector<path_instruction>{get_key_path(), get_named_path("t\t")},
                       "t");

  // filed name is 3 chars: 't', '\\', 't'; path is 2 chars: 't', '\t'
  // unescaped filed name is 2 chars: 't', '\t'
  test_get_json_object("               { \"t\\t\"  :   \"t\" }         ",
                       std::vector<path_instruction>{get_key_path(), get_named_path("t\t")},
                       "t");

  // filed name is 2 chars: 't', '\t'; path is 2 chars: 't', '\t'
  // because of allowing control char, '\t' char can be in the string directly without escape
  // According to conventional JSON format, '\t' char can be in string directly without escape
  test_get_json_object("               { \"t\t\"  :   \"t\" }         ",
                       std::vector<path_instruction>{get_key_path(), get_named_path("t\t")},
                       "t");

  // filed name is 3 chars: 't', '\\', 't'; path is 2 chars: 't', '\t'
  test_get_json_object("               { 't\\t'  :   't' }         ",
                       std::vector<path_instruction>{get_key_path(), get_named_path("t\t")},
                       "t");

  // path is 3 chars: 't', '\\', 't'
  test_get_json_object_fail("               { 't\\t'  :   't' }         ",
                            std::vector<path_instruction>{get_key_path(), get_named_path("t\\t")});
}

/**
 * https://github.com/NVIDIA/spark-rapids/issues/10194
 */
TEST_F(GetJsonObjectTests, TestIssue_10194)
{
  test_get_json_object_fail(R"(      {"url":"http://test.com",,}      )",
                            std::vector<path_instruction>{get_key_path(), get_named_path("url")});
}

/**
 * https://github.com/NVIDIA/spark-rapids/issues/9033
 */
TEST_F(GetJsonObjectTests, TestIssue_9033)
{
  test_get_json_object("      {\"A\": \"B\"}      ",
                       std::vector<path_instruction>{get_key_path(), get_named_path("A")},
                       "B");

  test_get_json_object("      {\"A\": \"B\nB\"}      ",
                       std::vector<path_instruction>{get_key_path(), get_named_path("A")},
                       "B\nB");

  test_get_json_object("      {\"A\": \"\\u7CFB\\u7D71\"}      ",
                       std::vector<path_instruction>{get_key_path(), get_named_path("A")},
                       "系統");

  test_get_json_object("      {\"A\": \"\\u7CFB\t\\u7D71\"}      ",
                       std::vector<path_instruction>{get_key_path(), get_named_path("A")},
                       "系\t統");
}

TEST_F(GetJsonObjectTests, Test_paths_depth_10)
{
  test_get_json_object(
    "{\"k1\":{\"k2\":{\"k3\":{\"k4\":{\"k5\":{\"k6\":{\"k7\":{\"k8\":{\"k9\":{\"k10\":\"v10\"}}}}}}"
    "}}}}",
    std::vector<path_instruction>{
      get_key_path(), get_named_path("k1"), get_key_path(), get_named_path("k2"),
      get_key_path(), get_named_path("k3"), get_key_path(), get_named_path("k4"),
      get_key_path(), get_named_path("k5"), get_key_path(), get_named_path("k6"),
      get_key_path(), get_named_path("k7"), get_key_path(), get_named_path("k8"),
      get_key_path(), get_named_path("k9"), get_key_path(), get_named_path("k10")},
    "v10");
}
