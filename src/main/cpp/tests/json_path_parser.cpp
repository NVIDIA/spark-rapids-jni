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
using spark_rapids_jni::detail::JsonPathParser;
using spark_rapids_jni::detail::path_instruction;
using spark_rapids_jni::detail::path_instruction_type;

JsonPathParser parser;

std::string serialize(std::string s)
{
  auto instructions = parser.parse(s);
  if (!instructions) { return "Invalid path"; }
  std::string result = "";
  for (const auto& instruction : *instructions) {
    switch (instruction.type) {
      case path_instruction_type::SUBSCRIPT: result += "SUBSCRIPT "; break;
      case path_instruction_type::WILDCARD: result += "WILDCARD "; break;
      case path_instruction_type::KEY: result += "KEY "; break;
      case path_instruction_type::INDEX:
        result += "INDEX: " + std::to_string(instruction.index) + " ";
        break;
      case path_instruction_type::NAMED:
        auto name = instruction.name.data();
        auto len  = size_t(instruction.name.size_bytes());
        auto str  = std::string(name, len);
        result += "NAMED: " + str + " ";
        break;
    }
  }
  return result;
}

std::vector<std::pair<std::string, std::string>> test_cases = {
  {"$.a", "KEY NAMED: a "},
  {"$.owner", "KEY NAMED: owner "},
  {"$."
   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
   "aaaa",
   "KEY NAMED: "
   "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
   "aaaa "},
  {"$.", "Invalid path"},
  {"$.store.fruit[0]", "KEY NAMED: store KEY NAMED: fruit SUBSCRIPT INDEX: 0 "},
  {"$.store.fruit[0].type",
   "KEY NAMED: store KEY NAMED: fruit SUBSCRIPT INDEX: 0 KEY NAMED: type "},
  {"$.store.fruit[-1].color", "Invalid path"},
  {"$['a']", "KEY NAMED: a "},
  {"$['a']['b']", "KEY NAMED: a KEY NAMED: b "},
  {"$['a']['b']['c']['d']['e']['f']['g']['h']['i']['j']['k']['l']['m']['n']['o']['p']['q']['r']['s'"
   "]['t']['u']['v']['w']['x']['y']['z']",
   "KEY NAMED: a KEY NAMED: b KEY NAMED: c KEY NAMED: d KEY NAMED: e KEY NAMED: f KEY NAMED: g KEY "
   "NAMED: h KEY NAMED: i KEY NAMED: j KEY NAMED: k KEY NAMED: l KEY NAMED: m KEY NAMED: n KEY "
   "NAMED: o KEY NAMED: p KEY NAMED: q KEY NAMED: r KEY NAMED: s KEY NAMED: t KEY NAMED: u KEY "
   "NAMED: v KEY NAMED: w KEY NAMED: x KEY NAMED: y KEY NAMED: z "},
  {"$.store.basket[0][2].b",
   "KEY NAMED: store KEY NAMED: basket SUBSCRIPT INDEX: 0 SUBSCRIPT INDEX: 2 KEY NAMED: b "},
  {"$.store.basket[0][-1].c", "Invalid path"},
  {"$.store.basket[0][-1].c[0]", "Invalid path"},
  {"$", ""},
  {"$['?']", "Invalid path"},
  {"$$", "Invalid path"},
  {"$[?]", "Invalid path"},
  {"$[?(@.length-1)]", "Invalid path"},
  {"$.     a", "KEY NAMED: a "},
  {"$.     o   .   b . c", "KEY NAMED: o    KEY NAMED: b  KEY NAMED: c "},
  {"$.     o   .   b . c   [   0   ]", "Invalid path"},
  {"$[   'a'   ]", "Invalid path"},
  {"$['    a      ']['     b']['c     ']", "KEY NAMED: a       KEY NAMED: b KEY NAMED: c      "},
  {"$['a.b.c']", "KEY NAMED: a.b.c "},
  {"$['a.b.c'][0]", "KEY NAMED: a.b.c SUBSCRIPT INDEX: 0 "},
  {"$[[*]]", "Invalid path"},
  {"$['[*]']", "KEY NAMED: [*] "},
  {"$[0[*]", "Invalid path"},
  {"$a.aa", "Invalid path"},
  {"$?.a", "Invalid path"},
  {"$??", "Invalid path"},
  {"$['?']", "Invalid path"},
  {"$['aa?aa']", "Invalid path"},
  {"$['a'][2000000000000000000]", "KEY NAMED: a SUBSCRIPT INDEX: 2000000000000000000 "},
  {"#['a']", "Invalid path"},
  {"$[*][*][*][*][*][*][*][*][*][*][*][*][*][*][*]",
   "SUBSCRIPT WILDCARD SUBSCRIPT WILDCARD SUBSCRIPT WILDCARD SUBSCRIPT WILDCARD SUBSCRIPT WILDCARD "
   "SUBSCRIPT WILDCARD SUBSCRIPT WILDCARD SUBSCRIPT WILDCARD SUBSCRIPT WILDCARD SUBSCRIPT WILDCARD "
   "SUBSCRIPT WILDCARD SUBSCRIPT WILDCARD SUBSCRIPT WILDCARD SUBSCRIPT WILDCARD SUBSCRIPT "
   "WILDCARD "},
  {"$.*", "WILDCARD "},
  {"$['*']['*']['*']['*']['*']['*']['*']['*']['*']['*']['*']['*']['*']['*']",
   "WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD "
   "WILDCARD WILDCARD WILDCARD WILDCARD "},
  {"$.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*",
   "WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD "
   "WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD WILDCARD "
   "WILDCARD WILDCARD WILDCARD "},
  {"$..[a]", "Invalid path"},
  {"$..a", "Invalid path"},
  {"$['']", "Invalid path"},
  {"$[''].a", "Invalid path"},
  {"$.  *", "KEY NAMED: * "},
  {"$[  *  ]", "Invalid path"},
  {"$.??['a']", "KEY NAMED: ?? KEY NAMED: a "},
  {"$[']']", "KEY NAMED: ] "},
  {"$.]", "KEY NAMED: ] "},
  {"$['[1]']", "KEY NAMED: [1] "},
  {"$['1]']", "KEY NAMED: 1] "},
  {"$.*'a'", "Invalid path"},
  {"$[*'a']", "Invalid path"},
  {"$[*1]", "Invalid path"},
  {"$'a'", "Invalid path"},
  {"$.[*]", "Invalid path"},
  {"$[*]aa", "Invalid path"},
  {"$[*]123", "Invalid path"},
  {"$']", "Invalid path"},
  {"$.'a']", "KEY NAMED: 'a'] "},
  {"$['aa*aa']", "KEY NAMED: aa*aa "},
  {"$.aa*aa", "KEY NAMED: aa*aa "},
  {"$.2]", "KEY NAMED: 2] "},
  {"", "Invalid path"},
  {"$.a2]??", "KEY NAMED: a2]?? "},
  {"$.*a", "Invalid path"},
  {"$['*a']", "KEY NAMED: *a "},
  {"$.11]", "KEY NAMED: 11] "},
  {"$.1]", "KEY NAMED: 1] "},
  {"$.[1]", "Invalid path"},
  {"$[1]", "SUBSCRIPT INDEX: 1 "},
  {"$'*'", "Invalid path"},
  {"$['.']", "KEY NAMED: . "},
  {"$['.a']", "KEY NAMED: .a "},
  {"$['12']", "KEY NAMED: 12 "},
  {"$. .a", "Invalid path"},
  {"$. a .", "Invalid path"},
  {"$. ", "Invalid path"},
  {"$.'*'", "KEY NAMED: '*' "},
  {"$.系统", "KEY NAMED: 系统 "},
  {"$['系统']", "KEY NAMED: 系统 "},
  {"$['系统'][2]", "KEY NAMED: 系统 SUBSCRIPT INDEX: 2 "},
  {"$'a''b'", "Invalid path"}};

TEST_F(JsonParserTests, TestJsonPathParser)
{
  for (const auto& test_case : test_cases) {
    std::string path     = test_case.first;
    std::string expected = test_case.second;
    std::string result   = serialize(path);
    ASSERT_EQ(result, expected);
  }
}