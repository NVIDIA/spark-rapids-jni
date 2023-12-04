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

#include <parse_uri.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

struct ParseURIProtocolTests : public cudf::test::BaseFixture {};
struct ParseURIHostTests : public cudf::test::BaseFixture {};

enum class test_types {
  SIMPLE,
  SPARK_EDGES,
  IPv6,
  IPv4,
  UTF8,
};

namespace {
cudf::test::strings_column_wrapper get_test_data(test_types t)
{
  switch (t) {
    case test_types::SIMPLE:
      return cudf::test::strings_column_wrapper({
        "https://www.nvidia.com/s/uri?param1=2",
        "http://www.nvidia.com",
        "file://path/to/a/cool/file",
        "smb://network/path/to/file",
        "http:/www.nvidia.com",
        "file:path/to/a/cool/file",
        "/network/path/to/file",
        "nvidia.com",
        "www.nvidia.com/s/uri",
      });

    case test_types::SPARK_EDGES:
      return cudf::test::strings_column_wrapper(
        {"https://nvidia.com/https&#://nvidia.com",
         "https://http://www.nvidia.com",
         "filesystemmagicthing://bob.yaml",
         "nvidia.com:8080",
         "http://thisisinvalid.data/due/to-the_character%s/inside*the#url`~",
         "file:/absolute/path",
         "//www.nvidia.com",
         "#bob",
         "#this%doesnt#make//sense://to/me",
         "HTTP:&bob",
         "/absolute/path",
         "http://%77%77%77.%4EV%49%44%49%41.com",
         "https:://broken.url",
         "https://www.nvidia.com/q/This%20is%20a%20query",
         "https://www.nvidia.com/\x93path/path/to/file",
         "http://?",
         "http://??",
         "http://\?\?/",
         "http://#",
         "http://user:pass@host/file;param?query;p2",
         "http://[1:2:3:4:5:6:7::]",
         "http://[::2:3:4:5:6:7:8]",
         "http://[fe80::7:8%eth0]",
         "http://[fe80::7:8%1]",
         "http://foo.bar/abc/\\\\\\http://foo.bar/abc.gif\\\\\\",
         "www.nvidia.com:8100/servlet/"
         "impc.DisplayCredits?primekey_in=2000041100:05:14115240636",
         "https://nvidia.com/2Ru15Ss ",
         "http://www.nvidia.com/plugins//##",
         "www.nvidia.com:81/Free.fr/L7D9qw9X4S-aC0&amp;D4X0/Panels&amp;solutionId=0X54a/"
         "cCdyncharset=UTF-8&amp;t=01wx58Tab&amp;ps=solution/"
         "ccmd=_help&amp;locale0X1&amp;countrycode=MA/",
         "http://www.nvidia.com/tags.php?%2F88\323\351\300ึณวน\331\315\370%2F",
         "http://www.nvidia.com//wp-admin/includes/index.html#9389#123",
         "http://www.nvidia.com/"
         "object.php?object=ะก-\320%9Fะฑ-ะฟ-ะก\321%82\321%80ะตะป\321%8Cะฝะฐ-\321%83ะป-\320%"
         "97ะฐะฒะพะด\321%81ะบะฐ\321%8F.html&sid=5",
         "http://www.nvidia.com/picshow.asp?id=106&mnid=5080&classname=\271\253ืฐฦช",
         "http://-.~_!$&'()*+,;=:%40:80%2f::::::@nvidia.com:443",
         "http://userid:password@example.com:8080/"});
    case test_types::IPv6:
      return cudf::test::strings_column_wrapper({
        "https://[fe80::]",
        "https://[2001:0db8:85a3:0000:0000:8a2e:0370:7334]",
        "https://[2001:0DB8:85A3:0000:0000:8A2E:0370:7334]",
        "https://[2001:db8::1:0]",
        "http://[2001:db8::2:1]",
        "https://[::1]",
        "https://[2001:db8:85a3:8d3:1319:8a2e:370:7348]:443",
        "https://[2001:db8:3333:4444:5555:6666:1.2.3.4]/path/to/file",
        "https://[2001:db8:3333:4444:5555:6666:7777:8888:1.2.3.4]/path/to/file",
        "https://[::db8:3333:4444:5555:6666:1.2.3.4]/path/to/file]",  // this is valid, but spark
                                                                      // doesn't think so
      });
    case test_types::IPv4:
      return cudf::test::strings_column_wrapper({
        "https://192.168.1.100/",
        "https://192.168.1.100:8443/",
        "https://192.168.1.100.5/",
        "https://192.168.1/",
        "https://280.100.1.1/",
        "https://182.168..100/path/to/file",
      });
    case test_types::UTF8:
      return cudf::test::strings_column_wrapper({
        "https://nvidia.com/%4EV%49%44%49%41",
        "http://%77%77%77.%4EV%49%44%49%41.com",
        "http://✪↩d⁚f„⁈.ws/123",
        "https:// /path/to/file",
      });
    default: CUDF_FAIL("Test type unsupported!"); return cudf::test::strings_column_wrapper();
  }
}
}  // namespace

TEST_F(ParseURIProtocolTests, Simple)
{
  auto col    = get_test_data(test_types::SIMPLE);
  auto result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected(
    {"https", "http", "file", "smb", "http", "file", "", "", ""}, {1, 1, 1, 1, 1, 1, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(ParseURIProtocolTests, SparkEdges)
{
  auto col    = get_test_data(test_types::SPARK_EDGES);
  auto result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected({"https",
                                               "https",
                                               "filesystemmagicthing",
                                               "nvidia.com",
                                               "",
                                               "file",
                                               "",
                                               "",
                                               "",
                                               "HTTP",
                                               "",
                                               "http",
                                               "https",
                                               "https",
                                               "",
                                               "http",
                                               "http",
                                               "http",
                                               "http",
                                               "http",
                                               "http",
                                               "http",
                                               "http",
                                               "http",
                                               "",
                                               "www.nvidia.com",
                                               "",
                                               "",
                                               "www.nvidia.com",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "http",
                                               "http"},
                                              {1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1,
                                               1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(ParseURIProtocolTests, IP6)
{
  auto col    = get_test_data(test_types::IPv6);
  auto result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected(
    {"https", "https", "https", "https", "http", "https", "https", "https", "", ""},
    {1, 1, 1, 1, 1, 1, 1, 1, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(ParseURIProtocolTests, IP4)
{
  auto col    = get_test_data(test_types::IPv4);
  auto result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected(
    {"https", "https", "https", "https", "https", "https"});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(ParseURIProtocolTests, UTF8)
{
  auto col    = get_test_data(test_types::UTF8);
  auto result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected({"https", "http", "http", ""}, {1, 1, 1, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(ParseURIHostTests, Simple)
{
  auto col    = get_test_data(test_types::SIMPLE);
  auto result = spark_rapids_jni::parse_uri_to_host(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected(
    {"www.nvidia.com", "www.nvidia.com", "path", "network", "", "", "", "", ""},
    {1, 1, 1, 1, 0, 0, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(ParseURIHostTests, SparkEdges)
{
  auto col    = get_test_data(test_types::SPARK_EDGES);
  auto result = spark_rapids_jni::parse_uri_to_host(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected({"nvidia.com",
                                               "http",
                                               "bob.yaml",
                                               "",
                                               "",
                                               "",
                                               "www.nvidia.com",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "www.nvidia.com",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "host",
                                               "[1:2:3:4:5:6:7::]",
                                               "[::2:3:4:5:6:7:8]",
                                               "[fe80::7:8%eth0]",
                                               "[fe80::7:8%1]",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "",
                                               "nvidia.com",
                                               "example.com"},
                                              {1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                               0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(ParseURIHostTests, IP6)
{
  auto col    = get_test_data(test_types::IPv6);
  auto result = spark_rapids_jni::parse_uri_to_host(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected({"[fe80::]",
                                               "[2001:0db8:85a3:0000:0000:8a2e:0370:7334]",
                                               "[2001:0DB8:85A3:0000:0000:8A2E:0370:7334]",
                                               "[2001:db8::1:0]",
                                               "[2001:db8::2:1]",
                                               "[::1]",
                                               "[2001:db8:85a3:8d3:1319:8a2e:370:7348]",
                                               "[2001:db8:3333:4444:5555:6666:1.2.3.4]",
                                               "",
                                               ""},
                                              {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(ParseURIHostTests, IP4)
{
  auto col    = get_test_data(test_types::IPv4);
  auto result = spark_rapids_jni::parse_uri_to_host(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected({"192.168.1.100", "192.168.1.100", "", "", "", ""},
                                              {1, 1, 0, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(ParseURIHostTests, UTF8)
{
  auto col    = get_test_data(test_types::UTF8);
  auto result = spark_rapids_jni::parse_uri_to_host(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected({"nvidia.com", "", "", ""}, {1, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}
