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

TEST_F(ParseURIProtocolTests, Simple)
{
  cudf::test::strings_column_wrapper col({
    "https://www.google.com/s/uri?param1=2",
    "http://www.yahoo.com",
    "file://path/to/a/cool/file",
    "smb://network/path/to/file",
    "http:/www.yahoo.com",
    "file:path/to/a/cool/file",
  });
  auto result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected({"https", "http", "file", "smb", "http", "file"});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(ParseURIProtocolTests, Negatives)
{
  cudf::test::strings_column_wrapper col({
    "https//www.google.com/s/uri?param1=2",
    "/network/path/to/file",
    "yahoo.com",
    "www.google.com/s/uri",
  });
  auto result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected({"", "", "", ""}, {0, 0, 0, 0});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(ParseURIProtocolTests, SparkEdges)
{
  cudf::test::strings_column_wrapper col(
    {"https://google.com/https&#://google.com",
     "https://http://www.google.com",
     "filesystemmagicthing://bob.yaml",
     "google.com:8080",
     "http://thisisinvalid.data/due/to-the_character%s/inside*the#url`~",
     "file:/absolute/path",
     "//www.google.com",
     "#bob",
     "#this%doesnt#make//sendse://to/me",
     "HTTP:&bob",
     "/absolute/path",
     "http://%77%77%77.%67o%6Fgle.com",
     "https:://broken.url",
     "https://www.google.com/q/This%20is%20a%20query"});

  auto result = spark_rapids_jni::parse_uri_to_protocol(cudf::strings_column_view{col});

  cudf::test::strings_column_wrapper expected({"https",
                                               "https",
                                               "filesystemmagicthing",
                                               "google.com",
                                               "",
                                               "file",
                                               "",
                                               "",
                                               "",
                                               "HTTP",
                                               "",
                                               "http",
                                               "https",
                                               "https"},
                                              {1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}