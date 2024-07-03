/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <string>
#include <vector>

struct SubstringIndexTest : public::test::BaseFixture {};

TEST_F(StringsSliceTest, Error)
{
  cudf::test::strings_column_wrapper strings{"this string intentionally left blank"};
  auto strings_view = cudf::strings_column_view(strings);
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, 0, 0, 0), cudf::logic_error);

  auto delim_col = cudf::test::strings_column_wrapper({"", ""});
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, cudf::strings_column_view{delim_col}, -1),
               cudf::logic_error);

  auto indexes = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2});
  EXPECT_THROW(cudf::strings::slice_strings(strings_view, indexes, indexes), cudf::logic_error);

@@ -299,16 +295,10 @@ TEST_F(StringsSliceTest, ZeroSizeStringsColumn)
  auto results = cudf::strings::slice_strings(strings_view, 1, 2);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("foo"), 1);
  cudf::test::expect_column_empty(results->view());

  cudf::column_view starts_column(cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0);
  cudf::column_view stops_column(cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0);
  results = cudf::strings::slice_strings(strings_view, starts_column, stops_column);
  cudf::test::expect_column_empty(results->view());

  results = cudf::strings::slice_strings(strings_view, strings_view, 1);
  cudf::test::expect_column_empty(results->view());
}


TEST_F(StringsSliceTest, AllEmpty)
@@ -317,250 +307,8 @@ TEST_F(StringsSliceTest, AllEmpty)
  auto strings_view = cudf::strings_column_view(strings_col);
  auto exp_results  = cudf::column_view(strings_col);

  auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("e"), -1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  results = cudf::strings::slice_strings(strings_view, strings_view, -1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
}

TEST_F(StringsSliceTest, EmptyDelimiter)
{
  auto strings_col = cudf::test::strings_column_wrapper(
    {"Héllo", "thesé", "", "lease", "tést strings", ""}, {true, true, false, true, true, true});
  ;
  auto strings_view = cudf::strings_column_view(strings_col);

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                        {true, true, false, true, true, true});

  auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar(""), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);

  auto delim_col = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                      {true, false, true, false, true, false});

  results = cudf::strings::slice_strings(strings_view, cudf::strings_column_view{delim_col}, 1);
  auto results = cudf::strings::slice_strings(strings_view, 0, -1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
}

TEST_F(StringsSliceTest, ZeroCount)
{
  auto strings_col = cudf::test::strings_column_wrapper(
    {"Héllo", "thesé", "", "lease", "tést strings", ""}, {true, true, false, true, true, true});
  ;
  auto strings_view = cudf::strings_column_view(strings_col);

  auto exp_results = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                        {true, true, false, true, true, true});

  auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("é"), 0);
  results = cudf::strings::slice_strings(strings_view, 0, -1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);

  auto delim_col = cudf::test::strings_column_wrapper({"", "", "", "", "", ""},
                                                      {true, false, true, false, true, false});

  results = cudf::strings::slice_strings(strings_view, cudf::strings_column_view{delim_col}, 0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
}

TEST_F(StringsSliceTest, SearchScalarDelimiter)
{
  auto strings_col = cudf::test::strings_column_wrapper(
    {"Héllo", "thesé", "", "lease", "tést strings", ""}, {true, true, false, true, true, true});
  ;
  auto strings_view = cudf::strings_column_view(strings_col);

  {
    auto exp_results = cudf::test::strings_column_wrapper({"H", "thes", "", "lease", "t", ""},
                                                          {true, true, false, true, true, true});

    auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("é"), 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto exp_results = cudf::test::strings_column_wrapper(
      {"llo", "", "", "lease", "st strings", ""}, {true, true, false, true, true, true});

    auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("é"), -1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("é"), 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings_col);
  }

  {
    auto results = cudf::strings::slice_strings(strings_view, cudf::string_scalar("é"), -2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, strings_col);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Hello LLollooogh", "oopppllo", "", "oppollo", "polo lop apploo po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper({"Hello LL", "o", "", "opp", "pol", ""},
                                                          {true, true, false, true, true, true});

    auto results =
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar("o"), 2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Hello LLollooogh", "oopppllo", "", "oppollo", "polo lop apploo po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper({"ogh", "pppllo", "", "llo", " po", ""},
                                                          {true, true, false, true, true, true});

    auto results =
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar("o"), -2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééééé", "poloéé lopéé applooéé po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééé", "poloéé lopéé apploo", ""},
      {true, true, false, true, true, true});

    auto results =
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar("éé"), 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééééé", "poloéé lopéé applooéé po", ""},
      {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"Héllo HélloHéllo", "Hélloééééé", "", "éééé", " lopéé applooéé po", ""},
      {true, true, false, true, true, true});

    auto results =
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar("éé"), -3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper({"www.yahoo.com",
                                                    "www.apache..org",
                                                    "tennis...com",
                                                    "nvidia....com",
                                                    "google...........com",
                                                    "microsoft...c.....co..m"});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"www.yahoo.com", "www.apache.", "tennis..", "nvidia..", "google..", "microsoft.."});

    auto results =
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar("."), 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper({"www.yahoo.com",
                                                    "www.apache..org",
                                                    "tennis..com",
                                                    "nvidia....com",
                                                    "google...........com",
                                                    ".",
                                                    "microsoft...c.....co..m"});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"www.yahoo.com", "www.apache..org", "tennis..com", "..com", "..com", ".", "co..m"});

    auto results =
      cudf::strings::slice_strings(cudf::strings_column_view{col0}, cudf::string_scalar(".."), -2);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }
}

TEST_F(StringsSliceTest, SearchColumnDelimiter)
{
  {
    auto col0 = cudf::test::strings_column_wrapper(
      {"H™élloi ™◎oo™ff™", "thesé", "", "lease™", "tést strings", "™"},
      {true, true, false, true, true, true});
    auto delim_col = cudf::test::strings_column_wrapper({"™", "™", "", "e", "t", "™"});

    auto exp_results = cudf::test::strings_column_wrapper({"H", "thesé", "", "l", "", ""},
                                                          {true, true, false, true, true, true});

    auto results = cudf::strings::slice_strings(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper({"H™élloﬀ ﬀﬀi ™◎ooﬀ™ff™",
                                                    "tﬀﬀhﬀesé",
                                                    "",
                                                    "lﬀ fooﬀ ffﬀ eaﬀse™",
                                                    "tést ﬀstri.nﬀgs",
                                                    "ﬀﬀ ™ ﬀﬀ ﬀ"},
                                                   {true, true, false, true, true, true});
    auto delim_col = cudf::test::strings_column_wrapper({"ﬀ™", "ﬀ", "", "ﬀ ", "t", "ﬀ ™"});

    auto exp_results = cudf::test::strings_column_wrapper(
      {"ff™", "esé", "", "eaﬀse™", "ri.nﬀgs", " ﬀﬀ ﬀ"}, {true, true, false, true, true, true});

    auto results = cudf::strings::slice_strings(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, -1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper({"H™élloﬀ ﬀﬀi fooﬀ™ barﬀ™ gooﬀ™ ™◎ooﬀ™ff™",
                                                    "tﬀﬀhﬀesé",
                                                    "",
                                                    "lﬀ fooﬀ ffﬀ eaﬀse™",
                                                    "tést ﬀ™ffﬀ™ﬀ™ffﬀstri.ﬀ™ffﬀ™nﬀgs",
                                                    "ﬀﬀ ™ ﬀﬀ ﬀ™ ﬀ™ﬀ™ﬀ™ ﬀ™ﬀ™ ﬀ"},
                                                   {true, true, false, true, true, true});
    auto delim_col = cudf::test::strings_column_wrapper({"ﬀ™", "ﬀ", "", "e ", "ﬀ™ff", "ﬀ™ﬀ™"},
                                                        {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper({"H™élloﬀ ﬀﬀi fooﬀ™ barﬀ™ goo",
                                                           "tﬀﬀh",
                                                           "",
                                                           "lﬀ fooﬀ ffﬀ eaﬀse™",
                                                           "tést ﬀ™ffﬀ™ﬀ™ffﬀstri.",
                                                           "ﬀﬀ ™ ﬀﬀ ﬀ™ ﬀ™ﬀ™ﬀ™ ﬀ™ﬀ™ ﬀ"},
                                                          {true, true, false, true, true, true});

    auto results = cudf::strings::slice_strings(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, 3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }

  {
    auto col0 = cudf::test::strings_column_wrapper({"H™élloﬀ ﬀﬀi fooﬀ™ barﬀ™ gooﬀ™ ™◎ooﬀ™ff™",
                                                    "tﬀﬀhﬀesé",
                                                    "",
                                                    "lﬀ fooﬀ ffﬀ eaﬀse™",
                                                    "tést ﬀ™ffﬀ™ﬀ™ffﬀstri.ﬀ™ffﬀ™nﬀgs",
                                                    "ﬀﬀ ™ ﬀﬀ ﬀ™ ﬀ™ﬀ™ﬀ™ ﬀ™ﬀ™ ﬀ"});
    auto delim_col = cudf::test::strings_column_wrapper({"ﬀ™", "ﬀ", "", "e ", "ﬀ™ff", "ﬀ™ﬀ™"},
                                                        {true, true, false, true, true, true});

    auto exp_results = cudf::test::strings_column_wrapper({" gooﬀ™ ™◎ooﬀ™ff™",
                                                           "ﬀhﬀesé",
                                                           "",
                                                           "lﬀ fooﬀ ffﬀ eaﬀse™",
                                                           "ﬀ™ﬀ™ffﬀstri.ﬀ™ffﬀ™nﬀgs",
                                                           "ﬀﬀ ™ ﬀﬀ ﬀ™ ﬀ™ﬀ™ﬀ™ ﬀ™ﬀ™ ﬀ"});

    auto results = cudf::strings::slice_strings(
      cudf::strings_column_view{col0}, cudf::strings_column_view{delim_col}, -3);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, exp_results);
  }
}
