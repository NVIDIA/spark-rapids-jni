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

#include <cast_string.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/strings/convert/convert_floats.hpp>

#include <limits>
#include <rmm/device_uvector.hpp>

using namespace cudf;

struct FloatToStringTests : public cudf::test::BaseFixture {};

TEST_F(StringsConvertTest, FromFloats32)
{
  std::vector<float> h_floats{100,
                              654321.25,
                              -12761.125,
                              0,
                              5,
                              -4,
                              std::numeric_limits<float>::quiet_NaN(),
                              839542223232.79,
                              -0.0};
  std::vector<char const*> h_expected{
    "100.0", "654321.25", "-12761.125", "0.0", "5.0", "-4.0", "NaN", "8.395422433e+11", "-0.0"};

  cudf::test::fixed_width_column_wrapper<float> floats(
    h_floats.begin(),
    h_floats.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::from_floats(floats);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}

TEST_F(StringsConvertTest, FromFloats64)
{
  std::vector<double> h_floats{100,
                               654321.25,
                               -12761.125,
                               0,
                               5,
                               -4,
                               std::numeric_limits<double>::quiet_NaN(),
                               839542223232.794248339,
                               -0.0};
  std::vector<char const*> h_expected{
    "100.0", "654321.25", "-12761.125", "0.0", "5.0", "-4.0", "NaN", "8.395422232e+11", "-0.0"};

  cudf::test::fixed_width_column_wrapper<double> floats(
    h_floats.begin(),
    h_floats.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::from_floats(floats);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}