/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <number_converter.hpp>

using namespace cudf;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};

struct ConvertTests : public cudf::test::BaseFixture {};

/**
 * @brief Refer to Spark NumberConverterSuite.scala
 */
// TEST_F(ConvertTests, NormalCase)
// {
//   auto const input_strings = cudf::test::strings_column_wrapper{
//     " 3 ", "-15 ", "  -15 ", " big", "9223372036854775807 ", "   11abc"};

//   auto const from_base = cudf::test::fixed_width_column_wrapper<int32_t>{10, 10, 10, 36, 36, 10};

//   auto const to_base = cudf::test::fixed_width_column_wrapper<int32_t>{2, -16, 16, 16, 16, 16};

//   auto results = spark_rapids_jni::convert_cv_cv_cv(
//     strings_column_view(input_strings), from_base, to_base, cudf::get_default_stream());

//   auto const expected = cudf::test::strings_column_wrapper{
//     "11", "-F", "FFFFFFFFFFFFFFF1", "3A48", "FFFFFFFFFFFFFFFF", "B"};

//   cudf::test::print(*results);

//   printf("hello!!\n");

//   cudf::test::print(expected);
//   CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
// }

TEST_F(ConvertTests, NormalCase2)
{
  auto const input_strings = cudf::test::strings_column_wrapper{"9223372036854775807 "};

  auto const from_base = cudf::test::fixed_width_column_wrapper<int32_t>{36};

  auto const to_base = cudf::test::fixed_width_column_wrapper<int32_t>{16};

  auto results = spark_rapids_jni::convert_cv_cv_cv(
    strings_column_view(input_strings), from_base, to_base, cudf::get_default_stream());

  auto const expected = cudf::test::strings_column_wrapper{"FFFFFFFFFFFFFFFF"};

  cudf::test::print(*results);

  printf("hello!!\n");

  cudf::test::print(expected);
  // CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*results, expected, verbosity);
}
