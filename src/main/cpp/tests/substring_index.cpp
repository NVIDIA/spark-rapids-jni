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

#include <substring_index.hpp>

#include <string>
#include <vector>

using namespace cudf;

struct SubstringIndexTest : public ::test::BaseFixture {};

TEST_F(SubstringIndexTest, Error)
{
  cudf::test::strings_column_wrapper strings{"this string intentionally left blank"};
  auto strings_view = cudf::strings_column_view(strings);
  auto delim_col    = cudf::test::strings_column_wrapper({"", ""});
  EXPECT_THROW(
    spark_rapids_jni::substring_index(strings_view, cudf::strings_column_view{delim_col}, -1),
    cudf::logic_error);
}

// TEST_F(SubstringIndexTest, ZeroSizeStringsColumn)
//   auto results = cudf::strings::slice_strings(strings_view, 1, 2);
//   cudf::test::expect_column_empty(results->view());
//
//   results = spark_rapids_jni::substring_index((strings_view, cudf::string_scalar("foo"), 1);
//   cudf::test::expect_column_empty(results->view());
//
//   cudf::column_view starts_column(cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0);
//   cudf::column_view stops_column(cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0);
//   results = spark_rapids_jni::substring_index((strings_view, starts_column, stops_column);
//   cudf::test::expect_column_empty(results->view());
//
//   results = spark_rapids_jni::substring_index((strings_view, strings_view, 1);
//   cudf::test::expect_column_empty(results->view());
// }
