/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>

#include <json_utils.hpp>

#include <limits>

using namespace cudf;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};

struct FloatToStringTests : public cudf::test::BaseFixture {};

TEST_F(FloatToStringTests, T1)
{
  auto const strs =
    cudf::test::strings_column_wrapper{"{\"a\": \"x\"}\"a\": \"y\"}", "   {\"b\": \"y", "{}"};
  auto out =
    spark_rapids_jni::from_json_to_raw_map(cudf::strings_column_view{strs}, true, true, true, true);

  cudf::test::print(*out);
  printf("null count: %d\n", out->null_count());
}

TEST_F(FloatToStringTests, T2)
{
  auto const source = cudf::io::source_info("/home/nghiat/TMP/df.parquet/1.parquet");
  auto const opts   = cudf::io::parquet_reader_options::builder(source).build();
  auto const& table = cudf::io::read_parquet(opts).tbl;

  auto const a = table->get_column(0);
  cudf::test::print(a.view());

  auto out = spark_rapids_jni::from_json_to_raw_map(
    cudf::strings_column_view{a.view()}, true, true, true, true);

  cudf::test::print(*out);
  printf("null count: %d\n", out->null_count());
}
