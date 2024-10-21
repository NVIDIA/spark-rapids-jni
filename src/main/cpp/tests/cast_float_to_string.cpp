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

#include <rmm/device_uvector.hpp>

#include <json_utils.hpp>

using namespace cudf;

struct FloatToStringTests : public cudf::test::BaseFixture {};

TEST_F(FloatToStringTests, FromFloats32)
{
  auto const input = cudf::test::strings_column_wrapper{R"("26/08/2015")"};
  auto out         = spark_rapids_jni::remove_quotes(input, true);

  // cudf::test::print(out->view());
}
