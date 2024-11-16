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

#include <cudf/io/json.hpp>

#include <rmm/device_uvector.hpp>

#include <json_utils.hpp>

using namespace cudf;

struct FloatToStringTests : public cudf::test::BaseFixture {};

TEST_F(FloatToStringTests, FromFloats32)
{
  std::string json_string = R"({"c2": [19]})";

  {
    cudf::io::json_reader_options in_options =
      cudf::io::json_reader_options::builder(
        cudf::io::source_info{json_string.data(), json_string.size()})
        .lines(true)
        .recovery_mode(cudf::io::json_recovery_mode_t::RECOVER_WITH_NULL)
        .normalize_whitespace(true)
        .mixed_types_as_string(true)
        .keep_quotes(true)
        .experimental(true)
        .strict_validation(true)
        .prune_columns(true);

    cudf::io::schema_element dtype_schema{cudf::data_type{cudf::type_id::STRUCT},
                                          {
                                            {"c2",
                                             {data_type{cudf::type_id::LIST},
                                              {{"element",
                                                {data_type{cudf::type_id::STRUCT},
                                                 {
                                                   {"name", {data_type{cudf::type_id::STRING}}},
                                                   {"abc", {data_type{cudf::type_id::STRING}}},
                                                   {"class", {data_type{cudf::type_id::STRING}}},
                                                 },
                                                 {{"name", "abc", "class"}}}}}}},
                                          },
                                          {{"c2"}}};
    in_options.set_dtypes(dtype_schema);

    auto const parsed_table_with_meta = cudf::io::read_json(in_options);
    // auto const& parsed_meta           = parsed_table_with_meta.metadata;
    auto parsed_columns = parsed_table_with_meta.tbl->release();
    for (auto& col : parsed_columns) {
      cudf::test::print(*col);
    }
  }
}
