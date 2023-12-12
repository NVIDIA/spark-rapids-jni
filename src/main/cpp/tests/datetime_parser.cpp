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

#include <cassert>
#include <cstring>

#include <datetime_parser.hpp>

//

#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

using timestamp_col = cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep>;

struct DateTimeParserTest : public cudf::test::BaseFixture
{
};

TEST_F(DateTimeParserTest, ParseTimestamp)
{
  auto v = (2023L * 365L * 86400L + 11L * 30L * 86400L + 5L * 86400L + 3L * 3600L + 4L * 60L + 55L) * 1000000L;
  auto const ts_col = timestamp_col{v, v, v + 123456};

  auto const ts_strings =
      cudf::test::strings_column_wrapper{"2023-11-05T03:04:55Z",
                                        "2023-11-05T03:04:55 ",
                                        "2023-11-05T03:04:55.123456   "};
  auto const parsed_ts =
      spark_rapids_jni::string_to_timestamp(cudf::strings_column_view(ts_strings),
                                            "Z",
                                            true,
                                            true);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(ts_col, *(parsed_ts.first));
}
