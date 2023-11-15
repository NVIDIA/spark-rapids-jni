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

#include "timezones.hpp"

#include <climits>

#include <cudf/wrappers/timestamps.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>
 
using days_col = 
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_s::rep>;

using seconds_col = 
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_s, cudf::timestamp_s::rep>;

using millis_col = 
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_ms, cudf::timestamp_s::rep>;

using micros_col = 
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_s::rep>;

class TimeZoneTest : public cudf::test::BaseFixture {
protected:
  void SetUp() override
  {
    transitions = make_transitions_table();
  }
  std::unique_ptr<cudf::table> transitions;

private:
  std::unique_ptr<cudf::table> make_transitions_table()
  {
    auto instants_from_utc_col = cudf::test::fixed_width_column_wrapper<int64_t>({LONG_MIN, LONG_MIN, -1585904400L, -933667200L, -922093200L, -908870400L, -888829200L, -650019600L, 515527200L, 558464400L, 684867600L});
    auto instants_to_utc_col = cudf::test::fixed_width_column_wrapper<int64_t>({LONG_MIN, LONG_MIN, -1585904400L, -933634800L, -922064400L, -908838000L, -888796801L, -649990800L, 515559600L, 558493200L, 684896400L});
    auto utc_offsets_col = cudf::test::fixed_width_column_wrapper<int32_t>({18000, 29143, 28800, 32400, 28800, 32400, 28800, 28800, 32400, 28800, 28800});
    auto struct_column = 
        cudf::test::structs_column_wrapper{{instants_from_utc_col, instants_to_utc_col, utc_offsets_col}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
    auto offsets =
        cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 11};
    auto list_nullmask = std::vector<bool>(1, 1);
    auto [null_mask, null_count] =
        cudf::test::detail::make_null_mask(list_nullmask.begin(), list_nullmask.end());
    auto list_column = cudf::make_lists_column(
        2, offsets.release(), struct_column.release(), null_count, std::move(null_mask));
    auto columns = std::vector<std::unique_ptr<cudf::column>>{};
    columns.push_back(std::move(list_column));
    return std::make_unique<cudf::table>(std::move(columns));
  }
};

TEST_F(TimeZoneTest, ConvertToUTCSeconds)
{
    auto const ts_col = seconds_col{
      -1262260800L,
      -908838000L,
      -908840700L,
      -888800400L,
      -888799500L,
      0L,
      1699566167L,
      568036800L,
    };
    // check the converted to utc version
    auto const expected = seconds_col{
      -1262289600L,
      -908870400L,
      -908869500L,
      -888832800L,
      -888831900L,
      -28800L,
      1699537367L,
      568008000L
    };
    auto const actual = spark_rapids_jni::convert_timestamp_to_utc(
      ts_col,
      *transitions,
      1,
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertToUTCMilliseconds)
{
    auto const ts_col = millis_col{
      -1262260800000L,
      -908838000000L,
      -908840700000L,
      -888800400000L,
      -888799500000L,
      0L,
      1699571634312L,
      568036800000L,
    };
    // check the converted to utc version
    auto const expected = millis_col{
      -1262289600000L,
      -908870400000L,
      -908869500000L,
      -888832800000L,
      -888831900000L,
      -28800000L,
      1699542834312L,
      568008000000L
    };
    auto const actual = spark_rapids_jni::convert_timestamp_to_utc(
      ts_col,
      *transitions,
      1,
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertToUTCMicroseconds)
{
    auto const ts_col = micros_col{
      -1262260800000000L,
      -908838000000000L,
      -908840700000000L,
      -888800400000000L,
      -888799500000000L,
      0L,
      1699571634312000L,
      568036800000000L,
    };
    // check the converted to utc version
    auto const expected = micros_col{
      -1262289600000000L,
      -908870400000000L,
      -908869500000000L,
      -888832800000000L,
      -888831900000000L,
      -28800000000L,
      1699542834312000L,
      568008000000000L
    };
    auto const actual = spark_rapids_jni::convert_timestamp_to_utc(
      ts_col,
      *transitions,
      1,
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertFromUTCSeconds)
{
    auto const ts_col = seconds_col{
      -1262289600L,
      -908870400L,
      -908869500L,
      -888832800L,
      -888831900L,
      0L,
      1699537367L,
      568008000L
    };
    // check the converted to utc version
    auto const expected = seconds_col{
      -1262260800L,
      -908838000L,
      -908837100L,
      -888800400L,
      -888799500L,
      28800L,
      1699566167L,
      568036800L,
    };
    auto const actual = spark_rapids_jni::convert_utc_timestamp_to_timezone(
      ts_col,
      *transitions,
      1,
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertFromUTCMilliseconds)
{
    auto const ts_col = millis_col{
      -1262289600000L,
      -908870400000L,
      -908869500000L,
      -888832800000L,
      -888831900000L,
      0L,
      1699542834312L,
      568008000000L
    };
    // check the converted to timezone version
    auto const expected = millis_col{
      -1262260800000L,
      -908838000000L,
      -908837100000L,
      -888800400000L,
      -888799500000L,
      28800000L,
      1699571634312L,
      568036800000L,
    };
    auto const actual = spark_rapids_jni::convert_utc_timestamp_to_timezone(
      ts_col,
      *transitions,
      1,
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}

TEST_F(TimeZoneTest, ConvertFromUTCMicroseconds)
{
    auto const ts_col = micros_col{
      -1262289600000000L,
      -908870400000000L,
      -908869500000000L,
      -888832800000000L,
      -888831900000000L,
      0L,
      1699542834312000L,
      568008000000000L
    };
    // check the converted to timezone version
    auto const expected = micros_col{
      -1262260800000000L,
      -908838000000000L,
      -908837100000000L,
      -888800400000000L,
      -888799500000000L,
      28800000000L,
      1699571634312000L,
      568036800000000L,
    };
    auto const actual = spark_rapids_jni::convert_utc_timestamp_to_timezone(
      ts_col,
      *transitions,
      1,
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *actual);
}