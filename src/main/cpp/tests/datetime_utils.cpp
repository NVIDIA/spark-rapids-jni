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

// #include <datetime_utils.hpp>
#include "/home/nghiat/Devel/jni/src/main/cpp/src/datetime_utils.hpp"
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

using days_col = cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>;
using micros_col =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_us, cudf::timestamp_us::rep>;

struct TimestampRebaseTest : public cudf::test::BaseFixture {};

TEST_F(TimestampRebaseTest, DayTimestamp) {
  auto const ts_col = days_col{-719162, -354285, -141714, -141438, -141437, -141432,
                               -141427, -31463,  -31453,  -1,      0,       18335};

  // Check the correctness of timestamp values. They should be the instants as given in ts_strings.
  {
    auto const ts_strings =
        cudf::test::strings_column_wrapper{"0001-01-01", "1000-01-01", "1582-01-01", "1582-10-04",
                                           "1582-10-05", // After Julian but before Gregorian
                                           "1582-10-10", // After Julian but before Gregorian
                                           "1582-10-15", // Gregorian cutover day
                                           "1883-11-10", "1883-11-20", "1969-12-31",
                                           "1970-01-01", // The epoch day
                                           "2020-03-14"};
    auto const parsed_ts =
        cudf::strings::to_timestamps(cudf::strings_column_view(ts_strings),
                                     cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, "%Y-%m-%d");
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(ts_col, *parsed_ts);
  }

  // Check the rebased values.
  {
    auto const rebased = cudf::jni::gregorian_to_julian(ts_col);
    auto const expected = days_col{-719164, -354280, -141704, -141428, -141427, -141427,
                                   -141427, -31463,  -31453,  -1,      0,       18335};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *rebased, cudf::test::debug_output_level::ALL_ERRORS);
  }
}

TEST_F(TimestampRebaseTest, NegativeDayTimestamp) {
  // Negative years cannot be parsed by cudf from strings.
  auto const ts_col = days_col{
      -1121294, // -1100-1-1
      -1100777, // -1044-3-5
      -735535   // -44-3-5
  };
  auto const rebased = cudf::jni::gregorian_to_julian(ts_col);
  auto const expected = days_col{-1121305, -1100787, -735537};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *rebased);
}

TEST_F(TimestampRebaseTest, MicroTimestamp) {
  auto const ts_col = micros_col{-62135593076345679L,
                                 -30610213078876544L,
                                 -12244061221876544L,
                                 -12220243200000000L,
                                 -12219292799000001L,
                                 -45446999900L,
                                 1L,
                                 1584178381500000L};

  // Check the correctness of ts_val. It should be the instant as given in ts_string.
  {

    auto const ts_string = cudf::test::strings_column_wrapper{
        "0001-01-01 01:02:03.654321", "1000-01-01 03:02:01.123456",
        "1582-01-01 07:52:58.123456", "1582-10-04 00:00:00.000000",
        "1582-10-15 00:00:00.999999", // Gregorian cutover day
        "1969-12-31 11:22:33.000100",
        "1970-01-01 00:00:00.000001", // The epoch day
        "2020-03-14 09:33:01.500000"};
    auto const parsed_ts = cudf::strings::to_timestamps(
        cudf::strings_column_view(ts_string),
        cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS}, "%Y-%m-%d %H:%M:%S.%6fz");
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(ts_col, *parsed_ts);
  }

  // Check the rebased values.
  {
    auto const rebased = cudf::jni::gregorian_to_julian(ts_col);
    auto const expected = micros_col{-62135765876345679L,
                                     -30609781078876544L,
                                     -12243197221876544L,
                                     -12219379200000000L,
                                     -12219292799000001L,
                                     -45446999900L,
                                     1L,
                                     1584178381500000L};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *rebased);
  }
}

#if 1
TEST_F(TimestampRebaseTest, x) {
  // Check the correctness of ts_val. It should be the instant as given in ts_string.
  {
    auto const ts_strings = cudf::test::strings_column_wrapper{"1582-10-10"};
    auto const parsed_ts =
        cudf::strings::to_timestamps(cudf::strings_column_view(ts_strings),
                                     cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, "%Y-%m-%d");
    auto x = cudf::bit_cast(parsed_ts->view(), cudf::data_type{cudf::type_id::INT32});
    cudf::test::print(x);
  }
}
#endif
