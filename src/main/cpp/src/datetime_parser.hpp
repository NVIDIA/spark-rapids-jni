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

#include <cudf/strings/strings_column_view.hpp>

namespace spark_rapids_jni {

/**
 * represents local date time in a time zone.
 */
struct timestamp_components {
  int32_t year; // max 6 digits
  int8_t month;
  int8_t day;
  int8_t hour;
  int8_t minute;
  int8_t second;
  int32_t microseconds;
};

thrust::pair<timestamp_components, cudf::string_view>
parse_string_to_timestamp_components_tz(cudf::string_view timestamp_str,
                                        cudf::string_view default_time_zone);

/**
 *
 * Trims and parses timestamp string column to a timestamp components column and a time zone
 * column, then create timestamp column
 * Refer to: https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
 * org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L394
 *
 * @param input input string column view.
 * @param default_time_zone if input string does not contain a time zone, use this time zone.
 * @returns timestamp components column and time zone string.
 * be empty.
 */
std::unique_ptr<cudf::column> parse_string_to_timestamp(cudf::strings_column_view const &input,
                                                        cudf::string_view default_time_zone);

/**
 *
 * Refer to `SparkDateTimeUtils.stringToTimestampWithoutTimeZone`
 */
std::unique_ptr<cudf::column>
string_to_timestamp_without_time_zone(cudf::strings_column_view const &input, bool allow_time_zone);

/**
 *
 * Refer to `SparkDateTimeUtils.stringToTimestamp`
 */
std::unique_ptr<cudf::column> string_to_timestamp(cudf::strings_column_view const &input,
                                                  cudf::string_view time_zone);

} // namespace spark_rapids_jni
