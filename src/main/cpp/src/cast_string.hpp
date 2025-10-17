/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#pragma once

#include "version.hpp"

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <memory>

namespace spark_rapids_jni {

struct cast_error : public std::runtime_error {
  /**
   * @brief Constructs a cast_error with the error message.
   *
   * @param message Message to be associated with the exception
   */
  cast_error(cudf::size_type row_number, std::string const& string_with_error)
    : std::runtime_error("casting error"),
      _row_number(row_number),
      _string_with_error(string_with_error)
  {
  }

  /**
   * @brief Get the row number of the error
   *
   * @return cudf::size_type row number
   */
  [[nodiscard]] cudf::size_type get_row_number() const { return _row_number; }

  /**
   * @brief Get the string that caused a parsing error
   *
   * @return char const* const problematic string
   */
  [[nodiscard]] char const* get_string_with_error() const { return _string_with_error.c_str(); }

 private:
  cudf::size_type _row_number;
  std::string _string_with_error;
};

/**
 * @brief Convert a string column into an integer column.
 *
 * @param dtype Type of column to return.
 * @param string_col Incoming string column to convert to integers.
 * @param ansi_mode If true, strict conversion and throws on erorr.
 *                  If false, null invalid entries.
 * @param strip if true leading and trailing white space is ignored.
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> Integer column that was created from string_col.
 */
std::unique_ptr<cudf::column> string_to_integer(
  cudf::data_type dtype,
  cudf::strings_column_view const& string_col,
  bool ansi_mode,
  bool strip,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Convert a string column into an decimal column.
 *
 * @param precision precision of input data
 * @param scale scale of input data
 * @param string_col Incoming string column to convert to decimals.
 * @param ansi_mode If true, strict conversion and throws on erorr.
 *                  If false, null invalid entries.
 * @param strip if true leading and trailing white space is ignored.
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> Decimal column that was created from string_col.
 */
std::unique_ptr<cudf::column> string_to_decimal(
  int32_t precision,
  int32_t scale,
  cudf::strings_column_view const& string_col,
  bool ansi_mode,
  bool strip,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Convert a string column into an float column.
 *
 * @param dtype Type of column to return.
 * @param string_col Incoming string column to convert to floating point.
 * @param ansi_mode If true, strict conversion and throws on error.
 *                  If false, null invalid entries.
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> Floating point column that was created from string_col.
 */
std::unique_ptr<cudf::column> string_to_float(
  cudf::data_type dtype,
  cudf::strings_column_view const& string_col,
  bool ansi_mode,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> format_float(
  cudf::column_view const& input,
  int const digits,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> float_to_string(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> decimal_to_non_ansi_string(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> long_to_binary_string(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Parse a timestamp string column into an intermediate struct column.
 * The output column is a struct column with 7 children:
 * - Parse Result type: 0 Success, 1 invalid e.g. year is 7 digits 1234567
 * - seconds part of parsed UTC timestamp
 * - microseconds part of parsed UTC timestamp
 * - Timezone type: 0 unspecified, 1 fixed type, 2 other type, 3 invalid
 * - Timezone offset for fixed type, only applies to fixed type
 * - Timezone is DST, only applies to other type
 * - Timezone index to `GpuTimeZoneDB.transitions` table
 *
 * @param input The input String column contains timestamp strings
 * @param default_tz_index The default timezone index to `GpuTimeZoneDB` transition table.
 * @param default_epoch_day Default epoch day to use if just time, e.g.:
 *   "T00:00:00Z" will use the default_epoch_day, e.g.: the result will be "2025-05-05T00:00:00Z"
 * @param tz_name_to_index_map Timezone info column: STRUCT<tz_name: string, index_to_tz_table: int>
 * @param tz_info_table Timezone transitions table from `GpuTimeZoneDB`, first column is fixed
 * offset, second column is DST rules.
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return a struct column constains 7 columns described above.
 */
std::unique_ptr<cudf::column> parse_timestamp_strings(
  cudf::strings_column_view const& input,
  cudf::size_type default_tz_index,
  int64_t default_epoch_day,
  cudf::column_view const& tz_name_to_index_map,
  cudf::table_view const& tz_info_table,
  spark_rapids_jni::spark_system const& spark_system,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Parse date string column to date column, first trim the input strings.
 * Refer to https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
 * org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L298
 *
 * Allowed formats:
 *   `[+-]yyyy*`
 *   `[+-]yyyy*-[m]m`
 *   `[+-]yyyy*-[m]m-[d]d`
 *   `[+-]yyyy*-[m]m-[d]d `
 *   `[+-]yyyy*-[m]m-[d]d *`
 *   `[+-]yyyy*-[m]m-[d]dT*`
 *
 * @param input The input String column contains date strings
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 */
std::unique_ptr<cudf::column> parse_strings_to_date(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
