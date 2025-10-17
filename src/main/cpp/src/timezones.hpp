/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>

namespace spark_rapids_jni {

/**
 * @brief Convert input column timestamps in current timezone to UTC
 *
 * The transition rules are in enclosed in a table, and the index corresponding to the
 * current timezone is given.
 *
 * This method is the inverse of convert_utc_timestamp_to_timezone.
 *
 * @param input the column of input timestamps in the current timezone
 * @param timezone_info the table of transitions for all timezones
 * @param tz_index the index of the row in `timezone_info` corresponding to the current timezone
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned timestamp column's memory
 */
std::unique_ptr<cudf::column> convert_timestamp_to_utc(
  cudf::column_view const& input,
  cudf::table_view const& timezone_info,
  cudf::size_type const tz_index,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Convert input column timestamps in UTC to specified timezone
 *
 * The timezone info is in enclosed in a table, and the index corresponding to the
 * specific timezone is given.
 *
 * This method is the inverse of convert_timestamp_to_utc.
 *
 * @param input the column of input timestamps in UTC
 * @param timezone_info the table of timezone info for all timezones
 * @param tz_index the index of the row in `timezone_info` corresponding to the specific timezone
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned timestamp column's memory
 */
std::unique_ptr<cudf::column> convert_utc_timestamp_to_timezone(
  cudf::column_view const& input,
  cudf::table_view const& timezone_info,
  cudf::size_type const tz_index,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Convert input column timestamps in UTC to specified timezone
 *
 * The timezone info is in enclosed in a table, and the indices corresponding to the
 * specific timezone is given.
 *
 * This method is the inverse of convert_timestamp_to_utc.
 *
 * @param input the column of input timestamps in UTC
 * @param timezone_info the table of timezone info for all timezones
 * @param tz_indices the indices of the timezones,
 * each index is the row in `timezone_info` corresponding to the specific timezone
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned timestamp column's memory
 */
std::unique_ptr<cudf::column> convert_utc_timestamp_to_timezone(
  cudf::column_view const& input,
  cudf::table_view const& transitions,
  cudf::column_view const& tz_indices,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Convert input column timestamps in multiple timezones to UTC.
 *
 * Note: The input timestamps are splited into seconds and microseconds columns to handle special
 * cases: before conversion the timestamp is overflow, but after conversion it is valid.
 *
 * @param input_seconds the seconds column for the input timestamps
 * @param input_microseconds the microseconds column for the input timestamps
 * @param invalid is the timestamp invalid
 * @param tz_type timezone type: fixed offset or other type
 * @param tz_offset timezone offsets, only apply to fixed offset timezone
 * @param transitions the table of transitions for all timezones
 * @param tz_indices the timezone index to transitions, if tz_type is not fixed offset,
 * use this column
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned timestamp column's memory
 *
 * @return a column of timestamps in microseconds
 */
std::unique_ptr<cudf::column> convert_timestamp_to_utc(
  cudf::column_view const& input_seconds,
  cudf::column_view const& input_microseconds,
  cudf::column_view const& invalid,
  cudf::column_view const& tz_type,
  cudf::column_view const& tz_offset,
  cudf::table_view const& transitions,
  cudf::column_view const tz_indices,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource());

}  // namespace spark_rapids_jni