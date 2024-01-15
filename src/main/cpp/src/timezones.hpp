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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

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
 * @param transitions the table of transitions for all timezones
 * @param tz_index the index of the row in `transitions` corresponding to the current timezone
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned timestamp column's memory
 */
std::unique_ptr<cudf::column> convert_timestamp_to_utc(
  cudf::column_view const& input,
  cudf::table_view const& transitions,
  cudf::size_type tz_index,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Convert input column timestamps in UTC to specified timezone
 *
 * The transition rules are in enclosed in a table, and the index corresponding to the
 * specific timezone is given.
 *
 * This method is the inverse of convert_timestamp_to_utc.
 *
 * @param input the column of input timestamps in UTC
 * @param transitions the table of transitions for all timezones
 * @param tz_index the index of the row in `transitions` corresponding to the specific timezone
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned timestamp column's memory
 */
std::unique_ptr<cudf::column> convert_utc_timestamp_to_timezone(
  cudf::column_view const& input,
  cudf::table_view const& transitions,
  cudf::size_type tz_index,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Add intervals to a column of timestamps.
 *
 * The transition rules are in enclosed in a table, and the index corresponding to the
 * specific timezone is given.
 *
 * @param input the column of input timestamps in UTC
 * @param intervals the column of intervals to add to the input timestamps
 * @param transitions the table of transitions for all timezones
 * @param tz_index the index of the row in `transitions` corresponding to the specific timezone
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned timestamp column's memory
 */
std::unique_ptr<cudf::column> time_add(
  cudf::column_view const& input,
  int64_t const& duration,
  cudf::table_view const& transitions,
  cudf::size_type tz_index,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> time_add(
  cudf::column_view const& input,
  cudf::column_view const& intervals,
  cudf::table_view const& transitions,
  cudf::size_type tz_index,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni