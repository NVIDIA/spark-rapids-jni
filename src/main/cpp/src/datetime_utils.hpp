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

#pragma once

#include <cudf/datetime.hpp>

namespace spark_rapids_jni {
std::unique_ptr<cudf::column> rebase_gregorian_to_julian(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::column> rebase_julian_to_gregorian(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::column> truncate(
  cudf::column_view const& datetime,
  cudf::column_view const& format,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::column> truncate(
  cudf::column_view const& datetime,
  std::string const& format,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * Calculates the difference in years between the epoch year (1970) and the given date column.
 * E.g.: for date '1971-01-01', the result would be 1: (1 year after epoch year)
 *
 * @param date The input date column.
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @param mr Device memory resource to use for allocations.
 * @return A column of type INT32 containing the year differences from epoch.
 */
std::unique_ptr<cudf::column> compute_year_diff(
  cudf::column_view const& date_input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * Calculates the difference in months between the epoch month (1970-01) and the given date column.
 * E.g.: for date '1971-02-01', the result would be 13: (1 year and 1 month after epoch month)
 *
 * @param date The input date column.
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @param mr Device memory resource to use for allocations.
 * @return A column of type INT32 containing the month differences from epoch.
 */
std::unique_ptr<cudf::column> compute_month_diff(
  cudf::column_view const& date_input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::column> compute_day_diff(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::column> compute_hour_diff(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
