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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace spark_rapids_jni {

// TODO: replace rmm::mr::get_current_device_resource() by cudf

std::unique_ptr<cudf::column> from_json_to_raw_map(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::tuple<std::unique_ptr<cudf::column>, std::unique_ptr<rmm::device_buffer>, char> concat_json(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> make_structs(
  std::vector<cudf::column_view> const& input,
  cudf::column_view const& is_null,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

struct json_schema_element {
  cudf::data_type type;

  std::vector<std::pair<std::string, json_schema_element>> child_types;
};

std::unique_ptr<cudf::column> convert_types(
  cudf::table_view const& input,
  std::vector<std::pair<std::string, json_schema_element>> const& schema,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> cast_strings_to_booleans(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> cast_strings_to_decimals(
  cudf::column_view const& input,
  int precision,
  int scale,
  bool is_us_locale,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> cast_strings_to_integers(
  cudf::column_view const& input,
  cudf::data_type output_type,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> cast_strings_to_dates(
  cudf::column_view const& input,
  std::string const& date_regex,
  std::string const& date_format,
  bool error_if_invalid,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> remove_quotes(
  cudf::column_view const& input,
  bool nullify_if_not_quoted,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> remove_quotes_for_floats(
  cudf::column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
