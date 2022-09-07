/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf/types.hpp>

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

std::unique_ptr<cudf::column> string_to_integer(
  cudf::data_type dtype,
  cudf::strings_column_view const& string_col,
  bool ansi_mode,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
