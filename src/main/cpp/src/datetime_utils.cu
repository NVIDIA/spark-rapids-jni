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

#include "datetime_utils.hpp"

#include <cudf/copying.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>


#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace {

std::unique_ptr<cudf::column> rebase_to_julian_days(cudf::column_view const& input) {
  if(input.size()==0) {
    return cudf::empty_like(input);
  }
  auto output = std::make_unique<cudf::column>();


  return output;
}

std::unique_ptr<cudf::column> rebase_to_julian_micros(cudf::column_view const& input) {
  if(input.size()==0) {
    return cudf::empty_like(input);
  }
  auto output = std::make_unique<cudf::column>();


  return output;
}

std::unique_ptr<cudf::column> rebase_to_julian_millis(cudf::column_view const& input) {
  if(input.size()==0) {
    return cudf::empty_like(input);
  }
  auto output = std::make_unique<cudf::column>();


  return output;
}

}

namespace cudf::jni {

std::unique_ptr<cudf::column> rebase_to_julian(cudf::column_view const& input) {
  auto const type = input.type().id();
  CUDF_EXPECTS(type == cudf::type_id::TIMESTAMP_DAYS ||
               type == cudf::type_id::TIMESTAMP_MILLISECONDS ||
               type == cudf::type_id::TIMESTAMP_MICROSECONDS,
               "The input is not a valid date/timestamp type to rebase.");

  if(input.size() == 0) {
    return cudf::empty_like(input);
  }

  if(type==cudf::type_id::TIMESTAMP_DAYS) {
    return rebase_to_julian_days(input);
  }
  if(type==cudf::type_id::TIMESTAMP_MILLISECONDS) {
    return rebase_to_julian_millis(input);
  }
  return rebase_to_julian_micros(input);
}

}  // namespace cudf::jni
