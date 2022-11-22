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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cstddef>

namespace cudf::jni {

std::unique_ptr<cudf::table>
multiply_decimal128(cudf::column_view const &a, cudf::column_view const &b, int32_t product_scale,
                    rmm::cuda_stream_view stream = rmm::cuda_stream_default);

std::unique_ptr<cudf::table>
divide_decimal128(cudf::column_view const &a, cudf::column_view const &b, int32_t quotient_scale,
                  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

std::unique_ptr<cudf::table>
add_decimal128(cudf::column_view const &a, cudf::column_view const &b, int32_t quotient_scale,
                  rmm::cuda_stream_view stream = rmm::cuda_stream_default);
} // namespace cudf::jni
