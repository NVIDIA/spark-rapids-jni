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

#include <cudf/aggregation.hpp>

namespace spark_rapids_jni {

std::unique_ptr<cudf::host_udf_base> create_test_reduction_host_udf();

std::unique_ptr<cudf::host_udf_base> create_test_segmented_reduction_host_udf();

std::unique_ptr<cudf::host_udf_base> create_test_groupby_host_udf();

}  // namespace spark_rapids_jni
