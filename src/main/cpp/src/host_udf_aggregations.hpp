/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cudf/aggregation/host_udf.hpp>

namespace spark_rapids_jni {

cudf::host_udf_base* create_hllpp_reduction_host_udf(int precision);
cudf::host_udf_base* create_hllpp_reduction_merge_host_udf(int precision);
cudf::host_udf_base* create_hllpp_groupby_host_udf(int precision);
cudf::host_udf_base* create_hllpp_groupby_merge_host_udf(int precision);

cudf::host_udf_base* create_central_moment_groupby_host_udf();
cudf::host_udf_base* create_central_moment_groupby_merge_host_udf();

}  // namespace spark_rapids_jni
