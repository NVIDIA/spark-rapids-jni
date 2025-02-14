/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"
#include "map.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

extern "C" {

JNIEXPORT jlong Java_com_nvidia_spark_rapids_jni_Map_sort(
  JNIEnv* env, jclass, jlong map_haldle, jboolean is_descending, jboolean is_null_smallest)
{
  JNI_NULL_CHECK(env, map_haldle, "column is null", 0);

  try {
    cudf::jni::auto_set_device(env);
    auto sort_order = is_descending ? cudf::order::DESCENDING : cudf::order::ASCENDING;
    auto null_order = is_null_smallest ? cudf::null_order::BEFORE : cudf::null_order::AFTER;
    cudf::column_view const& map_view = *reinterpret_cast<cudf::column_view const*>(map_haldle);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::sort_map_column(map_view, sort_order, null_order));
  }

  CATCH_STD(env, 0);
}
}
