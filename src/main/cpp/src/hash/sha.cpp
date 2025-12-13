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

#include "../hash.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>

namespace spark_rapids_jni { // TODO: Add namespace hash.

// TODO: Generalize.
std::unique_ptr<cudf::column> sha224_nulls_preserved(cudf::column_view const& input,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr) {
  if (input.is_empty()) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});
  }

  if (input.has_nulls()) {
    auto hash_from_cudf = cudf::hashing::sha224(cudf::table_view{{input}}, stream); // TODO: std::forward.
    hash_from_cudf->set_null_mask(cudf::copy_bitmask(input, stream), input.null_count());
    return cudf::purge_nonempty_nulls(*hash_from_cudf, stream, mr);
  }
  else {
    return cudf::hashing::sha224(cudf::table_view{{input}}, stream, mr);
  }
}
  
}  // namespace spark_rapids_jni;