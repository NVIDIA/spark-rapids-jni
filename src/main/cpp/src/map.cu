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

namespace spark_rapids_jni {

std::unique_ptr<cudf::column> sort_map_column(cudf::column_view const& map_column,
                                              cudf::order sort_order,
                                              cudf::null_order null_order,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto const& lists_of_structs = cudf::lists_column_view(map_column);
  auto const structs           = lists_of_structs.child();
  CUDF_EXPECTS(structs.type().id() == cudf::type_id::STRUCT,
               "maps_column_view input must have exactly 1 child (STRUCT) column.");
  CUDF_EXPECTS(structs.num_children() == 2,
               "maps_column_view key-value struct must have exactly 2 children.");
  auto keys     = structs.child(0);
  auto values   = structs.child(1);
  auto segments = lists_of_structs.offsets();

  auto sorted = cudf::segmented_sort_by_key(cudf::table_view{{keys, values}},
                                            cudf::table_view{{keys}},
                                            segments,
                                            {sort_order},
                                            {null_order},
                                            stream,
                                            mr);
  stream.synchronize();
  std::vector<std::unique_ptr<cudf::column>> k_v = sorted->release();

  auto mask = cudf::create_null_mask(keys.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  stream.synchronize();

  auto sorted_struct =
    cudf::make_structs_column(keys.size(), std::move(k_v), 0, std::move(mask), stream, mr);
  stream.synchronize();

  // clone segments
  auto copied_segements = cudf::make_numeric_column(cudf::data_type(segments.type().id()),
                                                    segments.size(),
                                                    cudf::mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);
  stream.synchronize();

  CUDF_CUDA_TRY(cudaMemcpyAsync(copied_segements->mutable_view().data<int32_t>(),
                                segments.data<int32_t>(),
                                segments.size() * sizeof(int32_t),
                                cudaMemcpyDeviceToDevice,
                                stream.value()));
  stream.synchronize();

  printf("mydebug: success \n");

  return cudf::make_lists_column(lists_of_structs.size(),
                                 std::move(copied_segements),  // offsets
                                 std::move(sorted_struct),     // child column
                                 lists_of_structs.null_count(),
                                 cudf::copy_bitmask(lists_of_structs.parent(), stream, mr),
                                 stream,
                                 mr);
}

}  // namespace spark_rapids_jni
