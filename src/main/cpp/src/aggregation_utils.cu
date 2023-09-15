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

#include "aggregation_utils.hpp"

//
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>

//
#include <thrust/iterator/permutation_iterator.h>

namespace spark_rapids_jni {

namespace {

struct percentile_fn {
  template <typename T> constexpr bool is_supported() {
    return std::is_arithmetic_v<T> || cudf::is_fixed_point<T>() || cudf::is_chrono<T>();
  }

  template <typename T, typename... Args>
  std::enable_if_t<!is_supported<T>(), std::unique_ptr<cudf::column>> operator()(Args &&...) {
    CUDF_FAIL("Unsupported type in histogram-to-percentile evaluation.");
  }

  template <typename T>
  std::enable_if_t<is_supported<T>(), std::unique_ptr<cudf::column>>
  operator()(cudf::size_type const *const ordered_indices, cudf::column_device_view const &data,
             cudf::column_device_view const &frequencies,
             cudf::device_span<double const> percentages, rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource *mr) {
    auto output = std::make_unique<cudf::column>(input.type(), percentages.size(),
                                                 cudf::mask_state::UNALLOCATED, stream, mr);
    if (output->size() == 0) {
      return output;
    }

    // Returns nulls for empty input.
    if (input.is_empty()) {
      output->set_null_mask(
          cudf::detail::create_null_mask(output->size(), cudf::mask_state::ALL_NULL, stream, mr),
          output->size());
      return output;
    }

    auto const sorted_input_it =
        thrust::make_permutation_iterator(input.begin<T>(), ordered_indices);
    thrust::transform(rmm::exec_policy(stream), percentages.begin(), percentages.end(),
                      output->mutable_view().begin<T>(),
                      [sorted_input_it, size = input.size()] __device__(double percentage) {
                        //
                        return T{};
                      });

    return output;
  }
};

} // namespace

std::unique_ptr<cudf::column> percentile_from_histogram(cudf::column_view const &input,
                                                        std::vector<double> const &percentages,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::mr::device_memory_resource *mr) {

  //    CUDF_EXPECTS(input.)

  CUDF_EXPECTS(input.type().id() == cudf::type_id::STRUCT && input.num_children() == 2,
               "The input histogram must be a structs column having two children.");
  CUDF_EXPECTS(!input.has_nulls() && !input.child(0).has_nulls() && !input.child(1).has_nulls(),
               "The input column and its children must not have nulls.");
  CUDF_EXPECTS(input.child(1).type().id() == cudf::type_id::INT64,
               "The second child of the input column must be INT64 type.");

  auto const default_mr = rmm::mr::get_current_device_resource();

  auto const ordered_indices =
      cudf::detail::sorted_order(cudf::table_view{{input}}, {}, {}, stream, default_mr);
  auto const d_data = cudf::column_device_view::create(
      cudf::structs_column_view{input}.get_sliced_child(0), stream);
  auto const d_frequencies = cudf::column_device_view::create(
      cudf::structs_column_view{input}.get_sliced_child(1), stream);
  auto const d_percentages =
      cudf::detail::make_device_uvector_sync(percentages, stream, default_mr);

  return type_dispatcher(input.type(), percentile_fn{},
                         ordered_indices->view().begin<cudf::size_type>(), d_data, d_frequencies,
                         d_percentages, stream, mr);
}

} // namespace spark_rapids_jni
