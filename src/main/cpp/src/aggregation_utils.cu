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
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>

//
#include <thrust/binary_search.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <type_traits>

namespace spark_rapids_jni {

namespace {

template <typename ElementIterator> //
struct percentile_fn {
  __device__ double operator()(double percentile) const {
    auto const max_positions = accumulated_counts.back() - 1L;
    auto const position = static_cast<double>(max_positions) * percentile;

    auto const lower = static_cast<int64_t>(floor(position));
    auto const higher = static_cast<int64_t>(ceil(position));

    auto const lower_index = search_counts(lower + 1);
    auto const lower_element = sorted_input[lower_index];
    if (higher == lower) {
      return lower_element;
    }

    auto const higher_index = search_counts(higher + 1);
    auto const higher_element = sorted_input[higher_index];
    if (higher_element == lower_element) {
      return lower_element;
    }

    return (higher - position) * lower_element + (position - lower) * higher_element;
  }

  percentile_fn(cudf::device_span<int64_t const> const accumulated_counts_,
                ElementIterator const sorted_input_)
      : accumulated_counts{accumulated_counts_}, sorted_input{sorted_input_} {}

private:
  __device__ cudf::size_type search_counts(int64_t position) const {
    auto const it = thrust::lower_bound(thrust::seq, accumulated_counts.begin(),
                                        accumulated_counts.end(), position);
    return static_cast<cudf::size_type>(thrust::distance(accumulated_counts.begin(), it));
  }

  cudf::device_span<int64_t const> const accumulated_counts;
  ElementIterator const sorted_input;
};

struct percentile_dispatcher {
  template <typename T> static constexpr bool is_supported() { return std::is_arithmetic_v<T>; }

  template <typename T, typename... Args>
  std::enable_if_t<!is_supported<T>(), std::unique_ptr<cudf::column>> operator()(Args &&...) const {
    CUDF_FAIL("Unsupported type in histogram-to-percentile evaluation.");
  }

  template <typename T, CUDF_ENABLE_IF(is_supported<T>())>
  std::unique_ptr<cudf::column>
  operator()(cudf::size_type const *const ordered_indices, cudf::column_device_view const &data,
             cudf::column_device_view const &counts,
             cudf::device_span<int64_t const> accumulated_counts,
             cudf::column_device_view const & percentages, rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource *mr) const {
    auto output =
        cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64}, percentages.size(),
                                  cudf::mask_state::UNALLOCATED, stream, mr);
    if (output->size() == 0) {
      return output;
    }

    // Returns nulls for empty input.
    if (data.size() == 0) {
      output->set_null_mask(
          cudf::detail::create_null_mask(output->size(), cudf::mask_state::ALL_NULL, stream, mr),
          output->size());
      return output;
    }

    auto const sorted_input_it =
        thrust::make_permutation_iterator(data.begin<T>(), ordered_indices);
    thrust::transform(rmm::exec_policy(stream), percentages.begin<double>(), percentages.end<double>(),
                      output->mutable_view().begin<double>(),
                      percentile_fn{accumulated_counts, sorted_input_it});

    return output;
  }
};

} // namespace

std::unique_ptr<cudf::column> percentile_from_histogram(cudf::column_view const &input,
                                                        cudf::column_view const &percentages,
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
  auto const d_counts = cudf::column_device_view::create(
      cudf::structs_column_view{input}.get_sliced_child(1), stream);
  auto const d_percentages= cudf::column_device_view::create(
              percentages, stream);
//  auto const d_percentages =
//      cudf::detail::make_device_uvector_sync(percentages, stream, default_mr);

  auto const counts = cudf::structs_column_view{input}.get_sliced_child(1);
  auto const sorted_counts = thrust::make_permutation_iterator(
      counts.begin<int64_t>(), ordered_indices->view().begin<cudf::size_type>());
  auto const d_accumulated_counts = [&] {
    auto output = rmm::device_uvector<int64_t>(counts.size(), stream, default_mr);
    thrust::inclusive_scan(rmm::exec_policy(stream), sorted_counts, sorted_counts + counts.size(),
                           output.begin());
    return output;
  }();

  return type_dispatcher(input.child(0).type(), percentile_dispatcher{},
                         ordered_indices->view().begin<cudf::size_type>(), *d_data, *d_counts,
                         d_accumulated_counts, *d_percentages, stream, mr);
}

} // namespace spark_rapids_jni
