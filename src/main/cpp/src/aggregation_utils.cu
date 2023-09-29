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
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>

//
#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/scan.h>

//
#include <type_traits>

namespace spark_rapids_jni {

namespace {

template <typename ElementIterator, typename ValidityIterator> //
struct percentile_fn {
  __device__ void operator()(cudf::size_type const idx) const {
    auto const histogram_idx = idx / percentages.size();
    printf("idx: %d, histogram idx : %d\n", idx, (int)histogram_idx);

    auto const start = offsets[histogram_idx];

    // If the last element is null: ignore it.
    auto const try_end = offsets[histogram_idx + 1];
    auto const end = sorted_validity[try_end - 1] ? try_end : try_end - 1;

    printf("start %d, end: %d\n", start, end);

    for (int i = start; i < end; ++i) {
      printf("  %d - count: %d\n", i, (int)accumulated_counts[i]);
    }

    // This should never happen here, but let's check it for sure.
    if (start == end) {

      printf("start == end\n");

      return;
    }

    auto const max_positions = accumulated_counts[end - 1] - 1L;
    auto const percentage = percentages[idx - histogram_idx * percentages.size()];
    auto const position = static_cast<double>(max_positions) * percentage;
    auto const lower = static_cast<int64_t>(floor(position));
    auto const higher = static_cast<int64_t>(ceil(position));

    printf("max pos: %d, position: %f\n", (int)max_positions, (float)position);

    auto const lower_index = search_counts(lower + 1, start, end);
    auto const lower_element = sorted_input[start + lower_index];

    printf("lower idx: %d\n", lower_index);

    if (higher == lower) {
      printf("out el: %f\n", (float)lower_element);

      output[idx] = lower_element;
      return;
    }

    auto const higher_index = search_counts(higher + 1, start, end);

    printf("higher_idx: %d\n", higher_index);

    auto const higher_element = sorted_input[start + higher_index];
    if (higher_element == lower_element) {

      printf("out el 2: %f\n", (float)lower_element);

      output[idx] = lower_element;
      return;
    }

    printf("position: %f, lower idx: %d, higher idx: %d,  lower el: %f, higher el: %f \n",
           (float)position, lower_index, higher_index, (float)lower_element, (float)higher_element);

    output[idx] = (higher - position) * lower_element + (position - lower) * higher_element;
  }

  percentile_fn(cudf::size_type const *const offsets_, ElementIterator const sorted_input_,
                ValidityIterator const sorted_validity_,
                cudf::device_span<int64_t const> const accumulated_counts_,
                cudf::device_span<double const> const percentages_, double *const output_)
      : offsets{offsets_}, sorted_input{sorted_input_}, sorted_validity{sorted_validity_},
        accumulated_counts{accumulated_counts_}, percentages{percentages_}, output{output_} {}

private:
  __device__ cudf::size_type search_counts(int64_t position, cudf::size_type start,
                                           cudf::size_type end) const {
    auto const it = thrust::lower_bound(thrust::seq, accumulated_counts.begin() + start,
                                        accumulated_counts.begin() + end, position);
    return static_cast<cudf::size_type>(thrust::distance(accumulated_counts.begin(), it));
  }

  cudf::size_type const *const offsets;
  ElementIterator const sorted_input;
  ValidityIterator const sorted_validity;
  cudf::device_span<int64_t const> const accumulated_counts;
  cudf::device_span<double const> const percentages;
  double *const output;
};

struct percentile_dispatcher {
  template <typename T> static constexpr bool is_supported() { return std::is_arithmetic_v<T>; }

  template <typename T, typename... Args>
  std::enable_if_t<!is_supported<T>(), std::unique_ptr<cudf::column>> operator()(Args &&...) const {
    CUDF_FAIL("Unsupported type in histogram-to-percentile evaluation.");
  }

  template <typename T, CUDF_ENABLE_IF(is_supported<T>())>
  std::unique_ptr<cudf::column> operator()(
      cudf::size_type const *const offsets, cudf::size_type const *const ordered_indices,
      cudf::column_device_view const &data, cudf::device_span<int64_t const> accumulated_counts,
      cudf::device_span<double const> percentages, bool has_null, cudf::size_type num_histograms,
      rmm::cuda_stream_view stream, rmm::mr::device_memory_resource *mr) const {

    auto out_percentiles =
        cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                  num_histograms * static_cast<cudf::size_type>(percentages.size()),
                                  cudf::mask_state::UNALLOCATED, stream, mr);

    if (out_percentiles->size() == 0) {

      printf("out empty\n");
      return out_percentiles;
    }

    // Returns nulls for empty input.
    if (data.size() == 0 || (data.size() == 1 && has_null)) {
      out_percentiles->set_null_mask(cudf::detail::create_null_mask(out_percentiles->size(),
                                                                    cudf::mask_state::ALL_NULL,
                                                                    stream, mr),
                                     out_percentiles->size());
    } else {
      auto const sorted_input_it =
          thrust::make_permutation_iterator(data.begin<T>(), ordered_indices);

      if (has_null) {
        auto const sorted_validity_it = thrust::make_permutation_iterator(
            cudf::detail::make_validity_iterator(data), ordered_indices);
        thrust::for_each(rmm::exec_policy(stream), thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(
                             num_histograms * static_cast<cudf::size_type>(percentages.size())),
                         percentile_fn{offsets, sorted_input_it, sorted_validity_it,
                                       accumulated_counts, percentages,
                                       out_percentiles->mutable_view().begin<double>()});
      } else {
        auto const sorted_validity_it = thrust::make_constant_iterator(true);
        thrust::for_each(rmm::exec_policy(stream), thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(
                             num_histograms * static_cast<cudf::size_type>(percentages.size())),
                         percentile_fn{offsets, sorted_input_it, sorted_validity_it,
                                       accumulated_counts, percentages,
                                       out_percentiles->mutable_view().begin<double>()});
      }
    }

    return out_percentiles;
  }
};

void check_input(cudf::column_view const &input, std::vector<double> const &percentages) {
  CUDF_EXPECTS(input.type().id() == cudf::type_id::LIST, "The input column must be of type LIST.",
               std::invalid_argument);

  auto const child = input.child(cudf::lists_column_view::child_column_index);
  CUDF_EXPECTS(!child.has_nulls(), "Child of the input column must not have nulls.",
               std::invalid_argument);
  CUDF_EXPECTS(child.type().id() == cudf::type_id::STRUCT && child.num_children() == 2,
               "The input column has invalid histogram format.", std::invalid_argument);
  CUDF_EXPECTS(!child.child(1).has_nulls(), "The input column has invalid histogram format.",
               std::invalid_argument);
  CUDF_EXPECTS(child.child(1).type().id() == cudf::type_id::INT64,
               "The input column has invalid histogram format.", std::invalid_argument);

  CUDF_EXPECTS(static_cast<std::size_t>(input.size()) * percentages.size() <=
                   static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
               "Size of output exceeds the column size limit.", std::overflow_error);
}

// Wrap the input column in a lists column, to satisfy the requirement type in Spark.
std::unique_ptr<cudf::column> wrap_in_list(std::unique_ptr<cudf::column> &&input,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource *mr) {
  if (input->size() == 0) {
    return cudf::lists::detail::make_empty_lists_column(input->type(), stream, mr);
  }

  auto const sizes_itr = thrust::constant_iterator<cudf::size_type>(input->size());
  auto offsets =
      std::get<0>(cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + 1, stream, mr));
  return cudf::make_lists_column(1, std::move(offsets), std::move(input), 0, rmm::device_buffer{},
                                 stream, mr);
}

} // namespace

std::unique_ptr<cudf::column> percentile_from_histogram(cudf::column_view const &input,
                                                        std::vector<double> const &percentages,
                                                        bool output_as_list,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::mr::device_memory_resource *mr) {
  check_input(input, percentages);

  auto const default_mr = rmm::mr::get_current_device_resource();

  auto const lists_cv = cudf::lists_column_view{input};
  auto const histograms = lists_cv.get_sliced_child(stream);
  auto const data_col = cudf::structs_column_view{histograms}.get_sliced_child(0);
  auto const counts_col = cudf::structs_column_view{histograms}.get_sliced_child(1);
  CUDF_EXPECTS(data_col.null_count() <= 1,
               "Each histogram must contain no more than one null element.", std::invalid_argument);

  printf("data size: %d, null: %d\n", data_col.size(), data_col.null_count());

  // Attach histogram labels to the input histogram elements.
  auto histogram_labels = rmm::device_uvector<cudf::size_type>(histograms.size(), stream);
  cudf::detail::label_segments(lists_cv.offsets_begin(), lists_cv.offsets_end(),
                               histogram_labels.begin(), histogram_labels.end(), stream);
  auto const labels_cv = cudf::column_view{cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                           static_cast<cudf::size_type>(histogram_labels.size()),
                                           histogram_labels.data(), nullptr, 0};
  auto const labeled_histograms = cudf::table_view{{labels_cv, histograms}};
  // The order of sorted elements within each histogram.
  auto const ordered_indices = cudf::detail::sorted_order(
      labeled_histograms, std::vector<cudf::order>{},
      std::vector<cudf::null_order>{cudf::null_order::AFTER, cudf::null_order::AFTER}, stream,
      default_mr);

  auto const d_data = cudf::column_device_view::create(data_col, stream);
  auto const d_percentages =
      cudf::detail::make_device_uvector_sync(percentages, stream, default_mr);

  auto const has_null = data_col.null_count() > 0;
  auto const sorted_counts = thrust::make_permutation_iterator(
      counts_col.begin<int64_t>(), ordered_indices->view().begin<cudf::size_type>());
  auto const d_accumulated_counts = [&] {
    auto output = rmm::device_uvector<int64_t>(counts_col.size(), stream, default_mr);
    thrust::inclusive_scan(rmm::exec_policy(stream), sorted_counts,
                           sorted_counts + counts_col.size(), output.begin());
    return output;
  }();

  auto out_percentiles =
      type_dispatcher(data_col.type(), percentile_dispatcher{}, lists_cv.offsets_begin(),
                      ordered_indices->view().begin<cudf::size_type>(), *d_data,
                      d_accumulated_counts, d_percentages, has_null, input.size(), stream, mr);

  if (output_as_list) {
    return wrap_in_list(std::move(out_percentiles), stream, mr);
  }
  return out_percentiles;
}

} // namespace spark_rapids_jni
