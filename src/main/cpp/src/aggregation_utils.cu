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
#include <cudf/detail/copy.hpp>
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
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/scan.h>

//
#include <type_traits>

namespace spark_rapids_jni {

namespace {

template <typename ElementIterator, typename ValidityIterator> //
struct fill_percentile_fn {
  __device__ void operator()(cudf::size_type const idx) const {
    auto const histogram_idx = idx / percentages.size();

    // If a histogram has null element, it never has more than one null (as the histogram
    // only stores unique elements) and that null is sorted to stay at the end.
    // We need to ignore null thus we will shift the end point if we see a null.
    auto const start = offsets[histogram_idx];
    auto const try_end = offsets[histogram_idx + 1];
    auto const all_valid = sorted_validity[try_end - 1];
    auto const end = all_valid ? try_end : try_end - 1;

    // If the end point after shifting coincides with the start point, we don't have any
    // other valid element.
    auto const has_all_nulls = start >= end;

    auto const percentage_idx = idx % percentages.size();
    if (out_validity && percentage_idx == 0) {
      // If the histogram only contains null elements, the output percentile will be null.
      out_validity[histogram_idx] = has_all_nulls ? 0 : 1;
    }

    if (has_all_nulls) {
      return;
    }

    auto const max_positions = accumulated_counts[end - 1] - 1L;
    auto const percentage = percentages[percentage_idx];
    auto const position = static_cast<double>(max_positions) * percentage;
    auto const lower = static_cast<int64_t>(floor(position));
    auto const higher = static_cast<int64_t>(ceil(position));

    auto const lower_index = search_counts(lower + 1, start, end);
    auto const lower_element = sorted_input[start + lower_index];
    if (higher == lower) {
      output[idx] = lower_element;
      return;
    }

    auto const higher_index = search_counts(higher + 1, start, end);
    auto const higher_element = sorted_input[start + higher_index];
    if (higher_element == lower_element) {
      output[idx] = lower_element;
      return;
    }

    output[idx] = (higher - position) * lower_element + (position - lower) * higher_element;
  }

  fill_percentile_fn(cudf::size_type const *const offsets_, ElementIterator const sorted_input_,
                     ValidityIterator const sorted_validity_,
                     cudf::device_span<int64_t const> const accumulated_counts_,
                     cudf::device_span<double const> const percentages_, double *const output_,
                     int8_t *const out_validity_)
      : offsets{offsets_}, sorted_input{sorted_input_}, sorted_validity{sorted_validity_},
        accumulated_counts{accumulated_counts_}, percentages{percentages_}, output{output_},
        out_validity{out_validity_} {}

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
  int8_t *const out_validity;
};

struct percentile_dispatcher {
  template <typename T> static constexpr bool is_supported() { return std::is_arithmetic_v<T>; }

  using output_type =
      std::tuple<std::unique_ptr<cudf::column>, rmm::device_buffer, cudf::size_type>;

  template <typename T, typename... Args>
  std::enable_if_t<!is_supported<T>(), output_type> operator()(Args &&...) const {
    CUDF_FAIL("Unsupported type in histogram-to-percentile evaluation.");
  }

  template <typename T, CUDF_ENABLE_IF(is_supported<T>())>
  output_type operator()(cudf::size_type const *const offsets,
                         cudf::size_type const *const ordered_indices,
                         cudf::column_device_view const &data,
                         cudf::device_span<int64_t const> accumulated_counts,
                         cudf::device_span<double const> percentages, bool has_null,
                         cudf::size_type num_histograms, rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource *mr) const {

    // Currently, the output type is always double.
    auto percentiles =
        cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                  num_histograms * static_cast<cudf::size_type>(percentages.size()),
                                  cudf::mask_state::UNALLOCATED, stream, mr);

    if (percentiles->size() == 0) {
      return {std::move(percentiles), rmm::device_buffer{}, 0};
    }

    // Returns all nulls for totally empty input.
    if (data.size() == 0) {
      percentiles->set_null_mask(cudf::detail::create_null_mask(
                                     percentiles->size(), cudf::mask_state::ALL_NULL, stream, mr),
                                 percentiles->size());
      return {std::move(percentiles), rmm::device_buffer{}, 0};
    }

    auto const fill_percentile = [&](auto const sorted_validity_it, auto const out_validity) {
      auto const sorted_input_it =
          thrust::make_permutation_iterator(data.begin<T>(), ordered_indices);
      thrust::for_each(rmm::exec_policy(stream), thrust::make_counting_iterator(0),
                       thrust::make_counting_iterator(
                           num_histograms * static_cast<cudf::size_type>(percentages.size())),
                       fill_percentile_fn{
                           offsets, sorted_input_it, sorted_validity_it, accumulated_counts,
                           percentages, percentiles->mutable_view().begin<double>(), out_validity});
    };

    if (!has_null) {
      fill_percentile(thrust::make_constant_iterator(true), nullptr);
    } else {
      auto const sorted_validity_it = thrust::make_permutation_iterator(
          cudf::detail::make_validity_iterator(data), ordered_indices);
      auto out_validities = rmm::device_uvector<int8_t>(num_histograms, stream,
                                                        rmm::mr::get_current_device_resource());
      fill_percentile(sorted_validity_it, out_validities.begin());

      auto [null_mask, null_count] = cudf::detail::valid_if(
          out_validities.begin(), out_validities.end(), thrust::identity{}, stream, mr);
      if (null_count > 0) {
        return {std::move(percentiles), std::move(null_mask), null_count};
      }
    }

    return {std::move(percentiles), rmm::device_buffer{}, 0};
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
               "Size of output exceeds cudf column size limit.", std::overflow_error);
}

// Wrap the input column in a lists column, to satisfy the requirement type in Spark.
std::unique_ptr<cudf::column> wrap_in_list(std::unique_ptr<cudf::column> &&input,
                                           rmm::device_buffer null_mask, cudf::size_type null_count,
                                           cudf::size_type num_histograms,
                                           cudf::size_type num_percentages,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource *mr) {
  if (input->size() == 0) {
    return cudf::lists::detail::make_empty_lists_column(input->type(), stream, mr);
  }

  auto const sizes_itr = thrust::make_constant_iterator(num_percentages);
  auto offsets =
      std::get<0>(cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + 1, stream, mr));
  auto output = cudf::make_lists_column(num_histograms, std::move(offsets), std::move(input),
                                        null_count, std::move(null_mask), stream, mr);
  if (null_count > 0) {
    output->set_null_mask(std::move(null_mask), null_count);
    return cudf::detail::purge_nonempty_nulls(output->view(), stream, mr);
  }

  return output;
}

} // namespace

std::unique_ptr<cudf::column> percentile_from_histogram(cudf::column_view const &input,
                                                        std::vector<double> const &percentages,
                                                        bool output_as_list,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::mr::device_memory_resource *mr) {
  check_input(input, percentages);

  auto const lcv_histograms = cudf::lists_column_view{input};
  auto const histograms = lcv_histograms.get_sliced_child(stream);
  auto const data_col = cudf::structs_column_view{histograms}.get_sliced_child(0);
  auto const counts_col = cudf::structs_column_view{histograms}.get_sliced_child(1);

  auto const default_mr = rmm::mr::get_current_device_resource();
  auto const d_data = cudf::column_device_view::create(data_col, stream);
  auto const d_percentages =
      cudf::detail::make_device_uvector_sync(percentages, stream, default_mr);

  // Attach histogram labels to the input histogram elements.
  auto histogram_labels = rmm::device_uvector<cudf::size_type>(histograms.size(), stream);
  cudf::detail::label_segments(lcv_histograms.offsets_begin(), lcv_histograms.offsets_end(),
                               histogram_labels.begin(), histogram_labels.end(), stream);
  auto const labels_cv = cudf::column_view{cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                           static_cast<cudf::size_type>(histogram_labels.size()),
                                           histogram_labels.data(), nullptr, 0};
  auto const labeled_histograms = cudf::table_view{{labels_cv, histograms}};
  // Find the order of segmented sort elements within each histogram list.
  auto const ordered_indices = cudf::detail::sorted_order(
      labeled_histograms, std::vector<cudf::order>{cudf::order::ASCENDING, cudf::order::ASCENDING},
      std::vector<cudf::null_order>{cudf::null_order::AFTER, cudf::null_order::AFTER}, stream,
      default_mr);

  auto const sorted_counts = thrust::make_permutation_iterator(
      counts_col.begin<int64_t>(), ordered_indices->view().begin<cudf::size_type>());
  auto const d_accumulated_counts = [&] {
    auto output = rmm::device_uvector<int64_t>(counts_col.size(), stream, default_mr);
    thrust::inclusive_scan(rmm::exec_policy(stream), sorted_counts,
                           sorted_counts + counts_col.size(), output.begin());
    return output;
  }();

  auto [percentiles, null_mask, null_count] = type_dispatcher(
      data_col.type(), percentile_dispatcher{}, lcv_histograms.offsets_begin(),
      ordered_indices->view().begin<cudf::size_type>(), *d_data, d_accumulated_counts,
      d_percentages, data_col.has_nulls(), input.size(), stream, mr);

  if (output_as_list) {
    return wrap_in_list(std::move(percentiles), std::move(null_mask), null_count,
                        lcv_histograms.size(), static_cast<cudf::size_type>(percentages.size()),
                        stream, mr);
  }
  percentiles->set_null_mask(std::move(null_mask), null_count);
  return std::move(percentiles);
}

} // namespace spark_rapids_jni
