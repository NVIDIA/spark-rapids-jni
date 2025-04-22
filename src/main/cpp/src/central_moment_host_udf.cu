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

#include <cudf/aggregation/host_udf.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace spark_rapids_jni {

using namespace cudf;

namespace {
enum class merge_aggregate { YES, NO };

template <typename ResultType, typename Iterator>
struct m2_transform {
  column_device_view const d_values;
  Iterator const values_iter;
  ResultType const* d_means;
  size_type const* d_group_labels;

  __device__ ResultType operator()(size_type const idx) const noexcept
  {
    if (d_values.is_null(idx)) { return 0.0; }

    auto const x         = static_cast<ResultType>(values_iter[idx]);
    auto const group_idx = d_group_labels[idx];
    auto const mean      = d_means[group_idx];
    auto const diff      = x - mean;
    return diff * diff;
  }
};

template <typename ResultType, typename Iterator>
void compute_m2_fn(column_device_view const& values,
                   Iterator values_iter,
                   cudf::device_span<size_type const> group_labels,
                   ResultType const* d_means,
                   ResultType* d_result,
                   rmm::cuda_stream_view stream)
{
  auto m2_fn = m2_transform<ResultType, decltype(values_iter)>{
    values, values_iter, d_means, group_labels.data()};
  auto const itr = thrust::counting_iterator<size_type>(0);
  // Using a temporary buffer for intermediate transform results instead of
  // using the transform-iterator directly in thrust::reduce_by_key
  // improves compile-time significantly.
  auto m2_vals = rmm::device_uvector<ResultType>(values.size(), stream);
  thrust::transform(rmm::exec_policy(stream), itr, itr + values.size(), m2_vals.begin(), m2_fn);

  thrust::reduce_by_key(rmm::exec_policy(stream),
                        group_labels.begin(),
                        group_labels.end(),
                        m2_vals.begin(),
                        thrust::make_discard_iterator(),
                        d_result);
}

struct m2_functor {
  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, std::unique_ptr<column>> operator()(
    column_view const& values,
    column_view const& group_means,
    cudf::device_span<size_type const> group_labels,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    using result_type = cudf::detail::target_type_t<T, aggregation::Kind::M2>;
    auto result       = make_numeric_column(data_type(type_to_id<result_type>()),
                                      group_means.size(),
                                      mask_state::UNALLOCATED,
                                      stream,
                                      mr);

    auto const values_dv_ptr = column_device_view::create(values, stream);
    auto const d_values      = *values_dv_ptr;
    auto const d_means       = group_means.data<result_type>();
    auto const d_result      = result->mutable_view().data<result_type>();

    if (!cudf::is_dictionary(values.type())) {
      auto const values_iter = d_values.begin<T>();
      compute_m2_fn(d_values, values_iter, group_labels, d_means, d_result, stream);
    } else {
      auto const values_iter =
        cudf::dictionary::detail::make_dictionary_iterator<T>(*values_dv_ptr);
      compute_m2_fn(d_values, values_iter, group_labels, d_means, d_result, stream);
    }

    // M2 column values should have the same bitmask as means's.
    if (group_means.nullable()) {
      result->set_null_mask(cudf::detail::copy_bitmask(group_means, stream, mr),
                            group_means.null_count());
    }

    return result;
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic_v<T>, std::unique_ptr<column>> operator()(Args&&...)
  {
    CUDF_FAIL("Only numeric types are supported in M2 groupby aggregation");
  }
};

std::unique_ptr<column> group_m2(column_view const& values,
                                 column_view const& group_means,
                                 cudf::device_span<size_type const> group_labels,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto values_type = cudf::is_dictionary(values.type())
                       ? cudf::dictionary_column_view(values).keys().type()
                       : values.type();

  return type_dispatcher(values_type, m2_functor{}, values, group_means, group_labels, stream, mr);
}

/**
 * @brief Struct to store partial results for merging.
 */
template <class result_type>
struct partial_result {
  double count;
  result_type mean;
  result_type M2;
};

/**
 * @brief Functor to accumulate (merge) all partial results corresponding to the same key into a
 * final result storing in a member variable. It performs merging for the partial results of
 * `COUNT_VALID`, `MEAN`, and `M2` at the same time.
 */
template <class result_type>
struct accumulate_fn {
  partial_result<result_type> merge_vals;

  void __device__ operator()(partial_result<result_type> const& partial_vals) noexcept
  {
    if (partial_vals.count == 0) { return; }

    auto const n_ab  = merge_vals.count + partial_vals.count;
    auto const delta = partial_vals.mean - merge_vals.mean;
    merge_vals.M2 += partial_vals.M2 + (delta * delta) *
                                         static_cast<result_type>(merge_vals.count) *
                                         static_cast<result_type>(partial_vals.count) / n_ab;
    merge_vals.mean =
      (merge_vals.mean * merge_vals.count + partial_vals.mean * partial_vals.count) / n_ab;
    merge_vals.count = n_ab;
  }
};

/**
 * @brief Functor to merge partial results of `COUNT_VALID`, `MEAN`, and `M2` aggregations
 * for a given group (key) index.
 */
template <class result_type>
struct merge_fn {
  size_type const* const d_offsets;
  double const* const d_counts;
  result_type const* const d_means;
  result_type const* const d_M2s;

  auto __device__ operator()(size_type const group_idx) noexcept
  {
    auto const start_idx = d_offsets[group_idx], end_idx = d_offsets[group_idx + 1];

    // This case should never happen, because all groups are non-empty as the results of
    // aggregation. Here we just to make sure we cover this case.
    if (start_idx == end_idx) {
      return thrust::make_tuple(double{0}, result_type{0}, result_type{0}, int8_t{0});
    }

    // If `(n = d_counts[idx]) > 0` then `d_means[idx] != null` and `d_M2s[idx] != null`.
    // Otherwise (`n == 0`), these value (mean and M2) will always be nulls.
    // In such cases, reading `mean` and `M2` from memory will return garbage values.
    // By setting these values to zero when `n == 0`, we can safely merge the all-zero tuple without
    // affecting the final result.
    auto get_partial_result = [&] __device__(size_type idx) {
      {
        auto const n = d_counts[idx];
        return n > 0 ? partial_result<result_type>{n, d_means[idx], d_M2s[idx]}
                     : partial_result<result_type>{double{0}, result_type{0}, result_type{0}};
      };
    };

    // Firstly, store tuple(count, mean, M2) of the first partial result in an accumulator.
    auto accumulator = accumulate_fn<result_type>{get_partial_result(start_idx)};

    // Then, accumulate (merge) the remaining partial results into that accumulator.
    for (auto idx = start_idx + 1; idx < end_idx; ++idx) {
      accumulator(get_partial_result(idx));
    }

    // Get the final result after merging.
    auto const& merge_vals = accumulator.merge_vals;

    // If there are all nulls in the partial results (i.e., sum of all valid counts is
    // zero), then the output is a null.
    auto const is_valid = int8_t{merge_vals.count > 0};

    return thrust::make_tuple(merge_vals.count, merge_vals.mean, merge_vals.M2, is_valid);
  }
};

std::unique_ptr<column> group_merge_m2(column_view const& values,
                                       cudf::device_span<size_type const> group_offsets,
                                       size_type num_groups,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(values.type().id() == type_id::STRUCT,
               "Input to `group_merge_m2` must be a structs column.");
  CUDF_EXPECTS(values.num_children() == 3,
               "Input to `group_merge_m2` must be a structs column having 3 children columns.");

  using result_type = id_to_type<type_id::FLOAT64>;
  static_assert(
    std::is_same_v<cudf::detail::target_type_t<result_type, aggregation::Kind::M2>, result_type>);
  CUDF_EXPECTS(values.child(0).type().id() == type_id::FLOAT64 &&
                 values.child(1).type().id() == type_to_id<result_type>() &&
                 values.child(2).type().id() == type_to_id<result_type>(),
               "Input to `group_merge_m2` must be a structs column having children columns "
               "containing tuples of (M2_value, mean, valid_count).");

  auto result_counts = make_numeric_column(
    data_type(type_to_id<double>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto result_means = make_numeric_column(
    data_type(type_to_id<result_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto result_M2s = make_numeric_column(
    data_type(type_to_id<result_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto validities = rmm::device_uvector<int8_t>(num_groups, stream);

  // Perform merging for all the aggregations. Their output (and their validity data) are written
  // out concurrently through an output zip iterator.
  using iterator_tuple  = thrust::tuple<double*, result_type*, result_type*, int8_t*>;
  using output_iterator = thrust::zip_iterator<iterator_tuple>;
  auto const out_iter =
    output_iterator{thrust::make_tuple(result_counts->mutable_view().template data<double>(),
                                       result_means->mutable_view().template data<result_type>(),
                                       result_M2s->mutable_view().template data<result_type>(),
                                       validities.begin())};

  auto const count_valid = values.child(0);
  auto const mean_values = values.child(1);
  auto const M2_values   = values.child(2);
  auto const iter        = thrust::make_counting_iterator<size_type>(0);

  auto const fn = merge_fn<result_type>{group_offsets.begin(),
                                        count_valid.template begin<double>(),
                                        mean_values.template begin<result_type>(),
                                        M2_values.template begin<result_type>()};
  thrust::transform(rmm::exec_policy(stream), iter, iter + num_groups, out_iter, fn);

  // Generate bitmask for the output.
  // Only mean and M2 values can be nullable. Count column must be non-nullable.
  auto [null_mask, null_count] =
    cudf::detail::valid_if(validities.begin(), validities.end(), cuda::std::identity{}, stream, mr);
  if (null_count > 0) {
    result_means->set_null_mask(null_mask, null_count, stream);   // copy null_mask
    result_M2s->set_null_mask(std::move(null_mask), null_count);  // take over null_mask
  }

  // Output is a structs column containing the merged values of `COUNT_VALID`, `MEAN`, and `M2`.
  std::vector<std::unique_ptr<column>> out_columns;
  out_columns.emplace_back(std::move(result_counts));
  out_columns.emplace_back(std::move(result_means));
  out_columns.emplace_back(std::move(result_M2s));
  auto result = cudf::make_structs_column(
    num_groups, std::move(out_columns), 0, rmm::device_buffer{0, stream, mr}, stream, mr);

  return result;
}
}  // namespace

namespace detail {
struct central_moment_groupby_udf : cudf::groupby_host_udf {
  central_moment_groupby_udf(merge_aggregate is_merge_) : is_merge(is_merge_) {}

  /**
   * @brief Perform the main groupby computation.
   */
  [[nodiscard]] std::unique_ptr<cudf::column> operator()(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    CUDF_FUNC_RANGE();

    auto const grouped_values = get_grouped_values();
    if (grouped_values.size() == 0) { return get_empty_output(stream, mr); }

    if (is_merge == merge_aggregate::NO) {
      auto const group_mean =
        compute_aggregation(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
      auto const grouped_labels = get_group_labels();
      return group_m2(grouped_values, group_mean, grouped_labels, stream, mr);
    }
    return group_merge_m2(grouped_values, get_group_offsets(), get_num_groups(), stream, mr);
  }

  /**
   * @brief Create an empty column when the input is empty.
   */
  [[nodiscard]] std::unique_ptr<cudf::column> get_empty_output(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT64}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  [[nodiscard]] bool is_equal(cudf::host_udf_base const& other) const override
  {
    auto o = dynamic_cast<central_moment_groupby_udf const*>(&other);
    return o != nullptr && o->is_merge == this->is_merge;
  }

  [[nodiscard]] std::size_t do_hash() const override
  {
    return 31 * (31 * std::hash<std::string>{}({"central_moment_groupby_udf"})) +
           static_cast<int>(is_merge);
  }

  [[nodiscard]] std::unique_ptr<cudf::host_udf_base> clone() const override
  {
    return std::make_unique<central_moment_groupby_udf>(is_merge);
  }

 private:
  merge_aggregate is_merge;
};

}  // namespace detail

cudf::host_udf_base* create_central_moment_groupby_host_udf()
{
  return new detail::central_moment_groupby_udf(merge_aggregate::NO);
}

cudf::host_udf_base* create_central_moment_groupby_merge_host_udf()
{
  return new detail::central_moment_groupby_udf(merge_aggregate::YES);
}

}  // namespace spark_rapids_jni
