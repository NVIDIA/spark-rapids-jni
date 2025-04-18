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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

namespace spark_rapids_jni {

namespace {
enum class merge_aggregate { YES, NO };

auto create_output_column(cudf::size_type num_rows,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  // Output is a structs column containing 3 double type children: `n`, `avg`, and `m2`.
  auto const create_double_column = [&] {
    return cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT64}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  };
  std::vector<std::unique_ptr<cudf::column>> children;
  children.emplace_back(create_double_column());
  children.emplace_back(create_double_column());
  children.emplace_back(create_double_column());
  return cudf::make_structs_column(
    num_rows, std::move(children), 0, rmm::device_buffer{0, stream, mr}, stream, mr);
}

/**
 * @brief Compute the squared difference between the values and mean.
 */
template <typename ValidIter, typename ValueType>
struct sqr_diff_fn {
  ValidIter is_valid;
  ValueType const* grouped_values;
  double const* group_avg;
  cudf::size_type const* value_group_index;

  __device__ double operator()(cudf::size_type const idx) const
  {
    if constexpr (!std::is_same_v<ValidIter, void*>) {
      if (!is_valid[idx]) { return 0; }
    }
    auto const x         = static_cast<double>(grouped_values[idx]);
    auto const group_idx = value_group_index[idx];
    auto const diff      = x - group_avg[group_idx];
    return diff * diff;
  }
};

struct central_moment {
  template <typename T>
  std::enable_if_t<std::is_arithmetic_v<T>, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& grouped_values,
    cudf::column_view const& group_count,
    cudf::column_view const& group_sum,
    cudf::device_span<cudf::size_type const> value_group_index,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    CUDF_FUNC_RANGE();

    auto const num_groups = group_count.size();
    auto output           = create_output_column(num_groups, stream, mr);
    auto const out_n      = output->child(0).mutable_view().data<double>();
    auto const out_avg    = output->child(1).mutable_view().data<double>();
    auto const out_m2     = output->child(2).mutable_view().data<double>();

    // Convert the count values into double and compute the average values in double type.
    using SumType = cudf::detail::target_type_t<T, cudf::aggregation::Kind::SUM>;
    thrust::transform(rmm::exec_policy_nosync(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(num_groups),
                      thrust::make_zip_iterator(out_n, out_avg),
                      [d_count = group_count.begin<cudf::size_type>(),
                       d_sum   = group_sum.begin<SumType>()] __device__(cudf::size_type const idx)
                        -> thrust::tuple<double, double> {
                        auto const count = d_count[idx];
                        if (count == 0) { return {0, 0}; }
                        auto const n = static_cast<double>(d_count[idx]);
                        return {n, static_cast<double>(d_sum[idx]) / n};
                      });

    auto sqr_diffs = rmm::device_uvector<double>(grouped_values.size(), stream);
    if (grouped_values.has_nulls()) {
      auto const d_values_ptr = cudf::column_device_view::create(grouped_values, stream);
      auto const is_valid_it  = cudf::detail::make_validity_iterator<false>(*d_values_ptr);
      thrust::tabulate(
        rmm::exec_policy_nosync(stream),
        sqr_diffs.begin(),
        sqr_diffs.end(),
        sqr_diff_fn{is_valid_it, grouped_values.begin<T>(), out_avg, value_group_index.begin()});
    } else {
      thrust::tabulate(rmm::exec_policy_nosync(stream),
                       sqr_diffs.begin(),
                       sqr_diffs.end(),
                       sqr_diff_fn<void*, T>{
                         nullptr, grouped_values.begin<T>(), out_avg, value_group_index.begin()});
    }

    thrust::reduce_by_key(rmm::exec_policy_nosync(stream),
                          value_group_index.begin(),
                          value_group_index.end(),
                          sqr_diffs.begin(),
                          thrust::make_discard_iterator(),
                          out_m2);
    return output;
  }

  template <typename T, typename... Args>
  std::enable_if_t<!std::is_arithmetic_v<T>, std::unique_ptr<cudf::column>> operator()(Args&&...)
  {
    CUDF_FAIL("Only numeric types are supported in CentralMoment host_udf aggregation");
  }
};

struct merge_fn {
  cudf::size_type const* const group_offsets;
  double const* const group_n;
  double const* const group_avg;
  double const* const group_m2;

  thrust::tuple<double, double, double> __device__ operator()(cudf::size_type const group_idx) const
  {
    double n{0};
    double avg{0};
    double m2{0};
    for (auto idx = group_offsets[group_idx], end = group_offsets[group_idx + 1]; idx < end;
         ++idx) {
      auto const partial_n = group_n[idx];
      if (partial_n == 0) { continue; }
      auto const partial_avg = group_avg[idx];
      auto const partial_m2  = group_m2[idx];
      auto const new_n       = n + partial_n;
      auto const delta       = partial_avg - avg;
      m2 += partial_m2 + delta * delta * n * partial_n / new_n;
      avg = (avg * n + partial_avg * partial_n) / new_n;
      n   = new_n;
    }
    return {n, avg, m2};
  }
};

std::unique_ptr<cudf::column> merge_central_moment(
  cudf::column_view const& grouped_values,
  cudf::device_span<cudf::size_type const> group_offsets,
  cudf::size_type num_groups,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(
    grouped_values.type().id() == cudf::type_id::STRUCT && grouped_values.num_children() == 3 &&
      std::all_of(grouped_values.child_begin(),
                  grouped_values.child_end(),
                  [](auto const& child) { return child.type().id() == cudf::type_id::FLOAT64; }),
    "Input to MergeCentralMoment must be a structs column having 3 children of type double.");

  auto const in_n   = grouped_values.child(0).data<double>();
  auto const in_avg = grouped_values.child(1).data<double>();
  auto const in_m2  = grouped_values.child(2).data<double>();

  auto output        = create_output_column(num_groups, stream, mr);
  auto const out_n   = output->child(0).mutable_view().data<double>();
  auto const out_avg = output->child(1).mutable_view().data<double>();
  auto const out_m2  = output->child(2).mutable_view().data<double>();

  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(num_groups),
                    thrust::make_zip_iterator(out_n, out_avg, out_m2),
                    merge_fn{group_offsets.data(), in_n, in_avg, in_m2});
  return output;
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

    if (is_merge == merge_aggregate::YES) {
      return merge_central_moment(
        grouped_values, get_group_offsets(), get_num_groups(), stream, mr);
    }

    auto const group_count = compute_aggregation(
      cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
    auto const group_sum =
      compute_aggregation(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    return cudf::type_dispatcher(grouped_values.type(),
                                 central_moment{},
                                 grouped_values,
                                 group_count,
                                 group_sum,
                                 get_group_labels(),
                                 stream,
                                 mr);
  }

  /**
   * @brief Create an empty column when the input is empty.
   */
  [[nodiscard]] std::unique_ptr<cudf::column> get_empty_output(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    return create_output_column(0, stream, mr);
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
