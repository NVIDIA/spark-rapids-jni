/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include "hyper_log_log_plus_plus.hpp"
#include "hyper_log_log_plus_plus_host_udf.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace spark_rapids_jni {

namespace {

template <typename cudf_aggregation>
struct hllpp_udf : cudf::host_udf_base {
  static_assert(std::is_same_v<cudf_aggregation, cudf::reduce_aggregation> ||
                std::is_same_v<cudf_aggregation, cudf::groupby_aggregation>);

  hllpp_udf(int precision_, bool is_merge_) : precision(precision_), is_merge(is_merge_) {}

  [[nodiscard]] input_data_attributes get_required_data() const override
  {
    if constexpr (std::is_same_v<cudf_aggregation, cudf::reduce_aggregation>) {
      return {reduction_data_attribute::INPUT_VALUES};
    } else {
      return {groupby_data_attribute::GROUPED_VALUES,
              groupby_data_attribute::GROUP_OFFSETS,
              groupby_data_attribute::GROUP_LABELS};
    }
  }

  [[nodiscard]] output_type operator()(host_udf_input const& udf_input,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr) const override
  {
    if constexpr (std::is_same_v<cudf_aggregation, cudf::reduce_aggregation>) {
      // reduce
      auto const& input_values =
        std::get<cudf::column_view>(udf_input.at(reduction_data_attribute::INPUT_VALUES));
      if (input_values.size() == 0) { return get_empty_output(std::nullopt, stream, mr); }
      if (is_merge) {
        // reduce intermidate result, input_values are struct of long columns
        return spark_rapids_jni::reduce_merge_hyper_log_log_plus_plus(
          input_values, precision, stream, mr);
      } else {
        return spark_rapids_jni::reduce_hyper_log_log_plus_plus(
          input_values, precision, stream, mr);
      }
    } else {
      // groupby
      auto const& group_values =
        std::get<cudf::column_view>(udf_input.at(groupby_data_attribute::GROUPED_VALUES));
      if (group_values.size() == 0) { return get_empty_output(std::nullopt, stream, mr); }
      auto const group_offsets = std::get<cudf::device_span<cudf::size_type const>>(
        udf_input.at(groupby_data_attribute::GROUP_OFFSETS));
      int num_groups          = group_offsets.size() - 1;
      auto const group_labels = std::get<cudf::device_span<cudf::size_type const>>(
        udf_input.at(groupby_data_attribute::GROUP_LABELS));
      if (is_merge) {
        // group by intermidate result, group_values are struct of long columns
        return spark_rapids_jni::group_merge_hyper_log_log_plus_plus(
          group_values, num_groups, group_labels, precision, stream, mr);
      } else {
        return spark_rapids_jni::group_hyper_log_log_plus_plus(
          group_values, num_groups, group_labels, precision, stream, mr);
      }
    }
  }

  /**
   * @brief create an empty struct scalar
   */
  [[nodiscard]] output_type get_empty_output(
    [[maybe_unused]] std::optional<cudf::data_type> output_dtype,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override
  {
    int num_registers       = 1 << precision;
    int num_long_cols       = num_registers / REGISTERS_PER_LONG + 1;
    auto const results_iter = cudf::detail::make_counting_transform_iterator(
      0, [&](int i) { return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}); });
    auto children =
      std::vector<std::unique_ptr<cudf::column>>(results_iter, results_iter + num_long_cols);

    if constexpr (std::is_same_v<cudf_aggregation, cudf::reduce_aggregation>) {
      // reduce
      auto host_results_view_iter = thrust::make_transform_iterator(
        children.begin(), [](auto const& results_column) { return results_column->view(); });
      auto views      = std::vector<cudf::column_view>(host_results_view_iter,
                                                  host_results_view_iter + num_long_cols);
      auto table_view = cudf::table_view{views};
      auto table      = cudf::table(table_view);
      return std::make_unique<cudf::struct_scalar>(std::move(table), true, stream, mr);
    } else {
      // groupby
      return cudf::make_structs_column(0,
                                       std::move(children),
                                       0,                     // null count
                                       rmm::device_buffer{},  // null mask
                                       stream,
                                       mr);
    }
  }

  [[nodiscard]] bool is_equal(host_udf_base const& other) const override
  {
    auto o = dynamic_cast<hllpp_udf const*>(&other);
    return o != nullptr && o->precision == this->precision;
  }

  [[nodiscard]] std::size_t do_hash() const override
  {
    return 31 * (31 * std::hash<std::string>{}({"hllpp_udf"}) + precision) + is_merge;
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<hllpp_udf>(precision, is_merge);
  }

  int precision;
  bool is_merge;
};

}  // namespace

std::unique_ptr<cudf::host_udf_base> create_hllpp_reduction_host_udf(int precision)
{
  return std::make_unique<hllpp_udf<cudf::reduce_aggregation>>(precision, /*is_merge*/ false);
}

std::unique_ptr<cudf::host_udf_base> create_hllpp_reduction_merge_host_udf(int precision)
{
  return std::make_unique<hllpp_udf<cudf::reduce_aggregation>>(precision, /*is_merge*/ true);
}

std::unique_ptr<cudf::host_udf_base> create_hllpp_groupby_host_udf(int precision)
{
  return std::make_unique<hllpp_udf<cudf::groupby_aggregation>>(precision, /*is_merge*/ false);
}

std::unique_ptr<cudf::host_udf_base> create_hllpp_groupby_merge_host_udf(int precision)
{
  return std::make_unique<hllpp_udf<cudf::groupby_aggregation>>(precision, /*is_merge*/ true);
}

}  // namespace spark_rapids_jni
