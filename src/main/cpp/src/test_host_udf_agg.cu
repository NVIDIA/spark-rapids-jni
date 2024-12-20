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

#include "aggregation_utils.hpp"

#include <cudf/aggregation/host_udf.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
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

/**
 * @brief A host-based UDF implementation for reduction.
 *
 * The aggregation computes `sum(value^2, for value in group)` (this is sum of squared).
 */
struct host_udf_reduction_example : cudf::host_udf_base {
  host_udf_reduction_example() = default;

  [[nodiscard]] output_t get_empty_output(
    [[maybe_unused]] std::optional<cudf::data_type> output_dtype,
    [[maybe_unused]] rmm::cuda_stream_view stream,
    [[maybe_unused]] rmm::device_async_resource_ref mr) const override
  {
    CUDF_EXPECTS(output_dtype.has_value(), "Data type for the reduction result must be specified.");
    return cudf::make_default_constructed_scalar(output_dtype.value(), stream, mr);
  }

  [[nodiscard]] output_t operator()(input_map_t const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr) const override
  {
    auto const& values =
      std::get<cudf::column_view>(input.at(reduction_data_attribute::INPUT_VALUES));
    auto const output_dtype =
      std::get<cudf::data_type>(input.at(reduction_data_attribute::OUTPUT_DTYPE));
    return cudf::double_type_dispatcher(
      values.type(), output_dtype, reduce_fn{this}, input, stream, mr);
  }

  [[nodiscard]] std::size_t do_hash() const override
  {
    // Just return the same hash for all instances of this class.
    return std::size_t{12345};
  }

  [[nodiscard]] bool is_equal(host_udf_base const& other) const override
  {
    // Just check if the other object is also instance of this class.
    return dynamic_cast<host_udf_reduction_example const*>(&other) != nullptr;
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_reduction_example>();
  }

  struct reduce_fn {
    // Store pointer to the parent class so we can call its functions.
    host_udf_reduction_example const* parent;

    // For simplicity, this example only accepts a single type input and output.
    using InputType  = double;
    using OutputType = int64_t;

    template <typename T,
              typename U,
              typename... Args,
              CUDF_ENABLE_IF(!std::is_same_v<InputType, T> || !std::is_same_v<OutputType, U>)>
    output_t operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input/output type.");
    }

    template <typename T,
              typename U,
              CUDF_ENABLE_IF(std::is_same_v<InputType, T>&& std::is_same_v<OutputType, U>)>
    output_t operator()(input_map_t const& input,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr) const
    {
      auto const& values =
        std::get<cudf::column_view>(input.at(reduction_data_attribute::INPUT_VALUES));
      auto const output_dtype =
        std::get<cudf::data_type>(input.at(reduction_data_attribute::OUTPUT_DTYPE));
      CUDF_EXPECTS(output_dtype == cudf::data_type{cudf::type_to_id<OutputType>()},
                   "Invalid output type.");
      if (values.size() == 0) { return parent->get_empty_output(output_dtype, stream, mr); }

      auto const input_init_value =
        std::get<std::optional<std::reference_wrapper<cudf::scalar const>>>(
          input.at(reduction_data_attribute::INIT_VALUE));
      auto const init_value = [&]() -> InputType {
        if (input_init_value.has_value() && input_init_value.value().get().is_valid(stream)) {
          auto const numeric_init_scalar =
            dynamic_cast<cudf::numeric_scalar<InputType> const*>(&input_init_value.value().get());
          CUDF_EXPECTS(numeric_init_scalar != nullptr, "Invalid init scalar for reduction.");
          return numeric_init_scalar->value(stream);
        }
        return InputType{0};
      }();

      auto const values_dv_ptr = cudf::column_device_view::create(values, stream);
      auto const result        = thrust::transform_reduce(rmm::exec_policy(stream),
                                                   thrust::make_counting_iterator(0),
                                                   thrust::make_counting_iterator(values.size()),
                                                   transform_fn{*values_dv_ptr},
                                                   static_cast<OutputType>(init_value),
                                                   thrust::plus<>{});

      auto output = cudf::make_numeric_scalar(output_dtype, stream, mr);
      static_cast<cudf::scalar_type_t<OutputType>*>(output.get())->set_value(result, stream);
      return output;
    }

    struct transform_fn {
      cudf::column_device_view values;
      OutputType __device__ operator()(cudf::size_type idx) const
      {
        if (values.is_null(idx)) { return OutputType{0}; }
        auto const val = static_cast<OutputType>(values.element<InputType>(idx));
        return val * val;
      }
    };
  };
};

/**
 * @brief A host-based UDF implementation for segmented reduction.
 *
 * The aggregation computes `sum(value^2, for value in group)` (this is sum of squared).
 */
struct host_udf_segmented_reduction_example : cudf::host_udf_base {
  host_udf_segmented_reduction_example() = default;

  [[nodiscard]] output_t get_empty_output(
    [[maybe_unused]] std::optional<cudf::data_type> output_dtype,
    [[maybe_unused]] rmm::cuda_stream_view stream,
    [[maybe_unused]] rmm::device_async_resource_ref mr) const override
  {
    CUDF_EXPECTS(output_dtype.has_value(),
                 "Data type for the segmented reduction result must be specified.");
    return cudf::make_default_constructed_scalar(output_dtype.value(), stream, mr);
  }

  [[nodiscard]] output_t operator()(input_map_t const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr) const override
  {
    auto const& values =
      std::get<cudf::column_view>(input.at(segmented_reduction_data_attribute::INPUT_VALUES));
    auto const output_dtype =
      std::get<cudf::data_type>(input.at(segmented_reduction_data_attribute::OUTPUT_DTYPE));
    return cudf::double_type_dispatcher(
      values.type(), output_dtype, segmented_reduce_fn{this}, input, stream, mr);
  }

  [[nodiscard]] std::size_t do_hash() const override
  {
    // Just return the same hash for all instances of this class.
    return std::size_t{12345};
  }

  [[nodiscard]] bool is_equal(host_udf_base const& other) const override
  {
    // Just check if the other object is also instance of this class.
    return dynamic_cast<host_udf_segmented_reduction_example const*>(&other) != nullptr;
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_segmented_reduction_example>();
  }

  struct segmented_reduce_fn {
    // Store pointer to the parent class so we can call its functions.
    host_udf_segmented_reduction_example const* parent;

    // For simplicity, this example only accepts a single type input and output.
    using InputType  = double;
    using OutputType = int64_t;

    template <typename T,
              typename U,
              typename... Args,
              CUDF_ENABLE_IF(!std::is_same_v<InputType, T> || !std::is_same_v<OutputType, U>)>
    output_t operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input/output type.");
    }

    template <typename T,
              typename U,
              CUDF_ENABLE_IF(std::is_same_v<InputType, T>&& std::is_same_v<OutputType, U>)>
    output_t operator()(input_map_t const& input,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr) const
    {
      auto const& values =
        std::get<cudf::column_view>(input.at(segmented_reduction_data_attribute::INPUT_VALUES));
      auto const output_dtype =
        std::get<cudf::data_type>(input.at(segmented_reduction_data_attribute::OUTPUT_DTYPE));
      CUDF_EXPECTS(output_dtype == cudf::data_type{cudf::type_to_id<OutputType>()},
                   "Invalid output type.");
      auto const offsets = std::get<cudf::device_span<cudf::size_type const>>(
        input.at(segmented_reduction_data_attribute::OFFSETS));
      CUDF_EXPECTS(offsets.size() > 0, "Invalid offsets.");
      auto const num_segments = static_cast<cudf::size_type>(offsets.size()) - 1;

      if (values.size() == 0) {
        if (num_segments <= 0) { return parent->get_empty_output(output_dtype, stream, mr); }
        return cudf::make_numeric_column(
          output_dtype, num_segments, cudf::mask_state::ALL_NULL, stream, mr);
      }

      auto const input_init_value =
        std::get<std::optional<std::reference_wrapper<cudf::scalar const>>>(
          input.at(segmented_reduction_data_attribute::INIT_VALUE));
      auto const init_value = [&]() -> InputType {
        if (input_init_value.has_value() && input_init_value.value().get().is_valid(stream)) {
          auto const numeric_init_scalar =
            dynamic_cast<cudf::numeric_scalar<InputType> const*>(&input_init_value.value().get());
          CUDF_EXPECTS(numeric_init_scalar != nullptr, "Invalid init scalar for reduction.");
          return numeric_init_scalar->value(stream);
        }
        return InputType{0};
      }();

      auto const null_handling =
        std::get<cudf::null_policy>(input.at(segmented_reduction_data_attribute::NULL_POLICY));
      auto const values_dv_ptr = cudf::column_device_view::create(values, stream);
      auto output              = cudf::make_numeric_column(
        output_dtype, num_segments, cudf::mask_state::UNALLOCATED, stream);

      // Store row index if it is valid, otherwise store a negative value denoting a null row.
      rmm::device_uvector<cudf::size_type> valid_idx(num_segments, stream);

      thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_segments),
        thrust::make_zip_iterator(output->mutable_view().begin<OutputType>(), valid_idx.begin()),
        transform_fn{*values_dv_ptr, offsets, static_cast<OutputType>(init_value), null_handling});

      auto const valid_idx_cv = cudf::column_view{
        cudf::data_type{cudf::type_id::INT32}, num_segments, valid_idx.begin(), nullptr, 0};
      return std::move(cudf::gather(cudf::table_view{{output->view()}},
                                    valid_idx_cv,
                                    cudf::out_of_bounds_policy::NULLIFY,
                                    stream,
                                    mr)
                         ->release()
                         .front());
    }

    struct transform_fn {
      cudf::column_device_view values;
      cudf::device_span<cudf::size_type const> offsets;
      OutputType init_value;
      cudf::null_policy null_handling;

      thrust::tuple<OutputType, cudf::size_type> __device__ operator()(cudf::size_type idx) const
      {
        auto const start = offsets[idx];
        auto const end   = offsets[idx + 1];

        auto constexpr invalid_idx = cuda::std::numeric_limits<cudf::size_type>::lowest();
        if (start == end) { return {OutputType{0}, invalid_idx}; }

        auto sum = init_value;
        for (auto i = start; i < end; ++i) {
          if (values.is_null(i)) {
            if (null_handling == cudf::null_policy::INCLUDE) { sum += init_value * init_value; }
            continue;
          }
          auto const val = static_cast<OutputType>(values.element<InputType>(i));
          sum += val * val;
        }
        auto const segment_size = end - start;
        return {static_cast<OutputType>(segment_size) * sum, idx};
      }
    };
  };
};

/**
 * @brief A host-based UDF implementation for groupby.
 *
 * For each group of values, the aggregation computes
 * `(group_idx + 1) * group_sum_of_squares - group_max * group_sum`.
 */
struct host_udf_groupby_example : cudf::host_udf_base {
  host_udf_groupby_example() = default;

  [[nodiscard]] data_attribute_set_t get_required_data() const override
  {
    // We need grouped values, group offsets, group labels, and also results from groups'
    // MAX and SUM aggregations.
    return {groupby_data_attribute::GROUPED_VALUES,
            groupby_data_attribute::GROUP_OFFSETS,
            groupby_data_attribute::GROUP_LABELS,
            cudf::make_max_aggregation<cudf::groupby_aggregation>(),
            cudf::make_sum_aggregation<cudf::groupby_aggregation>()};
  }

  [[nodiscard]] output_t get_empty_output(
    [[maybe_unused]] std::optional<cudf::data_type> output_dtype,
    [[maybe_unused]] rmm::cuda_stream_view stream,
    [[maybe_unused]] rmm::device_async_resource_ref mr) const override
  {
    return cudf::make_empty_column(
      cudf::data_type{cudf::type_to_id<typename groupby_fn::OutputType>()});
  }

  [[nodiscard]] output_t operator()(input_map_t const& input,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr) const override
  {
    auto const& values =
      std::get<cudf::column_view>(input.at(groupby_data_attribute::GROUPED_VALUES));
    return cudf::type_dispatcher(values.type(), groupby_fn{this}, input, stream, mr);
  }

  [[nodiscard]] std::size_t do_hash() const override
  {
    // Just return the same hash for all instances of this class.
    return std::size_t{12345};
  }

  [[nodiscard]] bool is_equal(host_udf_base const& other) const override
  {
    // Just check if the other object is also instance of this class.
    return dynamic_cast<host_udf_groupby_example const*>(&other) != nullptr;
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_groupby_example>();
  }

  struct groupby_fn {
    // Store pointer to the parent class so we can call its functions.
    host_udf_groupby_example const* parent;

    // For simplicity, this example only accepts double input and always produces double output.
    using InputType  = double;
    using OutputType = double;

    template <typename T, typename... Args, CUDF_ENABLE_IF(!std::is_same_v<InputType, T>)>
    output_t operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input type.");
    }

    template <typename T, CUDF_ENABLE_IF(std::is_same_v<InputType, T>)>
    output_t operator()(input_map_t const& input,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr) const
    {
      auto const& values =
        std::get<cudf::column_view>(input.at(groupby_data_attribute::GROUPED_VALUES));
      if (values.size() == 0) { return parent->get_empty_output(std::nullopt, stream, mr); }

      auto const offsets = std::get<cudf::device_span<cudf::size_type const>>(
        input.at(groupby_data_attribute::GROUP_OFFSETS));
      CUDF_EXPECTS(offsets.size() > 0, "Invalid offsets.");
      auto const num_groups    = static_cast<int>(offsets.size()) - 1;
      auto const group_indices = std::get<cudf::device_span<cudf::size_type const>>(
        input.at(groupby_data_attribute::GROUP_LABELS));
      auto const group_max = std::get<cudf::column_view>(
        input.at(cudf::make_max_aggregation<cudf::groupby_aggregation>()));
      auto const group_sum = std::get<cudf::column_view>(
        input.at(cudf::make_sum_aggregation<cudf::groupby_aggregation>()));

      auto const values_dv_ptr = cudf::column_device_view::create(values, stream);
      auto const output = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<OutputType>()},
                                                    num_groups,
                                                    cudf::mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);

      // Store row index if it is valid, otherwise store a negative value denoting a null row.
      rmm::device_uvector<cudf::size_type> valid_idx(num_groups, stream);

      thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_groups),
        thrust::make_zip_iterator(output->mutable_view().begin<OutputType>(), valid_idx.begin()),
        transform_fn{*values_dv_ptr,
                     offsets,
                     group_indices,
                     group_max.begin<InputType>(),
                     group_sum.begin<InputType>()});

      auto const valid_idx_cv = cudf::column_view{
        cudf::data_type{cudf::type_id::INT32}, num_groups, valid_idx.begin(), nullptr, 0};
      return std::move(cudf::gather(cudf::table_view{{output->view()}},
                                    valid_idx_cv,
                                    cudf::out_of_bounds_policy::NULLIFY,
                                    stream,
                                    mr)
                         ->release()
                         .front());
    }

    struct transform_fn {
      cudf::column_device_view values;
      cudf::device_span<cudf::size_type const> offsets;
      cudf::device_span<cudf::size_type const> group_indices;
      InputType const* group_max;
      InputType const* group_sum;

      thrust::tuple<OutputType, cudf::size_type> __device__ operator()(cudf::size_type idx) const
      {
        auto const start = offsets[idx];
        auto const end   = offsets[idx + 1];

        auto constexpr invalid_idx = cuda::std::numeric_limits<cudf::size_type>::lowest();
        if (start == end) { return {OutputType{0}, invalid_idx}; }

        auto sum_sqr = OutputType{0};
        bool has_valid{false};
        for (auto i = start; i < end; ++i) {
          if (values.is_null(i)) { continue; }
          has_valid      = true;
          auto const val = static_cast<OutputType>(values.element<InputType>(i));
          sum_sqr += val * val;
        }

        if (!has_valid) { return {OutputType{0}, invalid_idx}; }
        return {static_cast<OutputType>(group_indices[start] + 1) * sum_sqr -
                  static_cast<OutputType>(group_max[idx]) * static_cast<OutputType>(group_sum[idx]),
                idx};
      }
    };
  };
};

}  // namespace

std::unique_ptr<cudf::host_udf_base> create_test_reduction_host_udf()
{
  return std::make_unique<host_udf_reduction_example>();
}

std::unique_ptr<cudf::host_udf_base> create_test_segmented_reduction_host_udf()
{
  return std::make_unique<host_udf_segmented_reduction_example>();
}

std::unique_ptr<cudf::host_udf_base> create_test_groupby_host_udf()
{
  return std::make_unique<host_udf_groupby_example>();
}

}  // namespace spark_rapids_jni
