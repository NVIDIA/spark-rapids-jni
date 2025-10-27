/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <average_example.hpp>
#include <average_example_host_udf.hpp>

#include <vector>

using namespace cudf;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};

using int32_col = cudf::test::fixed_width_column_wrapper<int32_t>;

struct AverageExampleTests : public cudf::test::BaseFixture {};

namespace {

/**
 * @brief Concatenate struct scalars into a struct column.
 * @param scalar_col_ptrs Pointers to the columns in scalars, this span appends the column pointes
 * of scalar one by one, the size of the vector is num_scalars * 2.
 * @param num_scalars Number of struct scalars
 */
CUDF_KERNEL void concat_struct_scalars_to_struct_column_kernel(
  cudf::device_span<int32_t const*> scalar_col_ptrs,
  int num_scalars,
  cudf::device_span<int32_t*> output)
{
  for (auto col = 0; col < 2; ++col) {
    for (auto scalar_idx = 0; scalar_idx < num_scalars; ++scalar_idx) {
      auto flattened_col_idx  = scalar_idx * 2 + col;
      output[col][scalar_idx] = scalar_col_ptrs[flattened_col_idx][0];
    }
  }
}

/**
 * @brief Flatten columns in scalars into a vector.
 */
std::vector<int32_t const*> get_column_ptrs_from_struct_scalars(
  std::vector<std::unique_ptr<cudf::scalar>>& scalars)
{
  std::vector<int32_t const*> col_ptrs(2 * scalars.size());
  int idx = 0;
  for (auto const& s : scalars) {
    auto const struct_scalar_ptr = dynamic_cast<cudf::struct_scalar*>(s.get());
    auto const table_view        = struct_scalar_ptr->view();
    for (auto const& col_view : table_view) {
      col_ptrs[idx++] = col_view.data<int32_t>();
    }
  }
  return col_ptrs;
}

/**
 * @brief Make a struct column from multiple scalars with checks: each scalar is a struct(int, int)
 */
std::unique_ptr<cudf::column> make_struct_column_from_scalars(
  std::vector<std::unique_ptr<cudf::scalar>>& scalars,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // asserts
  for (auto const& s : scalars) {
    EXPECT_EQ(s->type().id(), cudf::type_id::STRUCT);
    auto const struct_scalar_ptr = dynamic_cast<cudf::struct_scalar*>(s.get());
    auto const table_view        = struct_scalar_ptr->view();
    EXPECT_EQ(2, table_view.num_columns());
    for (auto const& col_view : table_view) {
      EXPECT_EQ(col_view.type().id(), cudf::type_id::INT32);
    }
  }

  // get column pointers from struct scalars
  auto col_ptrs   = get_column_ptrs_from_struct_scalars(scalars);
  auto d_col_ptrs = cudf::detail::make_device_uvector(col_ptrs, stream, mr);

  // create output columns
  auto const results_iter = cudf::detail::make_counting_transform_iterator(0, [&](int i) {
    return cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                     scalars.size(),  // num_rows
                                     cudf::mask_state::UNALLOCATED,
                                     stream,
                                     mr);
  });
  auto children = std::vector<std::unique_ptr<cudf::column>>(results_iter, results_iter + 2);
  auto host_results_pointer_iter =
    thrust::make_transform_iterator(children.begin(), [](auto const& results_column) {
      return results_column->mutable_view().template data<int32_t>();
    });
  auto host_results_pointers =
    std::vector<int32_t*>(host_results_pointer_iter, host_results_pointer_iter + children.size());
  auto d_output = cudf::detail::make_device_uvector(host_results_pointers, stream, mr);

  // concatenate struct scalars into a struct column
  concat_struct_scalars_to_struct_column_kernel<<<1, 1, 0, stream.value()>>>(
    d_col_ptrs, scalars.size(), d_output);

  // create struct column
  return cudf::make_structs_column(scalars.size(),        // num_rows
                                   std::move(children),
                                   0,                     // null count
                                   rmm::device_buffer{},  // null mask
                                   stream);
}

/**
 * @brief Make a struct column from a single scalar with checks: each scalar is a struct(int, int)
 */
std::unique_ptr<cudf::column> make_struct_column_from_scalar(std::unique_ptr<cudf::scalar>& scalar,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::scalar>> scalars;
  scalars.push_back(std::move(scalar));
  return make_struct_column_from_scalars(scalars, stream, mr);
}

}  // namespace

TEST_F(AverageExampleTests, Group)
{
  // 1. Create data
  auto const keys_col    = int32_col{1, 2, 3, 1, 2, 3, 1, 2, 3};
  auto const values_col1 = int32_col{1, 2, 3, 1, 2, 3, 1, 2, 3};
  auto const values_col2 = int32_col{1, 2, 3, 1, 2, 3, 1, 2, 3};
  auto const values_col3 = int32_col{1, 2, 3, 1, 2, 3, 1, 2, 3};

  // 2. Execute groupby
  auto agg1 =
    cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_average_example_groupby_host_udf()));
  auto agg2 =
    cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_average_example_groupby_host_udf()));
  auto agg3 =
    cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_average_example_groupby_host_udf()));

  std::vector<cudf::groupby::aggregation_request> agg_requests;
  agg_requests.emplace_back();
  agg_requests[0].values = values_col1;
  agg_requests[0].aggregations.push_back(std::move(agg1));
  agg_requests.emplace_back();
  agg_requests[1].values = values_col2;
  agg_requests[1].aggregations.push_back(std::move(agg2));
  agg_requests.emplace_back();
  agg_requests[2].values = values_col3;
  agg_requests[2].aggregations.push_back(std::move(agg3));

  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys_col}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});
  auto const grp_result = gb_obj.aggregate(agg_requests, cudf::test::get_default_stream());

  // each grouped result struct(sum, count) has 3 rows:
  // key=1 -> sum=3, count=3
  // key=2 -> sum=6, count=3
  // key=3 -> sum=9, count=3
  auto const grouped_ret_for_vals1 = grp_result.second[0].results[0]->view();
  auto const grouped_ret_for_vals2 = grp_result.second[1].results[0]->view();
  auto const grouped_ret_for_vals3 = grp_result.second[2].results[0]->view();

  // 3. Execute merge multiple struct(sum, count)
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  // each result is 3 rows, concat to 9 rows.
  auto const sum_and_count = cudf::concatenate(
    std::vector<cudf::column_view>{
      grouped_ret_for_vals1, grouped_ret_for_vals2, grouped_ret_for_vals3},
    stream,
    mr);
  auto merge_agg =
    cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_average_example_groupby_merge_host_udf()));
  std::vector<cudf::groupby::aggregation_request> merge_requests;
  merge_requests.emplace_back();
  merge_requests[0].values = sum_and_count->view();
  merge_requests[0].aggregations.push_back(std::move(merge_agg));
  cudf::groupby::groupby gb_obj2(
    cudf::table_view({keys_col}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});
  auto const grp_result2 = gb_obj2.aggregate(merge_requests, cudf::test::get_default_stream());
  // input: 9 rows struct(sum, count)
  // key=1 -> sum=3, count=3
  // key=2 -> sum=6, count=3
  // key=3 -> sum=9, count=3
  // key=1 -> sum=3, count=3
  // key=2 -> sum=6, count=3
  // key=3 -> sum=9, count=3
  // key=1 -> sum=3, count=3
  // key=2 -> sum=6, count=3
  // key=3 -> sum=9, count=3
  // output: 3 rows struct(sum, count)
  // keys=1 -> sum = 9, count = 9
  // keys=2 -> sum = 18, count = 9
  // keys=3 -> sum = 27, count = 9
  auto const& merged = grp_result2.second[0].results[0];

  // 4. Compute average from sum and count
  // result is a int column with 3 rows: [1, 2, 3]
  auto const result = spark_rapids_jni::compute_average_example(*merged, stream, mr);

  // 5. Check result
  auto const expected = int32_col{1, 2, 3};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result, verbosity);
}

TEST_F(AverageExampleTests, Reduction)
{
  // 1. Create data
  auto const vals1 = int32_col{1, 2, 3};
  auto const vals2 = int32_col{2, 4, 6};
  auto stream      = cudf::get_default_stream();
  auto mr          = cudf::get_current_device_resource_ref();

  // 2. Execute reduce
  auto const reduce_agg =
    cudf::make_host_udf_aggregation<cudf::reduce_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_average_example_reduction_host_udf()));
  std::vector<std::unique_ptr<cudf::scalar>> reduced_scalars;
  for (size_t i = 0; i < 8; i++) {
    if (i % 2 == 0) {
      auto reduced =
        cudf::reduce(vals1, *reduce_agg, cudf::data_type{cudf::type_id::STRUCT}, stream, mr);
      EXPECT_TRUE(reduced->is_valid());
      reduced_scalars.push_back(std::move(reduced));
    } else {
      auto reduced =
        cudf::reduce(vals2, *reduce_agg, cudf::data_type{cudf::type_id::STRUCT}, stream, mr);
      EXPECT_TRUE(reduced->is_valid());
      reduced_scalars.push_back(std::move(reduced));
    }
  }

  // 3. reduce all the struct(sum, count) into one
  auto const input_for_merge = make_struct_column_from_scalars(reduced_scalars, stream, mr);
  auto const merge_agg =
    cudf::make_host_udf_aggregation<cudf::reduce_aggregation>(std::unique_ptr<cudf::host_udf_base>(
      spark_rapids_jni::create_average_example_reduction_merge_host_udf()));

  auto reduce_merged =
    cudf::reduce(*input_for_merge, *merge_agg, cudf::data_type{cudf::type_id::STRUCT}, stream, mr);
  EXPECT_TRUE(reduce_merged->is_valid());

  // 4. compute average from sum scalar and count scalar
  auto const merged = make_struct_column_from_scalar(reduce_merged, stream, mr);
  auto const result = spark_rapids_jni::compute_average_example(*merged, stream, mr);

  // 5. check result
  auto const expected = int32_col{3};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}
