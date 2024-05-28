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

#include "case_when.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <thrust/transform.h>

// #include <cudf/detail/iterator.cuh>
// #include <cudf/detail/utilities/vector_factories.hpp>

// #include <rmm/device_uvector.hpp>

namespace spark_rapids_jni {
namespace detail {

/**
 * Find the first column index with true value from bool columns for a specified row
 */
struct select_first_true_fn {
  cudf::table_device_view const d_table;

  /**
   * bool columns number should be the size of case when branches.
   * For case when that does not have else branch, max returned index is column number which means
   * use default null path.
   *
   * e.g.:
   * CASE WHEN 'a' THEN 'A' END , max index is 1 which means use null
   * CASE WHEN 'a' THEN 'A' ELSE '_' END, max index is also 1 which means use else value '_'
   */
  __device__ cudf::size_type operator()(std::size_t row_idx)
  {
    auto col_num                     = d_table.num_columns();
    bool found_true                  = false;
    cudf::size_type first_true_index = col_num;
    for (auto col_idx = 0; !found_true && col_idx < col_num; col_idx++) {
      if (d_table.column(col_idx).element<bool>(row_idx)) {
        found_true       = true;
        first_true_index = col_idx;
      }
    }
    return first_true_index;
  }
};

std::unique_ptr<cudf::column> select_first_true_index(cudf::table_view const& when_bool_columns,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  // checks
  auto const num_columns = when_bool_columns.num_columns();
  CUDF_EXPECTS(num_columns > 0, "At least one column must be specified");
  auto const row_count = when_bool_columns.num_rows();
  if (row_count == 0)  // empty begets empty
    return cudf::make_empty_column(cudf::type_id::INT32);

  auto ret = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, row_count, cudf::mask_state::ALL_VALID, stream, mr);

  auto d_table = cudf::table_device_view::create(when_bool_columns, stream);

  // select first true index
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(row_count),
                    ret->mutable_view().begin<cudf::size_type>(),
                    select_first_true_fn{*d_table});
  return ret;
}

/**
 * Exedute SQL `case when` semantic.
 * If there are multiple branches and each branch uses scalar to generator value,
 * then it's fast to use this function because it does not generate temp string columns.
 * Regular logic in RAPIDs Acceselorator invokes multiple cudf::copy_if_else to generate
 * multiple temp string columns which are time-consuming.
 *
 * E.g.:
 *   SQL is:
 *     select
 *        case
 *          when bool_1_expr then "value_1"
 *          when bool_2_expr then "value_2"
 *          when bool_3_expr then "value_3"
 *          else "value_else"
 *        end
 *      from tab
 *
 *   E.g.: the input data is:
 *     bool column 1: [true,  false, false, false]  // the result of
 bool_1_expr.executeColumnar
 *     bool column 2: [false, true,  false, flase]  // the result of
 bool_2_expr.executeColumnar
 *     bool column 3: [false, false, true,  flase]  // the result of
 bool_3_expr.executeColumnar
 *     scalars in SQL are "value_1", "value_1", "value_1", "value_else"
 *
 *   Output will be:
 *     ["value_1", "value_2", "value_3", "value_else"]
 *
 * Note: the number of bool columns is greater than the number of scalars by 1.
 * size(when_bool_columns) == (row of then_and_else_scalars) + 1
 *
 */

std::unique_ptr<cudf::column> select_from_index(
  cudf::strings_column_view const& then_and_else_scalar_column,
  cudf::column_view const& select_index_column,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  cudf::size_type num_of_rows   = select_index_column.size();
  cudf::size_type num_of_scalar = then_and_else_scalar_column.size();

  // create device views
  auto d_scalars =
    *(cudf::column_device_view::create(then_and_else_scalar_column.parent(), stream));
  auto d_select_index = *(cudf::column_device_view::create(select_index_column, stream));

  // Select <str_ptr, size> pairs from multiple scalars according to select index
  using str_view = thrust::pair<char const*, cudf::size_type>;
  rmm::device_uvector<str_view> indices(num_of_rows, stream);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(num_of_rows),
                    indices.begin(),
                    [d_scalars, num_of_scalar, d_select_index] __device__(cudf::size_type row_idx) {
                      // select scalar according to index
                      cudf::size_type scalar_idx = d_select_index.element<cudf::size_type>(row_idx);

                      // NOTE a case: scalar_idx == num_of_scalar, e.g.:
                      // CASE WHEN 'a' THEN 'A' END , max index is 1 which means use null
                      // CASE WHEN 'a' THEN 'A' ELSE '_' END, max index is also 1 which means use
                      // else value '_'
                      if (scalar_idx < num_of_scalar) {
                        auto const d_str = d_scalars.element<cudf::string_view>(scalar_idx);
                        return str_view{d_str.data(), d_str.size_bytes()};
                      } else {
                        return str_view{nullptr, 0};
                      }
                    });

  // create final string column from string index pairs
  return cudf::strings::detail::make_strings_column(indices.begin(), indices.end(), stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> select_first_true_index(cudf::table_view const& when_bool_columns,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr)
{
  return detail::select_first_true_index(when_bool_columns, stream, mr);
}

std::unique_ptr<cudf::column> select_from_index(
  cudf::strings_column_view const& then_and_else_scalars_column,
  cudf::column_view const& select_index_column,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return detail::select_from_index(then_and_else_scalars_column, select_index_column, stream, mr);
}

}  // namespace spark_rapids_jni
