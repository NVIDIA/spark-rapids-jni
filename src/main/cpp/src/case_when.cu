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

namespace spark_rapids_jni {
namespace detail {

/**
 * Select the column index for the first true in bool columns for the specified row
 */
struct select_first_true_fn {
  // bool columns stores the results of executing `when` expressions
  cudf::table_device_view const d_table;

  /**
   * The number of bool columns is the size of case when branches.
   * Note: reuturned index may be out of bound, valid bound is [0, col_num)
   * When returning col_num index, it means final result is NULL value or ELSE value.
   *
   * e.g.:
   * CASE WHEN 'a' THEN 'A' END
   *   The number of bool columns is 1
   *   The number of scalars is 1
   *   Max index is 1 which means using NULL(all when exprs are false).
   * CASE WHEN 'a' THEN 'A' ELSE '_' END
   *   The number of bool columns is 1
   *   The number of scalars is 2
   *   Max index is also 1 which means using else value '_'
   */
  __device__ cudf::size_type operator()(std::size_t row_idx)
  {
    auto col_num                     = d_table.num_columns();
    bool found_true                  = false;
    cudf::size_type first_true_index = col_num;
    for (auto col_idx = 0; !found_true && col_idx < col_num; col_idx++) {
      auto const& col = d_table.column(col_idx);
      if (!col.is_null(row_idx) && col.element<bool>(row_idx)) {
        // Predicate is true and not null
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

  // make output column
  auto ret = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, row_count, cudf::mask_state::ALL_VALID, stream, mr);

  // select first true index
  auto d_table = cudf::table_device_view::create(when_bool_columns, stream);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(row_count),
                    ret->mutable_view().begin<cudf::size_type>(),
                    select_first_true_fn{*d_table});
  return ret;
}

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

  // Select <str_ptr, str_size> pairs from multiple scalars according to select index
  using str_view = thrust::pair<char const*, cudf::size_type>;
  rmm::device_uvector<str_view> indices(num_of_rows, stream);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(num_of_rows),
                    indices.begin(),
                    [d_scalars, num_of_scalar, d_select_index] __device__(cudf::size_type row_idx) {
                      // select scalar according to index
                      cudf::size_type scalar_idx = d_select_index.element<cudf::size_type>(row_idx);

                      // return <str_ptr, str_size> pair
                      if (scalar_idx < num_of_scalar && !d_scalars.is_null(scalar_idx)) {
                        auto const d_str = d_scalars.element<cudf::string_view>(scalar_idx);
                        return str_view{d_str.data(), d_str.size_bytes()};
                      } else {
                        // index is out of bound, use NULL, for more details refer to comments in
                        // `select_first_true_fn`
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
