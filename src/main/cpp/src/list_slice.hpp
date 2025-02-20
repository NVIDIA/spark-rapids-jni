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

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace spark_rapids_jni {

/**
 * @brief Slices each row of a lists column according to the requested `start` and `length`.
 *
 * The indices start at 1, or from the end if negative(can not be zero). If any index in start is
 * outside [-n, n] for a row (where n is the number of elements in that row), the result for that
 * row is an empty list.
 *
 * If length is zero, the result for that row is an empty list. If length exceeds the number of
 * elements from the real start index(0 to n - 1) to the end of the row, the result for that row
 * is the elements from the real start index to the end of the row.
 *
 * @code{.pseudo}
 * input_column = [
 *   [1, 2, 3, 4],
 *   [5, 6, 7],
 *   [8, 9]
 * ]
 *
 * start = 2, length = 2
 *
 * result = [
 *   [2, 3],
 *   [6, 7],
 *   [9]
 * ]
 * @endcode
 *
 * @throws cudf::logic_error if @p start is zero
 * @throws cudf::logic_error if @p length is negative
 *
 * @param input The input lists column to slice
 * @param start The index of the first element to slice in each row(1-based, negative for reverse)
 * @param length The number of elements to slice in each row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate all returned device memory
 * @return The result column with elements in each row sliced according to @p start and @p length
 */
std::unique_ptr<cudf::column> list_slice(
  cudf::lists_column_view const& input,
  cudf::size_type const start,
  cudf::size_type const length,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Slices each row of a lists column according to the requested `start` and `length`.
 *
 * The indices start at 1, or from the end if negative(can not be zero). If any index in start is
 * outside [-n, n] for a row (where n is the number of elements in that row), the result for that
 * row is an empty list.
 *
 * If length is zero, the result for that row is an empty list. If length exceeds the number of
 * elements from the real start index(0 to n - 1) to the end of the row, the result for that row
 * is the elements from the real start index to the end of the row.
 *
 * Null handling: For each row, if the corresponding input or length element is null, the result row
 * will be null.
 *
 * @code{.pseudo}
 * input_column = [
 *   [1, 2, 3, 4],
 *   [5, 6, 7],
 *   [8, 9]
 * ]
 *
 * start = -2, length = [3, 2, 1]
 *
 * result = [
 *   [3, 4],
 *   [6, 7],
 *   [8]
 * ]
 * @endcode
 *
 * @throws cudf::logic_error if the sizes of @p input, @p length columns are not equal
 * @throws cudf::logic_error if @p start is zero
 * @throws cudf::logic_error if @p length column contains negative values
 *
 * @param input The input lists column to slice
 * @param start The index of the first element to slice in each row(1-based, negative for reverse)
 * @param length The number of elements to slice in each row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate all returned device memory
 * @return The result column with elements in each row sliced according to @p start and @p length
 */
std::unique_ptr<cudf::column> list_slice(
  cudf::lists_column_view const& input,
  cudf::size_type const start,
  cudf::column_view const& length,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Slices each row of a lists column according to the requested `start` and `length`.
 *
 * The indices start at 1, or from the end if negative(can not be zero). If any index in start is
 * outside [-n, n] for a row (where n is the number of elements in that row), the result for that
 * row is an empty list.
 *
 * If length is zero, the result for that row is an empty list. If length exceeds the number of
 * elements from the real start index(0 to n - 1) to the end of the row, the result for that row
 * is the elements from the real start index to the end of the row.
 *
 * Null handling: For each row, if the corresponding input or start element is null, the result row
 * will be null.
 *
 * @code{.pseudo}
 * input_column = [
 *   [1, 2, 3, 4],
 *   [5, 6, 7],
 *   [8, 9]
 * ]
 *
 * start = [2, 1, -1], length = 2
 *
 * result = [
 *   [2, 3],
 *   [5, 6],
 *   [9]
 * ]
 * @endcode
 *
 * @throws cudf::logic_error if the sizes of @p input, @p start columns are not equal
 * @throws cudf::logic_error if @p start column contains zeros
 * @throws cudf::logic_error if @p length is negative
 *
 * @param input The input lists column to slice
 * @param start The index of the first element to slice in each row(1-based, negative for reverse)
 * @param length The number of elements to slice in each row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate all returned device memory
 * @return The result column with elements in each row sliced according to @p start and @p length
 */
std::unique_ptr<cudf::column> list_slice(
  cudf::lists_column_view const& input,
  cudf::column_view const& start,
  cudf::size_type const length,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Slices each row of a lists column according to the requested `start` and `length`.
 *
 * The indices start at 1, or from the end if negative(can not be zero). If any index in start is
 * outside [-n, n] for a row (where n is the number of elements in that row), the result for that
 * row is an empty list.
 *
 * If length is zero, the result for that row is an empty list. If length exceeds the number of
 * elements from the real start index(0 to n - 1) to the end of the row, the result for that row
 * is the elements from the real start index to the end of the row.
 *
 * Null handling: For each row, if the corresponding input, start, or length element is null, the
 * result row will be null.
 *
 * @code{.pseudo}
 * input_column = [
 *   [1, 2, 3, 4],
 *   [5, 6, 7],
 *   [8, 9]
 * ]
 *
 * start = [2, 1, -1], length = [2, 2, 3]
 *
 * result = [
 *   [2, 3],
 *   [5, 6],
 *   [9]
 * ]
 * @endcode
 *
 * @throws cudf::logic_error if the sizes of @p input, @p start columns are not equal
 * @throws cudf::logic_error if the sizes of @p input, @p length columns are not equal
 * @throws cudf::logic_error if @p start column contains zeros
 * @throws cudf::logic_error if @p length column contains negative values
 *
 * @param input The input lists column to slice
 * @param start The index of the first element to slice in each row(1-based, negative for reverse)
 * @param length The number of elements to slice in each row
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate all returned device memory
 * @return The result column with elements in each row sliced according to @p start and @p length
 */
std::unique_ptr<cudf::column> list_slice(
  cudf::lists_column_view const& input,
  cudf::column_view const& start,
  cudf::column_view const& length,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace spark_rapids_jni
