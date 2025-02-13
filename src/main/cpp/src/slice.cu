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

#include "slice.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/lists/filling.hpp>
#include <cudf/lists/gather.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>

using namespace cudf;

namespace spark_rapids_jni {

namespace detail {

namespace {

void assert_start_is_not_zero(column_view const& start, rmm::cuda_stream_view stream)
{
  bool start_valid = thrust::all_of(rmm::exec_policy(stream),
                                    start.begin<int32_t>(),
                                    start.end<int32_t>(),
                                    [] __device__(int32_t x) { return x != 0; });
  CUDF_EXPECTS(start_valid, "Invalid start value: start must not be 0");
}

void assert_length_is_not_negative(column_view const& length, rmm::cuda_stream_view stream)
{
  bool length_valid = thrust::all_of(rmm::exec_policy(stream),
                                     length.begin<int32_t>(),
                                     length.end<int32_t>(),
                                     [] __device__(int32_t x) { return x >= 0; });
  CUDF_EXPECTS(length_valid, "Invalid length value: length must be >= 0");
}

struct int_iterator_from_scalar {
  int32_t value;
  int_iterator_from_scalar(int32_t v) : value(v) {}
  __device__ int32_t operator()(cudf::thread_index_type const index) const
  {
    // ignore index, always return the same value
    return value;
  }
};

struct int_iterator_from_pointer {
  int32_t const* pointer;
  int_iterator_from_pointer(int32_t const* p) : pointer(p) {}
  __device__ int32_t operator()(cudf::thread_index_type const index) const
  {
    return pointer[index];
  }
};

template <typename SRART_ITERATOR, typename LENGTH_ITERATOR>
CUDF_KERNEL void compute_starts_and_sizes_kernel(size_type const* offsets_of_input_lists,
                                                 size_type const num_rows,
                                                 SRART_ITERATOR const start_iterator,
                                                 LENGTH_ITERATOR const length_iterator,
                                                 size_type* d_starts,  // indices for each sub list
                                                 size_type* d_sizes)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= num_rows) { return; }

  auto const length_of_list = offsets_of_input_lists[tid + 1] - offsets_of_input_lists[tid];
  auto start                = start_iterator(tid);
  auto const length         = length_iterator(tid);

  start = start < 0 ? length_of_list + start : start - 1;
  if (start < 0 || start >= length_of_list) {
    d_starts[tid] = 0;
    d_sizes[tid]  = 0;
    return;
  }
  d_starts[tid] = start;
  d_sizes[tid]  = cuda::std::min(length_of_list - start, length);
}

template <typename SRART_ITERATOR, typename LENGTH_ITERATOR>
auto generate_starts_and_sizes(size_type const* offsets_of_input_lists,
                               size_type const num_rows,
                               SRART_ITERATOR const start_iterator,
                               LENGTH_ITERATOR const length_iterator,
                               rmm::cuda_stream_view stream)
{
  auto starts              = make_numeric_column(data_type{type_id::INT32},
                                    num_rows,
                                    mask_state::UNALLOCATED,
                                    stream,
                                    rmm::mr::get_current_device_resource());
  auto sizes               = make_numeric_column(data_type{type_id::INT32},
                                   num_rows,
                                   mask_state::UNALLOCATED,
                                   stream,
                                   rmm::mr::get_current_device_resource());
  constexpr int block_size = 256;
  auto grid                = cudf::detail::grid_1d{num_rows, block_size};
  compute_starts_and_sizes_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(
    offsets_of_input_lists,
    num_rows,
    start_iterator,
    length_iterator,
    starts->mutable_view().data<int32_t>(),
    sizes->mutable_view().data<int32_t>());
  return std::make_pair(std::move(starts), std::move(sizes));
}

std::unique_ptr<cudf::column> legal_slice(lists_column_view const& input,
                                          column_view const& starts,
                                          column_view const& sizes,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const gather_map =
    cudf::lists::sequences(starts, sizes, stream, rmm::mr::get_current_device_resource());
  cudf::lists_column_view const gather_map_view{*gather_map};

  return cudf::lists::segmented_gather(
    input, gather_map_view, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> slice(lists_column_view const& input,
                                    size_type const start,
                                    size_type const length,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(start != 0, "Invalid start value: start must not be 0");
  CUDF_EXPECTS(length >= 0, "Invalid length value: length must be >= 0");

  auto const num_rows = input.size();

  if (num_rows == 0) { return make_empty_column(data_type{type_id::LIST}); }

  auto [starts, sizes] = generate_starts_and_sizes(input.offsets_begin(),
                                                   num_rows,
                                                   int_iterator_from_scalar(start),
                                                   int_iterator_from_scalar(length),
                                                   stream);

  return legal_slice(input, starts->view(), sizes->view(), stream, mr);
}

std::unique_ptr<cudf::column> slice(lists_column_view const& input,
                                    size_type const start,
                                    column_view const& length,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.size();
  CUDF_EXPECTS(num_rows == length.size(), "Input and length size mismatch");

  CUDF_EXPECTS(start != 0, "Invalid start value: start must not be 0");
  assert_length_is_not_negative(length, stream);

  if (num_rows == 0) { return make_empty_column(data_type{type_id::LIST}); }

  auto [starts, sizes] =
    generate_starts_and_sizes(input.offsets_begin(),
                              num_rows,
                              int_iterator_from_scalar(start),
                              int_iterator_from_pointer(length.data<int32_t>()),
                              stream);

  return legal_slice(input, starts->view(), sizes->view(), stream, mr);
}

std::unique_ptr<cudf::column> slice(lists_column_view const& input,
                                    column_view const& start,
                                    size_type const length,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.size();
  CUDF_EXPECTS(num_rows == start.size(), "Input and start size mismatch");

  assert_start_is_not_zero(start, stream);
  CUDF_EXPECTS(length >= 0, "Invalid length value: length must be >= 0");

  if (num_rows == 0) { return make_empty_column(data_type{type_id::LIST}); }

  auto [starts, sizes] = generate_starts_and_sizes(input.offsets_begin(),
                                                   num_rows,
                                                   int_iterator_from_pointer(start.data<int32_t>()),
                                                   int_iterator_from_scalar(length),
                                                   stream);

  return legal_slice(input, starts->view(), sizes->view(), stream, mr);
}

std::unique_ptr<cudf::column> slice(lists_column_view const& input,
                                    column_view const& start,
                                    column_view const& length,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.size();
  CUDF_EXPECTS(num_rows == start.size(), "Input and start size mismatch");
  CUDF_EXPECTS(input.size() == length.size(), "Input and length size mismatch");

  assert_start_is_not_zero(start, stream);
  assert_length_is_not_negative(length, stream);

  if (num_rows == 0) { return make_empty_column(data_type{type_id::LIST}); }

  auto [starts, sizes] =
    generate_starts_and_sizes(input.offsets_begin(),
                              num_rows,
                              int_iterator_from_pointer(start.data<int32_t>()),
                              int_iterator_from_pointer(length.data<int32_t>()),
                              stream);

  return legal_slice(input, starts->view(), sizes->view(), stream, mr);
}

}  // namespace detail

// external API
std::unique_ptr<cudf::column> slice(lists_column_view const& input,
                                    size_type const start,
                                    size_type const length,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice(input, start, length, stream, mr);
}

std::unique_ptr<cudf::column> slice(lists_column_view const& input,
                                    size_type const start,
                                    column_view const& length,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice(input, start, length, stream, mr);
}

std::unique_ptr<cudf::column> slice(lists_column_view const& input,
                                    column_view const& start,
                                    size_type const length,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice(input, start, length, stream, mr);
}

std::unique_ptr<cudf::column> slice(lists_column_view const& input,
                                    column_view const& start,
                                    column_view const& length,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice(input, start, length, stream, mr);
}

}  // namespace spark_rapids_jni
