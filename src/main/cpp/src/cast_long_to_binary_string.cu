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

#include "cast_string.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace spark_rapids_jni {

namespace detail {
namespace {
CUDF_KERNEL void compute_output_size_kernel(cudf::column_device_view d_longs,
                                            cudf::size_type* d_sizes)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= d_longs.size()) { return; }
  if (d_longs.is_null(tid)) {
    d_sizes[tid] = 0;
    return;
  }
  // If the value is 0, the size should be 1
  d_sizes[tid] = max(1, 64 - __clzll(d_longs.element<int64_t>(tid)));
}

CUDF_KERNEL void long_to_binary_string_kernel(cudf::column_device_view d_longs,
                                              char* d_chars,
                                              cudf::detail::input_offsetalator d_offsets,
                                              cudf::size_type num_threads_per_row)
{
  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const row_idx = tid / num_threads_per_row;
  if (row_idx >= d_longs.size()) { return; }

  auto const str_len = d_offsets[row_idx + 1] - d_offsets[row_idx];
  if (str_len == 0) { return; }

  auto const value               = d_longs.element<int64_t>(row_idx);
  auto const lane_idx            = tid % num_threads_per_row;
  auto const num_bits_per_thread = 64 / num_threads_per_row;
  auto const first_byte_index    = lane_idx * num_bits_per_thread;

  char* d_buffer = d_chars + d_offsets[row_idx];
  for (auto i = 0; i < num_bits_per_thread; ++i) {
    auto const byte_index = first_byte_index + i;
    if (byte_index >= str_len) { return; }
    auto const num_shifts = str_len - 1 - byte_index;
    d_buffer[byte_index]  = '0' + ((value & (1UL << num_shifts)) >> num_shifts);
  }
}

}  // namespace

std::unique_ptr<cudf::column> long_to_binary_string(cudf::column_view const& input,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);

  CUDF_EXPECTS(input.type().id() == cudf::type_id::INT64, "Input column must be long type");

  auto const strings_count = input.size();
  auto const d_column      = cudf::column_device_view::create(input, stream);

  // The following code is adapted from `cudf::strings::detail::make_strings_children()`
  // Compute the output sizes
  auto output_sizes         = rmm::device_uvector<cudf::size_type>(strings_count, stream);
  cudf::size_type* d_sizes  = output_sizes.data();
  auto constexpr block_size = 256;
  auto grid                 = cudf::detail::grid_1d{strings_count, block_size};
  compute_output_size_kernel<<<grid.num_blocks, block_size, 0, stream.value()>>>(*d_column,
                                                                                 d_sizes);

  // Convert the sizes to offsets
  auto [offsets, bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);
  cudf::detail::input_offsetalator d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // Now build the chars column
  rmm::device_uvector<char> chars(bytes, stream, mr);
  cudf::experimental::prefetch::detail::prefetch("gather", chars, stream);
  char* d_chars = chars.data();

  // Fill in the chars data
  auto constexpr num_threads_per_row = 32;
  auto new_grid = cudf::detail::grid_1d{strings_count * num_threads_per_row, block_size};
  if (bytes > 0) {
    long_to_binary_string_kernel<<<new_grid.num_blocks, block_size, 0, stream.value()>>>(
      *d_column, d_chars, d_offsets, num_threads_per_row);
  }

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input, stream, mr));
}

}  // namespace detail

// external API
std::unique_ptr<cudf::column> long_to_binary_string(cudf::column_view const& input,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::long_to_binary_string(input, stream, mr);
}

}  // namespace spark_rapids_jni
