/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/null_mask.hpp>

#include <cub/warp/warp_reduce.cuh>

#include <cooperative_groups.h>

using namespace cudf;

namespace spark_rapids_jni {

namespace detail {

constexpr auto NUM_THREADS{256};

constexpr bool is_whitespace(char const chr)
{
  switch (chr) {
    case ' ':
    case '\r':
    case '\t':
    case '\n': return true;
    default: return false;
  }
}

template <typename T>
__global__ void string_to_integer_kernel(T* out,
                                         bitmask_type* validity,
                                         const char* const chars,
                                         offset_type const* offsets,
                                         size_type num_rows)
{
  auto const group = cooperative_groups::this_thread_block();
  auto const warp  = cooperative_groups::tiled_partition<cudf::detail::warp_size>(group);

  // each thread takes a row and marches it and builds the integer for that row
  auto const row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) { return; }
  auto const active        = cooperative_groups::coalesced_threads();
  auto const row_start     = offsets[row];
  auto const len           = offsets[row + 1] - row_start;
  bool valid               = len > 0;
  T thread_val             = 0;
  int i                    = 0;
  T sign                   = 1;
  constexpr bool is_signed = std::is_signed_v<T>;

  if (valid) {
    // skip leading whitespace
    while (is_whitespace(chars[i + row_start])) {
      i++;
    }

    // check for leading +-
    if constexpr (is_signed) {
      if (chars[i + row_start] == '+' || chars[i + row_start] == '-') {
        if (chars[i + row_start] == '-') { sign = -1; }
        i++;
      }
    }

    bool truncating = false;
    for (int c = i; c < len; ++c) {
      auto const chr = chars[c + row_start];
      if (chr == '.') {
        // Values are truncated after a decimal point. However, invalid characters AFTER this
        // decimal point will still invalidate this entry.
        truncating = true;
      } else {
        if (chr > '9' || chr < '0') {
          // invalid character in string!
          valid = false;
        }
      }

      if (!truncating) {
        if (c != i) thread_val *= 10;
        thread_val += chr - '0';
      }
    }

    out[row] = is_signed && sign < 1 ? thread_val * sign : thread_val;
  }

  auto const validity_int32 = warp.ballot(static_cast<int>(valid));
  if (warp.thread_rank() == 0) {
    validity[warp.meta_group_rank() + blockIdx.x * warp.meta_group_size()] = validity_int32;
  }
}

struct row_valid_fn {
  bool __device__ operator()(size_type const row) { return not col.is_valid(row); }
  column_device_view col;
};

struct string_to_integer_impl {
  template <typename T, typename std::enable_if_t<is_numeric<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const& string_col,
                                     bool ansi_mode,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    rmm::device_uvector<T> data(string_col.size(), stream);
    auto const num_words = bitmask_allocation_size_bytes(string_col.size()) / sizeof(bitmask_type);
    rmm::device_uvector<bitmask_type> null_mask(num_words, stream);

    dim3 const blocks(util::div_rounding_up_unsafe(string_col.size(), detail::NUM_THREADS));
    dim3 const threads{detail::NUM_THREADS};

    detail::string_to_integer_kernel<<<blocks, threads, 0, stream.value()>>>(
      data.data(),
      null_mask.data(),
      string_col.chars().data<char const>(),
      string_col.offsets().data<offset_type>(),
      string_col.size());

    auto col = std::make_unique<column>(
      data_type{type_to_id<T>()}, string_col.size(), data.release(), null_mask.release());

    if (ansi_mode) {
      auto const num_nulls = col->null_count();
      if (num_nulls > 0) {
        auto const cdv         = column_device_view::create(*col, stream);
        auto const first_error = thrust::find_if(rmm::exec_policy(stream),
                                                 thrust::make_counting_iterator(0),
                                                 thrust::make_counting_iterator(col->size()),
                                                 row_valid_fn{*cdv});

        offset_type string_bounds[2];
        cudaMemcpyAsync(&string_bounds,
                        &string_col.offsets().data<offset_type>()[*first_error],
                        sizeof(offset_type) * 2,
                        cudaMemcpyDeviceToHost,
                        stream.value());
        stream.synchronize();

        std::string dest;
        dest.resize(string_bounds[1] - string_bounds[0]);

        cudaMemcpyAsync(dest.data(),
                        &string_col.chars().data<char const>()[string_bounds[0]],
                        string_bounds[1] - string_bounds[0],
                        cudaMemcpyDeviceToHost,
                        stream.value());
        stream.synchronize();

        CUDF_EXPECTS(num_nulls == 0,
                     "String column had " + std::to_string(num_nulls) +
                       " parse errors, first error was line " + std::to_string(*first_error + 1) +
                       ": '" + dest + "'!");
      }
    }

    return col;
  }

  template <typename T, typename std::enable_if_t<!is_numeric<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const& string_col,
                                     bool ansi_mode,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    CUDF_FAIL("Invalid integer column type");
  }
};

}  // namespace detail

std::unique_ptr<column> string_to_integer(data_type dtype,
                                          strings_column_view const& string_col,
                                          bool ansi_mode,
                                          rmm::cuda_stream_view stream,
                                          rmm::mr::device_memory_resource* mr)
{
  return type_dispatcher(
    dtype, detail::string_to_integer_impl{}, string_col, ansi_mode, stream, mr);
}

}  // namespace spark_rapids_jni
