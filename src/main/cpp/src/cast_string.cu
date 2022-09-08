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
                                         bitmask_type const* incoming_null_mask,
                                         size_type num_rows,
                                         bool ansi_mode)
{
  auto const group = cooperative_groups::this_thread_block();
  auto const warp  = cooperative_groups::tiled_partition<cudf::detail::warp_size>(group);

  // each thread takes a row and marches it and builds the integer for that row
  auto const row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) { return; }
  auto const active      = cooperative_groups::coalesced_threads();
  auto const row_start   = offsets[row];
  auto const len         = offsets[row + 1] - row_start;
  bool const valid_entry = incoming_null_mask == nullptr || bit_is_set(incoming_null_mask, row);
  bool valid             = valid_entry && len > 0;
  T thread_val           = 0;
  int i                  = 0;
  T sign                 = 1;
  constexpr bool is_signed_type = std::is_signed_v<T>;

  if (valid) {
    // skip leading whitespace
    while (i < len && is_whitespace(chars[i + row_start])) {
      i++;
    }

    // check for leading +-
    if constexpr (is_signed_type) {
      if (i < len && (chars[i + row_start] == '+' || chars[i + row_start] == '-')) {
        if (chars[i + row_start] == '-') { sign = -1; }
        i++;
      }
    }

    // if there is no data left, this is invalid
    if (i == len) { valid = false; }

    bool truncating          = false;
    bool trailing_whitespace = false;
    for (int c = i; c < len; ++c) {
      auto const chr = chars[c + row_start];
      // only whitespace is allowed after we find trailing whitespace
      if (trailing_whitespace && !is_whitespace(chr)) {
        valid = false;
        break;
      } else if (!truncating && chr == '.' && !ansi_mode) {
        // Values are truncated after a decimal point. However, invalid characters AFTER this
        // decimal point will still invalidate this entry.
        truncating = true;
      } else {
        if (chr > '9' || chr < '0') {
          if (is_whitespace(chr) && c != i) {
            trailing_whitespace = true;
          } else {
            // invalid character in string!
            valid = false;
            break;
          }
        }
      }

      if (!truncating && !trailing_whitespace) {
        if (c != i) {
          if (is_signed_type && sign < 0) {
            auto constexpr minval = std::numeric_limits<T>::min() / 10;
            if (thread_val < minval) {
              // overflow
              valid = false;
              break;
            }
          } else {
            auto constexpr maxval = std::numeric_limits<T>::max() / 10;
            if (thread_val > maxval) {
              // overflow
              valid = false;
              break;
            }
          }

          thread_val *= 10;
        }
        if (is_signed_type && sign < 0) {
          auto const c       = chr - '0';
          auto const min_val = std::numeric_limits<T>::min() + c;
          if (thread_val < min_val) {
            // overflow
            valid = false;
            break;
          }
          thread_val -= c;
        } else {
          auto const c       = chr - '0';
          auto const max_val = std::numeric_limits<T>::max() - c;
          if (thread_val > max_val) {
            // overflow
            valid = false;
            break;
          }
          thread_val += c;
        }
      }
    }

    out[row] = thread_val;
  }

  auto const validity_int32 = warp.ballot(static_cast<int>(valid));
  if (warp.thread_rank() == 0) {
    validity[warp.meta_group_rank() + blockIdx.x * warp.meta_group_size()] = validity_int32;
  }
}

struct row_valid_fn {
  bool __device__ operator()(size_type const row)
  {
    return not bit_is_set(null_mask, row) &&
           (string_col_null_mask == nullptr || bit_is_set(string_col_null_mask, row));
  }

  bitmask_type const* null_mask;
  bitmask_type const* string_col_null_mask;
};

struct string_to_integer_impl {
  template <typename T, typename std::enable_if_t<is_numeric<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const& string_col,
                                     bool ansi_mode,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    rmm::device_uvector<T> data(string_col.size(), stream, mr);
    auto const num_words = bitmask_allocation_size_bytes(string_col.size()) / sizeof(bitmask_type);
    rmm::device_uvector<bitmask_type> null_mask(num_words, stream, mr);

    dim3 const blocks(util::div_rounding_up_unsafe(string_col.size(), detail::NUM_THREADS));
    dim3 const threads{detail::NUM_THREADS};

    detail::string_to_integer_kernel<<<blocks, threads, 0, stream.value()>>>(
      data.data(),
      null_mask.data(),
      string_col.chars().data<char const>(),
      string_col.offsets().data<offset_type>(),
      string_col.null_mask(),
      string_col.size(),
      ansi_mode);

    auto col = std::make_unique<column>(
      data_type{type_to_id<T>()}, string_col.size(), data.release(), null_mask.release());

    if (ansi_mode) {
      auto const num_nulls      = col->null_count();
      auto const incoming_nulls = string_col.null_count();
      auto const num_errors     = num_nulls - incoming_nulls;
      if (num_errors > 0) {
        auto const first_error =
          thrust::find_if(rmm::exec_policy(stream),
                          thrust::make_counting_iterator(0),
                          thrust::make_counting_iterator(col->size()),
                          row_valid_fn{col->view().null_mask(), string_col.null_mask()});

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

        throw cast_error(*first_error, dest);
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

/**
 * @brief Convert a string column into an integer column.
 *
 * @param dtype Type of column to return.
 * @param string_col Incoming string column to convert to integers.
 * @param ansi_mode If true, strict conversion and throws on erorr.
 *                  If false, null invalid entries.
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> Integer column that was created from string_col.
 */
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
