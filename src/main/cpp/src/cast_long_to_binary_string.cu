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
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace spark_rapids_jni {

namespace detail {
namespace {

template <typename LongType>
struct long_to_binary_string_fn {
  cudf::column_device_view d_longs;
  cudf::size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ cudf::size_type compute_output_size(LongType value)
  {
    return max(64 - __clzll(value), 1);  // If the value is 0, the output size should be 1
  }

  __device__ void long_to_binary_string(cudf::size_type idx)
  {
    auto const value = d_longs.element<LongType>(idx);
    char* d_buffer   = d_chars + d_offsets[idx];
    for (auto i = d_sizes[idx] - 1; i >= 0; --i) {
      *d_buffer++ = value & (1LL << i) ? '1' : '0';
    }
  }

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_longs.is_null(idx)) {
      if (d_chars == nullptr) { d_sizes[idx] = 0; }
      return;
    }
    if (d_chars != nullptr) {
      long_to_binary_string(idx);
    } else {
      d_sizes[idx] = compute_output_size(d_longs.element<LongType>(idx));
    }
  }
};

struct dispatch_long_to_binary_string_fn {
  template <typename LongType, CUDF_ENABLE_IF(std::is_same_v<LongType, std::int64_t>)>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    auto const d_column = cudf::column_device_view::create(input, stream);

    auto [offsets, chars] = cudf::strings::detail::make_strings_children(
      long_to_binary_string_fn<LongType>{*d_column}, input.size(), stream, mr);

    return cudf::make_strings_column(input.size(),
                                     std::move(offsets),
                                     chars.release(),
                                     input.null_count(),
                                     cudf::detail::copy_bitmask(input, stream, mr));
  }

  template <typename LongType, CUDF_ENABLE_IF(not std::is_same_v<LongType, std::int64_t>)>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const&,
                                           rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref) const
  {
    CUDF_FAIL("Values for long_to_binary_string function must be a long type.");
  }
};

}  // namespace

std::unique_ptr<cudf::column> long_to_binary_string(cudf::column_view const& input,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);
  return type_dispatcher(input.type(), dispatch_long_to_binary_string_fn{}, input, stream, mr);
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
