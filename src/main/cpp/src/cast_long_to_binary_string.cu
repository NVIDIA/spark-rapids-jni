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
struct long_to_binary_string_fn {
  cudf::column_device_view d_longs;
  cudf::size_type* d_sizes;
  char* d_chars;
  cudf::detail::input_offsetalator d_offsets;

  __device__ void long_to_binary_string(cudf::size_type idx)
  {
    auto const value = d_longs.element<int64_t>(idx);
    char* d_buffer   = d_chars + d_offsets[idx];
    for (auto i = d_sizes[idx] - 1; i >= 0; --i) {
      *d_buffer++ = '0' + ((value & (1UL << i)) >> i);
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
      // If the value is 0, the size should be 1
      d_sizes[idx] = max(1, 64 - __clzll(d_longs.element<int64_t>(idx)));
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> long_to_binary_string(cudf::column_view const& input,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);

  CUDF_EXPECTS(input.type().id() == cudf::type_id::INT64, "Input column must be long type");

  auto const d_column = cudf::column_device_view::create(input, stream);

  auto [offsets, chars] = cudf::strings::detail::make_strings_children(
    long_to_binary_string_fn{*d_column}, input.size(), stream, mr);

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
