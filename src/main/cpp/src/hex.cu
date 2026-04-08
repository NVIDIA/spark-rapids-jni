/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include "nvtx_ranges.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace spark_rapids_jni {

namespace detail {
namespace {

/**
 * @brief Functor for converting each byte of a string to its 2-character hex representation.
 *
 * Uses the standard two-pass pattern required by make_strings_children:
 * - Pass 1 (d_chars == nullptr): compute output size for each row (input bytes * 2)
 * - Pass 2 (d_chars != nullptr): write uppercase hex characters to output buffer
 */
struct bytes_to_hex_fn {
  cudf::column_device_view d_strings;
  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    auto const nbytes = d_str.size_bytes();
    if (d_chars) {
      auto* out = d_chars + d_offsets[idx];
      auto* in  = reinterpret_cast<uint8_t const*>(d_str.data());
      for (cudf::size_type i = 0; i < nbytes; ++i) {
        uint8_t const byte = in[i];
        uint8_t const hi   = byte >> 4;
        uint8_t const lo   = byte & 0x0F;
        out[i * 2]         = hi < 10 ? '0' + hi : 'A' + (hi - 10);
        out[i * 2 + 1]     = lo < 10 ? '0' + lo : 'A' + (lo - 10);
      }
    } else {
      d_sizes[idx] = nbytes * 2;
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> bytes_to_hex(cudf::strings_column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::type_id::STRING); }

  auto const d_column = cudf::column_device_view::create(input.parent(), stream);
  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    bytes_to_hex_fn{*d_column}, input.size(), stream, mr);

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets_column),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> bytes_to_hex(cudf::strings_column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  SRJ_FUNC_RANGE();
  return detail::bytes_to_hex(input, stream, mr);
}

}  // namespace spark_rapids_jni
