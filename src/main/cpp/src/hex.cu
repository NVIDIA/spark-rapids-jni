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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace spark_rapids_jni {

namespace detail {
namespace {

/**
 * @brief Functor to compute output string sizes from input offsets.
 *
 * Each input byte produces 2 hex characters, so output size = input byte count * 2.
 * This avoids a separate sizing kernel by deriving sizes directly from offsets.
 */
struct output_size_fn {
  cudf::detail::input_offsetalator d_input_offsets;
  cudf::size_type col_offset;

  __device__ cudf::size_type operator()(cudf::size_type idx) const
  {
    return static_cast<cudf::size_type>(
      (d_input_offsets[col_offset + idx + 1] - d_input_offsets[col_offset + idx]) * 2);
  }
};

/**
 * @brief Functor to convert each byte of a string to its 2-character hex representation.
 */
struct write_hex_fn {
  cudf::column_device_view d_strings;
  cudf::detail::input_offsetalator d_offsets;
  char* d_chars;

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    auto const nbytes = d_str.size_bytes();
    auto* out         = d_chars + d_offsets[idx];
    auto const* in    = reinterpret_cast<uint8_t const*>(d_str.data());
    for (cudf::size_type i = 0; i < nbytes; ++i) {
      uint8_t const byte = in[i];
      uint8_t const hi   = byte >> 4;
      uint8_t const lo   = byte & 0x0F;
      out[i * 2]         = hi + (hi < 10 ? '0' : 'A' - 10);
      out[i * 2 + 1]     = lo + (lo < 10 ? '0' : 'A' - 10);
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> bytes_to_hex(cudf::strings_column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return cudf::make_empty_column(cudf::type_id::STRING); }

  // Build output offsets directly from input offsets — no sizing kernel needed.
  auto const d_input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets());
  auto sizes_itr = cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0}, output_size_fn{d_input_offsets, input.offset()});
  auto [offsets_column, total_bytes] = cudf::strings::detail::make_offsets_child_column(
    sizes_itr, sizes_itr + input.size(), stream, mr);

  // Write hex chars in a single pass.
  auto chars = rmm::device_uvector<char>(total_bytes, stream, mr);
  if (total_bytes > 0) {
    auto const d_column = cudf::column_device_view::create(input.parent(), stream);
    auto const d_out_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());
    thrust::for_each(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     thrust::make_counting_iterator(input.size()),
                     write_hex_fn{*d_column, d_out_offsets, chars.data()});
  }

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
