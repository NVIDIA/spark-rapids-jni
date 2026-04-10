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
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace spark_rapids_jni {

namespace detail {
namespace {

/**
 * @brief Functor to convert each byte of a string to its 2-character hex representation.
 *
 * Output position is derived from the input string pointer, avoiding a separate
 * output offsets array read: out_offset = (in_ptr - chars_begin) * 2.
 */
struct write_hex_fn {
  cudf::column_device_view d_strings;
  char const* d_chars_begin;  // start of input chars buffer
  char* d_out;

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    auto const nbytes = d_str.size_bytes();
    auto const* in    = reinterpret_cast<uint8_t const*>(d_str.data());
    auto* out         = d_out + (d_str.data() - d_chars_begin) * 2;
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

  auto const num_strings = input.size();

  // Output offsets = input offsets * 2 (each byte produces 2 hex chars).
  // This avoids both a sizing kernel and cudf detail APIs.
  auto const d_in_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets());
  auto offsets_column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                  num_strings + 1,
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    d_in_offsets,
                    d_in_offsets + num_strings + 1,
                    offsets_column->mutable_view().data<int32_t>(),
                    cuda::proclaim_return_type<int32_t>(
                      [] __device__(int64_t offset) { return static_cast<int32_t>(offset * 2); }));

  // Total output bytes = total input char bytes * 2.
  auto const total_bytes = input.chars_size(stream) * 2;

  // Write hex chars in a single pass.
  auto chars = rmm::device_uvector<char>(total_bytes, stream, mr);
  if (total_bytes > 0) {
    auto const d_column = cudf::column_device_view::create(input.parent(), stream);
    thrust::for_each(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator(cudf::size_type{0}),
                     thrust::make_counting_iterator(num_strings),
                     write_hex_fn{*d_column, input.chars_begin(stream), chars.data()});
  }

  return cudf::make_strings_column(num_strings,
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
