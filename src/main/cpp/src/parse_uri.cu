/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "parse_uri.hpp"

#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>

namespace spark_rapids_jni {

using namespace cudf;

namespace detail {
namespace {

// utility to validate a character is valid in a URI
constexpr bool is_valid_character(char ch, bool alphanum_only)
{
  if (alphanum_only) {
    if (ch >= '-' && ch <= '9' && ch != '/') return true;  // 0-9 and .-
    if (ch >= 'A' && ch <= 'Z') return true;               // A-Z
    if (ch >= 'a' && ch <= 'z') return true;               // a-z
  } else {
    if (ch >= '!' && ch <= ';' && ch != '"') return true;  // 0-9 and !#%&'()*+,-./
    if (ch >= '=' && ch <= 'Z' && ch != '>') return true;  // A-Z and =?@
    if (ch >= '_' && ch <= 'z' && ch != '`') return true;  // a-z and _
  }
  return false;
}

/**
 * @brief Count the number of characters of each string after parsing the protocol.
 *
 * @tparam num_warps_per_threadblock Number of warps in a threadblock. This template argument must
 * match the launch configuration, i.e. the kernel must be launched with
 * `num_warps_per_threadblock * cudf::detail::warp_size` threads per threadblock.
 * @tparam char_block_size Number of characters which will be loaded into the shared memory at a
 * time.
 *
 * @param in_strings Input string column
 * @param out_counts Number of characters in each decode URL
 * @param out_validity Bitmask of validity data, updated in funcion
 */
template <size_type num_warps_per_threadblock, size_type char_block_size>
__global__ void parse_uri_protocol_char_counter(column_device_view const in_strings,
                                                size_type* const out_counts,
                                                bitmask_type* out_validity)
{
  __shared__ char temporary_buffer[num_warps_per_threadblock][char_block_size];
  __shared__ typename cub::WarpScan<int8_t>::TempStorage cub_storage[num_warps_per_threadblock];
  __shared__ bool found_token[num_warps_per_threadblock];

  auto const global_thread_id = cudf::detail::grid_1d::global_thread_id();
  auto const global_warp_id   = static_cast<size_type>(global_thread_id / cudf::detail::warp_size);
  auto const local_warp_id    = static_cast<size_type>(threadIdx.x / cudf::detail::warp_size);
  auto const warp_lane        = static_cast<size_type>(threadIdx.x % cudf::detail::warp_size);
  auto const nwarps     = static_cast<size_type>(gridDim.x * blockDim.x / cudf::detail::warp_size);
  char* in_chars_shared = temporary_buffer[local_warp_id];

  // Loop through strings, and assign each string to a warp.
  for (thread_index_type tidx = global_warp_id; tidx < in_strings.size(); tidx += nwarps) {
    auto const row_idx = static_cast<size_type>(tidx);
    if (in_strings.is_null(row_idx)) {
      if (warp_lane == 0) out_counts[row_idx] = 0;
      continue;
    }

    auto const in_string     = in_strings.element<string_view>(row_idx);
    auto const in_chars      = in_string.data();
    auto const string_length = in_string.size_bytes();
    auto const nblocks       = cudf::util::div_rounding_up_unsafe(string_length, char_block_size);
    size_type output_string_size = 0;

    // valid until proven otherwise
    bool valid{true};

    // Use the last thread of the warp to initialize `found_token` to false.
    if (warp_lane == cudf::detail::warp_size - 1) { found_token[local_warp_id] = false; }

    for (size_type block_idx = 0; block_idx < nblocks && valid; block_idx++) {
      auto const string_length_block =
        std::min(char_block_size, string_length - char_block_size * block_idx);

      // Each warp collectively loads input characters of the current block to the shared memory.
      for (auto char_idx = warp_lane; char_idx < string_length_block;
           char_idx += cudf::detail::warp_size) {
        auto const in_idx         = block_idx * char_block_size + char_idx;
        in_chars_shared[char_idx] = in_idx < string_length ? in_chars[in_idx] : 0;
      }

      __syncwarp();

      // `char_idx_start` represents the start character index of the current warp.
      for (size_type char_idx_start = 0; char_idx_start < string_length_block;
           char_idx_start += cudf::detail::warp_size) {
        auto const char_idx      = char_idx_start + warp_lane;
        char const* const ch_ptr = in_chars_shared + char_idx;

        // need to know if the character we are validating is before or after the token
        // as valid characters changes. Default to 1 to handle the case where we have
        // alreayd found the token and do not search for it again.
        int8_t out_tokens{1};
        if (!found_token[local_warp_id]) {
          // Warp-wise prefix sum to establish tokens of string.
          // All threads in the warp participate in the prefix sum, even if `char_idx` is beyond
          // `string_length_block`.
          int8_t const is_token = (char_idx < string_length_block && *ch_ptr == ':') ? 1 : 0;
          cub::WarpScan<int8_t>(cub_storage[local_warp_id]).InclusiveSum(is_token, out_tokens);
        }

        auto const before_token = out_tokens == 0;
        valid                   = valid && __ballot_sync(0xffffffff,
                                       (char_idx >= string_length_block ||
                                        is_valid_character(*ch_ptr, before_token))
                                                           ? 0
                                                           : 1) == 0;
        if (!valid) {
          // last thread in warp sets validity
          if (warp_lane == cudf::detail::warp_size - 1) {
            clear_bit(out_validity, row_idx);
            out_counts[row_idx] = 0;
          }
          break;
        }

        // if we have already found our token, no more string copy we only need to validate
        // characters
        if (!found_token[local_warp_id]) {
          // If the current character is before the token we will output the character.
          int8_t const out_size = (char_idx >= string_length_block || out_tokens > 0) ? 0 : 1;

          // Warp-wise prefix sum to establish output location of the current thread.
          // All threads in the warp participate in the prefix sum, even if `char_idx` is beyond
          // `string_length_block`.
          int8_t out_offset;
          cub::WarpScan<int8_t>(cub_storage[local_warp_id]).InclusiveSum(out_size, out_offset);

          // last thread of the warp updates offsets and token since it has the last offset and
          // token value
          if (warp_lane == cudf::detail::warp_size - 1) {
            output_string_size += out_offset;
            found_token[local_warp_id] = out_tokens > 0;
          }
        }

        __syncwarp();
      }
    }

    // last thread of the warp sets output size
    if (warp_lane == cudf::detail::warp_size - 1) {
      if (!found_token[local_warp_id]) {
        clear_bit(out_validity, row_idx);
        out_counts[row_idx] = 0;
      } else if (valid) {
        out_counts[row_idx] = output_string_size;
      }
    }
  }
}

/**
 * @brief Parse protocol and copy from the input string column to the output char buffer.
 *
 * @tparam num_warps_per_threadblock Number of warps in a threadblock. This template argument must
 * match the launch configuration, i.e. the kernel must be launched with
 * `num_warps_per_threadblock * cudf::detail::warp_size` threads per threadblock.
 * @tparam char_block_size Number of characters which will be loaded into the shared memory at a
 * time.
 *
 * @param in_strings Input string column
 * @param in_validity Validity vector of output column
 * @param out_chars Character buffer for the output string column
 * @param out_offsets Offset value of each string associated with `out_chars`
 */
template <size_type num_warps_per_threadblock, size_type char_block_size>
__global__ void parse_uri_to_protocol(column_device_view const in_strings,
                                      bitmask_type* in_validity,
                                      char* const out_chars,
                                      size_type const* const out_offsets)
{
  __shared__ char temporary_buffer[num_warps_per_threadblock][char_block_size];
  __shared__ typename cub::WarpScan<int8_t>::TempStorage cub_storage[num_warps_per_threadblock];
  __shared__ size_type out_idx[num_warps_per_threadblock];
  __shared__ bool found_token[num_warps_per_threadblock];

  auto const global_thread_id = cudf::detail::grid_1d::global_thread_id();
  auto const global_warp_id   = static_cast<size_type>(global_thread_id / cudf::detail::warp_size);
  auto const local_warp_id    = static_cast<size_type>(threadIdx.x / cudf::detail::warp_size);
  auto const warp_lane        = static_cast<size_type>(threadIdx.x % cudf::detail::warp_size);
  auto const nwarps     = static_cast<size_type>(gridDim.x * blockDim.x / cudf::detail::warp_size);
  char* in_chars_shared = temporary_buffer[local_warp_id];

  // Loop through strings, and assign each string to a warp
  for (thread_index_type tidx = global_warp_id; tidx < in_strings.size(); tidx += nwarps) {
    auto const row_idx = static_cast<size_type>(tidx);
    if (!bit_is_set(in_validity, row_idx)) { continue; }

    auto const in_string     = in_strings.element<string_view>(row_idx);
    auto const in_chars      = in_string.data();
    auto const string_length = in_string.size_bytes();
    auto out_chars_string    = out_chars + out_offsets[row_idx];
    auto const nblocks       = cudf::util::div_rounding_up_unsafe(string_length, char_block_size);

    // Use the last thread of the warp to initialize `out_idx` to 0 and `found_token` to false.
    if (warp_lane == cudf::detail::warp_size - 1) {
      out_idx[local_warp_id]     = 0;
      found_token[local_warp_id] = false;
    }

    __syncwarp();

    for (size_type block_idx = 0; block_idx < nblocks && !found_token[local_warp_id]; block_idx++) {
      auto const string_length_block =
        std::min(char_block_size, string_length - char_block_size * block_idx);

      // Each warp collectively loads input characters of the current block to shared memory.
      for (auto char_idx = warp_lane; char_idx < string_length_block;
           char_idx += cudf::detail::warp_size) {
        auto const in_idx         = block_idx * char_block_size + char_idx;
        in_chars_shared[char_idx] = in_idx >= 0 && in_idx < string_length ? in_chars[in_idx] : 0;
      }

      __syncwarp();

      // `char_idx_start` represents the start character index of the current warp.
      for (size_type char_idx_start = 0;
           char_idx_start < string_length_block && !found_token[local_warp_id];
           char_idx_start += cudf::detail::warp_size) {
        auto const char_idx      = char_idx_start + warp_lane;
        char const* const ch_ptr = in_chars_shared + char_idx;

        // Warp-wise prefix sum to establish tokens of string.
        // All threads in the warp participate in the prefix sum, even if `char_idx` is beyond
        // `string_length_block`.
        int8_t const is_token = (char_idx < string_length_block && *ch_ptr == ':') ? 1 : 0;
        int8_t out_tokens;
        cub::WarpScan<int8_t>(cub_storage[local_warp_id]).InclusiveSum(is_token, out_tokens);

        // If the current character is before the token we will output the character.
        int8_t const out_size = (char_idx >= string_length_block || out_tokens > 0) ? 0 : 1;

        // Warp-wise prefix sum to establish output location of the current thread.
        // All threads in the warp participate in the prefix sum, even if `char_idx` is beyond
        // `string_length_block`.
        int8_t out_offset;
        cub::WarpScan<int8_t>(cub_storage[local_warp_id]).ExclusiveSum(out_size, out_offset);

        // out_size of 1 means this thread writes a byte
        if (out_size == 1) { out_chars_string[out_idx[local_warp_id] + out_offset] = *ch_ptr; }

        // last thread of the warp updates the offset and the token
        if (warp_lane == cudf::detail::warp_size - 1) {
          out_idx[local_warp_id] += (out_offset + out_size);
          found_token[local_warp_id] = out_tokens > 0;
        }

        __syncwarp();
      }
    }
  }
}

}  // namespace

std::unique_ptr<column> parse_uri_to_protocol(strings_column_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = input.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  constexpr size_type num_warps_per_threadblock = 4;
  constexpr size_type threadblock_size = num_warps_per_threadblock * cudf::detail::warp_size;
  constexpr size_type char_block_size  = 256;
  auto const num_threadblocks =
    std::min(65536, cudf::util::div_rounding_up_unsafe(strings_count, num_warps_per_threadblock));

  auto offset_count    = strings_count + 1;
  auto const d_strings = column_device_view::create(input.parent(), stream);

  // build offsets column
  auto offsets_column = make_numeric_column(
    data_type{type_to_id<size_type>()}, offset_count, mask_state::UNALLOCATED, stream, mr);

  // copy null mask
  rmm::device_buffer null_mask =
    input.parent().nullable()
      ? cudf::detail::copy_bitmask(input.parent(), stream, mr)
      : cudf::detail::create_null_mask(input.size(), mask_state::ALL_VALID, stream, mr);

  // count number of bytes in each string after parsing and store it in offsets_column
  auto offsets_view         = offsets_column->view();
  auto offsets_mutable_view = offsets_column->mutable_view();
  parse_uri_protocol_char_counter<num_warps_per_threadblock, char_block_size>
    <<<num_threadblocks, threadblock_size, 0, stream.value()>>>(
      *d_strings,
      offsets_mutable_view.begin<size_type>(),
      reinterpret_cast<bitmask_type*>(null_mask.data()));

  // use scan to transform number of bytes into offsets
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         offsets_view.begin<size_type>(),
                         offsets_view.end<size_type>(),
                         offsets_mutable_view.begin<size_type>());

  // copy the total number of characters of all strings combined (last element of the offset column)
  // to the host memory
  auto out_chars_bytes = cudf::detail::get_value<size_type>(offsets_view, offset_count - 1, stream);

  // create the chars column
  auto chars_column = cudf::strings::detail::create_chars_child_column(out_chars_bytes, stream, mr);
  auto d_out_chars  = chars_column->mutable_view().data<char>();

  // parse and copy the characters from the input column to the output column
  parse_uri_to_protocol<num_warps_per_threadblock, char_block_size>
    <<<num_threadblocks, threadblock_size, 0, stream.value()>>>(
      *d_strings,
      reinterpret_cast<bitmask_type*>(null_mask.data()),
      d_out_chars,
      offsets_column->view().begin<size_type>());

  auto null_count =
    cudf::null_count(reinterpret_cast<bitmask_type*>(null_mask.data()), 0, strings_count);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask));
}

}  // namespace detail

// external API

std::unique_ptr<column> parse_uri_to_protocol(strings_column_view const& input,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::parse_uri_to_protocol(input, stream, mr);
}

}  // namespace spark_rapids_jni