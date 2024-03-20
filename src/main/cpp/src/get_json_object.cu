/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "get_json_object.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/json/json.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/optional.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>

namespace spark_rapids_jni {

namespace detail {
// namespace {

rmm::device_uvector<path_instruction> construct_path_commands(
  std::vector<std::tuple<path_instruction_type, std::string, int64_t>> const& instructions,
  cudf::string_scalar const& all_names_scalar,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  int name_pos = 0;

  // construct the path commands
  std::vector<path_instruction> path_commands;
  for (auto const& inst : instructions) {
    auto const& [type, name, index] = inst;
    switch (type) {
      case path_instruction_type::subscript:
        path_commands.emplace_back(path_instruction{path_instruction_type::subscript});
        break;
      case path_instruction_type::wildcard:
        path_commands.emplace_back(path_instruction{path_instruction_type::wildcard});
        break;
      case path_instruction_type::key:
        path_commands.emplace_back(path_instruction{path_instruction_type::key});
        path_commands.back().name =
          cudf::string_view(all_names_scalar.data() + name_pos, name.size());
        name_pos += name.size();
        break;
      case path_instruction_type::index:
        path_commands.emplace_back(path_instruction{path_instruction_type::index});
        path_commands.back().index = index;
        break;
      case path_instruction_type::named:
        path_commands.emplace_back(path_instruction{path_instruction_type::named});
        path_commands.back().name =
          cudf::string_view(all_names_scalar.data() + name_pos, name.size());
        name_pos += name.size();
        break;
      default: CUDF_FAIL("Invalid path instruction type");
    }
  }
  // convert to uvector
  return cudf::detail::make_device_uvector_sync(path_commands, stream, mr);
}

__device__ bool evaluate_path(json_parser<>& p,
                              json_generator<>& g,
                              write_style style,
                              path_instruction const* path_ptr,
                              int path_size)
{
  return path_evaluator::evaluate_path(p, g, style, path_ptr, path_size);
}

/**
 * @brief Parse a single json string using the provided command buffer
 *
 * @param j_parser The incoming json string and associated parser
 * @param path_ptr The command buffer to be applied to the string.
 * @param path_size Command buffer size
 * @param output Buffer used to store the results of the query
 * @returns A result code indicating success/fail/empty.
 */
__device__ inline bool parse_json_path(json_parser<>& j_parser,
                                       path_instruction const* path_ptr,
                                       size_t path_size,
                                       json_generator<>& output)
{
  j_parser.next_token();
  // JSON validation check
  if (json_token::ERROR == j_parser.get_current_token()) { return false; }

  return evaluate_path(j_parser, output, write_style::raw_style, path_ptr, path_size);
}

/**
 * @brief Parse a single json string using the provided command buffer
 *
 * This function exists primarily as a shim for debugging purposes.
 *
 * @param input The incoming json string
 * @param input_len Size of the incoming json string
 * @param commands The command buffer to be applied to the string. Always ends
 * with a path_operator_type::END
 * @param out_buf Buffer user to store the results of the query (nullptr in the
 * size computation step)
 * @param out_buf_size Size of the output buffer
 * @param options Options controlling behavior
 * @returns A pair containing the result code the output buffer.
 */
__device__ thrust::pair<bool, json_generator<>> get_json_object_single(
  char const* input,
  cudf::size_type input_len,
  path_instruction const* path_commands_ptr,
  int path_commands_size,
  char* out_buf,
  size_t out_buf_size)
{
  if (!out_buf) {
    // First step: preprocess sizes
    json_parser j_parser(input, input_len);
    json_generator generator(out_buf);
    bool success = parse_json_path(j_parser, path_commands_ptr, path_commands_size, generator);

    if (!success) {
      // generator may contain trash output, e.g.: generator writes some output,
      // then JSON format is invalid, the previous output becomes trash.
      // set output as zero to tell second step
      generator.set_output_len_zero();
    }
    return {success, generator};
  } else {
    // Second step: writes output
    // if output buf size is zero, pass in nullptr to avoid generator writing trash output
    char* actual_output = (0 == out_buf_size) ? nullptr : out_buf;
    json_parser j_parser(input, input_len);
    json_generator generator(actual_output);
    bool success = parse_json_path(j_parser, path_commands_ptr, path_commands_size, generator);
    return {success, generator};
  }
}

/**
 * @brief Kernel for running the JSONPath query.
 *
 * This kernel operates in a 2-pass way.  On the first pass, it computes
 * output sizes.  On the second pass it fills in the provided output buffers
 * (chars and validity)
 *
 * @param col Device view of the incoming string
 * @param commands JSONPath command buffer
 * @param output_offsets Buffer used to store the string offsets for the results
 * of the query
 * @param out_buf Buffer used to store the results of the query
 * @param out_validity Output validity buffer
 * @param out_valid_count Output count of # of valid bits
 * @param options Options controlling behavior
 */
template <int block_size>
__launch_bounds__(block_size) CUDF_KERNEL
  void get_json_object_kernel(cudf::column_device_view col,
                              path_instruction const* path_commands_ptr,
                              int path_commands_size,
                              cudf::size_type* d_sizes,
                              cudf::detail::input_offsetalator output_offsets,
                              thrust::optional<char*> out_buf,
                              thrust::optional<cudf::bitmask_type*> out_validity,
                              thrust::optional<cudf::size_type*> out_valid_count)
{
  auto tid          = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::thread_index_type{blockDim.x} * cudf::thread_index_type{gridDim.x};

  cudf::size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffff'ffffu, tid < col.size());
  while (tid < col.size()) {
    bool is_valid               = false;
    cudf::string_view const str = col.element<cudf::string_view>(tid);
    cudf::size_type output_size = 0;
    if (str.size_bytes() > 0) {
      char* dst = out_buf.has_value() ? out_buf.value() + output_offsets[tid] : nullptr;
      size_t const dst_size =
        out_buf.has_value() ? output_offsets[tid + 1] - output_offsets[tid] : 0;

      // process one single row
      auto [result, out] = get_json_object_single(
        str.data(), str.size_bytes(), path_commands_ptr, path_commands_size, dst, dst_size);
      output_size = out.get_output_len();
      if (result) { is_valid = true; }
    }

    // filled in only during the precompute step. during the compute step, the
    // offsets are fed back in so we do -not- want to write them out
    if (!out_buf.has_value()) { d_sizes[tid] = output_size; }

    // validity filled in only during the output step
    if (out_validity.has_value()) {
      uint32_t mask = __ballot_sync(active_threads, is_valid);
      // 0th lane of the warp writes the validity
      if (!(tid % cudf::detail::warp_size)) {
        out_validity.value()[cudf::word_index(tid)] = mask;
        warp_valid_count += __popc(mask);
      }
    }

    tid += stride;
    active_threads = __ballot_sync(active_threads, tid < col.size());
  }

  // sum the valid counts across the whole block
  if (out_valid_count) {
    cudf::size_type block_valid_count =
      cudf::detail::single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
    if (threadIdx.x == 0) { atomicAdd(out_valid_count.value(), block_valid_count); }
  }
}

std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  std::vector<std::tuple<path_instruction_type, std::string, int64_t>> const& instructions,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  if (col.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);

  // get a string buffer to store all the names and convert to device
  std::string all_names;
  for (auto const& inst : instructions) {
    all_names += std::get<1>(inst);
  }
  cudf::string_scalar all_names_scalar(all_names);
  // parse the json_path into a command buffer
  auto path_commands = construct_path_commands(instructions, all_names_scalar, stream, mr);

  // compute output sizes
  auto sizes = rmm::device_uvector<cudf::size_type>(
    col.size(), stream, rmm::mr::get_current_device_resource());
  auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(col.offsets());

  constexpr int block_size = 512;
  cudf::detail::grid_1d const grid{col.size(), block_size};
  auto cdv = cudf::column_device_view::create(col.parent(), stream);
  // preprocess sizes (returned in the offsets buffer)
  get_json_object_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(*cdv,
                                                                         path_commands.data(),
                                                                         path_commands.size(),
                                                                         sizes.data(),
                                                                         d_offsets,
                                                                         thrust::nullopt,
                                                                         thrust::nullopt,
                                                                         thrust::nullopt);

  // convert sizes to offsets
  auto [offsets, output_size] =
    cudf::strings::detail::make_offsets_child_column(sizes.begin(), sizes.end(), stream, mr);
  d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // allocate output string column
  rmm::device_uvector<char> chars(output_size, stream, mr);

  // potential optimization : if we know that all outputs are valid, we could
  // skip creating the validity mask altogether
  rmm::device_buffer validity =
    cudf::detail::create_null_mask(col.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  // compute results
  rmm::device_scalar<cudf::size_type> d_valid_count{0, stream};

  get_json_object_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *cdv,
      path_commands.data(),
      path_commands.size(),
      sizes.data(),
      d_offsets,
      chars.data(),
      static_cast<cudf::bitmask_type*>(validity.data()),
      d_valid_count.data());

  auto result = make_strings_column(col.size(),
                                    std::move(offsets),
                                    chars.release(),
                                    col.size() - d_valid_count.value(stream),
                                    std::move(validity));
  // unmatched array query may result in unsanitized '[' value in the result
  if (cudf::detail::has_nonempty_nulls(result->view(), stream)) {
    result = cudf::detail::purge_nonempty_nulls(result->view(), stream, mr);
  }
  return result;
}

// }  // namespace

}  // namespace detail

std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  std::vector<std::tuple<path_instruction_type, std::string, int64_t>> const& instructions,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return detail::get_json_object(col, instructions, stream, mr);
}

}  // namespace spark_rapids_jni
