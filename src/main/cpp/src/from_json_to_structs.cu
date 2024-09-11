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
#include "json_parser.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/io/json.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <numeric>
#include <unordered_set>

namespace spark_rapids_jni {

namespace detail {
#if 0
namespace test {
/**
 * @brief TODO
 */
template <int block_size, int min_block_per_sm>
__launch_bounds__(block_size, min_block_per_sm) CUDF_KERNEL
  void from_json_kernel(cudf::column_device_view input, std::size_t num_threads_per_row)
{
  auto const tidx    = cudf::detail::grid_1d::global_thread_id();
  auto const row_idx = tidx / num_threads_per_row;
  if (row_idx >= input.size()) { return; }

  auto const path_idx = tidx % num_threads_per_row;
  if (path_idx >= path_data.size()) { return; }

  auto const& path         = path_data[path_idx];
  char* const dst          = path.out_buf + path.offsets[row_idx];
  bool is_valid            = false;
  cudf::size_type out_size = 0;

  auto const str = input.element<cudf::string_view>(row_idx);
  if (str.size_bytes() > 0) {
    json_parser p{char_range{str}};
    thrust::tie(is_valid, out_size) =
      evaluate_path(p, path.path_commands, dst, max_path_depth_exceeded);

    // We did not terminate the `evaluate_path` function early to reduce complexity of the code.
    // Instead, if max depth was encountered, we've just continued the evaluation until here
    // then discard the output entirely.
    if (p.max_nesting_depth_exceeded()) {
      *max_path_depth_exceeded = 1;
      return;
    }

    auto const max_size = path.offsets[row_idx + 1] - path.offsets[row_idx];
    if (out_size > max_size) { *(path.has_out_of_bound) = 1; }
  }

  // Write out `nullptr` in the output string_view to indicate that the output is a null.
  // The situation `out_stringviews == nullptr` should only happen if the kernel is launched a
  // second time due to out-of-bound write in the first launch.
  if (path.out_stringviews) {
    path.out_stringviews[row_idx] = {is_valid ? dst : nullptr, out_size};
  }
}

/**
 * @brief A utility class to launch the main kernel.
 */
struct kernel_launcher {
  static void exec(cudf::column_device_view const& input,
                   cudf::device_span<json_path_processing_data> path_data,
                   int8_t* max_path_depth_exceeded,
                   rmm::cuda_stream_view stream)
  {
    // The optimal values for block_size and min_block_per_sm were found through testing,
    // which are either 128-8 or 256-4. The pair 128-8 seems a bit better.
    static constexpr int block_size       = 128;
    static constexpr int min_block_per_sm = 8;

    // The number of threads for processing one input row is at least one warp.
    auto const num_threads_per_row =
      cudf::util::div_rounding_up_safe(path_data.size(),
                                       static_cast<std::size_t>(cudf::detail::warp_size)) *
      cudf::detail::warp_size;
    auto const num_blocks = cudf::util::div_rounding_up_safe(num_threads_per_row * input.size(),
                                                             static_cast<std::size_t>(block_size));
    get_json_object_kernel<block_size, min_block_per_sm>
      <<<num_blocks, block_size, 0, stream.value()>>>(
        input, path_data, num_threads_per_row, max_path_depth_exceeded);
  }
};

int64_t calc_scratch_size(cudf::strings_column_view const& input,
                          cudf::detail::input_offsetalator const& in_offsets,
                          rmm::cuda_stream_view stream)
{
  auto const max_row_size = thrust::transform_reduce(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(input.size()),
    cuda::proclaim_return_type<int64_t>(
      [in_offsets] __device__(auto const idx) { return in_offsets[idx + 1] - in_offsets[idx]; }),
    int64_t{0},
    thrust::maximum{});

  // We will use scratch buffers to store the output strings without knowing their sizes.
  // Since we do not know their sizes, we need to allocate the buffer a bit larger than the input
  // size so that we will not write output strings into an out-of-bound position.
  // Checking out-of-bound needs to be performed in the main kernel to make sure we will not have
  // data corruption.
  auto const scratch_size = [&, max_row_size = max_row_size] {
    // Pad the scratch buffer by an additional size that is a multiple of max row size.
    auto constexpr padding_rows = 10;
    return input.chars_size(stream) + max_row_size * padding_rows;
  }();
  return scratch_size;
}

/**
 * @brief Error handling using error markers gathered after kernel launch.
 *
 * If the input JSON has nesting depth exceeds the maximum allowed value, an exception will be
 * thrown as it is unacceptable. Otherwise, out of bound write is checked and returned.
 *
 * @param error_check The array of markers to check for error
 * @return A boolean value indicating if there is any out of bound write
 */
bool check_error(cudf::detail::host_vector<int8_t> const& error_check)
{
  // The last value is to mark if nesting depth has exceeded.
  CUDF_EXPECTS(error_check.back() == 0,
               "The processed input has nesting depth exceeds depth limit.");

  // Do not use parallel check since we do not have many elements.
  // The last element is not related, but its value is already `0` thus just check until
  // the end of the array for simplicity.
  return std::none_of(
    error_check.cbegin(), error_check.cend(), [](auto const val) { return val != 0; });
}

std::vector<std::unique_ptr<cudf::column>> get_json_object_batch(
  cudf::column_device_view const& input,
  cudf::detail::input_offsetalator const& in_offsets,
  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>> const&
    json_paths,
  int64_t scratch_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const [d_json_paths, h_json_paths, d_inst_names, h_inst_names] =
    construct_path_commands(json_paths, stream);

  auto const num_outputs = json_paths.size();
  std::vector<std::unique_ptr<cudf::column>> output;

  // The error check array contains markers denoting if there is any out-of-bound write occurs
  // (first `num_outputs` elements), or if the nesting depth exceeded its limits (the last element).
  rmm::device_uvector<int8_t> d_error_check(num_outputs + 1, stream);
  auto const d_max_path_depth_exceeded = d_error_check.data() + num_outputs;

  std::vector<rmm::device_uvector<char>> scratch_buffers;
  std::vector<rmm::device_uvector<thrust::pair<char const*, cudf::size_type>>> out_stringviews;
  std::vector<json_path_processing_data> h_path_data;
  scratch_buffers.reserve(json_paths.size());
  out_stringviews.reserve(json_paths.size());
  h_path_data.reserve(json_paths.size());

  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    auto const& path = json_paths[idx];
    if (path.size() > MAX_JSON_PATH_DEPTH) {
      CUDF_FAIL("JSON Path has depth exceeds the maximum allowed value.");
    }

    scratch_buffers.emplace_back(rmm::device_uvector<char>(scratch_size, stream));
    out_stringviews.emplace_back(rmm::device_uvector<thrust::pair<char const*, cudf::size_type>>{
      static_cast<std::size_t>(input.size()), stream});

    h_path_data.emplace_back(json_path_processing_data{d_json_paths[idx],
                                                       in_offsets,
                                                       out_stringviews.back().data(),
                                                       scratch_buffers.back().data(),
                                                       d_error_check.data() + idx});
  }
  auto d_path_data = cudf::detail::make_device_uvector_async(
    h_path_data, stream, rmm::mr::get_current_device_resource());
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), d_error_check.begin(), d_error_check.end(), 0);

  kernel_launcher::exec(input, d_path_data, d_max_path_depth_exceeded, stream);
  auto h_error_check = cudf::detail::make_host_vector_sync(d_error_check, stream);
  auto has_no_oob    = check_error(h_error_check);

  // If we didn't see any out-of-bound write, everything is good so far.
  // Just gather the output strings and return.
  if (has_no_oob) {
    for (auto const& out_sview : out_stringviews) {
      output.emplace_back(cudf::make_strings_column(out_sview, stream, mr));
    }
    return output;
  }
  // From here, we had out-of-bound write. Although this is very rare, it may still happen.

  std::vector<std::pair<rmm::device_buffer, cudf::size_type>> out_null_masks_and_null_counts;
  std::vector<std::pair<std::unique_ptr<cudf::column>, int64_t>> out_offsets_and_sizes;
  std::vector<rmm::device_uvector<char>> out_char_buffers;
  std::vector<std::size_t> oob_indices;

  // Check validity from the stored char pointers.
  auto const validator = [] __device__(thrust::pair<char const*, cudf::size_type> const item) {
    return item.first != nullptr;
  };

  // Rebuild the data only for paths that had out of bound write.
  h_path_data.clear();
  for (std::size_t idx = 0; idx < num_outputs; ++idx) {
    auto const& out_sview = out_stringviews[idx];

    if (h_error_check[idx]) {
      oob_indices.emplace_back(idx);
      output.emplace_back(nullptr);  // just placeholder.

      out_null_masks_and_null_counts.emplace_back(
        cudf::detail::valid_if(out_sview.begin(), out_sview.end(), validator, stream, mr));

      // The string sizes computed in the previous kernel call will be used to allocate a new char
      // buffer to store the output.
      auto const size_it = cudf::detail::make_counting_transform_iterator(
        0,
        cuda::proclaim_return_type<cudf::size_type>(
          [string_pairs = out_sview.data()] __device__(auto const idx) {
            return string_pairs[idx].second;
          }));
      out_offsets_and_sizes.emplace_back(cudf::strings::detail::make_offsets_child_column(
        size_it, size_it + input.size(), stream, mr));
      out_char_buffers.emplace_back(
        rmm::device_uvector<char>(out_offsets_and_sizes.back().second, stream, mr));

      h_path_data.emplace_back(
        json_path_processing_data{d_json_paths[idx],
                                  cudf::detail::offsetalator_factory::make_input_iterator(
                                    out_offsets_and_sizes.back().first->view()),
                                  nullptr /*out_stringviews*/,
                                  out_char_buffers.back().data(),
                                  d_error_check.data() + idx});
    } else {
      output.emplace_back(cudf::make_strings_column(out_sview, stream, mr));
    }
  }
  // These buffers are no longer needed.
  scratch_buffers.clear();
  out_stringviews.clear();

  // Push data to the GPU and launch the kernel again.
  d_path_data = cudf::detail::make_device_uvector_async(
    h_path_data, stream, rmm::mr::get_current_device_resource());
  thrust::uninitialized_fill(
    rmm::exec_policy(stream), d_error_check.begin(), d_error_check.end(), 0);
  kernel_launcher::exec(input, d_path_data, d_max_path_depth_exceeded, stream);
  h_error_check = cudf::detail::make_host_vector_sync(d_error_check, stream);
  has_no_oob    = check_error(h_error_check);

  // The last kernel call should not encounter any out-of-bound write.
  // If OOB is still detected, there must be something wrong happened.
  CUDF_EXPECTS(has_no_oob, "Unexpected out-of-bound write in get_json_object kernel.");

  for (std::size_t idx = 0; idx < oob_indices.size(); ++idx) {
    auto const out_idx = oob_indices[idx];
    output[out_idx] =
      cudf::make_strings_column(input.size(),
                                std::move(out_offsets_and_sizes[idx].first),
                                out_char_buffers[idx].release(),
                                out_null_masks_and_null_counts[idx].second,
                                std::move(out_null_masks_and_null_counts[idx].first));
  }
  return output;
}

std::unique_ptr<cudf::column> from_json_to_struct_bk(cudf::strings_column_view const& input,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  auto const num_outputs = json_paths.size();

  // Input is empty or all nulls - just return all null columns.
  if (input.is_empty() || input.size() == input.null_count()) {
    std::vector<std::unique_ptr<cudf::column>> output;
    for (std::size_t idx = 0; idx < num_outputs; ++idx) {
      output.emplace_back(std::make_unique<cudf::column>(input.parent(), stream, mr));
    }
    return output;
  }

  std::vector<std::size_t> sorted_indices(json_paths.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);  // Fill with 0, 1, 2, ...

  // Sort indices based on the corresponding paths.
  std::sort(sorted_indices.begin(), sorted_indices.end(), [&json_paths](size_t i, size_t j) {
    return json_paths[i] < json_paths[j];
  });

  auto const in_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  auto const scratch_size = calc_scratch_size(input, in_offsets, stream);
  if (memory_budget_bytes <= 0 && parallel_override <= 0) {
    parallel_override = static_cast<int>(sorted_indices.size());
  }
  auto const d_input_ptr = cudf::column_device_view::create(input.parent(), stream);
  std::vector<std::unique_ptr<cudf::column>> output(num_outputs);

  std::vector<cudf::host_span<std::tuple<path_instruction_type, std::string, int32_t> const>> batch;
  std::vector<std::size_t> output_ids;

  std::size_t starting_path = 0;
  while (starting_path < num_outputs) {
    std::size_t at = starting_path;
    batch.resize(0);
    output_ids.resize(0);
    if (parallel_override > 0) {
      int count = 0;
      while (at < num_outputs && count < parallel_override) {
        auto output_location = sorted_indices[at];
        batch.emplace_back(json_paths[output_location]);
        output_ids.push_back(output_location);
        at++;
        count++;
      }
    } else {
      long budget = 0;
      while (at < num_outputs && budget < memory_budget_bytes) {
        auto output_location = sorted_indices[at];
        batch.emplace_back(json_paths[output_location]);
        output_ids.push_back(output_location);
        at++;
        budget += scratch_size;
      }
    }
    auto tmp = get_json_object_batch(*d_input_ptr, in_offsets, batch, scratch_size, stream, mr);
    for (std::size_t i = 0; i < tmp.size(); i++) {
      std::size_t out_i = output_ids[i];
      output[out_i]     = std::move(tmp[i]);
    }
    starting_path = at;
  }
  return output;
}

}  // namespace test
#endif

void travel_path(
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>>& paths,
  std::vector<std::tuple<path_instruction_type, std::string, int32_t>>& current_path,
  std::unordered_set<std::size_t>& keep_quotes,
  std::string const& name,
  cudf::io::schema_element const& column_schema)
{
  current_path.emplace_back(path_instruction_type::NAMED, name, -1);
  if (column_schema.child_types.size() == 0) {  // leaf of the schema
    if (cudf::is_fixed_width(column_schema.type)) {
      // TODO: comment
      keep_quotes.insert(paths.size());
    }
    printf("column_schema type: %d\n", static_cast<int>(column_schema.type.id()));
    paths.push_back(current_path);  // this will copy
  } else {
    if (column_schema.type.id() != cudf::type_id::STRUCT) {
      CUDF_FAIL("Unsupported column type in schema");
    }

    paths.push_back(current_path);  // this will copy

    auto const last_path_size = paths.size();
    for (auto const& [child_name, child_schema] : column_schema.child_types) {
      travel_path(paths, current_path, keep_quotes, child_name, child_schema);
    }
  }
  current_path.pop_back();
}

std::pair<std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>>,
          std::unordered_set<std::size_t>>
flatten_schema_to_paths(std::vector<std::pair<std::string, cudf::io::schema_element>> const& schema)
{
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> paths;
  std::unordered_set<std::size_t> keep_quotes;

  std::vector<std::tuple<path_instruction_type, std::string, int32_t>> current_path;
  std::for_each(schema.begin(), schema.end(), [&](auto const& kv) {
    travel_path(paths, current_path, keep_quotes, kv.first, kv.second);
  });

  return {std::move(paths), std::move(keep_quotes)};
}

void assemble_column(std::size_t& column_order,
                     std::vector<std::unique_ptr<cudf::column>>& output,
                     std::vector<std::unique_ptr<cudf::column>>& read_columns,
                     std::string const& name,
                     cudf::io::schema_element const& column_schema,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  if (column_schema.child_types.size() == 0) {  // leaf of the schema
    output.emplace_back(std::move(read_columns[column_order]));
    ++column_order;
  } else {
    if (column_schema.type.id() != cudf::type_id::STRUCT) {
      CUDF_FAIL("Unsupported column type in schema");
    }

    std::vector<std::unique_ptr<cudf::column>> children;
    for (auto const& [child_name, child_schema] : column_schema.child_types) {
      assemble_column(column_order, children, read_columns, child_name, child_schema, stream, mr);
    }

    auto const null_count = read_columns[column_order]->null_count();
    auto const null_mask  = std::move(read_columns[column_order]->release().null_mask);
    ++column_order;

    // TODO: generate null mask from input.
    auto const num_rows = children.front()->size();
    output.emplace_back(cudf::make_structs_column(
      num_rows, std::move(children), null_count, std::move(*null_mask), stream, mr));
  }
}

std::vector<std::unique_ptr<cudf::column>> assemble_output(
  std::vector<std::pair<std::string, cudf::io::schema_element>> const& schema,
  std::vector<std::unique_ptr<cudf::column>>& read_columns,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> output;
  output.reserve(read_columns.size());

  std::size_t column_order{0};
  std::for_each(schema.begin(), schema.end(), [&](auto const& kv) {
    assemble_column(column_order, output, read_columns, kv.first, kv.second, stream, mr);
  });

  return output;
}

// Extern
std::vector<std::unique_ptr<cudf::column>> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const&
    json_paths,
  std::unordered_set<std::size_t> const& keep_quotes,
  int64_t memory_budget_bytes,
  int32_t parallel_override,
  bool allow_leading_zero_numbers,
  bool allow_non_numeric_numbers,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

std::vector<std::unique_ptr<cudf::column>> from_json_to_structs(
  cudf::strings_column_view const& input,
  std::vector<std::pair<std::string, cudf::io::schema_element>> const& schema,
  bool allow_leading_zero_numbers,
  bool allow_non_numeric_numbers,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  printf("line %d\n", __LINE__);
  fflush(stdout);
  auto const [json_paths, keep_quotes] = flatten_schema_to_paths(schema);

  printf("line %d\n", __LINE__);
  fflush(stdout);

#if 1
  for (auto const& path : json_paths) {
    printf("\n\npath: \n");
    for (auto node : path) {
      printf(".%s", std::get<1>(node).c_str());
    }
    printf("\n");
  }

  printf("keep quotes: \n");
  for (auto const i : keep_quotes) {
    printf("%d, ", (int)i);
  }
  printf("\n\n\n");
  fflush(stdout);

  auto ptr  = input.chars_begin(stream);
  auto size = input.chars_size(stream);
  std::vector<char> h_v(size);
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(h_v.data(), ptr, sizeof(char) * size, cudaMemcpyDefault, stream.value()));
  stream.synchronize();

  printf("input (size = %d): ", (int)size);
  for (auto c : h_v) {
    printf("%c", c);
  }
  printf("\n");

#endif

  auto tmp = get_json_object(input,
                             json_paths,
                             keep_quotes,
                             -1L,
                             -1,
                             allow_leading_zero_numbers,
                             allow_non_numeric_numbers,
                             stream,
                             mr);
  printf("line %d\n", __LINE__);
  fflush(stdout);

  if (1) {
    for (std::size_t i = 0; i < tmp.size(); ++i) {
      auto out  = cudf::strings_column_view{tmp[i]->view()};
      auto ptr  = out.chars_begin(stream);
      auto size = out.chars_size(stream);
      std::vector<char> h_v(size);
      CUDF_CUDA_TRY(
        cudaMemcpyAsync(h_v.data(), ptr, sizeof(char) * size, cudaMemcpyDefault, stream.value()));
      stream.synchronize();

      printf("out %d (size = %d): ", (int)i, (int)size);
      for (auto c : h_v) {
        printf("%c", c);
      }
      printf("\n");
    }
  }

  return assemble_output(schema, tmp, stream, mr);
}

}  // namespace detail

std::vector<std::unique_ptr<cudf::column>> from_json_to_structs(
  cudf::strings_column_view const& input,
  std::vector<std::pair<std::string, cudf::io::schema_element>> const& schema,
  bool allow_leading_zero_numbers,
  bool allow_non_numeric_numbers,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_json_to_structs(
    input, schema, allow_leading_zero_numbers, allow_non_numeric_numbers, stream, mr);
}

}  // namespace spark_rapids_jni
