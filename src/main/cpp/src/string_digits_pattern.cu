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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/atomic>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace spark_rapids_jni {

// /**
//  * @brief Check if `d_target` appears in a row in `d_strings`.
//  *
//  * This executes as a warp per string/row and performs well for longer strings.
//  * @see AVG_CHAR_BYTES_THRESHOLD
//  *
//  * @param d_strings Column of input strings
//  * @param d_target String to search for in each row of `d_strings`
//  * @param d_results Indicates which rows contain `d_target`
//  */
// __global__ static void string_digits_pattern_warp_parallel_fn(cudf::column_device_view const
// d_strings,
//                                           cudf::string_view const d_target,
//                                           bool* d_results)
// {
//   cudf::size_type const idx = static_cast<cudf::size_type>(threadIdx.x + blockIdx.x *
//   blockDim.x); using warp_reduce   = cub::WarpReduce<bool>;
//   __shared__ typename warp_reduce::TempStorage temp_storage;

//   if (idx >= (d_strings.size() * cudf::detail::warp_size)) { return; }

//   auto const str_idx  = idx / cudf::detail::warp_size;
//   auto const lane_idx = idx % cudf::detail::warp_size;
//   if (d_strings.is_null(str_idx)) { return; }
//   // get the string for this warp
//   auto const d_str = d_strings.element<cudf::string_view>(str_idx);
//   // each thread of the warp will check just part of the string
//   auto found = false;
//   for (auto i = static_cast<cudf::size_type>(idx % cudf::detail::warp_size);
//       !found && ((i + d_target.size_bytes()) <= d_str.size_bytes());
//       i += cudf::detail::warp_size) {
//     // check the target matches this part of the d_str data
//     if (d_target.compare(d_str.data() + i, d_target.size_bytes()) == 0) { found = true; }
//   }
//   auto const result = warp_reduce(temp_storage).Reduce(found, cub::Max());
//   if (lane_idx == 0) { d_results[str_idx] = result; }
// }

// std::unique_ptr<cudf::column> string_digits_pattern_warp_parallel(cudf::strings_column_view
// const& input,
//                                               cudf::string_scalar const& target,
//                                               rmm::cuda_stream_view stream,
//                                               rmm::mr::device_memory_resource* mr)
// {
//   CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
//   auto d_target = cudf::string_view(target.data(), target.size());

//   // create output column
//   auto results = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
//                                     input.size(),
//                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
//                                     input.null_count(),
//                                     stream,
//                                     mr);

//   // fill the output with `false` unless the `d_target` is empty
//   auto results_view = results->mutable_view();
//   thrust::fill(rmm::exec_policy(stream),
//               results_view.begin<bool>(),
//               results_view.end<bool>(),
//               d_target.empty());

//   if (!d_target.empty()) {
//     // launch warp per string
//     auto const d_strings     = cudf::column_device_view::create(input.parent(), stream);
//     constexpr int block_size = 256;
//     cudf::detail::grid_1d grid{input.size() * cudf::detail::warp_size, block_size};
//     string_digits_pattern_warp_parallel_fn<<<grid.num_blocks, grid.num_threads_per_block, 0,
//     stream.value()>>>(
//       *d_strings, d_target, results_view.data<bool>());
//   }
//   results->set_null_count(input.null_count());
//   return results;
// }

/**
 * @brief Utility to return a bool column indicating the presence of
 * a given target string in a strings column.
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam BoolFunction Return bool value given two strings.
 *
 * @param strings Column of strings to check for target.
 * @param target UTF-8 encoded string to check in strings column.
 * @param pfn Returns bool value if target is found in the given string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New BOOL column.
 */
template <typename BoolFunction>
std::unique_ptr<cudf::column> string_digits_pattern_fn(cudf::strings_column_view const& strings,
                                                       cudf::string_scalar const& target,
                                                       int const d,
                                                       BoolFunction pfn,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::mr::device_memory_resource* mr)
{
  auto strings_count = strings.size();
  if (strings_count == 0) return cudf::make_empty_column(cudf::type_id::BOOL8);

  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");

  auto d_target       = cudf::string_view(target.data(), target.size());
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column
  auto results      = make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // set the bool values by evaluating the passed function
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(strings_count),
                    d_results,
                    [d_strings, pfn, d_target, d] __device__(cudf::size_type idx) {
                      if (!d_strings.is_null(idx)) {
                        // printf("!!! in kernel, idx: %d\n", idx);
                        return bool{pfn(d_strings.element<cudf::string_view>(idx), d_target, d)};
                      }
                      return false;
                    });
  results->set_null_count(strings.null_count());
  return results;
}

std::unique_ptr<cudf::column> string_digits_pattern(cudf::strings_column_view const& input,
                                                    cudf::string_scalar const& target,
                                                    int const d,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  // // use warp parallel when the average string width is greater than the threshold
  // if ((input.null_count() < input.size()) &&
  //     ((input.chars_size(stream) / input.size()) > 64)) {
  //   return string_digits_pattern_warp_parallel(input, target, stream, mr);
  // }

  // benchmark measurements showed this to be faster for smaller strings
  auto pfn = [] __device__(cudf::string_view d_string, cudf::string_view d_target, int d) {
    // printf("!!! in kernel, start\n");
    int n = d_string.length(), m = d_target.length();
    // printf("!!! in kernel, n: %d, m: %d\n", n, m);
    for (int i = 0; i <= n - m - d; i++) {
      // printf("!!! in kernel, i: %d\n", i);
      bool match = true;
      for (int j = 0; j < m; j++) {
        // printf("!!! in kernel, j: %d\n", j);
        if (d_string[i + j] != d_target[j]) {
          // printf("!!! in kernel, match 1: false\n");
          match = false;
          break;
        }
      }
      if (match) {
        // printf("!!! in kernel, match 2: true\n");
        for (int j = 0; j < d; j++) {
          // printf("!!! in kernel, j: %d\n", j);
          if (d_string[i + m + j] < '0' || d_string[i + m + j] > '9') {
            // printf("!!! in kernel, match 3: false\n");
            match = false;
            break;
          }
        }
        if (match) {
          // printf("!!! in kernel, match 4: true\n");
          return true;
        }
      }
    }
    // printf("!!! in kernel, return");
    return false;
  };
  // printf("!!! before into string_digits_pattern_fn\n");
  return string_digits_pattern_fn(input, target, d, pfn, stream, mr);
}

}  // namespace spark_rapids_jni