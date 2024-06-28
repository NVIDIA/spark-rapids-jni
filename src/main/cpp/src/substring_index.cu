/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/detail/find.hpp>
#include <cudf/strings/detail/slice_strings.hpp>
#include <cudf/scalar/scalar.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

/**
 * @brief Kernel to update start and end positions for substring_index.
 */
template <bool forward>
struct substring_index_fn {
    column_device_view d_column;
    string_view d_delimiter;
    size_type count;
    size_type* d_start_positions;
    size_type* d_end_positions;

    __device__ void operator()(size_type idx) const {
        auto const& col_val = d_column.element<string_view>(idx);
        if (col_val.empty()) return;

        size_type pos = 0;
        size_type char_pos = string_view::npos;
        for (size_type i = 0; i < count; ++i) {
            char_pos = forward ? col_val.find(d_delimiter, pos) : col_val.rfind(d_delimiter, 0, pos);
            if (char_pos == string_view::npos) break;
            pos = forward ? char_pos + d_delimiter.size() : char_pos;
        }

        if (forward) {
            d_start_positions[idx] = (char_pos == string_view::npos) ? 0 : pos;
            d_end_positions[idx] = (char_pos == string_view::npos) ? col_val.size() : char_pos;
        } else {
            d_start_positions[idx] = (char_pos == string_view::npos) ? 0 : char_pos + d_delimiter.size();
            d_end_positions[idx] = (char_pos == string_view::npos) ? col_val.size() : pos;
        }
    }
};

}  // namespace

std::unique_ptr<column> substring_index(
    strings_column_view const& strings,
    string_scalar const& delimiter,
    size_type count,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
    CUDF_FUNC_RANGE();

    auto strings_count = strings.size();
    if (strings_count == 0) {
        return make_empty_column(data_type{type_id::STRING});
    }

    auto d_column = column_device_view::create(strings.parent(), stream);
    auto d_delimiter = string_view(delimiter.data(), delimiter.size());

    auto start_positions = make_numeric_column(
        data_type{type_to_id<size_type>()}, strings_count, mask_state::UNALLOCATED, stream, mr);
    auto end_positions = make_numeric_column(
        data_type{type_to_id<size_type>()}, strings_count, mask_state::UNALLOCATED, stream, mr);

    auto d_start_positions = start_positions->mutable_view().data<size_type>();
    auto d_end_positions = end_positions->mutable_view().data<size_type>();

    // Initialize start and end positions
    thrust::fill(rmm::exec_policy(stream), d_start_positions, d_start_positions + strings_count, 0);
    thrust::fill(rmm::exec_policy(stream), d_end_positions, d_end_positions + strings_count, -1);

    if (count == 0) {
        return make_empty_column(data_type{type_id::STRING});
    }

    // Compute the substring indices
    if (count > 0) {
        // Positive count: find positions from the left
        thrust::for_each_n(rmm::exec_policy(stream),
                           thrust::make_counting_iterator<size_type>(0),
                           strings_count,
                           substring_index_fn<true>{*d_column, d_delimiter, count, d_start_positions, d_end_positions});
    } else {
        // Negative count: find positions from the right
        thrust::for_each_n(rmm::exec_policy(stream),
                           thrust::make_counting_iterator<size_type>(0),
                           strings_count,
                           substring_index_fn<false>{*d_column, d_delimiter, -count, d_start_positions, d_end_positions});
    }

    // Extract the substrings using the computed start and end positions
    auto result_column = slice_strings(strings, start_positions->view(), end_positions->view(), stream, mr);
    return result_column;
}

} // namespace detail

std::unique_ptr<column> substring_index(
    strings_column_view const& strings,
    string_scalar const& delimiter,
    size_type count,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
{
    return detail::substring_index(strings, delimiter, count, stream, mr);
}

} // namespace strings
} // namespace cudf
