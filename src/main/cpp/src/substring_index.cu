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
#include "substring_index.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

using namespace cudf;

namespace spark_rapids_jni {

namespace detail {

namespace {

template <typename IndexIterator>
struct substring_from_fn {
  column_device_view const d_column;
  IndexIterator const starts;
  IndexIterator const stops;
  __device__ string_view operator()(size_type idx) const
  {
    if (d_column.is_null(idx)) { return string_view{nullptr, 0}; }
    auto const d_str  = d_column.template element<string_view>(idx);
    auto const length = d_str.length();
    auto const start  = std::max(starts[idx], 0);
    if (start >= length) { return string_view{}; }
    auto const stop = stops[idx];
    auto const end  = (((stop < 0) || (stop > length)) ? length : stop);
    return start < end ? d_str.substr(start, end - start) : string_view{};
  }
  substring_from_fn(column_device_view const& d_column, IndexIterator starts, IndexIterator stops)
    : d_column(d_column), starts(starts), stops(stops)
  {
  }
};

template <typename IndexIterator>
std::unique_ptr<cudf::column> compute_substrings_from_fn(cudf::column_device_view const& d_column,
                                                         IndexIterator starts,
                                                         IndexIterator stops,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  auto results = rmm::device_uvector<string_view>(d_column.size(), stream);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(d_column.size()),
                    results.begin(),
                    substring_from_fn{d_column, starts, stops});
  return cudf::make_strings_column(results, cudf::string_view{nullptr, 0}, stream, mr);
}

/**
 * @brief Compute slice indices for each string.
 *
 * When slice_strings is invoked with a delimiter string and a delimiter count, we need to
 * compute the start and end indices of the substring. This function accomplishes that.
 */
template <typename DelimiterItrT>
void compute_substring_indices(cudf::column_device_view const& d_column,
                               DelimiterItrT const delim_itr,
                               cudf::size_type delimiter_count,
                               cudf::size_type* start_char_pos,
                               cudf::size_type* end_char_pos,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref)
{
  auto strings_count = d_column.size();

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    strings_count,
    [delim_itr, delimiter_count, start_char_pos, end_char_pos, d_column] __device__(
      cudf::size_type idx) {
      auto const& delim_val_pair = delim_itr[idx];
      auto const& delim_val      = delim_val_pair.first;  // Don't use it yet

      // If the column value for this row is null, result is null.
      // If the delimiter count is 0, result is empty string.
      // If the global delimiter or the row specific delimiter is invalid or if it is empty, row
      // value is empty.
      if (d_column.is_null(idx) || !delim_val_pair.second || delim_val.empty()) return;
      auto const& col_val = d_column.element<cudf::string_view>(idx);

      // If the column value for the row is empty, the row value is empty.
      if (!col_val.empty()) {
        auto const col_val_len   = col_val.length();
        auto const delimiter_len = delim_val.length();

        auto nsearches           = (delimiter_count < 0) ? -delimiter_count : delimiter_count;
        bool const left_to_right = (delimiter_count > 0);

        cudf::size_type start_pos = start_char_pos[idx];
        cudf::size_type end_pos   = col_val_len;
        cudf::size_type char_pos  = -1;

        end_char_pos[idx] = col_val_len;

        for (auto i = 0; i < nsearches; ++i) {
          char_pos = left_to_right ? col_val.find(delim_val, start_pos)
                                   : col_val.rfind(delim_val, 0, end_pos);
          if (char_pos == string_view::npos) return;
          if (left_to_right)
            start_pos = char_pos + delimiter_len;
          else
            end_pos = char_pos;
        }
        if (left_to_right)
          end_char_pos[idx] = char_pos;
        else
          start_char_pos[idx] = end_pos + delimiter_len;
      }
    });
}

}  // namespace

template <typename DelimiterItrT>
std::unique_ptr<cudf::column> substring_index(cudf::strings_column_view const& strings,
                                              DelimiterItrT const delimiter_itr,
                                              cudf::size_type count,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto strings_count = strings.size();
  // If there aren't any rows, return an empty strings column
  if (strings_count == 0) { return cudf::make_empty_column(cudf::type_id::STRING); }

  // Compute the substring indices first
  auto start_chars_pos_vec =
    cudf::make_column_from_scalar(cudf::numeric_scalar<cudf::size_type>(0, true, stream),
                                  strings_count,
                                  stream,
                                  rmm::mr::get_current_device_resource());
  auto stop_chars_pos_vec =
    cudf::make_column_from_scalar(cudf::numeric_scalar<cudf::size_type>(0, true, stream),
                                  strings_count,
                                  stream,
                                  rmm::mr::get_current_device_resource());

  auto start_char_pos = start_chars_pos_vec->mutable_view().data<cudf::size_type>();
  auto end_char_pos   = stop_chars_pos_vec->mutable_view().data<cudf::size_type>();

  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  // If delimiter count is 0, the output column will contain empty strings
  if (count != 0) {
    // Compute the substring indices first
    compute_substring_indices(
      d_column, delimiter_itr, count, start_char_pos, end_char_pos, stream, mr);
  }

  // Extract the substrings using the indices next
  auto starts_iter =
    cudf::detail::indexalator_factory::make_input_iterator(start_chars_pos_vec->view());
  auto stops_iter =
    cudf::detail::indexalator_factory::make_input_iterator(stop_chars_pos_vec->view());
  return compute_substrings_from_fn(d_column, starts_iter, stops_iter, stream, mr);
}

std::unique_ptr<cudf::column> substring_index(cudf::strings_column_view const& strings,
                                              cudf::strings_column_view const& delimiters,
                                              cudf::size_type count,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(strings.size() == delimiters.size(),
               "Strings and delimiters column sizes do not match");
  auto delimiters_dev_view_ptr = cudf::column_device_view::create(delimiters.parent(), stream);
  auto delimiters_dev_view     = *delimiters_dev_view_ptr;
  return (delimiters_dev_view.nullable())
           ? detail::substring_index(
               strings,
               cudf::detail::make_pair_iterator<cudf::string_view, true>(delimiters_dev_view),
               count,
               stream,
               mr)
           : detail::substring_index(
               strings,
               cudf::detail::make_pair_iterator<cudf::string_view, false>(delimiters_dev_view),
               count,
               stream,
               mr);
}

}  // namespace detail

// external API

std::unique_ptr<cudf::column> substring_index(cudf::strings_column_view const& strings,
                                              cudf::string_scalar const& delimiter,
                                              cudf::size_type count,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::substring_index(strings,
                                 cudf::detail::make_pair_iterator<cudf::string_view>(delimiter),
                                 count,
                                 cudf::get_default_stream(),
                                 mr);
}

std::unique_ptr<cudf::column> substring_index(cudf::strings_column_view const& strings,
                                              cudf::strings_column_view const& delimiters,
                                              cudf::size_type count,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::substring_index(strings, delimiters, count, cudf::get_default_stream(), mr);
}

}  // namespace spark_rapids_jni
