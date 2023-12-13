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

#include "cast_string.hpp"
#include "ftos_converter.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace spark_rapids_jni {

namespace detail {
namespace {

template <typename FloatType>
struct format_float_fn {
  cudf::column_device_view d_floats;
  int digits;
  cudf::size_type* d_offsets;
  char* d_chars;

  __device__ cudf::size_type compute_output_size(FloatType value, int digits_) const
  {
    bool constexpr is_float = std::is_same_v<FloatType, float>;
    return static_cast<cudf::size_type>(
      ftos_converter::compute_format_float_size(static_cast<double>(value), digits_, is_float));
  }

  __device__ void format_float(cudf::size_type idx, int digits_) const
  {
    auto const value        = d_floats.element<FloatType>(idx);
    bool constexpr is_float = std::is_same_v<FloatType, float>;
    auto const output       = d_chars + d_offsets[idx];
    ftos_converter::format_float(static_cast<double>(value), digits_, is_float, output);
  }

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_floats.is_null(idx)) {
      if (d_chars == nullptr) { d_offsets[idx] = 0; }
      return;
    }
    if (d_chars != nullptr) {
      format_float(idx, digits);
    } else {
      d_offsets[idx] = compute_output_size(d_floats.element<FloatType>(idx), digits);
    }
  }
};

/**
 * @brief This dispatch method is for converting floats into strings.
 *
 * The template function declaration ensures only float types are allowed.
 */
struct dispatch_format_float_fn {
  template <typename FloatType, CUDF_ENABLE_IF(std::is_floating_point_v<FloatType>)>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& floats,
                                           int const digits,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr) const
  {
    auto const strings_count = floats.size();
    if (strings_count == 0) { return cudf::make_empty_column(cudf::type_id::STRING); }

    auto const input_ptr = cudf::column_device_view::create(floats, stream);

    auto [offsets, chars] = cudf::strings::detail::make_strings_children(
      format_float_fn<FloatType>{*input_ptr, digits}, strings_count, stream, mr);

    return cudf::make_strings_column(strings_count,
                                     std::move(offsets),
                                     std::move(chars),
                                     floats.null_count(),
                                     cudf::detail::copy_bitmask(floats, stream, mr));
  }

  // non-float types throw an exception
  template <typename T, CUDF_ENABLE_IF(not std::is_floating_point_v<T>)>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const&,
                                           int const,
                                           rmm::cuda_stream_view,
                                           rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("Values for format_float function must be a float type.");
  }
};

}  // namespace

// This will convert all float column types into a strings column.
std::unique_ptr<cudf::column> format_float(cudf::column_view const& floats,
                                           int const digits,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  return type_dispatcher(floats.type(), dispatch_format_float_fn{}, floats, digits, stream, mr);
}

}  // namespace detail

// external API
std::unique_ptr<cudf::column> format_float(cudf::column_view const& floats,
                                           int const digits,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::format_float(floats, digits, stream, mr);
}

}  // namespace spark_rapids_jni