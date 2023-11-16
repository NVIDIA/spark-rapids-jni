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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/convert/int_to_string.cuh>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <ftos_converter.cu>

using namespace cudf;

namespace spark_rapids_jni {

namespace detail {
namespace {

template <typename FloatType>
struct format_float_fn {
  column_device_view d_floats;
  int digits;
  size_type* d_offsets;
  char* d_chars;

  __device__ size_type compute_output_size(FloatType value, int digits)
  {
    ftos_converter fts;
    bool is_float = std::is_same_v<FloatType, float>;
    return static_cast<size_type>(fts.compute_ftos_size(static_cast<double>(value), digits, is_float));
  }

  __device__ void format_float(size_type idx, int digits)
  {
    FloatType value = d_floats.element<FloatType>(idx);
    ftos_converter fts;
    bool is_float = std::is_same_v<FloatType, float>;
    fts.format_float(static_cast<double>(value), digits, d_chars + d_offsets[idx], is_float);
  }

  __device__ void operator()(size_type idx)
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
  template <typename FloatType, std::enable_if_t<std::is_floating_point_v<FloatType>>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& floats,
                                     int digits,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    size_type strings_count = floats.size();
    auto column             = column_device_view::create(floats, stream);
    auto d_column           = *column;

    // copy the null mask
    rmm::device_buffer null_mask = cudf::detail::copy_bitmask(floats, stream, mr);

    auto [offsets, chars] =
      cudf::strings::detail::make_strings_children(format_float_fn<FloatType>{d_column, digits}, strings_count, stream, mr);

    return make_strings_column(strings_count,
                               std::move(offsets),
                               std::move(chars),
                               floats.null_count(),
                               std::move(null_mask));
  }

  // non-float types throw an exception
  template <typename T, std::enable_if_t<not std::is_floating_point_v<T>>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     int,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("Values for format_float function must be a float type.");
  }
};

}  // namespace

// This will convert all float column types into a strings column.
std::unique_ptr<column> format_float(column_view const& floats,
                                    int digits,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = floats.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  return type_dispatcher(floats.type(), dispatch_format_float_fn{}, floats, digits, stream, mr);
}

}  // namespace detail

// external API
std::unique_ptr<column> format_float(column_view const& floats, 
                                      int digits,
                                      rmm::cuda_stream_view stream, 
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::format_float(floats, digits, stream, mr);
}

}  // namespace spark_rapids_jni