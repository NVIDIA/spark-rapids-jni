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

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/transform.h>

#include <cuda/std/climits>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

using namespace cudf;

namespace spark_rapids_jni {

namespace detail {
namespace {
constexpr int max_zeros = 5;

template <typename DecimalType>
struct decimal_to_non_ansi_string_fn {
  column_device_view d_decimals;
  offset_type* d_offsets{};
  char* d_chars{};

  /**
   * @brief Calculates the size of the string required to convert the element, in base-10 format.
   *
   * @note This code does not properly handle a max negative decimal value and will overflow. This isn't an issue here because Spark will not use the full range of values and will never cause this issue.
   *
   * Output format is [-]integer.fraction
   */
  __device__ int32_t compute_output_size(DecimalType value)
  {
    auto const scale = d_decimals.type().scale();

    if (scale >= 0) return strings::detail::count_digits(value) + scale;

    auto const abs_value = numeric::detail::abs(value);
    auto const exp_ten   = numeric::detail::exp10<DecimalType>(-scale);
    auto const fraction  = strings::detail::count_digits(abs_value % exp_ten);
    auto const num_zeros = std::max(0, (-scale - fraction));

    return value == 0 && num_zeros > max_zeros && scale < 0
             ? static_cast<int32_t>(value < 0) +                       // sign if negative
                 strings::detail::count_digits(abs_value / exp_ten) +  // integer
                 2 +                                                   // E-
                 strings::detail::count_digits(num_zeros + 1)
             :  // number of zeros

             static_cast<int32_t>(value < 0) +                       // sign if negative
               strings::detail::count_digits(abs_value / exp_ten) +  // integer
               1 +                                                   // decimal point
               num_zeros +                                           // zeros padding
               fraction;                                             // size of fraction
  }

  /**
   * @brief Converts a decimal element into a string.
   *
   * @note This code does not properly handle a max negative decimal value and will overflow. This isn't an issue here because Spark will not use the full range of values and will never cause this issue.
   *
   * The value is converted into base-10 digits [0-9]
   * plus the decimal point and a negative sign prefix.
   */
  __device__ void decimal_to_non_ansi_string(size_type idx)
  {
    auto const value = d_decimals.element<DecimalType>(idx);
    auto const scale = d_decimals.type().scale();
    char* d_buffer   = d_chars + d_offsets[idx];

    if (scale >= 0) {
      d_buffer += strings::detail::integer_to_string(value, d_buffer);
      thrust::generate_n(thrust::seq, d_buffer, scale, []() { return '0'; });  // add zeros
      return;
    }

    // scale < 0
    // write format:   [-]integer.fraction
    // where integer  = abs(value) / (10^abs(scale))
    //       fraction = abs(value) % (10^abs(scale))
    if (value < 0) *d_buffer++ = '-';  // add sign
    auto const abs_value = numeric::detail::abs(value);
    auto const exp_ten   = numeric::detail::exp10<DecimalType>(-scale);
    auto const num_zeros =
      std::max(0, (-scale - strings::detail::count_digits(abs_value % exp_ten)));

    if (value == 0 && num_zeros > max_zeros) {
      *d_buffer++ = '0';
      *d_buffer++ = 'E';
      *d_buffer++ = '-';
      d_buffer += strings::detail::integer_to_string(num_zeros + 1, d_buffer);
      return;
    }

    d_buffer +=
      strings::detail::integer_to_string(abs_value / exp_ten, d_buffer);  // add the integer part
    *d_buffer++ = '.';                                                    // add decimal point

    thrust::generate_n(thrust::seq, d_buffer, num_zeros, []() { return '0'; });  // add zeros
    d_buffer += num_zeros;

    strings::detail::integer_to_string(abs_value % exp_ten, d_buffer);  // add the fraction part
  }

  __device__ void operator()(size_type idx)
  {
    if (d_decimals.is_null(idx)) {
      if (d_chars == nullptr) { d_offsets[idx] = 0; }
      return;
    }
    if (d_chars != nullptr) {
      decimal_to_non_ansi_string(idx);
    } else {
      d_offsets[idx] = compute_output_size(d_decimals.element<DecimalType>(idx));
    }
  }
};

/**
 * @brief The dispatcher functor for converting fixed-point values into strings.
 */
struct dispatch_decimal_to_non_ansi_string_fn {
  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    using DecimalType = device_storage_type_t<T>;  // underlying value type

    auto const d_column = column_device_view::create(input, stream);

    auto [offsets, chars] = strings::detail::make_strings_children(
      decimal_to_non_ansi_string_fn<DecimalType>{*d_column}, input.size(), stream, mr);

    return make_strings_column(input.size(),
                               std::move(offsets),
                               std::move(chars),
                               input.null_count(),
                               cudf::detail::copy_bitmask(input, stream, mr));
  }

  template <typename T, std::enable_if_t<not cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("Values for decimal_to_non_ansi_string function must be a decimal type.");
  }
};

}  // namespace

std::unique_ptr<column> decimal_to_non_ansi_string(column_view const& input,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);
  return type_dispatcher(input.type(), dispatch_decimal_to_non_ansi_string_fn{}, input, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> decimal_to_non_ansi_string(column_view const& input,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::decimal_to_non_ansi_string(input, stream, mr);
}

}  // namespace spark_rapids_jni
