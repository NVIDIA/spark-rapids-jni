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

struct ftos_converter {
  // significant digits is independent of scientific notation range
  // digits more than this may require using long values instead of ints
  static constexpr unsigned int significant_digits = 17;
  // maximum power-of-10 that will fit in 32-bits
  // static constexpr unsigned long long nine_digits = 1000000000;  // 1x10^9
  // static constexpr unsigned long long fifteen_digits = 1000000000000000;
  static constexpr unsigned long long sixteen_digits = 10000000000000000;
  // Range of numbers here is for normalizing the value.
  // If the value is above or below the following limits, the output is converted to
  // scientific notation in order to show (at most) the number of significant digits.
  static constexpr double upper_limit = 10000000;  // max is 1x10^7
  static constexpr double lower_limit = 0.001;      // printf uses scientific notation below this
  // Tables for doing normalization: converting to exponent form
  // IEEE double float has maximum exponent of 305 so these should cover everything
  double const upper10[9]  = {10, 100, 10000, 1e8, 1e16, 1e32, 1e64, 1e128, 1e256};
  double const lower10[9]  = {.1, .01, .0001, 1e-8, 1e-16, 1e-32, 1e-64, 1e-128, 1e-256};
  double const blower10[9] = {1.0, .1, .001, 1e-7, 1e-15, 1e-31, 1e-63, 1e-127, 1e-255};

  // utility for quickly converting known integer range to character array
  __device__ char* int2str(int value, char* output)
  {
    if (value == 0) {
      *output++ = '0';
      return output;
    }
    char buffer[significant_digits];  // should be big-enough for significant digits
    char* ptr = buffer;
    while (value > 0) {
      *ptr++ = (char)('0' + (value % 10));
      value /= 10;
    }
    while (ptr != buffer)
      *output++ = *--ptr;  // 54321 -> 12345
    return output;
  }

  /**
   * @brief Dissect a float value into integer, decimal, and exponent components.
   *
   * @return The number of decimal places.
   */
  __device__ int dissect_value(double value,
                               unsigned int& integer,
                               unsigned long long& decimal,
                               int& exp10)
  {
    // normalize step puts value between lower-limit and upper-limit
    // by adjusting the exponent up or down
    exp10 = 0;
    if (value > upper_limit) {
      int fx = 256;
      for (int idx = 8; idx >= 0; --idx) {
        if (value >= upper10[idx]) {
          value *= lower10[idx];
          exp10 += fx;
        }
        fx = fx >> 1;
      }
    } else if ((value > 0.0) && (value < lower_limit)) {
      int fx = 256;
      for (int idx = 8; idx >= 0; --idx) {
        if (value < blower10[idx]) {
          value *= upper10[idx];
          exp10 -= fx;
        }
        fx = fx >> 1;
      }
    }
    //
    // int decimal_places = significant_digits - (exp10? 2 : 1);
    // unsigned long long max_digits = (exp10? fifteen_digits : sixteen_digits);
    int decimal_places = significant_digits - 1;
    unsigned long long max_digits = sixteen_digits;
    double temp_value = value;
    while (temp_value < 1.0 && temp_value > 0.0) {
      max_digits *= 10;
      temp_value *= 10.0;
      decimal_places++;
    }
    integer                 = (unsigned int)value;
    for (unsigned int i = integer; i >= 10; i /= 10) {
      --decimal_places;
      max_digits /= 10;
    }
    double diff = value - (double)integer;
    double remainder = diff * (double)max_digits;
    decimal          = (unsigned long long)remainder;
    remainder -= (double)decimal;
    decimal += (unsigned long long)(2.0 * remainder); // round up
    if (decimal >= max_digits) {
      decimal = 0;
      ++integer;
      if (exp10 && (integer >= 10)) {
        ++exp10;
        integer = 1;
      }
    }
    //
    while ((decimal % 10) == 0 && (decimal_places > 0)) {
      decimal /= 10;
      --decimal_places;
    }
    return decimal_places;
  }

  /**
   * @brief Main kernel method for converting float value to char output array.
   *
   * Output need not be more than (significant_digits + 7) bytes:
   * 7 = 1 sign, 1 decimal point, 1 exponent ('e'), 1 exponent-sign, 3 digits for exponent
   *
   * @param value Float value to convert.
   * @param output Memory to write output characters.
   * @return Number of bytes written.
   */
  __device__ int float_to_string(double value, char* output)
  {
    // check for valid value
    if (std::isnan(value)) {
      memcpy(output, "NaN", 3);
      return 3;
    }
    bool bneg = false;
    if (signbit(value)) {  // handles -0.0 too
      value = -value;
      bneg  = true;
    }
    if (std::isinf(value)) {
      if (bneg) {
        memcpy(output, "-Infinity", 9);
      } else {
        memcpy(output, "Infinity", 8);
      }
      return bneg ? 9 : 8;
    }

    // dissect value into components
    unsigned int integer = 0;
    unsigned long long decimal = 0;
    int exp10          = 0;
    int decimal_places = dissect_value(value, integer, decimal, exp10);
    //
    // now build the string from the
    // components: sign, integer, decimal, exp10, decimal_places
    //
    // sign
    char* ptr = output;
    if (bneg) *ptr++ = '-';
    // integer
    ptr = int2str(integer, ptr);
    // decimal
    *ptr++ = '.';
    if (decimal_places) {
      char buffer[18];
      char* pb = buffer;
      while (decimal_places--) {
        *pb++ = (char)('0' + (decimal % 10));
        decimal /= 10;
      }
      while (pb != buffer)  // reverses the digits
        *ptr++ = *--pb;     // e.g. 54321 -> 12345
    } else
      *ptr++ = '0';  // always include at least .0
    // exponent
    if (exp10) {
      *ptr++ = 'E';
      if (exp10 < 0) {
        *ptr++ = '-';
        exp10  = -exp10;
      }
      // if (exp10 < 10) *ptr++ = '0';  // extra zero-pad
      ptr = int2str(exp10, ptr);
    }
    // done
    return (int)(ptr - output);  // number of bytes written
  }

  /**
   * @brief Compute how man bytes are needed to hold the output string.
   *
   * @param value Float value to convert.
   * @return Number of bytes required.
   */
  __device__ int compute_ftos_size(double value)
  {
    if (std::isnan(value)) return 3;  // NaN
    bool bneg = false;
    if (signbit(value)) {  // handles -0.0 too
      value = -value;
      bneg  = true;
    }
    if (std::isinf(value)) return 8 + (int)bneg;  // Inf

    // dissect float into parts
    unsigned int integer = 0;
    unsigned long long decimal = 0;
    int exp10          = 0;
    int decimal_places = dissect_value(value, integer, decimal, exp10);
    // now count up the components
    // sign
    int count = (int)bneg;
    // integer
    count += (int)(integer == 0);
    while (integer > 0) {
      integer /= 10;
      ++count;
    }  // log10(integer)
    // decimal
    ++count;  // decimal point
    if (decimal_places)
      count += decimal_places;
    else
      ++count;  // always include .0
    // exponent
    if (exp10) {
      count ++;  // 'e±'
      if (exp10 < 0) {
        count ++;
        exp10 = -exp10;
      }
      while (exp10 > 0) {
        exp10 /= 10;
        ++count;
      }  // log10(exp10)
    }
    return count;
  }
};

template <typename FloatType>
struct float_to_string_fn {
  column_device_view d_floats;
  size_type* d_offsets;
  char* d_chars;

  __device__ size_type compute_output_size(FloatType value)
  {
    ftos_converter fts;
    return static_cast<size_type>(fts.compute_ftos_size(static_cast<double>(value)));
  }

  __device__ void float_to_string(size_type idx)
  {
    FloatType value = d_floats.element<FloatType>(idx);
    ftos_converter fts;
    fts.float_to_string(static_cast<double>(value), d_chars + d_offsets[idx]);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_floats.is_null(idx)) {
      if (d_chars == nullptr) { d_offsets[idx] = 0; }
      return;
    }
    if (d_chars != nullptr) {
      float_to_string(idx);
    } else {
      d_offsets[idx] = compute_output_size(d_floats.element<FloatType>(idx));
    }
  }
};

/**
 * @brief This dispatch method is for converting floats into strings.
 *
 * The template function declaration ensures only float types are allowed.
 */
struct dispatch_float_to_string_fn {
  template <typename FloatType, std::enable_if_t<std::is_floating_point_v<FloatType>>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& floats,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    size_type strings_count = floats.size();
    auto column             = column_device_view::create(floats, stream);
    auto d_column           = *column;

    // copy the null mask
    rmm::device_buffer null_mask = cudf::detail::copy_bitmask(floats, stream, mr);

    auto [offsets, chars] =
      cudf::strings::detail::make_strings_children(float_to_string_fn<FloatType>{d_column}, strings_count, stream, mr);

    return make_strings_column(strings_count,
                               std::move(offsets),
                               std::move(chars),
                               floats.null_count(),
                               std::move(null_mask));
  }

  // non-float types throw an exception
  template <typename T, std::enable_if_t<not std::is_floating_point_v<T>>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("Values for float_to_string function must be a float type.");
  }
};

}  // namespace

// This will convert all float column types into a strings column.
std::unique_ptr<column> float_to_string(column_view const& floats,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = floats.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  return type_dispatcher(floats.type(), dispatch_float_to_string_fn{}, floats, stream, mr);
}

}  // namespace detail

// external API
std::unique_ptr<column> float_to_string(column_view const& floats, 
                                      rmm::cuda_stream_view stream, 
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::float_to_string(floats, stream, mr);
}

}  // namespace spark_rapids_jni