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
  // static constexpr unsigned int significant_digits_float = 9;
  // static constexpr unsigned int significant_digits_double = 17;
  // static constexpr unsigned int eight_digits = 100000000;  // 1x10^8
  static constexpr unsigned long long sixteen_digits = 10000000000000000; // 1x10^16
  // Range of numbers here is for normalizing the value.
  // If the value is above or below the following limits, the output is converted to
  // scientific notation in order to show (at most) the number of significant digits.
  static constexpr double upper_limit = 10000000;  // Spark's max is 1x10^7
  static constexpr double lower_limit = 0.001;      // printf uses scientific notation below this
  // Tables for doing normalization: converting to exponent form
  // IEEE double float has maximum exponent of 305 so these should cover everything
  double const upper10[9]  = {10, 100, 10000, 1e8, 1e16, 1e32, 1e64, 1e128, 1e256};
  double const lower10[9]  = {.1, .01, .0001, 1e-8, 1e-16, 1e-32, 1e-64, 1e-128, 1e-256};
  double const blower10[9] = {1.0, .1, .001, 1e-7, 1e-15, 1e-31, 1e-63, 1e-127, 1e-255};

  // // utility for quickly converting known integer range to character array
  // __device__ char* int2str(int value, char* output)
  // {
  //   if (value == 0) {
  //     *output++ = '0';
  //     return output;
  //   }
  //   char buffer[significant_digits_double];  // should be big-enough for significant digits
  //   char* ptr = buffer;
  //   while (value > 0) {
  //     *ptr++ = (char)('0' + (value % 10));
  //     value /= 10;
  //   }
  //   while (ptr != buffer)
  //     *output++ = *--ptr;  // 54321 -> 12345
  //   return output;
  // }

  // // Add separator every 3 digits for integer part
  // __device__ char* format_int(int value, char* output)
  // {
  //   if (value == 0) {
  //     *output++ = '0';
  //     return output;
  //   }
  //   char buffer[30];  // TODO: avoid hard-coded size
  //   char* ptr = buffer;
  //   int sep_count = 0;
  //   while (value > 0) {
  //     if (sep_count == 3) {
  //       *ptr++ = ',';
  //       sep_count = 0;
  //     }
  //     *ptr++ = (char)('0' + (value % 10));
  //     value /= 10;
  //     sep_count++;
  //   }
  //   while (ptr != buffer)
  //     *output++ = *--ptr;  // 543,21 -> 12,345
  //   return output;
  // }

  __device__ char* ll2str(long long n, char* result) {
    if (n == 0) {
      *result++ = '0';
      return result;
    }
    char buffer[18];  // should be big-enough for significant digits
    char* ptr = buffer;
    while (n > 0) {
      *ptr++ = (char)('0' + (n % 10));
      n /= 10;
    }
    while (ptr != buffer)
      *result++ = *--ptr;  // 54321 -> 12345
    return result;
  }

  // __device__ char* format_ll(long long n, char* result, char* dec_ptr, int& dec_pos, int exp10) {
  //   if (n == 0) {
  //     *result++ = '0';
  //     return result;
  //   }
  //   int sep_count = 0;
  //   char buffer[305];  // should be big-enough for significant digits
  //   char* ptr = buffer;
  //   while (n > 0) {
  //       if (sep_count == 3) {
  //           *ptr++ = ',';
  //           sep_count = 0;
  //       }
  //       *ptr++ = (char)('0' + (n % 10));
  //       n /= 10;
  //       sep_count++;
  //   }
  //   int len = dec_ptr - dec_str;
  //   int dec_pos = 0;
  //   while (exp10--) {
  //       if (sep_count == 3) {
  //           *ptr++ = ',';
  //           sep_count = 0;
  //       }
  //       if (dec_pos < len) {
  //         *ptr++ = dec_str[dec_pos++];
  //       } else {
  //         *ptr++ = '0';
  //       }
  //       sep_count++;
  //   }
  //   while (ptr != buffer) {
  //       *result++ = *--ptr;  // 54321 -> 12345
  //   }
  //   return result;
  // }

  // /**
  //  * @brief Dissect a float value into integer, decimal, and exponent components.
  //  *
  //  * @return The number of decimal places.
  //  */
  // __device__ int dissect_value(double value,
  //                              int digits,
  //                              unsigned int& integer,
  //                              unsigned long long& decimal,
  //                              int& exp10,
  //                              bool is_float = false)
  // {
  //   // normalize step puts value between lower-limit and upper-limit
  //   // by adjusting the exponent up or down
  //   exp10 = 0;
  //   if (value > upper_limit) {
  //     int fx = 256;
  //     for (int idx = 8; idx >= 0; --idx) {
  //       if (value >= upper10[idx]) {
  //         value *= lower10[idx];
  //         exp10 += fx;
  //       }
  //       fx = fx >> 1;
  //     }
  //   } else if ((value > 0.0) && (value < lower_limit)) {
  //     int fx = 256;
  //     for (int idx = 8; idx >= 0; --idx) {
  //       if (value < blower10[idx]) {
  //         value *= upper10[idx];
  //         exp10 -= fx;
  //       }
  //       fx = fx >> 1;
  //     }
  //   }
  //   //
  //   // int decimal_places = significant_digits - (exp10? 2 : 1);
  //   // unsigned long long max_digits = (exp10? fifteen_digits : sixteen_digits);
  //   int decimal_places = (is_float? significant_digits_float: significant_digits_double) - 1;
  //   unsigned long long max_digits = (is_float? eight_digits: sixteen_digits);
  //   double temp_value = value;
  //   while (temp_value < 1.0 && temp_value > 0.0) {
  //     max_digits *= 10;
  //     temp_value *= 10.0;
  //     decimal_places++;
  //   }
  //   integer                 = (unsigned int)value;
  //   for (unsigned int i = integer; i >= 10; i /= 10) {
  //     --decimal_places;
  //     max_digits /= 10;
  //   }
  //   double diff = value - (double)integer;
  //   double remainder = diff * (double)max_digits;
  //   decimal          = (unsigned long long)remainder;
  //   remainder -= (double)decimal;
  //   decimal += (unsigned long long)(2.0 * remainder); // round up
  //   if (decimal >= max_digits) {
  //     decimal = 0;
  //     ++integer;
  //     if (exp10 && (integer >= 10)) {
  //       ++exp10;
  //       integer = 1;
  //     }
  //   }
  //   //
  //   while ((decimal % 10) == 0 && (decimal_places > 0)) {
  //     decimal /= 10;
  //     --decimal_places;
  //   }
  //   return decimal_places;
  // }

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
  __device__ int format_float(double value, int digits, char* output, bool is_float)
  {
    // check for valid value
    if (std::isnan(value)) {
      memcpy(output, "NaN", 3);
      return 3;
    }
    bool const bneg = [&value]() {
      if (signbit(value)) {  // handles -0.0 too
        value = -value;
        return true;
      } else {
        return false;
      }
    }();
    if (std::isinf(value)) {
      if (bneg) {
        memcpy(output, "-Infinity", 9);
      } else {
        memcpy(output, "Infinity", 8);
      }
      return bneg ? 9 : 8;
    }

    // dissect value into components
    // unsigned int integer = 0;
    // unsigned long long decimal = 0;
    int exp10          = 0;
    // int decimal_places = dissect_value(value, digits, integer, decimal, exp10, is_float);
    //
    // now build the string from the
    // components: sign, integer, decimal, exp10, decimal_places
    //
    // sign
    char* ptr = output;
    if (bneg) *ptr++ = '-';
    // int exp10 = 0;
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
    // x * 10^exp10
    char dec_str[18];
    if (exp10 > 0) {
      long long int_part = static_cast<long long>(value);
      double decimal_double = value - double(int_part);
      long long dec_part = decimal_double * sixteen_digits;
      char* dec_ptr = ll2str(dec_part, dec_str);  
      // ptr = format_ll(int_part, ptr, dec_ptr, dec_pos, exp10);   
      if (int_part == 0) {
        *ptr++ = '0';
      } else {
        int sep_count = 0;
        char buffer[23];  // should be big-enough for significant digits
        char* buf_ptr = buffer;
        while (int_part > 0) {
          if (sep_count == 3) {
              *buf_ptr++ = ',';
              sep_count = 0;
          }
          *buf_ptr++ = (char)('0' + (int_part % 10));
          int_part /= 10;
          sep_count++;
        }
        while (buf_ptr != buffer) {
          *ptr++ = *--buf_ptr;  // 54321 -> 12345
        }
        int len = dec_ptr - dec_str;
        int dec_pos = 0;
        while (exp10--) {
          if (sep_count == 3) {
            *ptr++ = ',';
            sep_count = 0;
          }
          if (dec_pos < len) {
            *ptr++ = dec_str[dec_pos++];
          } else {
            *ptr++ = '0';
          }
          sep_count++;
        }
        *ptr++ = '.';
        while (digits--) {
          if (dec_pos < len) {
            *ptr++ = dec_str[dec_pos++];
          } else {
            *ptr++ = '0';
          }
        }
      }
    } else if (exp10 == 0) {
        long long int_part = static_cast<long long>(value);
        double decimal_double = value - double(int_part);
        long long dec_part = decimal_double * sixteen_digits;
        if (int_part == 0) {
          *ptr++ = '0';
        } else {
          int sep_count = 0;
          char buffer[23];  // should be big-enough for significant digits
          char* buf_ptr = buffer;
          while (int_part > 0) {
            if (sep_count == 3) {
              *buf_ptr++ = ',';
              sep_count = 0;
            }
            *buf_ptr++ = (char)('0' + (int_part % 10));
            int_part /= 10;
            sep_count++;
          }
          while (buf_ptr != buffer) {
            *ptr++ = *--buf_ptr;  // 54321 -> 12345
          }
        }
        // ptr = ll2str(int_part, ptr);
        *ptr++ = '.';
        char* dec_ptr = ll2str(dec_part, dec_str); 
        int len = dec_ptr - dec_str;
        int dec_pos = 0;
        while (digits--) {
            if (dec_pos < len) {
                *ptr++ =  dec_str[dec_pos++];
            } else {
                *ptr++ =  '0';
            }
        }
    } else {
        // exp10 < 0
        *ptr++ = '0';
        *ptr++ = '.';
        long long dec_part = value * sixteen_digits;
        char* dec_ptr = ll2str(dec_part, dec_str);
        int len = dec_ptr - dec_str;
        int dec_pos = 0;
        while (digits--) {
          if (exp10 < -1) {
            *ptr++ =  '0';
            exp10++;
          } else if (dec_pos < len) {
            *ptr++ =  dec_str[dec_pos++];
          } else {
            *ptr++ =  '0';
          }
        }
    }
    return int(ptr - output);
  }

  __device__ int int_part_len(double value)
  {
    int exp10 = 0;
    if (value > upper_limit) {
      int fx = 256;
      for (int idx = 8; idx >= 0; --idx) {
        if (value >= upper10[idx]) {
          value *= lower10[idx];
          exp10 += fx;
        }
        fx = fx >> 1;
      }
    }
    int cnt = 0;
    if (value == 0.0) {
      return 1;
    }
    while (value >= 1.0) {
      value /= 10.0;
      ++cnt;
    }
    if (exp10) {
      cnt += exp10;
    }
    return cnt;
  }

  /**
   * @brief Compute how man bytes are needed to hold the output string.
   *
   * @param value Float value to convert.
   * @return Number of bytes required.
   */
  __device__ int compute_ftos_size(double value, int digits, bool is_float)
  {
    if (std::isnan(value)) return 3;  // NaN
    bool const bneg = [&value]() {
      if (signbit(value)) {  // handles -0.0 too
        value = -value;
        return true;
      } else {
        return false;
      }
    }();
    if (std::isinf(value)) return 8 + (int)bneg;  // Inf

    int int_len = int_part_len(value);
    // sign
    int count = (int)bneg;
    // integer
    count += int_len;
    // decimal
    count += 1 + digits;
    int sep_count = 0;
    while (int_len > 0) { // speedup with math?
      if (sep_count == 3) {
        ++count;
        sep_count = 0;
      }
      int_len--;
      ++sep_count;
    }  // log10(integer)
    return count;
  }
};

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