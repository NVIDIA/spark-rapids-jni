/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "number_converter.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/types.hpp>

#include <thrust/for_each.h>
#include <thrust/logical.h>
#include <thrust/pair.h>

#include <cstdlib>  // For abs() function

namespace spark_rapids_jni {

namespace {

constexpr int MIN_BASE = 2;
constexpr int MAX_BASE = 36;

/**
 * @brief Trims space characters (ASCII 32)
 * @return The first non-space index and last non-space index pair
 */
__device__ thrust::pair<int, int> trim(char const* ptr, int len)
{
  int first = 0;
  int last  = len - 1;
  while (first < len && ptr[first] == ' ') {
    ++first;
  }
  while (last > first && ptr[last] == ' ') {
    --last;
  }
  return thrust::make_pair(first, last);
}

/**
 * @brief Convert byte value to char, assume base is in range [0, 35]
 * E.g.:
 *   0 => '0'
 *   ...
 *   9 => '9'
 *   10 => 'A'
 *   ...
 *   15 => 'F'
 *   ...
 *   35 => 'Z'
 * Note:
 *   For base ragne in [2, 36], the max char is 'Z'
 */
__device__ char byte_to_char(int byte_value)
{
  if (byte_value >= 0 && byte_value <= 9) {
    return static_cast<char>('0' + (byte_value - 0));
  } else if (byte_value >= 10 && byte_value <= 35) {
    return static_cast<char>('A' + (byte_value - 10));
  } else {
    cudf_assert(false);
  }
}

/**
 * @brief Convert char to byte value, it's a reverse of byte_to_char; If char is invalid
 * considering 'base', return -1.
 * E.g.: base = 36
 *   '0' => 0
 *   ...
 *   '9' => 9
 *   'A' or 'a' => 10
 *   ...
 *   'F' or 'f' => 15
 *   ...
 *   'Z' or 'z' => 35
 * E.g.: base = 8
 *   '8' => -1 // invalid char considering base
 */
__device__ int char_to_byte(char c, int base)
{
  if (c >= '0' && c <= '9' && c - '0' < base) {
    return c - '0';
  } else if (c >= 'A' && c <= 'Z' && c - 'A' + 10 < base) {
    return c - 'A' + 10;
  } else if (c >= 'a' && c <= 'z' && c - 'a' + 10 < base) {
    return c - 'a' + 10;
  } else {
    return -1;
  }
}

/**
 * @brief The types of result of convert function
 */
enum class result_type : int32_t { SUCCESS, OVERFLOW, NULL_VALUE };

/**
 * @brief Convert numbers in string representation between different number bases. It's a two
 phases
 * function, first phase calculates the length of the converted string, second phase converts the
 * string to the target base. It's the fist phase when `out` is nullptr and it's the second phase
 * when `out` is not nullptr. It first trims space characters (ASCII 32) from both sides. If
 * toBase>0 the result is unsigned, otherwise it is signed.
 *
 * This logic is borrowed from org.apache.spark.sql.catalyst.util.NumberConverter
 *
 * @return result_type and length pair
 *
 */
__device__ thrust::pair<result_type, int> convert(
  char const* ptr, int len, int from_base, int to_base, char* out, int out_len, bool ansi_mode)
{
  // trim spaces
  auto [first, last] = trim(ptr, len);
  if (last - first < 0) {
    // return null if the trimmed string is empty
    return thrust::make_pair(result_type::NULL_VALUE, 0);
  }

  // handle sign
  bool negative = false;
  if (ptr[first] == '-') {
    negative = true;
    ++first;
  }

  // convert to long value
  int64_t v  = 0;
  auto bound = static_cast<unsigned long>(-1L - from_base) / from_base;
  for (int char_idx = first; char_idx <= last; ++char_idx) {
    int b = char_to_byte(ptr[char_idx], from_base);
    if (b < 0) {
      // meet invalid char, ignore the suffix starting there
      break;
    }

    if (v < 0) {
      // if v < 0, which mean its sign(first) bit is 1, so v * base will cause
      // overflow since base is greater than 2 and v is considered as unsigned long
      if (ansi_mode) {
        // overflow for ansi mode, which means throw exception
        return thrust::make_pair(result_type::OVERFLOW, 0);
      } else {
        // overflow for non-ansi mode, which means null
        return thrust::make_pair(result_type::NULL_VALUE, 0);
      }
    }

    // check if v is greater or equal than bound
    // if v is greater than bound, v * base + base may cause overflow.
    if (v >= bound) {
      if (static_cast<unsigned long>(-1L - b) / from_base < v) {
        // if v > bound, which mean its sign(first) bit is 1, so v * base will cause
        // overflow since base is greater than 2 and v is considered as unsigned long
        if (ansi_mode) {
          // overflow for ansi mode, which means throw exception
          return thrust::make_pair(result_type::OVERFLOW, 0);
        } else {
          // overflow for non-ansi mode, which means null
          return thrust::make_pair(result_type::NULL_VALUE, 0);
        }
      }
    }

    v = v * from_base + b;
  }

  printf("v is %ld\n");

  if (negative && to_base > 0) {
    if (v < 0) {
      v = -1;
    } else {
      v = -v;
    }
  }
  if (to_base < 0 && v < 0) {
    v        = -v;
    negative = true;
  }

  // write out string representation
  auto to_base_abs = std::abs(to_base);
  int out_idx      = out_len - 1;
  uint64_t uv      = static_cast<uint64_t>(v);

  printf("uv is %ul\n", uv);

  if (uv == 0) {
    if (out != nullptr) {
      out[out_idx--] = '0';
    } else {
      --out_idx;
    }
  } else {
    while (uv != 0) {
      auto remainder = uv % to_base_abs;
      if (out != nullptr) {
        out[out_idx--] = byte_to_char(remainder);
      } else {
        --out_idx;
      }
      uv /= to_base_abs;
    }
  }

  // write out sign
  if (negative && to_base < 0) {
    if (out != nullptr) {
      out[out_idx--] = '-';
    } else {
      --out_idx;
    }
  }
  return thrust::make_pair(result_type::SUCCESS, out_len - 1 - out_idx);
}

struct const_base {
  int const base;

  __device__ int operator()(int idx) const { return base; }
};

struct base_iter {
  int const* bases;

  __device__ int operator()(int idx) const { return bases[idx]; }
};

template <typename FROM_BASE_ITERATOR, typename TO_BASE_ITERATOR, bool is_const_bases = false>
struct convert_fn {
  cudf::column_device_view input;
  FROM_BASE_ITERATOR from_base_iter;
  TO_BASE_ITERATOR to_base_iter;

  // For the first phase: calculate the lengths/nulls of the converted strings
  int* out_lens;
  int8_t* out_mask;

  // For the second phase: convert the strings to the target base
  char* out;
  int const* out_offsets;

  __device__ void operator()(int idx)
  {
    int from_base = from_base_iter(idx);
    int to_base   = to_base_iter(idx);

    if constexpr (!is_const_bases) {
      // if not const bases, check bases for each row
      if (from_base < MIN_BASE || from_base > MAX_BASE || std::abs(to_base) < MIN_BASE ||
          std::abs(to_base) > MAX_BASE) {
        // if base is invalid, return all nulls
        if (out == nullptr) { out_mask[idx] = 0; }
        return;
      }
    }

    if (input.is_null(idx)) {
      if (out == nullptr) {
        // first phase, set null/length
        out_lens[idx] = 0;
        out_mask[idx] = 0;
      }
    } else {
      auto str = input.element<cudf::string_view>(idx);
      if (out == nullptr) {
        // first phase, set null/length
        auto [ret_type, len] = convert(str.data(),
                                       str.length(),
                                       from_base,
                                       to_base,
                                       nullptr,
                                       -1,
                                       /*ansi mode*/ false);
        out_lens[idx]        = len;
        if (ret_type == result_type::NULL_VALUE) {
          out_mask[idx] = 0;
        } else {
          out_mask[idx] = 1;
        }
      } else {
        convert(str.data(),
                str.length(),
                from_base,
                to_base,
                out + out_offsets[idx],
                out_offsets[idx + 1] - out_offsets[idx],
                /*ansi mode*/ false);
      }
    }
  }
};

template <typename FROM_BASE_ITERATOR, typename TO_BASE_ITERATOR, bool is_const_bases = false>
std::unique_ptr<cudf::column> convert_impl(cudf::strings_column_view const& input,
                                           FROM_BASE_ITERATOR from_base,
                                           TO_BASE_ITERATOR to_base,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  if (input.size() == 0) return cudf::make_empty_column(cudf::type_id::STRING);

  // check base is in range: [2, 36]
  if constexpr (is_const_bases) {
    // if const bases, check bases only once
    if (from_base(0) < MIN_BASE || from_base(0) > MAX_BASE || std::abs(to_base(0)) < MIN_BASE ||
        std::abs(to_base(0)) > MAX_BASE) {
      cudf::string_scalar null_s("", false);
      // if base is invalid, return all nulls
      return cudf::make_column_from_scalar(null_s, input.size(), stream, mr);
    }
  }

  auto d_strings = cudf::column_device_view::create(input.parent(), stream);
  auto out_sizes = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                             input.size(),
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             cudf::get_current_device_resource_ref());
  auto out_mask  = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT8},
                                            input.size(),
                                            cudf::mask_state::UNALLOCATED,
                                            stream,
                                            cudf::get_current_device_resource_ref());

  // First phase: calculate the lengths/nulls of the converted strings
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   thrust::make_counting_iterator<cudf::size_type>(0),
                   thrust::make_counting_iterator<cudf::size_type>(input.size()),
                   convert_fn<FROM_BASE_ITERATOR, TO_BASE_ITERATOR, is_const_bases>{
                     *d_strings,
                     from_base,
                     to_base,
                     out_sizes->mutable_view().data<int>(),
                     out_mask->mutable_view().data<int8_t>(),
                     nullptr,
                     nullptr});
  // make null mask and null count
  auto [null_mask, null_count] = cudf::detail::valid_if(out_mask->view().begin<int8_t>(),
                                                        out_mask->view().end<int8_t>(),
                                                        thrust::identity<bool>{},
                                                        stream,
                                                        mr);
  // make offsets
  auto const sizes_input_it =
    cudf::detail::indexalator_factory::make_input_iterator(out_sizes->view());
  auto [offsets, n_chars] = cudf::detail::make_offsets_child_column(
    sizes_input_it, sizes_input_it + out_sizes->size(), stream, mr);

  // allocate chars memory
  rmm::device_uvector<char> chars(n_chars, stream, mr);
  cudf::experimental::prefetch::detail::prefetch("gather", chars, stream);

  // Second phase: convert the strings to the target base
  thrust::for_each(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(input.size()),
    convert_fn<FROM_BASE_ITERATOR, TO_BASE_ITERATOR, is_const_bases>{
      *d_strings, from_base, to_base, nullptr, nullptr, chars.data(), offsets->view().data<int>()});

  return cudf::make_strings_column(
    input.size(), std::move(offsets), chars.release(), null_count, std::move(null_mask));
}

/**
 * @brief Check if the convert function will cause overflow
 * @return true if overflow, false otherwise
 */
__device__ bool is_convert_overflow(char const* ptr, int len, int from_base, int to_base)
{
  auto pair = convert(ptr, len, from_base, to_base, nullptr, -1, /*ansi_mode*/ true);
  return pair.first == result_type::OVERFLOW;
}

template <typename FROM_BASE_ITERATOR, typename TO_BASE_ITERATOR, bool is_const_bases = false>
struct is_overflow_fn {
  cudf::column_device_view input;
  FROM_BASE_ITERATOR from_base_iter;
  TO_BASE_ITERATOR to_base_iter;

  __device__ bool operator()(int idx)
  {
    int from_base = from_base_iter(idx);
    int to_base   = to_base_iter(idx);

    if constexpr (!is_const_bases) {
      // if not const bases, check bases for each row
      if (from_base < MIN_BASE || from_base > MAX_BASE || std::abs(to_base) < MIN_BASE ||
          std::abs(to_base) > MAX_BASE) {
        // if base is invalid, return all nulls thus no overflow
        return false;
      }
    }

    if (input.is_null(idx)) {
      return false;
    } else {
      auto str = input.element<cudf::string_view>(idx);
      return is_convert_overflow(str.data(), str.length(), from_base, to_base);
    }
  }
};

template <typename FROM_BASE_ITERATOR, typename TO_BASE_ITERATOR, bool is_const_bases = false>
bool is_convert_overflow_impl(cudf::strings_column_view const& input,
                              FROM_BASE_ITERATOR from_base,
                              TO_BASE_ITERATOR to_base,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  if constexpr (is_const_bases) {
    // if not const bases, check bases only once
    if (from_base(0) < MIN_BASE || from_base(0) > MAX_BASE || std::abs(to_base(0)) < MIN_BASE ||
        std::abs(to_base(0)) > MAX_BASE) {
      // if base is invalid, return all nulls thus no overflow
      return false;
    }
  }

  auto d_strings = cudf::column_device_view::create(input.parent(), stream);
  return thrust::any_of(rmm::exec_policy_nosync(stream),
                        thrust::counting_iterator<cudf::size_type>(0),
                        thrust::counting_iterator<cudf::size_type>(input.size()),
                        is_overflow_fn<FROM_BASE_ITERATOR, TO_BASE_ITERATOR, is_const_bases>{
                          *d_strings, from_base, to_base});
}

}  // anonymous namespace

std::unique_ptr<cudf::column> convert_cv_cv_cv(cudf::strings_column_view const& input,
                                               cudf::column_view const& from_base,
                                               cudf::column_view const& to_base,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  return convert_impl<base_iter, base_iter, /*is_const_bases*/ false>(
    input, base_iter{from_base.data<int>()}, base_iter{to_base.data<int>()}, stream, mr);
}

std::unique_ptr<cudf::column> convert_cv_cv_s(cudf::strings_column_view const& input,
                                              cudf::column_view const& from_base,
                                              int const to_base,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  return convert_impl<base_iter, const_base, /*is_const_bases*/ false>(
    input, base_iter{from_base.data<int>()}, const_base{to_base}, stream, mr);
}

std::unique_ptr<cudf::column> convert_cv_s_cv(cudf::strings_column_view const& input,
                                              int const from_base,
                                              cudf::column_view const& to_base,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  return convert_impl<const_base, base_iter, /*is_const_bases*/ false>(
    input, const_base{from_base}, base_iter{to_base.data<int>()}, stream, mr);
}

std::unique_ptr<cudf::column> convert_cv_s_s(cudf::strings_column_view const& input,
                                             int const from_base,
                                             int const to_base,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  return convert_impl<const_base, const_base, /*is_const_bases*/ true>(
    input, const_base{from_base}, const_base{to_base}, stream, mr);
}

bool is_convert_overflow_cv_cv_cv(cudf::strings_column_view const& input,
                                  cudf::column_view const& from_base,
                                  cudf::column_view const& to_base,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  return is_convert_overflow_impl<base_iter, base_iter, /*is_const_bases*/ false>(
    input, base_iter{from_base.data<int>()}, base_iter{to_base.data<int>()}, stream, mr);
}

bool is_convert_overflow_cv_cv_s(cudf::strings_column_view const& input,
                                 cudf::column_view const& from_base,
                                 int const to_base,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  return is_convert_overflow_impl<base_iter, const_base, /*is_const_bases*/ false>(
    input, base_iter{from_base.data<int>()}, const_base{to_base}, stream, mr);
}

bool is_convert_overflow_cv_s_cv(cudf::strings_column_view const& input,
                                 int const from_base,
                                 cudf::column_view const& to_base,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  return is_convert_overflow_impl<const_base, base_iter, /*is_const_bases*/ false>(
    input, const_base{from_base}, base_iter{to_base.data<int>()}, stream, mr);
}

bool is_convert_overflow_cv_s_s(cudf::strings_column_view const& input,
                                int const from_base,
                                int const to_base,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  return is_convert_overflow_impl<const_base, const_base, /*is_const_bases*/ true>(
    input, const_base{from_base}, const_base{to_base}, stream, mr);
}

}  // namespace spark_rapids_jni
