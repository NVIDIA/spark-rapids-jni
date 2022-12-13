/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cub/warp/warp_reduce.cuh>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/detail/convert/string_to_float.cuh>
#include <cudf/utilities/bit.hpp>

using namespace cudf;

namespace spark_rapids_jni {

namespace detail {

__device__ __inline__ bool is_digit(char c) { return c >= '0' && c <= '9'; }

/**
 * @brief Identify if a character is whitespace.
 *
 * @param chr character to test
 * @return true if character is a whitespace character
 */
constexpr bool is_whitespace(char const chr)
{
  switch (chr) {
    case ' ':
    case '\r':
    case '\t':
    case '\n': return true;
    default: return false;
  }
}

template <typename T, size_type block_size>
class string_to_float {
 public:
  __device__ string_to_float(T* out,
                             bitmask_type* validity,
                             int32_t* ansi_except,
                             size_type* valid_count,
                             const char* const chars,
                             offset_type const* offsets,
                             uint64_t const* const ipow,
                             bitmask_type const* incoming_null_mask,
                             size_type const num_rows)
    : _out(out),
      _validity(validity),
      _ansi_except(ansi_except),
      _valid_count(valid_count),
      _chars(chars),
      _warp_id((threadIdx.x + (blockDim.x * blockIdx.x)) / 32),
      _row(_warp_id),
      _warp_lane((threadIdx.x + (blockDim.x * blockIdx.x)) % 32),
      _row_start(offsets[_row]),
      _len(offsets[_row + 1] - _row_start),
      _ipow(ipow),
      _incoming_null_mask(incoming_null_mask),
      _num_rows(num_rows)
  {
  }

  __device__ void operator()()
  {
    _bstart = 0;              // start position of the current batch
    _blen   = min(32, _len);  // length of the batch
    _bpos   = 0;              // current position within the current batch of chars for the warp
    _c      = _warp_lane < _blen ? _chars[_row_start + _warp_lane] : 0;

    if (_incoming_null_mask != nullptr && !bit_is_set(_incoming_null_mask, _row)) {
      _valid = false;
      compute_validity(_valid, _except);
      return;
    }

    remove_leading_whitespace();

    // check for + or -
    int sign = check_for_sign();

    // check for leading nan
    if (check_for_nan()) {
      _out[_row] = NAN;
      compute_validity(_valid, _except);
      return;
    }

    // check for inf / infinity
    if (check_for_inf()) {
      if (_warp_lane == 0) {
        _out[_row] =
          sign > 0 ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      }
      compute_validity(_valid, _except);
      return;
    }

    // parse the remainder as floating point.
    auto const [digits, exp_base] = parse_digits();
    if (!_valid) {
      compute_validity(_valid, _except);
      return;
    }

    // 0 / -0.
    if (digits == 0) {
      remove_leading_whitespace();
      if (_bpos < _blen) {
        _valid  = false;
        _except = true;
      }

      if (_warp_lane == 0) { _out[_row] = sign * static_cast<double>(0); }
      compute_validity(_valid, _except);
      return;
    }

    // parse any manual exponent
    auto const manual_exp = parse_manual_exp();
    if (!_valid) {
      compute_validity(_valid, _except);
      return;
    }

    check_trailing_bytes();
    if (!_valid) {
      compute_validity(_valid, _except);
      return;
    }

    // construct the final float value
    if (_warp_lane == 0) {
      // base value
      double digitsf = sign * static_cast<double>(digits);

      // exponent
      int exp_ten = exp_base + manual_exp;

      // final value
      if (exp_ten > std::numeric_limits<double>::max_exponent10) {
        _out[_row] = sign > 0 ? std::numeric_limits<double>::infinity()
                              : -std::numeric_limits<double>::infinity();
      } else {
        // make sure we don't produce a subnormal number.
        // - a normal number is one where the leading digit of the floating point rep is not zero.
        //      eg:   0.0123  represented as  1.23e-2
        //
        // - a denormalized number is one where the leading digit of the floating point rep is zero.
        //      eg:   0.0123 represented as   0.123e-1
        //
        // - a subnormal number is a denormalized number where if you tried to normalize it, the
        // exponent
        //   required would be smaller then the smallest representable exponent.
        //
        // https://en.wikipedia.org/wiki/Denormal_number
        //

        auto const subnormal_shift = std::numeric_limits<double>::min_exponent10 - exp_ten;
        if (subnormal_shift > 0) {
          // Handle subnormal values. Ensure that both base and exponent are
          // normal values before computing their product.
          int const num_digits = static_cast<int>(log10(static_cast<double>(digits))) + 1;
          digitsf = digitsf / exp10(static_cast<double>(num_digits - 1 + subnormal_shift));
          exp_ten += num_digits - 1;  // adjust exponent
          auto const exponent = exp10(static_cast<double>(exp_ten + subnormal_shift));
          _out[_row] = static_cast<T>(digitsf * exponent);
        } else {
          double const exponent = exp10(static_cast<double>(std::abs(exp_ten)));
          double const result   = exp_ten < 0 ? digitsf / exponent : digitsf * exponent;

          _out[_row] = static_cast<T>(result);
        }
      }
    }
    compute_validity(_valid, _except);
  }

 private:
  // shuffle down to remove whitespace
  __device__ void remove_leading_whitespace()
  {
    do {
      // skip any leading whitespace
      //
      auto const chars_left = _blen - _bpos;
      auto const non_whitespace_mask =
        __ballot_sync(0xffffffff, _warp_lane < chars_left && !is_whitespace(_c));
      auto const first_non_whitespace = __ffs(non_whitespace_mask) - 1;

      if (first_non_whitespace > 0) {
        _bpos += first_non_whitespace;
        _c = __shfl_down_sync(0xffffffff, _c, first_non_whitespace);
      } else if (non_whitespace_mask == 0) {
        //  all whitespace
        _bpos += chars_left;
      }

      if (_bpos == _blen) {
        _bstart += _blen;
        // nothing left to read?
        if (_bstart == _len) { break; }
        // read the next batch
        _bpos = 0;
        _blen = min(32, _len - _bstart);
        _c    = _warp_lane < _blen ? _chars[_row_start + _bstart + _warp_lane] : 0;
      } else {
        break;
      }
    } while (1);
  }

  // returns true if we encountered 'nan'
  // potentially changes:  valid/except
  __device__ bool check_for_nan()
  {
    auto const nan_mask = __ballot_sync(0xffffffff,
                                        (_warp_lane == 0 && (_c == 'N' || _c == 'n')) ||
                                          (_warp_lane == 1 && (_c == 'A' || _c == 'a')) ||
                                          (_warp_lane == 2 && (_c == 'N' || _c == 'n')));
    if (nan_mask == 0x7) {
      // if we start with 'nan', then even if we have other garbage character, this is a null row.
      //
      // if we're in ansi mode and this is not -precisely- nan, report that so that we can throw
      // an exception later.
      if (_len != 3) {
        _valid  = false;
        _except = _len != 3;
      }
      return true;
    }
    return false;
  }

  // returns 1 or -1 to indicate sign
  __device__ int check_for_sign()
  {
    auto const sign_mask = __ballot_sync(0xffffffff, _warp_lane == 0 && (_c == '+' || _c == '-'));
    int sign             = 1;
    if (sign_mask) {
      // NOTE: warp lane 0 is the only thread that ever reads `sign`, so technically it would be
      // valid to just check if(c == '-'), but that would leave other threads with an incorrect
      // value. if this code ever changes, that could lead to hard-to-find bugs.
      if (__ballot_sync(0xffffffff, _warp_lane == 0 && _c == '-')) { sign = -1; }
      _bpos++;
      _c = __shfl_down_sync(0xffffffff, _c, 1);
    }
    return sign;
  }

  // returns true if we encountered an inf
  // potentially changes:  valid
  __device__ bool check_for_inf()
  {
    // check for inf or infinity
    auto const inf_mask = __ballot_sync(0xffffffff,
                                        (_warp_lane == 0 && (_c == 'I' || _c == 'i')) ||
                                          (_warp_lane == 1 && (_c == 'N' || _c == 'n')) ||
                                          (_warp_lane == 2 && (_c == 'F' || _c == 'f')));
    if (inf_mask == 0x7) {
      _bpos += 3;
      _c = __shfl_down_sync(0xffffffff, _c, 3);

      // if we're at the end
      if (_bpos == _len) { return true; }

      // see if we have the whole word
      auto const infinity_mask = __ballot_sync(0xffffffff,
                                               (_warp_lane == 0 && (_c == 'I' || _c == 'i')) ||
                                                 (_warp_lane == 1 && (_c == 'N' || _c == 'n')) ||
                                                 (_warp_lane == 2 && (_c == 'I' || _c == 'i')) ||
                                                 (_warp_lane == 3 && (_c == 'T' || _c == 't')) ||
                                                 (_warp_lane == 4 && (_c == 'Y' || _c == 'y')));
      if (infinity_mask == 0x1f) {
        _bpos += 5;
        // if we're at the end
        if (_bpos == _len) { return true; }
      }

      // if we reach here for any reason, it means we have "inf" or "infinity" at the start of the
      // string but also have additional characters, making this whole thing bogus/null
      _valid = false;
      return true;
    }
    return false;
  }

  // parse the actual digits.  returns 64 bit digit holding value and exponent
  __device__ thrust::pair<uint64_t, int> parse_digits()
  {
    typedef cub::WarpReduce<uint64_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    // what we will need to compute the exponent
    uint64_t digits      = 0;
    int real_digits      = 0;  // total # of digits we've got stored in 'digits'
    int truncated_digits = 0;  // total # of digits we've had to truncate off
    // the # of total digits is (real_digits + truncated_digits)
    bool decimal    = false;  // whether or not we have a decimal
    int decimal_pos = 0;      // absolute decimal pos

    constexpr int max_safe_digits = 19;
    do {
      int num_chars = min(max_safe_digits, _blen - _bpos);

      // have we seen a valid digit yet?
      bool seen_valid_digit = false;

      // if our current sum is 0 and we don't have a decimal, strip leading
      // zeros.  handling cases such as
      // 0000001
      if (!decimal && digits == 0) {
        auto const zero_mask = __ballot_sync(0xffffffff, _warp_lane < num_chars && _c != '0');
        auto const nz_pos    = __ffs(zero_mask) - 1;
        if (nz_pos > 0) {
          num_chars -= nz_pos;
          _bpos += nz_pos;
          _c               = __shfl_down_sync(0xffffffff, _c, nz_pos);
          seen_valid_digit = true;
        }
      }

      // handle a decimal point
      auto const decimal_mask = __ballot_sync(0xffffffff, _warp_lane < num_chars && _c == '.');
      if (decimal_mask) {
        // if we have more than one decimal, this is an invalid value
        if (decimal || __popc(decimal_mask) > 1) {
          _valid  = false;
          _except = true;
          return {0, 0};
        }
        auto const dpos = __ffs(decimal_mask) - 1;  // 0th bit is reported as 1 by __ffs
        decimal_pos     = (dpos + real_digits);
        decimal         = true;

        // strip the decimal char out
        if (_warp_lane >= dpos) { _c = __shfl_down_sync(~((1 << dpos) - 1), _c, 1); }
        num_chars--;
      }

      // handle any chars that are not actually digits
      //
      auto const non_digit_mask =
        __ballot_sync(0xffffffff, _warp_lane < num_chars && !is_digit(_c));
      auto const first_non_digit = __ffs(non_digit_mask);

      num_chars = min(num_chars, first_non_digit > 0 ? first_non_digit - 1 : num_chars);

      if (decimal_pos > 0 && decimal_pos > num_chars + real_digits) {
        _valid  = false;
        _except = true;
        return {0, 0};
      }

      if (num_chars == 0 && _blen == _len) {
        if (!seen_valid_digit) {
          _valid  = false;
          _except = true;
        }
        return {0, 0};
      }

      // our local digit
      uint64_t const digit = _warp_lane < num_chars ? static_cast<uint64_t>(_c - '0') *
                                                        _ipow[(num_chars - _warp_lane) - 1]
                                                    : 0;

      // we may have to start truncating because we'd go past the 64 bit limit by adding the new
      // digits
      //
      // max uint64_t is 20 digits, so any 19 digit number is valid.
      // 2^64:  18,446,744,073,709,551,616
      //         9,999,999,999,999,999,999
      //
      // if the 20th digit would push us past that limit, we have to start truncating.
      // max_holding:  1,844,674,407,370,955,160
      // so     1,844,674,407,370,955,160 + 9    -> 18,446,744,073,709,551,609  -> ok
      //        1,844,674,407,370,955,160 + 1X   -> 18,446,744,073,709,551,61X  -> potentially rolls
      //        past the limit
      //
      constexpr uint64_t max_holding = (std::numeric_limits<uint64_t>::max() - 9) / 10;
      // if we're already past the max_holding, just truncate.
      // eg:    9,999,999,999,999,999,999
      if (digits > max_holding) {
        truncated_digits += num_chars;
      } else {
        // add as many digits to the running sum as we can.
        int const safe_count = min(max_safe_digits - real_digits, num_chars);
        if (safe_count > 0) {
          // only lane 0 will have the real value so we need to shfl it to the rest of the threads.
          digits = (digits * _ipow[safe_count]) +
                   __shfl_sync(0xffffffff, WarpReduce(temp_storage).Sum(digit, safe_count), 0);
          real_digits += safe_count;
        }

        // if we have more digits
        if (safe_count < num_chars) {
          // we're already past max_holding so we have to start truncating
          if (digits > max_holding) {
            truncated_digits += num_chars - safe_count;
          }
          // we may be able to add one more digit.
          else {
            auto const last_digit =
              static_cast<uint64_t>(__shfl_sync(0xffffffff, _c, safe_count) - '0');
            if ((digits * 10) + last_digit <= max_holding) {
              // we can add this final digit
              digits = (digits * 10) + last_digit;
              truncated_digits += num_chars - (safe_count - 1);
            }
            // everything else gets truncated
            else {
              truncated_digits += num_chars - safe_count;
            }
          }
        }
      }
      _bpos += num_chars + (decimal_mask > 0);

      // read the next batch of chars.
      if (_bpos == _blen) {
        _bstart += _blen;
        // nothing left to read?
        if (_bstart == _len) { break; }
        // read the next batch
        _bpos = 0;
        _blen = min(32, _len - _bstart);
        _c    = _warp_lane < _blen ? _chars[_row_start + _bstart + _warp_lane] : 0;
      } else {
        _c = __shfl_down_sync(0xffffffff, _c, num_chars);

        // if we encountered a non-digit, we're done
        if (first_non_digit) { break; }
      }
    } while (1);

    // 0 / -0.
    if (digits == 0) { return {0, 0}; }

    // the total amount of actual digits
    auto const total_digits = real_digits + truncated_digits;

    // exponent
    // any truncated digits are effectively just trailing zeros
    int exp_ten = (truncated_digits
                   // if we've got a decimal, shift left by it's position
                   - (decimal ? (total_digits - decimal_pos) : 0));
    return {digits, exp_ten};
  }

  // parse manually specified exponent.
  // potentially changes: valid
  __device__ int parse_manual_exp()
  {
    typedef cub::WarpReduce<uint64_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    // if we still have chars left, the only thing legal now is a manual exponent.
    // eg:  E-10
    //
    int manual_exp = 0;
    if (_bpos < _blen) {
      // read some trailing chars.

      auto const exp_mask =
        __ballot_sync(0xffffffff, (_warp_lane == 0 && (_c == 'E' || _c == 'e')));
      if (!exp_mask) { return 0; }
      auto const exp_sign_mask =
        __ballot_sync(0xffffffff, (_warp_lane == 1 && (_c == '-' || _c == '+')));
      auto const exp_sign =
        exp_sign_mask ? __ballot_sync(0xffffffff, _warp_lane == 1 && _c == '-') ? -1 : 1 : 1;
      auto const chars_to_skip = exp_sign_mask ? 2 : 1;
      _c                       = __shfl_down_sync(0xffffffff, _c, chars_to_skip);
      _bpos += chars_to_skip;

      // the largest valid exponent for a double is 4 digits (3 for floats).
      int const num_chars = min(4, _blen - _bpos);

      // handle any chars that are not actually digits
      //
      auto const non_digit_mask =
        __ballot_sync(0xffffffff, _warp_lane < num_chars && !is_digit(_c));
      auto const first_non_digit = __ffs(non_digit_mask);

      int const num_digits = first_non_digit > 0 ? first_non_digit - 1 : num_chars;

      if (num_digits == 0) {
        _valid  = false;
        _except = true;
        return 0;
      }

      uint64_t const digit = _warp_lane < num_digits ? static_cast<uint64_t>(_c - '0') *
                                                         _ipow[(num_digits - _warp_lane) - 1]
                                                     : 0;
      manual_exp           = WarpReduce(temp_storage).Sum(digit, num_digits) * exp_sign;
      _c                   = __shfl_down_sync(0xffffffff, _c, num_digits);
      _bpos += num_digits;
    }

    return manual_exp;
  }

  __device__ void check_trailing_bytes()
  {
    if (_blen - _bpos > 0) {
      // strip trailing f if it exists
      // f is a valid character at the end of a float string
      auto const f_mask = __ballot_sync(
        0xffffffff, (_warp_lane == 0 && (_c == 'F' || _c == 'f' || _c == 'd' || _c == 'D')));
      if (f_mask > 0) {
        _c = __shfl_down_sync(0xffffffff, _c, 1);
        _bpos++;
      }
    }

    // nothing trailing
    if (_blen - _bpos == 0) { return; }

    // strip any whitespace
    remove_leading_whitespace();

    // invalid characters in string
    if (_blen - _bpos > 0) {
      _valid  = false;
      _except = true;
    }
  }

  // sets validity bits, updates outgoing validity count for the block and potentially sets the
  // outgoing ansi_except field
  __device__ void compute_validity(bool const valid, bool const except = false)
  {
    // compute null count for the block. each warp processes one string, so lane 0
    // from each warp contributes 1 bit of validity
    size_type const block_valid_count =
      cudf::detail::single_lane_block_sum_reduce<block_size, 0>(valid ? 1 : 0);
    // 0th thread in each block updates the validity count and (optionally) the ansi_except flag
    if (threadIdx.x == 0) {
      atomicAdd(_valid_count, block_valid_count);

      if (_ansi_except && except) { atomicMax(_ansi_except, _num_rows - _row); }
    }

    // 0th thread in each warp updates the validity
    size_type const row_id = _warp_id;
    if (threadIdx.x % 32 == 0 && valid) {
      // uses atomics
      cudf::set_bit(_validity, row_id);
    }
  }

  T* _out;
  bitmask_type* _validity;
  int32_t* _ansi_except;
  size_type* _valid_count;
  const char* const _chars;
  size_type const _warp_id;
  size_type const _row;
  size_type const _warp_lane;
  size_type const _row_start;
  size_type const _len;
  size_type const _num_rows;
  uint64_t const* const _ipow;
  bitmask_type const* _incoming_null_mask;

  // shared/modified by the various parsing functions
  size_type _bstart;  // batch start within the entire string
  size_type _bpos;    // position with current batch
  size_type _blen;    // batch length;
  char _c;            // current character
  bool _valid  = true;
  bool _except = false;
};

template <typename T, size_type block_size>
__global__ void string_to_float_kernel(T* out,
                                       bitmask_type* validity,
                                       int32_t* ansi_except,
                                       size_type* valid_count,
                                       const char* const chars,
                                       offset_type const* offsets,
                                       bitmask_type const* incoming_null_mask,
                                       size_type const num_rows)
{
  size_type const tid = threadIdx.x + (blockDim.x * blockIdx.x);
  size_type const row = tid / 32;
  if (row >= num_rows) { return; }

  __shared__ uint64_t ipow[19];
  if (threadIdx.x == 0) {
    ipow[0]  = 1;
    ipow[1]  = 10;
    ipow[2]  = 100;
    ipow[3]  = 1000;
    ipow[4]  = 10000;
    ipow[5]  = 100000;
    ipow[6]  = 1000000;
    ipow[7]  = 10000000;
    ipow[8]  = 100000000;
    ipow[9]  = 1000000000;
    ipow[10] = 10000000000;
    ipow[11] = 100000000000;
    ipow[12] = 1000000000000;
    ipow[13] = 10000000000000;
    ipow[14] = 100000000000000;
    ipow[15] = 1000000000000000;
    ipow[16] = 10000000000000000;
    ipow[17] = 100000000000000000;
    ipow[18] = 1000000000000000000;
  }
  __syncthreads();

  // convert
  string_to_float<T, block_size>{
    out, validity, ansi_except, valid_count, chars, offsets, ipow, incoming_null_mask, num_rows}();
}

}  // namespace detail

/**
 * @brief Convert a string column into an float column.
 *
 * @param dtype Type of column to return.
 * @param string_col Incoming string column to convert to integers.
 * @param ansi_mode If true, strict conversion and throws on error.
 *                  If false, null invalid entries.
 * @param stream Stream on which to operate.
 * @param mr Memory resource for returned column
 * @return std::unique_ptr<column> Integer column that was created from string_col.
 */
std::unique_ptr<column> string_to_float(data_type dtype,
                                        strings_column_view const& string_col,
                                        bool ansi_mode,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(dtype == data_type{type_id::FLOAT32} || dtype == data_type{type_id::FLOAT64},
               "invalid float data type");
  if (string_col.size() == 0) { return cudf::make_empty_column(dtype); }

  auto out = cudf::make_numeric_column(dtype, string_col.size(), mask_state::ALL_NULL, stream, mr);

  using ScalarType = cudf::scalar_type_t<size_type>;
  auto valid_count = cudf::make_numeric_scalar(cudf::data_type(cudf::type_to_id<size_type>()));
  auto ansi_count  = cudf::make_numeric_scalar(cudf::data_type(cudf::type_to_id<size_type>()));
  static_cast<ScalarType*>(valid_count.get())->set_value(0, stream);
  if (ansi_mode) { static_cast<ScalarType*>(ansi_count.get())->set_value(-1, stream); }

  constexpr auto warps_per_block = 8;
  constexpr auto rows_per_block  = warps_per_block;
  auto const num_blocks = cudf::util::div_rounding_up_safe(string_col.size(), rows_per_block);
  auto const num_rows   = string_col.size();

  if (dtype == data_type{type_id::FLOAT32}) {
    detail::string_to_float_kernel<float, warps_per_block * 32>
      <<<num_blocks, warps_per_block * 32>>>(
        out->mutable_view().begin<float>(),
        out->mutable_view().null_mask(),
        ansi_mode ? static_cast<ScalarType*>(ansi_count.get())->data() : nullptr,
        static_cast<ScalarType*>(valid_count.get())->data(),
        string_col.chars().begin<char>(),
        string_col.offsets().begin<offset_type>(),
        string_col.null_mask(),
        num_rows);
  } else {
    detail::string_to_float_kernel<double, warps_per_block * 32>
      <<<num_blocks, warps_per_block * 32>>>(
        out->mutable_view().begin<double>(),
        out->mutable_view().null_mask(),
        ansi_mode ? static_cast<ScalarType*>(ansi_count.get())->data() : nullptr,
        static_cast<ScalarType*>(valid_count.get())->data(),
        string_col.chars().begin<char>(),
        string_col.offsets().begin<offset_type>(),
        string_col.null_mask(),
        num_rows);
  }

  if (ansi_mode) {
    auto const val = static_cast<ScalarType*>(ansi_count.get())->value(stream);
    if (val >= 0) {
      auto const error_row = num_rows - val;
      offset_type string_bounds[2];
      cudaMemcpyAsync(&string_bounds,
                      &string_col.offsets().data<offset_type>()[error_row],
                      sizeof(offset_type) * 2,
                      cudaMemcpyDeviceToHost,
                      stream.value());
      stream.synchronize();

      std::string dest;
      dest.resize(string_bounds[1] - string_bounds[0]);

      cudaMemcpyAsync(dest.data(),
                      &string_col.chars().data<char const>()[string_bounds[0]],
                      string_bounds[1] - string_bounds[0],
                      cudaMemcpyDeviceToHost,
                      stream.value());
      stream.synchronize();

      throw cast_error(error_row, dest);
    }
  }

  return out;
}

}  // namespace spark_rapids_jni