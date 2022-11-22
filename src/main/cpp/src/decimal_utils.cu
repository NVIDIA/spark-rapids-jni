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

#include "decimal_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cmath>
#include <cstddef>

namespace {

// Holds the 64-bit chunks of a 256-bit value
struct chunked256 {
  inline chunked256() = default;

  // sign-extend a 128-bit value into a chunked 256-bit value
  inline __device__ explicit chunked256(__int128_t x) {
    chunks[0] = static_cast<uint64_t>(x);
    __int128_t x_shifted = x >> 64;
    chunks[1] = static_cast<uint64_t>(x_shifted);
    chunks[2] = static_cast<uint64_t>(x_shifted >> 64);
    chunks[3] = chunks[2];
  }

  inline __device__ explicit chunked256(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    chunks[0] = d;
    chunks[1] = c;
    chunks[2] = b;
    chunks[3] = a;
  }

  inline __device__ uint64_t operator[](int i) const { return chunks[i]; }
  inline __device__ uint64_t &operator[](int i) { return chunks[i]; }
  inline __device__ int64_t sign() const { return static_cast<int64_t>(chunks[3]) >> 63; }

  inline __device__ void add(int a) {
    add(chunked256(static_cast<__int128_t>(a)));
  }

  inline __device__ void add(chunked256 const &a) {
    __uint128_t carry_and_sum = 0;
    for (int i = 0; i < 4; ++i) {
      carry_and_sum += static_cast<__uint128_t>(chunks[i]) + a.chunks[i];
      chunks[i] = static_cast<uint64_t>(carry_and_sum);
      carry_and_sum >>= 64;
    }
  }

  inline __device__ void negate() {
    for (int i = 0; i < 4; i++) {
      chunks[i] = ~chunks[i];
    }
    add(1);
  }

  inline __device__ bool lt_unsigned(chunked256 const &other) const {
    for (int i = 3; i >= 0; i--) {
      if (chunks[i] < other.chunks[i]) {
        return true;
      } else if (chunks[i] > other.chunks[i]) {
        return false;
      }
    }
    return false;
  }

  inline __device__ bool gte_unsigned(chunked256 const &other) const {
      return !lt_unsigned(other);
  }

  inline __device__ int leading_zeros() const {
    if (sign() < 0) {
      chunked256 tmp = *this;
      tmp.negate();
      return tmp.leading_zeros();
    }

    int ret = 0;
    for (int i = 3; i >= 0; i--) {
      if (chunks[i] == 0) {
        ret += 64;
      } else {
        ret += __clzll(chunks[i]);
        return ret;
      }
    }
  }

  inline __device__ bool fits_in_128_bits() const {
    // check for overflow by ensuring no significant bits will be lost when truncating to 128-bits
    int64_t sign = static_cast<int64_t>(chunks[1]) >> 63;
    return sign == static_cast<int64_t>(chunks[2]) && sign == static_cast<int64_t>(chunks[3]);
  }

  inline __device__ __int128_t as_128_bits() const {
    return (static_cast<__int128_t>(chunks[1]) << 64) | chunks[0];
  }
private:
  uint64_t chunks[4];
};

struct divmod256 {
  chunked256 quotient;
  __int128_t remainder;
};

// Perform a 256-bit multiply in 64-bit chunks
__device__ chunked256 multiply(chunked256 const &a, chunked256 const &b) {
  chunked256 r;
  __uint128_t mul;
  uint64_t carry = 0;
  for (int a_idx = 0; a_idx < 4; ++a_idx) {
    mul = static_cast<__uint128_t>(a[a_idx]) * b[0] + carry;
    r[a_idx] = static_cast<uint64_t>(mul);
    carry = static_cast<uint64_t>(mul >> 64);
  }
  for (int b_idx = 1; b_idx < 4; ++b_idx) {
    carry = 0;
    for (int a_idx = 0; a_idx < 4 - b_idx; ++a_idx) {
      int r_idx = a_idx + b_idx;
      mul = static_cast<__uint128_t>(a[a_idx]) * b[b_idx] + r[r_idx] + carry;
      r[r_idx] = static_cast<uint64_t>(mul);
      carry = static_cast<uint64_t>(mul >> 64);
    }
  }
  return r;
}

__device__ divmod256 divide_unsigned(chunked256 const &n, __int128_t const &d) {
  // TODO: FIXME this is long division, and so it is likely very slow...
  chunked256 q(0);
  __uint128_t r = 0;

  for (int i = 255; i >= 0; i--) {
    int block = i / 64;
    int bit = i % 64;
    int read = (int)((n[block] >> bit) & 0x01);
    r = r << 1;
    r = r | read;

    if (r >= d) {
      r = r - d;
      int64_t bit_set = 1L << bit;
      q[block] = q[block] | bit_set;
    }
  }
  return divmod256{q, static_cast<__int128_t>(r)};
}

__device__ divmod256 divide(chunked256 const &n, __int128_t const &d) {
  // We assume that d is not 0. This is because we do the zero check,
  // if needed before calling divide so we can set an overflow properly.
  bool const is_n_neg = n.sign() < 0;
  bool const is_d_neg = d < 0;
  // When computing the absolute value we don't need to worry about overflowing
  // beause we are dealing with decimal numbers that should not go to
  // the maximum value that can be held by d or n
  chunked256 abs_n = n;
  if (is_n_neg) {
    abs_n.negate();
  }

  __int128_t abs_d = is_d_neg ? -d : d;
  divmod256 result = divide_unsigned(abs_n, abs_d);

  if (is_d_neg != is_n_neg) {
    result.quotient.negate();
  }

  if (is_n_neg) {
    result.remainder = -result.remainder;
  }

  return result;
}

__device__ chunked256 round_from_remainder(chunked256 const &q, __int128_t const &r, 
        chunked256 const & n, __int128_t const &d) {
  // We are going to round if the abs value of the remainder is >= half of the divisor
  // but if we divide the divisor in half, we can lose data so instead we are going to
  // multiply the remainder by 2
  __int128_t const double_remainder = r << 1;

  // But this too can lose data if multiplying by 2 pushes off the top bit, it is a
  // little more complicated than that because of negative numbers. That is okay
  // because if we lose information when multiplying, then we know that the number
  // is in a range that would have us round because the divisor has to fit within
  // an __int128_t.

  bool const need_inc = ((double_remainder >> 1) != r) || // if we lost info or
      (double_remainder < 0 ? -double_remainder : double_remainder) >= // abs remainder is >=
      (d < 0 ? -d : d); // abs divisor

  // To know which way to round, more specifically when the quotient is 0
  // we keed to know what the sign of the quotient would have been. In this
  // case that happens if only one of the inputs was negative (xor)
  bool const is_n_neg = n.sign() < 0;
  bool const is_d_neg = d < 0;
  bool const round_down = is_n_neg != is_d_neg;

  int const round_inc = (need_inc ? (round_down ? -1 : 1) : 0);
  chunked256 ret = q;
  ret.add(round_inc);
  return ret;
}

/**
 * Divide n by d and do half up rounding based off of the remainder returned.
 */
__device__ chunked256 divide_and_round(chunked256 const &n, __int128_t const &d) {
  divmod256 div_result = divide(n, d);

  return round_from_remainder(div_result.quotient, div_result.remainder, n, d);
}

inline __device__ chunked256 pow_ten(int exp) {
  // Note that the body of this was generated using the following scala script
  /*
  import java.math.BigInteger
  import java.lang.Long.toHexString

  val lmax = new BigInteger(Long.MaxValue.toString())
  val mask = lmax.or(lmax.shiftLeft(64))

  def printAsInt128s(v: BigInteger): Unit = {
    val len = v.bitLength();
    System.out.print(s"0x${toHexString(v.shiftRight(192).longValue())}, ")
    System.out.print(s"0x${toHexString(v.shiftRight(128).longValue())}, ")
    System.out.print(s"0x${toHexString(v.shiftRight(64).longValue())}, ")
    System.out.print(s"0x${toHexString(v.longValue)}")
  }

  (0 until 77).foreach { exp =>
    val ret = BigInteger.TEN.pow(exp);
    System.out.println(s"    case $exp:")
    System.out.println(s"      //$ret")
    System.out.print("      return chunked256(")
    printAsInt128s(ret)
    System.out.println(");")
  }
  */ 
  switch(exp) {
    case 0:
      //1
      return chunked256(0x0, 0x0, 0x0, 0x1);
    case 1:
      //10
      return chunked256(0x0, 0x0, 0x0, 0xa);
    case 2:
      //100
      return chunked256(0x0, 0x0, 0x0, 0x64);
    case 3:
      //1000
      return chunked256(0x0, 0x0, 0x0, 0x3e8);
    case 4:
      //10000
      return chunked256(0x0, 0x0, 0x0, 0x2710);
    case 5:
      //100000
      return chunked256(0x0, 0x0, 0x0, 0x186a0);
    case 6:
      //1000000
      return chunked256(0x0, 0x0, 0x0, 0xf4240);
    case 7:
      //10000000
      return chunked256(0x0, 0x0, 0x0, 0x989680);
    case 8:
      //100000000
      return chunked256(0x0, 0x0, 0x0, 0x5f5e100);
    case 9:
      //1000000000
      return chunked256(0x0, 0x0, 0x0, 0x3b9aca00);
    case 10:
      //10000000000
      return chunked256(0x0, 0x0, 0x0, 0x2540be400);
    case 11:
      //100000000000
      return chunked256(0x0, 0x0, 0x0, 0x174876e800);
    case 12:
      //1000000000000
      return chunked256(0x0, 0x0, 0x0, 0xe8d4a51000);
    case 13:
      //10000000000000
      return chunked256(0x0, 0x0, 0x0, 0x9184e72a000);
    case 14:
      //100000000000000
      return chunked256(0x0, 0x0, 0x0, 0x5af3107a4000);
    case 15:
      //1000000000000000
      return chunked256(0x0, 0x0, 0x0, 0x38d7ea4c68000);
    case 16:
      //10000000000000000
      return chunked256(0x0, 0x0, 0x0, 0x2386f26fc10000);
    case 17:
      //100000000000000000
      return chunked256(0x0, 0x0, 0x0, 0x16345785d8a0000);
    case 18:
      //1000000000000000000
      return chunked256(0x0, 0x0, 0x0, 0xde0b6b3a7640000);
    case 19:
      //10000000000000000000
      return chunked256(0x0, 0x0, 0x0, 0x8ac7230489e80000);
    case 20:
      //100000000000000000000
      return chunked256(0x0, 0x0, 0x5, 0x6bc75e2d63100000);
    case 21:
      //1000000000000000000000
      return chunked256(0x0, 0x0, 0x36, 0x35c9adc5dea00000);
    case 22:
      //10000000000000000000000
      return chunked256(0x0, 0x0, 0x21e, 0x19e0c9bab2400000);
    case 23:
      //100000000000000000000000
      return chunked256(0x0, 0x0, 0x152d, 0x2c7e14af6800000);
    case 24:
      //1000000000000000000000000
      return chunked256(0x0, 0x0, 0xd3c2, 0x1bcecceda1000000);
    case 25:
      //10000000000000000000000000
      return chunked256(0x0, 0x0, 0x84595, 0x161401484a000000);
    case 26:
      //100000000000000000000000000
      return chunked256(0x0, 0x0, 0x52b7d2, 0xdcc80cd2e4000000);
    case 27:
      //1000000000000000000000000000
      return chunked256(0x0, 0x0, 0x33b2e3c, 0x9fd0803ce8000000);
    case 28:
      //10000000000000000000000000000
      return chunked256(0x0, 0x0, 0x204fce5e, 0x3e25026110000000);
    case 29:
      //100000000000000000000000000000
      return chunked256(0x0, 0x0, 0x1431e0fae, 0x6d7217caa0000000);
    case 30:
      //1000000000000000000000000000000
      return chunked256(0x0, 0x0, 0xc9f2c9cd0, 0x4674edea40000000);
    case 31:
      //10000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x7e37be2022, 0xc0914b2680000000);
    case 32:
      //100000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x4ee2d6d415b, 0x85acef8100000000);
    case 33:
      //1000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x314dc6448d93, 0x38c15b0a00000000);
    case 34:
      //10000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x1ed09bead87c0, 0x378d8e6400000000);
    case 35:
      //100000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x13426172c74d82, 0x2b878fe800000000);
    case 36:
      //1000000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0xc097ce7bc90715, 0xb34b9f1000000000);
    case 37:
      //10000000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x785ee10d5da46d9, 0xf436a000000000);
    case 38:
      //100000000000000000000000000000000000000
      return chunked256(0x0, 0x0, 0x4b3b4ca85a86c47a, 0x98a224000000000);
    case 39:
      //1000000000000000000000000000000000000000
      return chunked256(0x0, 0x2, 0xf050fe938943acc4, 0x5f65568000000000);
    case 40:
      //10000000000000000000000000000000000000000
      return chunked256(0x0, 0x1d, 0x6329f1c35ca4bfab, 0xb9f5610000000000);
    case 41:
      //100000000000000000000000000000000000000000
      return chunked256(0x0, 0x125, 0xdfa371a19e6f7cb5, 0x4395ca0000000000);
    case 42:
      //1000000000000000000000000000000000000000000
      return chunked256(0x0, 0xb7a, 0xbc627050305adf14, 0xa3d9e40000000000);
    case 43:
      //10000000000000000000000000000000000000000000
      return chunked256(0x0, 0x72cb, 0x5bd86321e38cb6ce, 0x6682e80000000000);
    case 44:
      //100000000000000000000000000000000000000000000
      return chunked256(0x0, 0x47bf1, 0x9673df52e37f2410, 0x11d100000000000);
    case 45:
      //1000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x2cd76f, 0xe086b93ce2f768a0, 0xb22a00000000000);
    case 46:
      //10000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x1c06a5e, 0xc5433c60ddaa1640, 0x6f5a400000000000);
    case 47:
      //100000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x118427b3, 0xb4a05bc8a8a4de84, 0x5986800000000000);
    case 48:
      //1000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0xaf298d05, 0xe4395d69670b12b, 0x7f41000000000000);
    case 49:
      //10000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x6d79f8232, 0x8ea3da61e066ebb2, 0xf88a000000000000);
    case 50:
      //100000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x446c3b15f9, 0x926687d2c40534fd, 0xb564000000000000);
    case 51:
      //1000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x2ac3a4edbbf, 0xb8014e3ba83411e9, 0x15e8000000000000);
    case 52:
      //10000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x1aba4714957d, 0x300d0e549208b31a, 0xdb10000000000000);
    case 53:
      //100000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x10b46c6cdd6e3, 0xe0828f4db456ff0c, 0x8ea0000000000000);
    case 54:
      //1000000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0xa70c3c40a64e6, 0xc51999090b65f67d, 0x9240000000000000);
    case 55:
      //10000000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x6867a5a867f103, 0xb2fffa5a71fba0e7, 0xb680000000000000);
    case 56:
      //100000000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x4140c78940f6a24, 0xfdffc78873d4490d, 0x2100000000000000);
    case 57:
      //1000000000000000000000000000000000000000000000000000000000
      return chunked256(0x0, 0x28c87cb5c89a2571, 0xebfdcb54864ada83, 0x4a00000000000000);
    case 58:
      //10000000000000000000000000000000000000000000000000000000000
      return chunked256(0x1, 0x97d4df19d6057673, 0x37e9f14d3eec8920, 0xe400000000000000);
    case 59:
      //100000000000000000000000000000000000000000000000000000000000
      return chunked256(0xf, 0xee50b7025c36a080, 0x2f236d04753d5b48, 0xe800000000000000);
    case 60:
      //1000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x9f, 0x4f2726179a224501, 0xd762422c946590d9, 0x1000000000000000);
    case 61:
      //10000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x639, 0x17877cec0556b212, 0x69d695bdcbf7a87a, 0xa000000000000000);
    case 62:
      //100000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x3e3a, 0xeb4ae1383562f4b8, 0x2261d969f7ac94ca, 0x4000000000000000);
    case 63:
      //1000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x26e4d, 0x30eccc3215dd8f31, 0x57d27e23acbdcfe6, 0x8000000000000000);
    case 64:
      //10000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x184f03, 0xe93ff9f4daa797ed, 0x6e38ed64bf6a1f01, 0x0);
    case 65:
      //100000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0xf31627, 0x1c7fc3908a8bef46, 0x4e3945ef7a25360a, 0x0);
    case 66:
      //1000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x97edd87, 0x1cfda3a5697758bf, 0xe3cbb5ac5741c64, 0x0);
    case 67:
      //10000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x5ef4a747, 0x21e864761ea97776, 0x8e5f518bb6891be8, 0x0);
    case 68:
      //100000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x3b58e88c7, 0x5313ec9d329eaaa1, 0x8fb92f75215b1710, 0x0);
    case 69:
      //1000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x25179157c9, 0x3ec73e23fa32aa4f, 0x9d3bda934d8ee6a0, 0x0);
    case 70:
      //10000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x172ebad6ddc, 0x73c86d67c5faa71c, 0x245689c107950240, 0x0);
    case 71:
      //100000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0xe7d34c64a9c, 0x85d4460dbbca8719, 0x6b61618a4bd21680, 0x0);
    case 72:
      //1000000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x90e40fbeea1d, 0x3a4abc8955e946fe, 0x31cdcf66f634e100, 0x0);
    case 73:
      //10000000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x5a8e89d752524, 0x46eb5d5d5b1cc5ed, 0xf20a1a059e10ca00, 0x0);
    case 74:
      //100000000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x3899162693736a, 0xc531a5a58f1fbb4b, 0x746504382ca7e400, 0x0);
    case 75:
      //1000000000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x235fadd81c2822b, 0xb3f07877973d50f2, 0x8bf22a31be8ee800, 0x0);
    case 76:
      //10000000000000000000000000000000000000000000000000000000000000000000000000000
      return chunked256(0x161bcca7119915b5, 0x764b4abe8652979, 0x7775a5f171951000, 0x0);
    default:
      // This is not a supported value...
      assert(0);
    }
}

// check that the divide is going to do the right thing
void check_scale_divisor(int source_scale, int target_scale) {
  int exponent = target_scale - source_scale;
  CUDF_EXPECTS(exponent <= cuda::std::numeric_limits<__int128_t>::digits10, "divisor too big");
}

inline __device__ int precision10(chunked256 value) {
    if (value.sign() < 0) {
      // we want to do this on positive numbers
      value.negate();
    }
    // TODO this is a horrible way to do this. We should at least
    // be able to approximate the log10 using the leading zeros similar to 
    // http://graphics.stanford.edu/~seander/bithacks.html and then start
    // the search around the guess.
    for (int i = 0; i <= 76; i++) {
      chunked256 tmp = pow_ten(i);
      if (tmp.gte_unsigned(value)) {
        return i;
      }
    }
    return -1;
}

__device__ chunked256 set_scale_and_round(chunked256 data, int old_scale, int new_scale) {
    if (old_scale != new_scale) {
        if (new_scale < old_scale) {
            int raise = old_scale - new_scale;
            int multiplier = pow_ten(raise).as_128_bits();
            data = multiply(data, chunked256(multiplier));
        } else {
            int drop = new_scale - old_scale;
            auto const diviser = pow_ten(drop).as_128_bits();
            data = divide_and_round(data, chunked256(diviser).as_128_bits());
        }
    }
    return data;
}

// Functor to add two DECIMAL128 columns with rounding and overflow detection.
struct dec128_add: public thrust::unary_function<cudf::size_type, __int128_t> {
  dec128_add(bool *overflows, cudf::mutable_column_view const &sum_view,
                    cudf::column_view const &a_col, cudf::column_view const &b_col)
      : overflows(overflows), a_data(a_col.data<__int128_t>()), b_data(b_col.data<__int128_t>()),
        add_data(sum_view.data<__int128_t>()),
        a_scale(a_col.type().scale()), b_scale(b_col.type().scale()),
        sum_scale(sum_view.type().scale()) {}

  __device__ __int128_t operator()(cudf::size_type const i) const {
    chunked256 const a(a_data[i]);
    chunked256 const b(b_data[i]);

    chunked256 working_a = a;
    chunked256 working_b = b;

    /*
    * The way Spark 3.4 does add is first it rescales the original numbers
    * and if there is no overflow then we are finished.
    * Otherwise there is an overflow so we add the number with the original scale
    * and set the target scale on the result
    */
    int intermediate_scale = min(a_scale, b_scale);
    if (a_scale != intermediate_scale) {
    printf("converting a_scale \n");
        working_a = set_scale_and_round(working_a, a_scale, intermediate_scale);
    }
    if (b_scale != intermediate_scale) {
    printf("converting b_scale \n");
        working_b = set_scale_and_round(working_b, b_scale, intermediate_scale);
    }

    chunked256 sum = working_a;
    sum.add(working_b);

    if (sum_scale != intermediate_scale) {
        sum = set_scale_and_round(sum, intermediate_scale, sum_scale);
    }

    overflows[i] = !sum.fits_in_128_bits();
    add_data[i] = sum.as_128_bits();
  }

private:

  // output column for overflow detected
  bool * const overflows;

  // input data for add
  __int128_t const * const a_data;
  __int128_t const * const b_data;
  __int128_t * const add_data;
  int const a_scale;
  int const b_scale;
  int const sum_scale;
};

// Functor to multiply two DECIMAL128 columns with rounding and overflow detection.
struct dec128_multiplier : public thrust::unary_function<cudf::size_type, __int128_t> {
  dec128_multiplier(bool *overflows, cudf::mutable_column_view const &product_view,
                    cudf::column_view const &a_col, cudf::column_view const &b_col)
      : overflows(overflows), a_data(a_col.data<__int128_t>()), b_data(b_col.data<__int128_t>()),
        product_data(product_view.data<__int128_t>()),
        a_scale(a_col.type().scale()), b_scale(b_col.type().scale()),
        prod_scale(product_view.type().scale()) {}

  __device__ __int128_t operator()(cudf::size_type const i) const {
    chunked256 const a(a_data[i]);
    chunked256 const b(b_data[i]);

    chunked256 product = multiply(a, b);

    // Spark does some really odd things that I personally think are a bug
    // https://issues.apache.org/jira/browse/SPARK-40129
    // But to match Spark we need to first round the result to a precision of 38
    // and this is specific to the value in the result of the multiply.
    // Then we need to round the result to the final scale that we care about.
    int dec_precision = precision10(product);
    int first_div_precision = dec_precision - 38;

    int mult_scale = a_scale + b_scale;
    if (first_div_precision > 0) {
      auto const first_div_scale_divisor = pow_ten(first_div_precision).as_128_bits();
      product = divide_and_round(product, first_div_scale_divisor);

      // a_scale and b_scale are negative. first_div_precision is not
      mult_scale = a_scale + b_scale + first_div_precision;
    }

    int exponent = prod_scale - mult_scale;
    if (exponent < 0) {
      // we need to multiply, but only if this will not overflow.
      int new_precision = precision10(product);
      if (new_precision - exponent > 38) {
        // this would overflow...
        overflows[i] = true;
        return;
      } else {
        auto const scale_mult = pow_ten( -exponent).as_128_bits();
        product = multiply(product, chunked256(scale_mult));
      }
    } else {
      auto const scale_divisor = pow_ten(exponent).as_128_bits();

      // scale and round to target scale
      if (scale_divisor != 1) {
        product = divide_and_round(product, scale_divisor);
      }
    }

    overflows[i] = !product.fits_in_128_bits();
    product_data[i] = product.as_128_bits();
  }

private:

  // output column for overflow detected
  bool * const overflows;

  // input data for multiply
  __int128_t const * const a_data;
  __int128_t const * const b_data;
  __int128_t * const product_data;
  int const a_scale;
  int const b_scale;
  int const prod_scale;
};

// Functor to divide two DECIMAL128 columns with rounding and overflow detection.
struct dec128_divider : public thrust::unary_function<cudf::size_type, __int128_t> {
  dec128_divider(bool *overflows, cudf::mutable_column_view const &quotient_view,
                    cudf::column_view const &a_col, cudf::column_view const &b_col)
      : overflows(overflows), a_data(a_col.data<__int128_t>()), b_data(b_col.data<__int128_t>()),
        quotient_data(quotient_view.data<__int128_t>()),
        a_scale(a_col.type().scale()), b_scale(b_col.type().scale()),
        quot_scale(quotient_view.type().scale()) {}

  __device__ __int128_t operator()(cudf::size_type const i) const {
    chunked256 n(a_data[i]);
    __int128_t const d(b_data[i]);

    // Divide by zero, not sure if we care or not, but...
    if (d == 0) {
      overflows[i] = true;
      quotient_data[i] = 0;
      return;
    }

    // The output scale of a divide is a_scale - b_scale. But we want an output scale of
    // quot_scale, so we need to multiply a by a set amount before we can do the divide.

    int n_shift_exp = quot_scale - (a_scale - b_scale);

    if (n_shift_exp > 0) {
      // In this case we have to divide twice to get the answer we want.
      // The first divide is a regular divide
      divmod256 const first_div_result = divide(n, d);

      // Ignore the remainder because we don't need it.
      auto const scale_divisor = pow_ten(n_shift_exp).as_128_bits();

      // The second divide gets the result into the scale that we care about and does the rounding.
      auto const result = divide_and_round(first_div_result.quotient, scale_divisor);

      overflows[i] = !result.fits_in_128_bits();
      quotient_data[i] = result.as_128_bits();
    } else if (n_shift_exp < -38) {
      // We need to do a multiply before we can divide, but the multiply might
      // overflow so we do a multiply then a divide and shift the result and
      // remainder over by the amount left to multiply. It is kind of like long
      // division, but base 10.

      // First multiply by 10^38 and divide to get a remainder
      n = multiply(n, chunked256(pow_ten(38)));

      auto const first_div_result = divide(n, d);
      chunked256 const first_div_r(first_div_result.remainder);

      //now we have to multiply each of these by how much is left
      int const remaining_exp = (-n_shift_exp) - 38;
      auto const scale_mult = pow_ten(remaining_exp);
      auto result = multiply(first_div_result.quotient, scale_mult);
      auto const scaled_div_r = multiply(first_div_r, scale_mult);

      // Now do a second divide on what is left
      auto const second_div_result = divide(scaled_div_r, d);
      result.add(second_div_result.quotient);

      // and finally round
      result = round_from_remainder(result, second_div_result.remainder, scaled_div_r, d);

      overflows[i] = !result.fits_in_128_bits();
      quotient_data[i] = result.as_128_bits();
    } else {
      // Regular multiply followed by a divide
      if (n_shift_exp < 0) {
        n = multiply(n, pow_ten(-n_shift_exp));
      }

      auto const result = divide_and_round(n, d);

      overflows[i] = !result.fits_in_128_bits();
      quotient_data[i] = result.as_128_bits();
    }
  }

private:

  // output column for overflow detected
  bool * const overflows;

  // input data for multiply
  __int128_t const * const a_data;
  __int128_t const * const b_data;
  __int128_t * const quotient_data;
  int const a_scale;
  int const b_scale;
  int const quot_scale;
};

} // anonymous namespace

namespace cudf::jni {

std::unique_ptr<cudf::table>
multiply_decimal128(cudf::column_view const &a, cudf::column_view const &b, int32_t product_scale,
                    rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(a.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  CUDF_EXPECTS(b.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  auto const num_rows = a.size();
  CUDF_EXPECTS(num_rows == b.size(), "inputs have mismatched row counts");
  auto [result_null_mask, result_null_count] = cudf::detail::bitmask_and(cudf::table_view{{a, b}}, stream);
  std::vector<std::unique_ptr<cudf::column>> columns;
  // copy the null mask here, as it will be used again later
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8}, num_rows,
                                                  rmm::device_buffer(result_null_mask, stream), result_null_count, stream));
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL128, product_scale}, num_rows, std::move(result_null_mask), result_null_count, stream));
  auto overflows_view = columns[0]->mutable_view();
  auto product_view = columns[1]->mutable_view();
  check_scale_divisor(a.type().scale() + b.type().scale(), product_scale);
  thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(num_rows),
                    product_view.begin<__int128_t>(),
                    dec128_multiplier(overflows_view.begin<bool>(), product_view, a, b));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table>
divide_decimal128(cudf::column_view const &a, cudf::column_view const &b, int32_t quotient_scale,
                  rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(a.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  CUDF_EXPECTS(b.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  auto const num_rows = a.size();
  CUDF_EXPECTS(num_rows == b.size(), "inputs have mismatched row counts");
  auto [result_null_mask, result_null_count] = cudf::detail::bitmask_and(cudf::table_view{{a, b}}, stream);
  std::vector<std::unique_ptr<cudf::column>> columns;
  // copy the null mask here, as it will be used again later
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8}, num_rows,
                                                  rmm::device_buffer(result_null_mask, stream), result_null_count, stream));
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL128, quotient_scale}, num_rows, std::move(result_null_mask), result_null_count, stream));
  auto overflows_view = columns[0]->mutable_view();
  auto quotient_view = columns[1]->mutable_view();
  thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(num_rows),
                    quotient_view.begin<__int128_t>(),
                    dec128_divider(overflows_view.begin<bool>(), quotient_view, a, b));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table>
add_decimal128(cudf::column_view const &a, cudf::column_view const &b, int32_t target_scale,
                  rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(a.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  CUDF_EXPECTS(b.type().id() == cudf::type_id::DECIMAL128, "not a DECIMAL128 column");
  auto const num_rows = a.size();
  CUDF_EXPECTS(num_rows == b.size(), "inputs have mismatched row counts");
  auto [result_null_mask, result_null_count] = cudf::detail::bitmask_and(cudf::table_view{{a, b}}, stream);
  std::vector<std::unique_ptr<cudf::column>> columns;
  // copy the null mask here, as it will be used again later
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8}, num_rows,
                                                  rmm::device_buffer(result_null_mask, stream), result_null_count, stream));
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL128, target_scale}, num_rows, std::move(result_null_mask), result_null_count, stream));
  auto overflows_view = columns[0]->mutable_view();
  auto sum_view = columns[1]->mutable_view();
  thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(num_rows),
                    sum_view.begin<__int128_t>(),
                    dec128_add(overflows_view.begin<bool>(), sum_view, a, b));
  return std::make_unique<cudf::table>(std::move(columns));
}
} // namespace cudf::jni
