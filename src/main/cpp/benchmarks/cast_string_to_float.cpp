/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cast_string.hpp>
#include <nvbench/nvbench.cuh>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

// Caller-side helper trigger gate (mirrors the device-side gate in
// cast_string_to_float.cu): `slow_path_eligible = (digits > 2^53) AND
// (|exp_ten| <= 19)`. Re-computed on the host from the same input strings so
// each benchmark axis can print a verifiable trigger rate before timing,
// instead of inferring it from input shape.
struct parsed_decimal {
  uint64_t digits;
  int exp_ten;
  bool parse_ok;
};

// Lightweight host-side decimal parser sufficient to evaluate the
// `slow_path_eligible` predicate for our benchmark inputs (sign / leading
// whitespace / NaN / Inf / hex etc. are not relevant for the inputs we
// construct here and are not handled — that matches what the device parser
// peels off before reaching the slow-path gate).
parsed_decimal parse_for_trigger_gate(std::string const& s)
{
  parsed_decimal out{0, 0, false};
  size_t i           = 0;
  bool seen_dot      = false;
  int frac_digits    = 0;
  bool any_digit     = false;
  __uint128_t digits = 0;

  if (i < s.size() && (s[i] == '+' || s[i] == '-')) { ++i; }

  for (; i < s.size(); ++i) {
    char c = s[i];
    if (c == '.') {
      if (seen_dot) { return out; }
      seen_dot = true;
      continue;
    }
    if (c == 'e' || c == 'E') { break; }
    if (c < '0' || c > '9') { return out; }
    digits = digits * 10 + static_cast<uint64_t>(c - '0');
    // Reject inputs whose accumulator no longer fits in uint64_t. `digits` is
    // __uint128_t so this is a direct high-half check rather than a
    // cross-width comparison.
    if ((digits >> 64) != 0) { return out; }
    any_digit = true;
    if (seen_dot) { ++frac_digits; }
  }
  if (!any_digit) { return out; }

  int manual_exp = 0;
  if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
    ++i;
    bool exp_neg = false;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
      exp_neg = (s[i] == '-');
      ++i;
    }
    if (i >= s.size()) { return out; }
    int e = 0;
    for (; i < s.size(); ++i) {
      char c = s[i];
      if (c < '0' || c > '9') { return out; }
      e = e * 10 + (c - '0');
    }
    manual_exp = exp_neg ? -e : e;
  }

  out.digits   = static_cast<uint64_t>(digits);
  out.exp_ten  = manual_exp - frac_digits;
  out.parse_ok = true;
  return out;
}

// Print the share of rows that satisfy the helper's caller-side gate.
// Run once per axis right before timing so the trigger rate is part of the
// benchmark transcript and the perf numbers cannot be misread. `sample_cap`
// is the maximum number of rows to inspect (the host parser is O(N) and we
// don't want to pay it for 100M-row axes).
void print_trigger_rate(std::vector<std::string> const& strings,
                        char const* tag,
                        size_t sample_cap = 1'000'000)
{
  size_t const inspect = std::min(strings.size(), sample_cap);
  size_t triggered     = 0;
  size_t parse_err     = 0;
  for (size_t i = 0; i < inspect; ++i) {
    auto const p = parse_for_trigger_gate(strings[i]);
    if (!p.parse_ok) {
      ++parse_err;
      continue;
    }
    int const abs_q = p.exp_ten < 0 ? -p.exp_ten : p.exp_ten;
    if (p.digits > (1ULL << 53) && abs_q <= 19) { ++triggered; }
  }
  double const pct =
    inspect == 0 ? 0.0 : 100.0 * static_cast<double>(triggered) / static_cast<double>(inspect);
  double const errpct =
    inspect == 0 ? 0.0 : 100.0 * static_cast<double>(parse_err) / static_cast<double>(inspect);
  std::printf("[helper-trigger] %s: %zu/%zu inspected = %.2f%% triggered (parse-err: %.2f%%)\n",
              tag,
              triggered,
              inspect,
              pct,
              errpct);
}

// Construct host strings that all satisfy the helper's caller-side gate:
// `digits > 2^53 AND |exp_ten| <= 19`. Random uint64 in [2^53+1, UINT64_MAX]
// gets printed via std::to_string, then an optional `e{q}` with
// q in [-19, 19] is appended. By construction every row exercises the
// correctly-rounded helper.
std::vector<std::string> make_helper_triggering_strings(cudf::size_type n_rows)
{
  std::mt19937_64 rng{0xC0FFEEULL};
  std::uniform_int_distribution<uint64_t> digit_dist{(1ULL << 53) + 1, UINT64_MAX};
  std::uniform_int_distribution<int> q_dist{-19, 19};

  std::vector<std::string> out;
  out.reserve(static_cast<size_t>(n_rows));
  for (cudf::size_type i = 0; i < n_rows; ++i) {
    uint64_t const digits = digit_dist(rng);
    int const q           = q_dist(rng);
    std::string s         = std::to_string(digits);
    if (q != 0) {
      s += 'e';
      s += std::to_string(q);
    }
    out.push_back(std::move(s));
  }
  return out;
}

// Pack a host vector of strings into a single contiguous chars buffer plus
// matching offsets. Used to upload to device for cudf::make_strings_column.
struct packed_host_strings {
  std::vector<char> chars;
  std::vector<int32_t> offsets;
};

packed_host_strings pack_host_strings(std::vector<std::string> const& strings)
{
  packed_host_strings out;
  out.offsets.reserve(strings.size() + 1);
  out.offsets.push_back(0);
  size_t total = 0;
  for (auto const& s : strings) {
    total += s.size();
  }
  out.chars.resize(total);
  size_t pos = 0;
  for (auto const& s : strings) {
    std::memcpy(out.chars.data() + pos, s.data(), s.size());
    pos += s.size();
    // Guard the int32 cast — cudf strings columns currently use int32 offsets.
    // If a future change bumps `num_rows` or string length past INT32_MAX,
    // fail loudly here instead of silently truncating to a negative offset.
    assert(pos <= static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    out.offsets.push_back(static_cast<int32_t>(pos));
  }
  return out;
}

// Build a non-nullable cudf STRING column from a host string vector without
// pulling in cudf::cudftestutil (which isn't linked into benchmarks unless
// BUILD_CUDF_TESTS=ON).
std::unique_ptr<cudf::column> string_column_from_host(std::vector<std::string> const& strings,
                                                      rmm::cuda_stream_view stream)
{
  auto const n = static_cast<cudf::size_type>(strings.size());
  if (n == 0) { return cudf::make_empty_column(cudf::type_id::STRING); }
  auto packed = pack_host_strings(strings);

  auto const mr = cudf::get_current_device_resource_ref();
  rmm::device_uvector<char> d_chars(packed.chars.size(), stream, mr);
  rmm::device_uvector<int32_t> d_offsets(packed.offsets.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_chars.data(),
                                packed.chars.data(),
                                packed.chars.size(),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_offsets.data(),
                                packed.offsets.data(),
                                packed.offsets.size() * sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  stream.synchronize();

  auto offsets_col = std::make_unique<cudf::column>(std::move(d_offsets), rmm::device_buffer{}, 0);
  return cudf::make_strings_column(
    n, std::move(offsets_col), d_chars.release(), 0, rmm::device_buffer{});
}

}  // namespace

// FP32 cast — the correctly-rounded helper does not run on this code path
// (the `if constexpr (is_same_v<T, double>)` gate skips it for float). Kept
// as the original FP32 baseline.
void string_to_float(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};
  auto const float_tbl  = create_random_table({cudf::type_id::FLOAT32}, row_count{n_rows});
  auto const float_col  = float_tbl->get_column(0);
  auto const string_col = cudf::strings::from_floats(float_col.view());

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto rows = spark_rapids_jni::string_to_float(cudf::data_type{cudf::type_id::FLOAT32},
                                                  string_col->view(),
                                                  false,
                                                  cudf::get_default_stream());
  });
}

NVBENCH_BENCH(string_to_float)
  .set_name("Strings to Float Cast (FP32 baseline)")
  .add_int64_axis("num_rows", {1 * 1024 * 1024, 100 * 1024 * 1024});

// FP64 cast where the correctly-rounded helper is NEVER entered. Inputs are
// produced by `cudf::strings::from_floats`, which caps significant digits at
// 10 via `nine_digits = 10^9` in `dissect_value` (cudf
// `convert_floats.cu:134`), so the parsed mantissa is at most ~10^10 — well
// below 2^53 (~9.007e15). The caller's `slow_path_eligible` is `false` for
// every row and the legacy `digits * exp10(exp_ten)` path runs. The trigger
// rate is not host-verified here (avoids reading the device output back to
// the host); instead it is asserted statically by `from_floats`'s contract.
// This is the regression-check baseline: cost on the FP64 path WITHOUT the
// new helper.
void string_to_double_helper_off(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};
  auto const float_tbl  = create_random_table({cudf::type_id::FLOAT64}, row_count{n_rows});
  auto const float_col  = float_tbl->get_column(0);
  auto const string_col = cudf::strings::from_floats(float_col.view());

  std::printf(
    "[helper-trigger] string_to_double_helper_off: 0/%d rows triggered "
    "(by construction — from_floats caps digits at ~10^10 < 2^53)\n",
    static_cast<int>(n_rows));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto rows = spark_rapids_jni::string_to_float(cudf::data_type{cudf::type_id::FLOAT64},
                                                  string_col->view(),
                                                  false,
                                                  cudf::get_default_stream());
  });
}

NVBENCH_BENCH(string_to_double_helper_off)
  .set_name("Strings to Double Cast (helper OFF — from_floats, digits <= 10^10)")
  .add_int64_axis("num_rows", {1 * 1024 * 1024, 100 * 1024 * 1024});

// FP64 cast where the correctly-rounded helper IS entered for every row.
// Strings are hand-built on the host so the parsed mantissa is always
// `> 2^53` and the explicit exponent is always in `[-19, 19]`, guaranteeing
// the caller's `slow_path_eligible` predicate is `true` for every row. A
// host-side re-parse over a sample prints the actual trigger rate before
// timing. This measures the worst case: every row pays the int128 helper
// cost.
void string_to_double_helper_on(nvbench::state& state)
{
  cudf::size_type const n_rows{(cudf::size_type)state.get_int64("num_rows")};
  auto const host_strings = make_helper_triggering_strings(n_rows);
  print_trigger_rate(host_strings, "string_to_double_helper_on");
  auto const stream     = cudf::get_default_stream();
  auto const string_col = string_column_from_host(host_strings, stream);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto rows = spark_rapids_jni::string_to_float(
      cudf::data_type{cudf::type_id::FLOAT64}, string_col->view(), false, stream);
  });
}

// Note: helper_on uses smaller row counts than the other axes because the
// handcrafted inputs average ~20 chars/row; at 100M rows the total chars
// would exceed `INT32_MAX` and overflow the strings column's int32 offsets.
// `from_floats`-based inputs (the other two axes) are ~10 chars/row so 100M
// fits.
NVBENCH_BENCH(string_to_double_helper_on)
  .set_name("Strings to Double Cast (helper ON — handcrafted digits>2^53, |q|<=19)")
  .add_int64_axis("num_rows", {1 * 1024 * 1024, 10 * 1024 * 1024});
