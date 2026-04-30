/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include "datetime_utils.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace spark_rapids_jni {

namespace {

// Whitespace test consistent with Spark UTF8String.trimAll for char input.
__device__ bool is_whitespace(unsigned char c) { return c <= 32 || c == 127; }

// Read exactly `n` digits starting at pos. Advances pos on success.
__device__ bool read_n_digits(unsigned char const* p, int& pos, int end, int n, int& v)
{
  if (pos + n > end) { return false; }
  v = 0;
  for (int i = 0; i < n; ++i) {
    int c = static_cast<int>(p[pos + i]) - '0';
    if (c < 0 || c > 9) { return false; }
    v = v * 10 + c;
  }
  pos += n;
  return true;
}

// Read 1 or 2 digits greedily. Advances pos on success.
__device__ bool read_1_or_2_digits(unsigned char const* p, int& pos, int end, int& v)
{
  if (pos >= end) { return false; }
  int c = static_cast<int>(p[pos]) - '0';
  if (c < 0 || c > 9) { return false; }
  v = c;
  ++pos;
  if (pos < end) {
    int c2 = static_cast<int>(p[pos]) - '0';
    if (c2 >= 0 && c2 <= 9) {
      v = v * 10 + c2;
      ++pos;
    }
  }
  return true;
}

__device__ bool try_parse_char(unsigned char const* p, int& pos, int end, unsigned char c)
{
  if (pos >= end || p[pos] != c) { return false; }
  ++pos;
  return true;
}

// 'T' or ' ' separator between date and time
__device__ bool expect_t_or_space(unsigned char const* p, int& pos, int end)
{
  if (pos >= end) { return false; }
  unsigned char c = p[pos];
  if (c != ' ' && c != 'T') { return false; }
  ++pos;
  return true;
}

// Skip [ \t]* between separator and next digit. Mirrors REMOVE_WHITESPACE_FROM_MONTH_DAY.
__device__ void skip_ht_whitespace(unsigned char const* p, int& pos, int end)
{
  while (pos < end && (p[pos] == ' ' || p[pos] == '\t')) { ++pos; }
}

// LEGACY trailing-character test: end-of-string OR a non-digit char.
__device__ bool legacy_trailing_ok(unsigned char const* p, int pos, int end)
{
  if (pos >= end) { return true; }
  unsigned char c = p[pos];
  return !(c >= '0' && c <= '9');
}

// CORRECTED trailing test: must be EOF (full anchored match).
__device__ bool corrected_trailing_ok(int pos, int end) { return pos == end; }

// Wire contract: each id mirrors `CastStrings.SimpleTimestampFormat#getId` on the Java side.
//
// "_LOWER" variants are for Spark patterns whose middle field is `mm` (minute) rather than
// `MM` (month) — `yyyy-mm-dd` and `yyyymmdd`. SimpleDateFormat reads the middle 2 digits as
// minute-of-hour and produces a `00:mm:00` wall-clock time on day 1 of January.
enum simple_ts_format : int32_t {
  C_YYYY_DASH_MM_DASH_DD          = 0,
  C_YYYY_SLASH_MM_SLASH_DD        = 1,
  C_YYYY_DASH_MM                  = 2,
  C_YYYY_SLASH_MM                 = 3,
  C_DD_SLASH_MM_SLASH_YYYY        = 4,
  C_YYYY_DASH_MM_DASH_DD_HH_MM_SS = 5,
  C_MM_DASH_DD                    = 6,
  C_MM_SLASH_DD                   = 7,
  C_DD_DASH_MM                    = 8,
  C_DD_SLASH_MM                   = 9,
  C_MM_SLASH_YYYY                 = 10,
  C_MM_DASH_YYYY                  = 11,
  C_MM_SLASH_DD_SLASH_YYYY        = 12,
  C_MM_DASH_DD_DASH_YYYY          = 13,
  C_MMYYYY                        = 14,

  L_YYYY_DASH_MM_DASH_DD            = 15,
  L_YYYY_SLASH_MM_SLASH_DD          = 16,
  L_DD_DASH_MM_DASH_YYYY            = 17,
  L_DD_SLASH_MM_SLASH_YYYY          = 18,
  L_YYYY_DASH_MM_DASH_DD_HH_MM_SS   = 19,
  L_YYYY_SLASH_MM_SLASH_DD_HH_MM_SS = 20,
  L_YYYYMMDD_HH_MM_SS               = 21,
  L_YYYYMMDD                        = 22,
  L_YYYY_DASH_MM_DASH_DD_LOWER      = 23,  // yyyy-mm-dd: middle field is minute
  L_YYYYMMDD_LOWER                  = 24,  // yyyymmdd:   middle field is minute
};

// Reject any string whose first non-[ \t] char is '\n', mirroring rejectLeadingNewlineThenStrip.
__device__ bool has_leading_newline(unsigned char const* p, int end)
{
  int probe = 0;
  while (probe < end && (p[probe] == ' ' || p[probe] == '\t')) { ++probe; }
  return probe < end && p[probe] == '\n';
}

// In-place trim: advance pos past leading whitespace, pull end back past trailing.
__device__ void trim(unsigned char const* p, int& pos, int& end)
{
  while (pos < end && is_whitespace(p[pos])) { ++pos; }
  while (pos < end && is_whitespace(p[end - 1])) { --end; }
}

// Year/month/day/hour/minute/second; defaults match cuDF strftime missing-field semantics.
struct parsed_dt {
  int year   = 1970;
  int month  = 1;
  int day    = 1;
  int hour   = 0;
  int minute = 0;
  int second = 0;
};

// CORRECTED-policy parse. The pattern must consume the entire (un-trimmed) input.
__device__ bool parse_corrected(simple_ts_format fmt,
                                unsigned char const* p,
                                int pos,
                                int end,
                                parsed_dt& d)
{
  switch (fmt) {
    case C_YYYY_DASH_MM_DASH_DD:
      return read_n_digits(p, pos, end, 4, d.year) && try_parse_char(p, pos, end, '-') &&
             read_n_digits(p, pos, end, 2, d.month) && try_parse_char(p, pos, end, '-') &&
             read_n_digits(p, pos, end, 2, d.day) && corrected_trailing_ok(pos, end);
    case C_YYYY_SLASH_MM_SLASH_DD:
      return read_n_digits(p, pos, end, 4, d.year) && try_parse_char(p, pos, end, '/') &&
             read_n_digits(p, pos, end, 2, d.month) && try_parse_char(p, pos, end, '/') &&
             read_n_digits(p, pos, end, 2, d.day) && corrected_trailing_ok(pos, end);
    case C_YYYY_DASH_MM:
      return read_n_digits(p, pos, end, 4, d.year) && try_parse_char(p, pos, end, '-') &&
             read_n_digits(p, pos, end, 2, d.month) && corrected_trailing_ok(pos, end);
    case C_YYYY_SLASH_MM:
      return read_n_digits(p, pos, end, 4, d.year) && try_parse_char(p, pos, end, '/') &&
             read_n_digits(p, pos, end, 2, d.month) && corrected_trailing_ok(pos, end);
    case C_DD_SLASH_MM_SLASH_YYYY:
      return read_n_digits(p, pos, end, 2, d.day) && try_parse_char(p, pos, end, '/') &&
             read_n_digits(p, pos, end, 2, d.month) && try_parse_char(p, pos, end, '/') &&
             read_n_digits(p, pos, end, 4, d.year) && corrected_trailing_ok(pos, end);
    case C_YYYY_DASH_MM_DASH_DD_HH_MM_SS:
      return read_n_digits(p, pos, end, 4, d.year) && try_parse_char(p, pos, end, '-') &&
             read_n_digits(p, pos, end, 2, d.month) && try_parse_char(p, pos, end, '-') &&
             read_n_digits(p, pos, end, 2, d.day) && expect_t_or_space(p, pos, end) &&
             read_n_digits(p, pos, end, 2, d.hour) && try_parse_char(p, pos, end, ':') &&
             read_n_digits(p, pos, end, 2, d.minute) && try_parse_char(p, pos, end, ':') &&
             read_n_digits(p, pos, end, 2, d.second) && corrected_trailing_ok(pos, end);
    case C_MM_DASH_DD:
      return read_n_digits(p, pos, end, 2, d.month) && try_parse_char(p, pos, end, '-') &&
             read_n_digits(p, pos, end, 2, d.day) && corrected_trailing_ok(pos, end);
    case C_MM_SLASH_DD:
      return read_n_digits(p, pos, end, 2, d.month) && try_parse_char(p, pos, end, '/') &&
             read_n_digits(p, pos, end, 2, d.day) && corrected_trailing_ok(pos, end);
    case C_DD_DASH_MM:
      return read_n_digits(p, pos, end, 2, d.day) && try_parse_char(p, pos, end, '-') &&
             read_n_digits(p, pos, end, 2, d.month) && corrected_trailing_ok(pos, end);
    case C_DD_SLASH_MM:
      return read_n_digits(p, pos, end, 2, d.day) && try_parse_char(p, pos, end, '/') &&
             read_n_digits(p, pos, end, 2, d.month) && corrected_trailing_ok(pos, end);
    case C_MM_SLASH_YYYY:
      return read_n_digits(p, pos, end, 2, d.month) && try_parse_char(p, pos, end, '/') &&
             read_n_digits(p, pos, end, 4, d.year) && corrected_trailing_ok(pos, end);
    case C_MM_DASH_YYYY:
      return read_n_digits(p, pos, end, 2, d.month) && try_parse_char(p, pos, end, '-') &&
             read_n_digits(p, pos, end, 4, d.year) && corrected_trailing_ok(pos, end);
    case C_MM_SLASH_DD_SLASH_YYYY:
      return read_n_digits(p, pos, end, 2, d.month) && try_parse_char(p, pos, end, '/') &&
             read_n_digits(p, pos, end, 2, d.day) && try_parse_char(p, pos, end, '/') &&
             read_n_digits(p, pos, end, 4, d.year) && corrected_trailing_ok(pos, end);
    case C_MM_DASH_DD_DASH_YYYY:
      return read_n_digits(p, pos, end, 2, d.month) && try_parse_char(p, pos, end, '-') &&
             read_n_digits(p, pos, end, 2, d.day) && try_parse_char(p, pos, end, '-') &&
             read_n_digits(p, pos, end, 4, d.year) && corrected_trailing_ok(pos, end);
    case C_MMYYYY:
      return read_n_digits(p, pos, end, 2, d.month) && read_n_digits(p, pos, end, 4, d.year) &&
             corrected_trailing_ok(pos, end);
    default: return false;
  }
}

// LEGACY-policy parse. Trims input, rejects leading newline, and folds the
// REMOVE_WHITESPACE_FROM_MONTH_DAY rewrite directly into the state machine.
// Trailing tail (anything non-digit after the matched prefix) is accepted.
__device__ bool parse_legacy(simple_ts_format fmt,
                             unsigned char const* p,
                             int pos,
                             int end,
                             parsed_dt& d)
{
  switch (fmt) {
    case L_YYYY_DASH_MM_DASH_DD: {
      if (!read_n_digits(p, pos, end, 4, d.year)) return false;
      if (!try_parse_char(p, pos, end, '-')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.month)) return false;
      if (!try_parse_char(p, pos, end, '-')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.day)) return false;
      return legacy_trailing_ok(p, pos, end);
    }
    case L_YYYY_SLASH_MM_SLASH_DD: {
      if (!read_n_digits(p, pos, end, 4, d.year)) return false;
      if (!try_parse_char(p, pos, end, '/')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.month)) return false;
      if (!try_parse_char(p, pos, end, '/')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.day)) return false;
      return legacy_trailing_ok(p, pos, end);
    }
    case L_DD_DASH_MM_DASH_YYYY: {
      if (!read_1_or_2_digits(p, pos, end, d.day)) return false;
      if (!try_parse_char(p, pos, end, '-')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.month)) return false;
      if (!try_parse_char(p, pos, end, '-')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_n_digits(p, pos, end, 4, d.year)) return false;
      return legacy_trailing_ok(p, pos, end);
    }
    case L_DD_SLASH_MM_SLASH_YYYY: {
      if (!read_1_or_2_digits(p, pos, end, d.day)) return false;
      if (!try_parse_char(p, pos, end, '/')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.month)) return false;
      if (!try_parse_char(p, pos, end, '/')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_n_digits(p, pos, end, 4, d.year)) return false;
      return legacy_trailing_ok(p, pos, end);
    }
    case L_YYYY_DASH_MM_DASH_DD_HH_MM_SS: {
      if (!read_n_digits(p, pos, end, 4, d.year)) return false;
      if (!try_parse_char(p, pos, end, '-')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.month)) return false;
      if (!try_parse_char(p, pos, end, '-')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.day)) return false;
      if (!expect_t_or_space(p, pos, end)) return false;
      if (!read_1_or_2_digits(p, pos, end, d.hour)) return false;
      if (!try_parse_char(p, pos, end, ':')) return false;
      if (!read_1_or_2_digits(p, pos, end, d.minute)) return false;
      if (!try_parse_char(p, pos, end, ':')) return false;
      if (!read_1_or_2_digits(p, pos, end, d.second)) return false;
      return legacy_trailing_ok(p, pos, end);
    }
    case L_YYYY_SLASH_MM_SLASH_DD_HH_MM_SS: {
      if (!read_n_digits(p, pos, end, 4, d.year)) return false;
      if (!try_parse_char(p, pos, end, '/')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.month)) return false;
      if (!try_parse_char(p, pos, end, '/')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.day)) return false;
      if (!expect_t_or_space(p, pos, end)) return false;
      if (!read_1_or_2_digits(p, pos, end, d.hour)) return false;
      if (!try_parse_char(p, pos, end, ':')) return false;
      if (!read_1_or_2_digits(p, pos, end, d.minute)) return false;
      if (!try_parse_char(p, pos, end, ':')) return false;
      if (!read_1_or_2_digits(p, pos, end, d.second)) return false;
      return legacy_trailing_ok(p, pos, end);
    }
    case L_YYYYMMDD_HH_MM_SS: {
      // 8-digit packed date (year(4)+month(2)+day(2)) then [ T] then time.
      // No separators inside the date part means no whitespace fold applies.
      if (!read_n_digits(p, pos, end, 4, d.year)) return false;
      if (!read_n_digits(p, pos, end, 2, d.month)) return false;
      if (!read_n_digits(p, pos, end, 2, d.day)) return false;
      if (!expect_t_or_space(p, pos, end)) return false;
      if (!read_1_or_2_digits(p, pos, end, d.hour)) return false;
      if (!try_parse_char(p, pos, end, ':')) return false;
      if (!read_1_or_2_digits(p, pos, end, d.minute)) return false;
      if (!try_parse_char(p, pos, end, ':')) return false;
      if (!read_1_or_2_digits(p, pos, end, d.second)) return false;
      return legacy_trailing_ok(p, pos, end);
    }
    case L_YYYYMMDD: {
      if (!read_n_digits(p, pos, end, 4, d.year)) return false;
      if (!read_n_digits(p, pos, end, 2, d.month)) return false;
      if (!read_n_digits(p, pos, end, 2, d.day)) return false;
      return legacy_trailing_ok(p, pos, end);
    }
    case L_YYYY_DASH_MM_DASH_DD_LOWER: {
      // yyyy-mm-dd with lowercase mm: middle field is minute, month stays at default 1.
      if (!read_n_digits(p, pos, end, 4, d.year)) return false;
      if (!try_parse_char(p, pos, end, '-')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.minute)) return false;
      if (!try_parse_char(p, pos, end, '-')) return false;
      skip_ht_whitespace(p, pos, end);
      if (!read_1_or_2_digits(p, pos, end, d.day)) return false;
      return legacy_trailing_ok(p, pos, end);
    }
    case L_YYYYMMDD_LOWER: {
      // yyyymmdd with lowercase mm: middle field is minute, month stays at default 1.
      if (!read_n_digits(p, pos, end, 4, d.year)) return false;
      if (!read_n_digits(p, pos, end, 2, d.minute)) return false;
      if (!read_n_digits(p, pos, end, 2, d.day)) return false;
      return legacy_trailing_ok(p, pos, end);
    }
    default: return false;
  }
}

__device__ bool is_legacy_format(simple_ts_format fmt) { return fmt >= L_YYYY_DASH_MM_DASH_DD; }

struct parse_simple_ts_fn {
  cudf::column_device_view d_strings;
  simple_ts_format fmt;
  bool* validity;
  cudf::timestamp_us* output;

  // Sets the row's null bit. The output value is left uninitialized — under cuDF's null mask
  // convention, masked-off positions are not read by downstream consumers.
  __device__ void set_invalid(cudf::size_type idx) const { validity[idx] = false; }

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) {
      set_invalid(idx);
      return;
    }
    auto const sv  = d_strings.element<cudf::string_view>(idx);
    auto const* p  = reinterpret_cast<unsigned char const*>(sv.data());
    int const size = sv.size_bytes();

    int pos = 0;
    int end = size;

    bool const legacy = is_legacy_format(fmt);
    if (legacy) {
      if (has_leading_newline(p, end)) {
        set_invalid(idx);
        return;
      }
      trim(p, pos, end);
      if (pos >= end) {
        set_invalid(idx);
        return;
      }
    }

    parsed_dt d{};
    bool ok = legacy ? parse_legacy(fmt, p, pos, end, d) : parse_corrected(fmt, p, pos, end, d);
    if (!ok) {
      set_invalid(idx);
      return;
    }

    if (!date_time_utils::is_valid_date_for_timestamp(d.year, d.month, d.day) ||
        !date_time_utils::is_valid_time(d.hour, d.minute, d.second, /*us*/ 0)) {
      set_invalid(idx);
      return;
    }

    int64_t const days    = date_time_utils::to_epoch_day(d.year, d.month, d.day);
    int64_t const seconds = days * 86400L + d.hour * 3600L + d.minute * 60L + d.second;
    int64_t result_us     = 0;
    if (overflow_checker::get_timestamp_overflow(seconds, /*us*/ 0, result_us)) {
      set_invalid(idx);
      return;
    }
    validity[idx] = true;
    output[idx]   = cudf::timestamp_us{cudf::duration_us{result_us}};
  }
};

}  // namespace

std::unique_ptr<cudf::column> parse_timestamp_strings_with_format(
  cudf::strings_column_view const& input,
  int32_t format_id,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_rows = input.size();
  if (num_rows == 0) {
    return cudf::make_empty_column(
      cudf::data_type{cudf::type_to_id<cudf::timestamp_us>()});
  }

  auto const fmt = static_cast<simple_ts_format>(format_id);

  auto const d_input = cudf::column_device_view::create(input.parent(), stream);
  auto result        = cudf::make_timestamp_column(
    cudf::data_type{cudf::type_to_id<cudf::timestamp_us>()},
    num_rows,
    rmm::device_buffer{},
    0,
    stream,
    mr);
  auto validity =
    rmm::device_uvector<bool>(num_rows, stream, cudf::get_current_device_resource_ref());

  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::make_counting_iterator(0),
                     num_rows,
                     parse_simple_ts_fn{*d_input,
                                        fmt,
                                        validity.begin(),
                                        result->mutable_view().begin<cudf::timestamp_us>()});

  auto [output_bitmask, null_count] =
    cudf::bools_to_mask(cudf::device_span<bool const>(validity), stream, mr);
  if (null_count) { result->set_null_mask(std::move(*output_bitmask.release()), null_count); }

  return result;
}

}  // namespace spark_rapids_jni
