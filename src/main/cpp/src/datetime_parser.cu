/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "datetime_parser.hpp"

#include <iostream>
#include <vector>

#include <cuda/std/cassert>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/valid_if.cuh>

#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/string_view.cuh>

#include <cudf/reduction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/optional.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

using column                   = cudf::column;
using column_device_view       = cudf::column_device_view;
using column_view              = cudf::column_view;
using lists_column_device_view = cudf::detail::lists_column_device_view;
using size_type                = cudf::size_type;
using string_view              = cudf::string_view;
using struct_view              = cudf::struct_view;
using table_view               = cudf::table_view;

namespace {

/**
 * Represents local date time in a time zone.
 */
struct timestamp_components {
  int32_t year;  // max 6 digits
  int8_t month;
  int8_t day;
  int8_t hour;
  int8_t minute;
  int8_t second;
  int32_t microseconds;
};

/**
 * Is white space
 */
__device__ __host__ inline bool is_whitespace(const char chr)
{
  switch (chr) {
    case ' ':
    case '\r':
    case '\t':
    case '\n': return true;
    default: return false;
  }
}

/**
 * Ported from Spark
 */
__device__ __host__ bool is_valid_digits(int segment, int digits)
{
  // A Long is able to represent a timestamp within [+-]200 thousand years
  const int constexpr maxDigitsYear = 6;
  // For the nanosecond part, more than 6 digits is allowed, but will be
  // truncated.
  return segment == 6 || (segment == 0 && digits >= 4 && digits <= maxDigitsYear) ||
         // For the zoneId segment(7), it's could be zero digits when it's a
         // region-based zone ID
         (segment == 7 && digits <= 2) ||
         (segment != 0 && segment != 6 && segment != 7 && digits > 0 && digits <= 2);
}

/**
 * We have to distinguish INVALID value with UNSUPPORTED value.
 * INVALID means the value is invalid in Spark SQL.
 * UNSUPPORTED means the value is valid in Spark SQL but not supported by rapids
 * yet. As for INVALID values, we treat them in the same as Spark SQL. As for
 * UNSUPPORTED values, we just throw cuDF exception.
 */
enum ParseResult { OK = 0, INVALID = 1, UNSUPPORTED = 2 };

template <bool with_timezone>
struct parse_timestamp_string_fn {
  column_device_view const d_strings;
  size_type default_tz_index;
  bool allow_tz_in_date_str = true;
  // The list column of transitions to figure out the correct offset
  // to adjust the timestamp. The type of the values in this column is
  // LIST<STRUCT<utcInstant: int64, tzInstant: int64, utcOffset: int32,
  // looseTzInstant: int64>>.
  thrust::optional<lists_column_device_view const> transitions = thrust::nullopt;
  thrust::optional<column_device_view const> tz_indices        = thrust::nullopt;

  __device__ thrust::tuple<cudf::timestamp_us, uint8_t> operator()(const cudf::size_type& idx) const
  {
    // inherit the null mask of the input column
    if (!d_strings.is_valid(idx)) {
      return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, ParseResult::INVALID);
    }

    auto const d_str = d_strings.element<cudf::string_view>(idx);

    timestamp_components ts_comp{};
    char const* tz_lit_ptr = nullptr;
    size_type tz_lit_len   = 0;
    switch (parse_string_to_timestamp_us(&ts_comp, &tz_lit_ptr, &tz_lit_len, d_str)) {
      case ParseResult::INVALID:
        return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, ParseResult::INVALID);
      case ParseResult::UNSUPPORTED:
        return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}},
                                  ParseResult::UNSUPPORTED);
      case ParseResult::OK:
      default: break;
    }

    if constexpr (!with_timezone) {
      // path without timezone, in which unix_timestamp is straightforwardly
      // computed
      auto const ts_unaligned = compute_epoch_us(ts_comp);
      return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{ts_unaligned}},
                                ParseResult::OK);
    }

    // path with timezone, in which timezone offset has to be determined before
    // computing unix_timestamp
    int64_t tz_offset;
    if (tz_lit_ptr == nullptr) {
      tz_offset = extract_timezone_offset(compute_loose_epoch_s(ts_comp), default_tz_index);
    } else {
      auto tz_view = string_view(tz_lit_ptr, tz_lit_len);
      // Firstly, try parsing as utc-like timezone rep
      if (auto [utc_offset, ret_code] = parse_utc_like_tz(tz_view); ret_code == 0) {
        tz_offset = utc_offset;
      } else if (ret_code == 1) {
        // Then, try parsing as region-based timezone ID
        auto tz_index = query_index_from_tz_db(tz_view);
        // tz_index < size(tzDB): found the ID in tzDB
        // size(tzDB) <= tz_index < size(tzIDs): found the ID but not supported
        // yet tz_index == size(tzIDs): invalid timezone ID
        if (tz_index > transitions->size()) {
          if (tz_index == tz_indices->size())
            return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}},
                                      ParseResult::INVALID);
          return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}},
                                    ParseResult::UNSUPPORTED);
        }
        tz_offset = extract_timezone_offset(compute_loose_epoch_s(ts_comp), tz_index);
      } else {
        // (ret_code == 2) quick path to mark value invalid
        return thrust::make_tuple(cudf::timestamp_us{cudf::duration_us{0}}, ParseResult::INVALID);
      }
    }

    // Compute the epoch as UTC timezone, then apply the timezone offset.
    auto const ts_unaligned = compute_epoch_us(ts_comp);

    return thrust::make_tuple(
      cudf::timestamp_us{cudf::duration_us{ts_unaligned - tz_offset * 1000000L}}, ParseResult::OK);
  }

  /**
   * TODO: support CST/PST/AST
   *
   * Parse UTC-like timezone representation such as: UTC+11:22:33, GMT-8:08:01.
   * This function is purposed to be fully align to Apache Spark's behavior. The
   * function returns the status along with the result: 0 - successfully parsed
   * the timezone offset 1 - not a valid UTC-like timezone representation, maybe
   * valid region-based rep 2 - not a valid timezone representation
   *
   * Valid patterns:
   *   with colon
   *     hh:mm      : ^(GMT|UTC)?[+-](\d|0[0-9]|1[0-8]):(\d|[0-5][0-9])
   *     hh:mm:ss   : ^(GMT|UTC)?[+-](\d|0[0-9]|1[0-8]):[0-5][0-9]:[0-5][0-9]
   *   without colon
   *     hh only    : ^(GMT|UTC)?[+-](\d|0[0-9]|1[0-8])
   *     hh:mm:(ss) : ^(GMT|UTC)?[+-](0[0-9]|1[0-8])([0-5][0-9])?([0-5][0-9])?
   *   special symbols:
   *                  ^(Z|CST|PST|AST|...)
   *
   *   additional restriction: 18:00:00 is the upper bound (which means 18:00:01
   * is invalid)
   */
  __device__ inline thrust::pair<int64_t, uint8_t> parse_utc_like_tz(
    string_view const& tz_lit) const
  {
    size_type len = tz_lit.size_bytes();

    char const* ptr = tz_lit.data();

    // try to parse Z
    if (*ptr == 'Z') {
      if (len > 1) return {0, 1};
      return {0, 0};
    }

    size_t char_offset = 0;
    // skip UTC|GMT if existing
    if (len > 2 && ((*ptr == 'G' && *(ptr + 1) == 'M' && *(ptr + 2) == 'T') ||
                    (*ptr == 'U' && *(ptr + 1) == 'T' && *(ptr + 2) == 'C'))) {
      char_offset = 3;
    }

    // return for the pattern UTC|GMT (without exact offset)
    if (len == char_offset) return {0, 0};

    // parse sign +|-
    char const sign_char = *(ptr + char_offset++);
    int64_t sign;
    if (sign_char == '+') {
      sign = 1L;
    } else if (sign_char == '-') {
      sign = -1L;
    } else {
      // if the rep starts with UTC|GMT, it can NOT be region-based rep
      return {0, char_offset < 3 ? 1 : 2};
    }

    // parse hh:mm:ss
    int64_t hms[3] = {0L, 0L, 0L};
    bool has_colon = false;
    for (size_type i = 0; i < 3; i++) {
      // deal with the first digit
      hms[i] = *(ptr + char_offset++) - '0';
      if (hms[i] < 0 || hms[i] > 9) return {0, 2};

      // deal with trailing single digit instant:
      //  hh(GMT+8) - valid
      //  mm(GMT+11:2) - must be separated from (h)h by `:`
      //  ss(GMT-11:22:3) - invalid
      if (len == char_offset) {
        if (i == 2 || (i == 1 && !has_colon)) return {0, 2};
        break;
      }

      // deal with `:`
      if (*(ptr + char_offset) == ':') {
        // 1. (i == 1) one_digit mm with ss is invalid (+11:2:3)
        // 2. (i == 2) one_digit ss is invalid (+11:22:3)
        // 3. trailing `:` is invalid (GMT+8:)
        if (i > 0 || len == ++char_offset) return {0, 2};
        has_colon = true;
        continue;
      }

      // deal with the second digit
      auto digit = *(ptr + char_offset++) - '0';
      if (digit < 0 || digit > 9) return {0, 2};
      hms[i] = hms[i] * 10 + digit;

      if (len == char_offset) break;
      // deal with `:`
      if (*(ptr + char_offset) == ':') {
        // trailing `:` is invalid (UTC+11:)
        if (len == ++char_offset) return {0, 2};
        has_colon = true;
      }
    }

    // the upper bound is 18:00:00 (regardless of sign)
    if (hms[0] > 18 || hms[1] > 59 || hms[2] > 59) return {0, 2};
    if (hms[0] == 18 && hms[1] + hms[2] > 0) return {0, 2};

    return {sign * (hms[0] * 3600L + hms[1] * 60L + hms[2]), 0};
  }

  /**
   * TODO: replace linear search with more efficient approach (like prefix tree)
   */
  __device__ inline int query_index_from_tz_db(string_view const& tz_lit) const
  {
    auto predicate = [tz = tz_indices, &tz_lit] __device__(auto const i) {
      return tz->element<string_view>(i) == tz_lit;
    };
    auto ret = thrust::find_if(thrust::seq,
                               thrust::make_counting_iterator(0),
                               thrust::make_counting_iterator(tz_indices->size()),
                               predicate);

    return *ret;
  }

  /**
   * Perform binary search to search out the timezone offset based on loose epoch
   * instants. Basically, this is the same approach as
   * `convert_timestamp_tz_functor`.
   */
  __device__ inline int64_t extract_timezone_offset(int64_t loose_epoch_second,
                                                    size_type tz_index) const
  {
    auto const& utc_offsets    = transitions->child().child(2);
    auto const& loose_instants = transitions->child().child(3);

    auto const local_transitions = cudf::list_device_view{*transitions, tz_index};
    auto const list_size         = local_transitions.size();

    auto const transition_times = cudf::device_span<int64_t const>(
      loose_instants.data<int64_t>() + local_transitions.element_offset(0),
      static_cast<size_t>(list_size));

    auto const it = thrust::upper_bound(
      thrust::seq, transition_times.begin(), transition_times.end(), loose_epoch_second);
    auto const idx         = static_cast<size_type>(thrust::distance(transition_times.begin(), it));
    auto const list_offset = local_transitions.element_offset(idx - 1);

    return static_cast<int64_t>(utc_offsets.element<int32_t>(list_offset));
  }

  /**
   * The formula to compute loose epoch from local time. The loose epoch is used
   * to search for the corresponding timezone offset of specific zone ID from
   * TimezoneDB. The target of loose epoch is to transfer local time to a number
   * which is proportional to the real timestamp as easily as possible. Loose
   * epoch, as a computation approach, helps us to align probe(kernel side) to
   * the TimezoneDB(Java side). Then, we can apply binary search based on loose
   * epoch instants of TimezoneDB to find out the correct timezone offset.
   */
  __device__ inline int64_t compute_loose_epoch_s(timestamp_components const& ts) const
  {
    return (ts.year * 400 + (ts.month - 1) * 31 + ts.day - 1) * 86400L + ts.hour * 3600L +
           ts.minute * 60L + ts.second;
  }

  /**
   * Leverage STL to convert local time to UTC unix_timestamp(in millisecond)
   */
  __device__ inline int64_t compute_epoch_us(timestamp_components const& ts) const
  {
    auto const ymd =  // chrono class handles the leap year calculations for us
      cuda::std::chrono::year_month_day(cuda::std::chrono::year{ts.year},
                                        cuda::std::chrono::month{static_cast<uint32_t>(ts.month)},
                                        cuda::std::chrono::day{static_cast<uint32_t>(ts.day)});
    auto days = cuda::std::chrono::sys_days(ymd).time_since_epoch().count();

    int64_t timestamp_s = (days * 24L * 3600L) + (ts.hour * 3600L) + (ts.minute * 60L) + ts.second;

    return timestamp_s * 1000000L + ts.microseconds;
  }

  /**
   * Ported from Spark:
   *   https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
   *   org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L394
   *
   * Parse a string with time zone to a timestamp.
   * The bool in the returned tuple is false if the parse failed.
   */
  __device__ inline ParseResult parse_string_to_timestamp_us(
    timestamp_components* ts_comp,
    char const** parsed_tz_ptr,
    size_type* parsed_tz_length,
    cudf::string_view const& timestamp_str) const
  {
    const char* curr_ptr = timestamp_str.data();
    const char* end_ptr  = curr_ptr + timestamp_str.size_bytes();

    // trim left
    while (curr_ptr < end_ptr && is_whitespace(*curr_ptr)) {
      ++curr_ptr;
    }
    // trim right
    while (curr_ptr < end_ptr - 1 && is_whitespace(*(end_ptr - 1))) {
      --end_ptr;
    }

    if (curr_ptr == end_ptr) { return ParseResult::INVALID; }

    const char* const bytes      = curr_ptr;
    const size_type bytes_length = end_ptr - curr_ptr;

    // segments stores: [year, month, day, hour, minute, seconds, microseconds, no_use_item, no_use_item]
    // the two tail items are no use, but here keeps them as Spark does
    int segments[]             = {1, 1, 1, 0, 0, 0, 0, 0, 0};
    int segments_len           = 9;
    int i                      = 0;
    int current_segment_value  = 0;
    int current_segment_digits = 0;
    size_t j                   = 0;
    int digits_milli           = 0;
    // bool just_time = false;
    thrust::optional<int> year_sign;
    if ('-' == bytes[j] || '+' == bytes[j]) {
      if ('-' == bytes[j]) {
        year_sign = -1;
      } else {
        year_sign = 1;
      }
      j += 1;
    }

    while (j < bytes_length) {
      char b           = bytes[j];
      int parsed_value = static_cast<int32_t>(b - '0');
      if (parsed_value < 0 || parsed_value > 9) {
        if (0 == j && 'T' == b) {
          // just_time = true;
          i += 3;
        } else if (i < 2) {
          if (b == '-') {
            if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
            segments[i]            = current_segment_value;
            current_segment_value  = 0;
            current_segment_digits = 0;
            i += 1;
          } else if (0 == i && ':' == b && !year_sign.has_value()) {
            // just_time = true;
            if (!is_valid_digits(3, current_segment_digits)) { return ParseResult::INVALID; }
            segments[3]            = current_segment_value;
            current_segment_value  = 0;
            current_segment_digits = 0;
            i                      = 4;
          } else {
            return ParseResult::INVALID;
          }
        } else if (2 == i) {
          if (' ' == b || 'T' == b) {
            if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
            segments[i]            = current_segment_value;
            current_segment_value  = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            return ParseResult::INVALID;
          }
        } else if (3 == i || 4 == i) {
          if (':' == b) {
            if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
            segments[i]            = current_segment_value;
            current_segment_value  = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            return ParseResult::INVALID;
          }
        } else if (5 == i || 6 == i) {
          if ('.' == b && 5 == i) {
            if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
            segments[i]            = current_segment_value;
            current_segment_value  = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            if (!is_valid_digits(i, current_segment_digits) || !allow_tz_in_date_str) {
              return ParseResult::INVALID;
            }
            segments[i]            = current_segment_value;
            current_segment_value  = 0;
            current_segment_digits = 0;
            i += 1;
            *parsed_tz_ptr = bytes + j;
            // strip the whitespace between timestamp and timezone
            while (*parsed_tz_ptr < end_ptr && is_whitespace(**parsed_tz_ptr))
              ++(*parsed_tz_ptr);
            *parsed_tz_length = end_ptr - *parsed_tz_ptr;
            break;
          }
          if (i == 6 && '.' != b) { i += 1; }
        } else {
          if (i < segments_len && (':' == b || ' ' == b)) {
            if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
            segments[i]            = current_segment_value;
            current_segment_value  = 0;
            current_segment_digits = 0;
            i += 1;
          } else {
            return ParseResult::INVALID;
          }
        }
      } else {
        if (6 == i) { digits_milli += 1; }
        // We will truncate the nanosecond part if there are more than 6 digits,
        // which results in loss of precision
        if (6 != i || current_segment_digits < 6) {
          current_segment_value = current_segment_value * 10 + parsed_value;
        }
        current_segment_digits += 1;
      }
      j += 1;
    }

    if (!is_valid_digits(i, current_segment_digits)) { return ParseResult::INVALID; }
    segments[i] = current_segment_value;

    while (digits_milli < 6) {
      segments[6] *= 10;
      digits_milli += 1;
    }

    segments[0] *= year_sign.value_or(1);
    // above is ported from Spark.

    // copy segments to equivalent kernel timestamp_components
    // Note: In order to keep above code is equivalent to Spark implementation,
    //       did not use `timestamp_components` directly to save values.
    ts_comp->year         = segments[0];
    ts_comp->month        = static_cast<int8_t>(segments[1]);
    ts_comp->day          = static_cast<int8_t>(segments[2]);
    ts_comp->hour         = static_cast<int8_t>(segments[3]);
    ts_comp->minute       = static_cast<int8_t>(segments[4]);
    ts_comp->second       = static_cast<int8_t>(segments[5]);
    ts_comp->microseconds = segments[6];

    return ParseResult::OK;
  }
};

/**
 * The common entrance of string_to_timestamp, which combines two paths:
 * with_timezone and without_timezone. This function returns the The
 * transitions, tz_indices and default_tz_index are only for handling inputs
 * with timezone. So, this function distinguish with_timezone calls from
 * without_timezone ones by checking if transitions and tz_indices are nullptr.
 *
 */
std::unique_ptr<cudf::column> to_timestamp(cudf::strings_column_view const& input,
                                           bool ansi_mode,
                                           bool allow_tz_in_date_str                   = true,
                                           size_type default_tz_index                  = 1000000000,
                                           cudf::column_view const* transitions        = nullptr,
                                           cudf::strings_column_view const* tz_indices = nullptr)
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = rmm::mr::get_current_device_resource();

  auto d_strings = cudf::column_device_view::create(input.parent(), stream);
  // column to store the result timestamp
  auto result_col =
    cudf::make_timestamp_column(cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS},
                                input.size(),
                                cudf::mask_state::UNALLOCATED,
                                stream,
                                mr);
  // column to store the status `ParseResult`
  auto result_valid_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::UINT8}, input.size(), cudf::mask_state::UNALLOCATED, stream, mr);

  if (transitions == nullptr || tz_indices == nullptr) {
    thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(input.size()),
      thrust::make_zip_iterator(
        thrust::make_tuple(result_col->mutable_view().begin<cudf::timestamp_us>(),
                           result_valid_col->mutable_view().begin<uint8_t>())),
      parse_timestamp_string_fn<false>{
        *d_strings, default_tz_index, allow_tz_in_date_str});
  } else {
    auto const ft_cdv_ptr    = column_device_view::create(*transitions, stream);
    auto const d_transitions = lists_column_device_view{*ft_cdv_ptr};
    auto d_tz_indices        = cudf::column_device_view::create(tz_indices->parent(), stream);

    thrust::transform(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(input.size()),
      thrust::make_zip_iterator(
        thrust::make_tuple(result_col->mutable_view().begin<cudf::timestamp_us>(),
                           result_valid_col->mutable_view().begin<uint8_t>())),
      parse_timestamp_string_fn<true>{
        *d_strings, default_tz_index, true, d_transitions, *d_tz_indices});
  }

  auto valid_view = result_valid_col->mutable_view();

  // throw cuDF exception if there exists any unsupported formats
  auto exception_exists =
    thrust::any_of(rmm::exec_policy(stream),
                   valid_view.begin<uint8_t>(),
                   valid_view.end<uint8_t>(),
                   [] __device__(uint8_t e) { return e == ParseResult::UNSUPPORTED; });
  if (exception_exists) { CUDF_FAIL("There exists unsupported timestamp schema!"); }

  // build the updated null mask and compute the null count
  auto [valid_bitmask, valid_null_count] = cudf::detail::valid_if(
    valid_view.begin<uint8_t>(),
    valid_view.end<uint8_t>(),
    [] __device__(uint8_t e) { return e == 0; },
    stream,
    mr);

  // `output null count > input null count` indicates that there are new null
  // values generated during the `to_timestamp` transaction to replace invalid
  // inputs.
  if (ansi_mode && input.null_count() < valid_null_count) { return nullptr; }

  result_col->set_null_mask(valid_bitmask, valid_null_count, stream);
  return result_col;
}

}  // namespace

namespace spark_rapids_jni {

/**
 * Parse string column with time zone to timestamp column,
 * Returns a pair of timestamp column and a bool indicates whether it successes.
 * If does not have time zone in string, use the default time zone.
 */
std::unique_ptr<cudf::column> string_to_timestamp_with_tz(
  cudf::strings_column_view const& input,
  cudf::column_view const& transitions,
  cudf::strings_column_view const& tz_indices,
  cudf::size_type default_tz_index,
  bool ansi_mode)
{
  if (input.size() == 0) { return nullptr; }
  return to_timestamp(
    input, ansi_mode, true, default_tz_index, &transitions, &tz_indices);
}

/**
 * Parse string column with time zone to timestamp column,
 * Returns a pair of timestamp column and a bool indicates whether it successes.
 * Do not use the time zone in string.
 * If allow_time_zone is false and string contains time zone, then the string is
 * invalid.
 */
std::unique_ptr<cudf::column> string_to_timestamp_without_tz(
  cudf::strings_column_view const& input,
  bool allow_time_zone,
  bool ansi_mode)
{
  if (input.size() == 0) { return nullptr; }
  return to_timestamp(input, ansi_mode, allow_time_zone);
}

}  // namespace spark_rapids_jni
