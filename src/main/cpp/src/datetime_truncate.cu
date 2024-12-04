/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "datetime_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace spark_rapids_jni {

namespace detail {

namespace {

enum class component : uint8_t {
  YEAR,
  QUARTER,
  MONTH,
  WEEK,
  DAY,
  HOUR,
  MINUTE,
  SECOND,
  MILLISECOND,
  MICROSECOND
};

template <typename Timestamp, component Component>
struct truncate_fn {
  __device__ inline Timestamp operator()(Timestamp const ts) const
  {
    // No truncation needed for microsecond timestamps.
    if constexpr (Component == component::MICROSECOND) { return ts; }

    using namespace cuda::std::chrono;

    auto const days_since_epoch = floor<days>(ts);
    auto time_since_midnight    = ts - days_since_epoch;
    if (time_since_midnight.count() < 0) { time_since_midnight += days(1); }

    auto const hrs_  = [&] { return duration_cast<hours>(time_since_midnight); };
    auto const mins_ = [&] { return duration_cast<minutes>(time_since_midnight) - hrs_(); };
    auto const secs_ = [&] {
      return duration_cast<seconds>(time_since_midnight) - hrs_() - mins_();
    };
    auto const millisecs_ = [&] {
      return duration_cast<milliseconds>(time_since_midnight) - hrs_() - mins_() - secs_();
    };
    auto const microsecs_ = [&] {
      return duration_cast<microseconds>(time_since_midnight) - hrs_() - mins_() - secs_() -
             millisecs_();
    };
    auto const nanosecs_ = [&] {
      return duration_cast<nanoseconds>(time_since_midnight) - hrs_() - mins_() - secs_() -
             millisecs_() - microsecs_();
    };

    auto const ymd = year_month_day(days_since_epoch);

    // The components that are common for both date and timestamp (type day and microsecond):
    // year, quarter, month, week.
    if constexpr (Component == component::YEAR) {
      return Timestamp{sys_days{year_month_day{ymd.year(), month{1}, day{1}}}};
    }
    if constexpr (Component == component::MONTH) {
      return Timestamp{sys_days{year_month_day{ymd.year(), ymd.month(), day{1}}}};
    }

    // The rest of the components are specific to timestamp (type microsecond):
    if constexpr (Component == component::DAY) { return Timestamp{sys_days{ymd}}; }
    if constexpr (Component == component::HOUR) {
      static_assert(cuda::std::is_same_v<Timestamp, cudf::timestamp_us>);
      return Timestamp{sys_days{ymd} + hrs_()};
    }
    if constexpr (Component == component::MINUTE) {
      static_assert(cuda::std::is_same_v<Timestamp, cudf::timestamp_us>);
      return Timestamp{sys_days{ymd} + mins_()};
    }
    if constexpr (Component == component::SECOND) {
      static_assert(cuda::std::is_same_v<Timestamp, cudf::timestamp_us>);
      return Timestamp{sys_days{ymd} + secs_()};
    }
    if constexpr (Component == component::MILLISECOND) {
      static_assert(cuda::std::is_same_v<Timestamp, cudf::timestamp_us>);
      return Timestamp{sys_days{ymd} + millisecs_()};
    }

    CUDF_UNREACHABLE("Invalid component to truncate.");
  }
};

template <typename Timestamp, component Component>
std::unique_ptr<cudf::column> truncate_component(cudf::column_view const& input,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::is_timestamp(input.type()), "Input column must be timestamp type");
  auto const size = input.size();
  if (size == 0) return cudf::make_empty_column(input.type());

  auto output = cudf::make_fixed_width_column(input.type(),
                                              size,
                                              cudf::detail::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    input.begin<Timestamp>(),
                    input.end<Timestamp>(),
                    output->mutable_view().begin<Timestamp>(),
                    truncate_fn<Timestamp, Component>{});

  // Null count was invalidated when calling to `mutable_view()`.
  output->set_null_count(input.null_count());
  return output;
}

template <component Component>
auto const truncate_date_component = truncate_component<cudf::timestamp_D, Component>;

template <component Component>
auto const truncate_timestamp_component = truncate_component<cudf::timestamp_us, Component>;

std::string to_upper(std::string const& input)
{
  auto const to_upper_char = [](char const c) { return ('a' <= c && c <= 'z') ? c ^ 0x20 : c; };
  std::string output;
  output.reserve(input.size());
  for (auto const c : input) {
    output.push_back(to_upper_char(c));
  }
  return output;
}

}  // namespace

std::unique_ptr<cudf::column> truncate_date(cudf::column_view const& input,
                                            std::string const& component,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_DAYS,
               "The input must be day timestamps.");
  auto const comp = to_upper(component);

  if (comp == "YEAR" || comp == "YYYY" || comp == "YY") {
    return truncate_date_component<component::YEAR>(input, stream, mr);
  }
  if (comp == "MONTH" || comp == "MM" || comp == "MON") {
    return truncate_date_component<component::MONTH>(input, stream, mr);
  }

  // QUARTER and WEEK are not yet supported.
  CUDF_FAIL("Unsupported truncating component.");
}

std::unique_ptr<cudf::column> truncate_timestamp(cudf::column_view const& input,
                                                 std::string const& component,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input.type().id() == cudf::type_id::TIMESTAMP_MICROSECONDS,
               "The input must be microsecond timestamps.");
  auto const comp = to_upper(component);

  if (comp == "YEAR" || comp == "YYYY" || comp == "YY") {
    return truncate_timestamp_component<component::YEAR>(input, stream, mr);
  }
  if (comp == "MONTH" || comp == "MM" || comp == "MON") {
    return truncate_timestamp_component<component::MONTH>(input, stream, mr);
  }
  if (comp == "DAY" || comp == "DD") {
    return truncate_timestamp_component<component::DAY>(input, stream, mr);
  }
  if (comp == "HOUR") { return truncate_timestamp_component<component::HOUR>(input, stream, mr); }
  if (comp == "MINUTE") {
    return truncate_timestamp_component<component::MINUTE>(input, stream, mr);
  }
  if (comp == "SECOND") {
    return truncate_timestamp_component<component::SECOND>(input, stream, mr);
  }
  if (comp == "MILLISECOND") {
    return truncate_timestamp_component<component::MILLISECOND>(input, stream, mr);
  }
  if (comp == "MICROSECOND") {
    return truncate_timestamp_component<component::MICROSECOND>(input, stream, mr);
  }

  // QUARTER and WEEK are not yet supported.
  CUDF_FAIL("Unsupported truncating component.");
}

}  // namespace detail

std::unique_ptr<cudf::column> truncate(cudf::column_view const& input,
                                       std::string const& component,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const type = input.type().id();
  CUDF_EXPECTS(
    type == cudf::type_id::TIMESTAMP_DAYS || type == cudf::type_id::TIMESTAMP_MICROSECONDS,
    "The input must be either day or microsecond timestamps.");

  if (input.size() == 0) { return cudf::make_empty_column(input.type()); }
  return type == cudf::type_id::TIMESTAMP_DAYS
           ? detail::truncate_date(input, component, stream, mr)
           : detail::truncate_timestamp(input, component, stream, mr);
}

}  // namespace spark_rapids_jni
