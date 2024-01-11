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

#include <cudf/strings/strings_column_view.hpp>

namespace spark_rapids_jni {

/**
 *
 * Trims and parses a timestamp string column with time zone suffix to a
 * timestamp column. e.g.: 1991-04-14T02:00:00Asia/Shanghai => 1991-04-13
 * 18:00:00
 *
 * Refer to: https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
 * org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L394
 *
 * Spark supports the following formats:
 * `[+-]yyyy*`
 * `[+-]yyyy*-[m]m`
 * `[+-]yyyy*-[m]m-[d]d`
 * `[+-]yyyy*-[m]m-[d]d `
 * `[+-]yyyy*-[m]m-[d]d [h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 * `[+-]yyyy*-[m]m-[d]dT[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 * `[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 * `T[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 *
 * Unlike Spark, Spark-Rapids only supports the following formats:
 * `[+-]yyyy*`
 * `[+-]yyyy*-[m]m`
 * `[+-]yyyy*-[m]m-[d]d`
 * `[+-]yyyy*-[m]m-[d]d `
 * `[+-]yyyy*-[m]m-[d]d [h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 * `[+-]yyyy*-[m]m-[d]dT[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 *
 * Spark supports the following zone id forms:
 *   - Z - Zulu time zone UTC+0
 *   - +|-[h]h:[m]m
 *   - A short id, see
 * https://docs.oracle.com/javase/8/docs/api/java/time/ZoneId.html#SHORT_IDS
 *   - An id with one of the prefixes UTC+, UTC-, GMT+, GMT-, UT+ or UT-,
 *     and a suffix in the formats:
 *     - +|-h[h]
 *     - +|-hh[:]mm
 *     - +|-hh:mm:ss
 *     - +|-hhmmss
 *  - Region-based zone IDs in the form `area/city`, such as `Europe/Paris`
 *
 * Unlike Spark, Spark-Rapids only supports the following time zones:
 *   - Z - Zulu time zone UTC+0
 *   - +|-[h]h:[m]m
 *   - Region-based zone IDs in the form `area/city`, such as `Europe/Paris`
 *
 *
 * @param input input string column view.
 * @param transitions TimezoneDB, the table of transitions contains all
 * information for timezones
 * @param tz_indices TimezoneDB index of region-based timezone IDs
 * @param special_datetime_lit cache of special date times
 * @param default_tz_index the index of default timezone in TimezoneDB, if input
 * date-like string does not contain a time zone (like: YYYY-MM-DD:hhmmss), use
 * this time zone.
 * @param ansi_mode whether enforce ANSI mode or not. If true, exception will be
 * thrown encountering any invalid inputs.
 * @returns the pointer of the timestamp result column, which points to nullptr
 * if there exists invalid inputs and ANSI mode is on.
 */
std::unique_ptr<cudf::column> string_to_timestamp_with_tz(
  cudf::strings_column_view const& input,
  cudf::column_view const& transitions,
  cudf::strings_column_view const& tz_indices,
  cudf::strings_column_view const& special_datetime_lit,
  cudf::size_type default_tz_index,
  bool ansi_mode);

/**
 *
 * Trims and parses a timestamp string column with time zone suffix to a
 * timestamp column. e.g.: 1991-04-14T02:00:00Asia/Shanghai => 1991-04-13
 * 18:00:00
 *
 * Refer to: https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
 * org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L394
 *
 * Spark supports the following formats:
 * `[+-]yyyy*`
 * `[+-]yyyy*-[m]m`
 * `[+-]yyyy*-[m]m-[d]d`
 * `[+-]yyyy*-[m]m-[d]d `
 * `[+-]yyyy*-[m]m-[d]d [h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 * `[+-]yyyy*-[m]m-[d]dT[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 * `[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 * `T[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 *
 * Unlike Spark, Spark-Rapids only supports the following formats:
 * `[+-]yyyy*`
 * `[+-]yyyy*-[m]m`
 * `[+-]yyyy*-[m]m-[d]d`
 * `[+-]yyyy*-[m]m-[d]d `
 * `[+-]yyyy*-[m]m-[d]d [h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 * `[+-]yyyy*-[m]m-[d]dT[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 *
 * Spark supports the following zone id forms:
 *   - Z - Zulu time zone UTC+0
 *   - +|-[h]h:[m]m
 *   - A short id, see
 * https://docs.oracle.com/javase/8/docs/api/java/time/ZoneId.html#SHORT_IDS
 *   - An id with one of the prefixes UTC+, UTC-, GMT+, GMT-, UT+ or UT-,
 *     and a suffix in the formats:
 *     - +|-h[h]
 *     - +|-hh[:]mm
 *     - +|-hh:mm:ss
 *     - +|-hhmmss
 *  - Region-based zone IDs in the form `area/city`, such as `Europe/Paris`
 *
 * Unlike Spark, Spark-Rapids only supports the following time zones:
 *   - Z - Zulu time zone UTC+0
 *   - +|-[h]h:[m]m
 *   - Region-based zone IDs in the form `area/city`, such as `Europe/Paris`
 *
 *
 * @param input input string column view.
 * @param special_datetime_lit cache of special date times
 * @param allow_time_zone whether allow time zone in the timestamp string. e.g.:
 *   1991-04-14T02:00:00Asia/Shanghai is invalid when do not allow time zone.
 * @param ansi_mode whether enforce ANSI mode or not. If true, exception will be
 * thrown encountering any invalid inputs.
 * @returns the pointer of the timestamp result column, which points to nullptr
 * if there exists invalid inputs and ANSI mode is on.
 */
std::unique_ptr<cudf::column> string_to_timestamp_without_tz(
  cudf::strings_column_view const& input,
  cudf::strings_column_view const& special_datetime_lit,
  bool allow_time_zone,
  bool ansi_mode);

}  // namespace spark_rapids_jni
