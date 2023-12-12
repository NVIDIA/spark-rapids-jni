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

#include <cudf/strings/strings_column_view.hpp>

namespace spark_rapids_jni {

/**
 *
 * Trims and parses a timestamp string column with time zone suffix to a timestamp column.
 * e.g.: 1991-04-14T02:00:00Asia/Shanghai => 1991-04-13 18:00:00
 *
 * Refer to: https://github.com/apache/spark/blob/v3.5.0/sql/api/src/main/scala/
 * org/apache/spark/sql/catalyst/util/SparkDateTimeUtils.scala#L394
 *
 * Formats are:
 *
 * `[+-]yyyy*`
 * `[+-]yyyy*-[m]m`
 * `[+-]yyyy*-[m]m-[d]d`
 * `[+-]yyyy*-[m]m-[d]d `
 * `[+-]yyyy*-[m]m-[d]d [h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 * `[+-]yyyy*-[m]m-[d]dT[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 * `[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 * `T[h]h:[m]m:[s]s.[ms][ms][ms][us][us][us][zone_id]`
 *
 * Spark supports the following zone id forms:
 *   - Z - Zulu time zone UTC+0
 *   - +|-[h]h:[m]m
 *   - A short id, see https://docs.oracle.com/javase/8/docs/api/java/time/ZoneId.html#SHORT_IDS
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
 * @param default_time_zone if input string does not contain a time zone, use this time zone.
 * @param allow_time_zone whether allow time zone in the timestamp string. e.g.: 
 *   1991-04-14T02:00:00Asia/Shanghai is invalid when do not allow time zone.
 * @param allow_special_expressions whether allow epoch, now, today, yesterday, tomorrow strings.
 * @param ansi_mode is ansi mode
 * @returns a timestamp column and a bool column. Bool column is empty if ansi mode is false, not empty otherwise.
 */
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
parse_string_to_timestamp(cudf::strings_column_view const &input,
                          std::string_view const &default_time_zone,
                          bool allow_time_zone,
                          bool allow_special_expressions,
                          bool ansi_mode);
} // namespace spark_rapids_jni
