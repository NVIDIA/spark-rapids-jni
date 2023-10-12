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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;

/**
 * Utility class for converting between column major and row major data
 */
public class DateTimeRebase {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Convert the given timestamps as a number of days or microseconds since the epoch instant
   * 1970-01-01T00:00:00Z to a local date-time in Proleptic Gregorian calendar, reinterpreting
   * the result as in Julian calendar, then compute the number of days or microseconds since the
   * epoch from that Julian local date-time.
   * This is to match with Apache Spark's `localRebaseGregorianToJulianDays` and
   * `rebaseGregorianToJulianMicros` functions with timezone fixed to UTC.
   */
  public static ColumnVector rebaseGregorianToJulian(ColumnView input) {
    return new ColumnVector(rebaseGregorianToJulian(input.getNativeView()));
  }

  private static native long rebaseGregorianToJulian(long nativeHandle);
}
