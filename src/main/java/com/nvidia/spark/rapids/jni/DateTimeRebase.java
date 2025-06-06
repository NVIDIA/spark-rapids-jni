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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;

/**
 * This will be removed after the plugin picks up DateTimeUtils class.
 */
public class DateTimeRebase {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static ColumnVector rebaseGregorianToJulian(ColumnView input) {
    return DateTimeUtils.rebaseGregorianToJulian(input);
  }

  public static ColumnVector rebaseJulianToGregorian(ColumnView input) {
    return DateTimeUtils.rebaseJulianToGregorian(input);
  }
}
