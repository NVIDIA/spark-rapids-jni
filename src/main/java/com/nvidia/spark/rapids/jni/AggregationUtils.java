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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.NativeDepsLoader;

public class AggregationUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Create a test host UDF for testing purposes.
   *
   * This will return two values: the first is the pointer to the host UDF, and the second is the
   * hash code of the host UDF.
   *
   * To create a host UDF aggregation, do this:
   * ```
   * long[] udfAndHash = AggregationUtils.createTestHostUDF();
   * new ai.rapids.cudf.HostUDFAggregation(udfAndHash[0], udfAndHash[1]);
   * ```
   */
  public static long[] createTestHostUDF() {
    return createNativeTestHostUDF();
  }

  private static native long[] createNativeTestHostUDF();
}
