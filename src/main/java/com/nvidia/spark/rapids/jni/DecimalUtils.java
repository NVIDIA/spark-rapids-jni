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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

public class DecimalUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }


  /**
   * Multiply two DECIMAL128 columns together into a DECIMAL128 product rounded to the specified
   * scale with overflow detection.
   * @param a            factor input, must match row count of the other factor input
   * @param b            factor input, must match row count of the other factor input
   * @param productScale scale to use for the product type
   * @return table containing a boolean column and a DECIMAL128 product column of the specified
   *         scale. The boolean value will be true if an overflow was detected for that row's
   *         DECIMAL128 product value. A null input row will result in a corresponding null output
   *         row.
   */
  public static Table multiply128(ColumnView a, ColumnView b, int productScale) {
    return new Table(multiply128(a.getNativeView(), b.getNativeView(), productScale));
  }

  /**
   * Divide two DECIMAL128 columns and produce a DECIMAL128 quotient rounded to the specified
   * scale with overflow detection.
   * @param a            factor input, must match row count of the other factor input
   * @param b            factor input, must match row count of the other factor input
   * @param quotientScale scale to use for the quotient type
   * @return table containing a boolean column and a DECIMAL128 quotient column of the specified
   *         scale. The boolean value will be true if an overflow was detected for that row's
   *         DECIMAL128 quotient value. A null input row will result in a corresponding null output
   *         row.
   */
  public static Table divide128(ColumnView a, ColumnView b, int quotientScale) {
    return new Table(divide128(a.getNativeView(), b.getNativeView(), quotientScale));
  }

  private static native long[] multiply128(long viewA, long viewB, int productScale);

  private static native long[] divide128(long viewA, long viewB, int quotientScale);
}
