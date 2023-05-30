/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
   * scale with overflow detection. This method considers a precision greater than 38 as overflow
   * even if the number still fits in a 128-bit representation.
   * @param a factor input, must match row count of the other factor input
   * @param b factor input, must match row count of the other factor input
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
   * scale with overflow detection. This method considers a precision greater than 38 as overflow
   * even if the number still fits in a 128-bit representation.
   * @param a factor input, must match row count of the other factor input
   * @param b factor input, must match row count of the other factor input
   * @param quotientScale scale to use for the quotient type
   * @return table containing a boolean column and a DECIMAL128 quotient column of the specified
   *         scale. The boolean value will be true if an overflow was detected for that row's
   *         DECIMAL128 quotient value. A null input row will result in a corresponding null output
   *         row.
   */
  public static Table divide128(ColumnView a, ColumnView b, int quotientScale) {
    return new Table(divide128(a.getNativeView(), b.getNativeView(), quotientScale, false));
  }

  /**
   * Divide two DECIMAL128 columns and produce a INT64 quotient with overflow detection.
   * This method considers an overflow if the 128-bit quotient of the original numbers overflows
   * and doesn't base the decision on the 64-bit number.
   * Example:
   * 451635271134476686911387864.48 div -961.110 = 2284624887606872042L
   * A positive number divided by a negative number resulting in a positive result is clearly
   * an overflow but Spark doesn't consider this an overflow as the 128-bit
   * answer (-469910073908789510993942.27973) is still within the expected 38 digits of
   * precision
   *
   * @param a factor input, must match row count of the other factor input
   * @param b factor input, must match row count of the other factor input
   * @return table containing a boolean column and a INT64 quotient column.
   *         The boolean value will be true if an overflow was detected for that row's
   *         INT64 quotient value. A null input row will result in a corresponding null output
   *         row.
   */
  public static Table integerDivide128(ColumnView a, ColumnView b) {
    return new Table(divide128(a.getNativeView(), b.getNativeView(), 0, true));
  }

  /**
   * Divide two DECIMAL128 columns and produce a DECIMAL128 remainder with overflow detection.
   * Example:
   * 451635271134476686911387864.48 % -961.110 = 775.233
   * 
   * Generally, this will never really overflow unless in the divide by zero case.
   * But it will detect an overflow in any case.
   *
   * @param a factor input, must match row count of the other factor input
   * @param b factor input, must match row count of the other factor input
   * @param remainderScale scale to use for the remainder type
   * @return table containing a boolean column and a DECIMAL128 remainder column.
   *         The boolean value will be true if an overflow was detected for that row's
   *         DECIMAL128 remainder value. A null input row will result in a corresponding null 
   *         output row.
   */
  public static Table remainder128(ColumnView a, ColumnView b, int remainderScale) {
    return new Table(remainder128(a.getNativeView(), b.getNativeView(), remainderScale));
  }

  /**
   * Subtract two DECIMAL128 columns and produce a DECIMAL128 result rounded to the specified
   * scale with overflow detection. This method considers a precision greater than 38 as overflow
   * even if the number still fits in a 128-bit representation.
   *
   * NOTE: This is very specific to Spark 3.4. This method is incompatible with previous versions
   * of Spark. We don't need this for versions prior to Spark 3.4
   *
   * @param a input, must match row count of the other input
   * @param b input, must match row count of the other input
   * @param targetScale scale to use for the result
   * @return table containing a boolean column and a DECIMAL128 result column of the specified
   *         scale. The boolean value will be true if an overflow was detected for that row's
   *         DECIMAL128 result value. A null input row will result in a corresponding null output
   *         row.
   */

  public static Table subtract128(ColumnView a, ColumnView b, int targetScale) {
    if (java.lang.Math.abs(a.getType().getScale() - b.getType().getScale()) > 77) {
      throw new IllegalArgumentException("The intermediate scale for calculating the result " +
          "exceeds 256-bit representation");
    }
    return new Table(subtract128(a.getNativeView(), b.getNativeView(), targetScale));
  }
  /**
   * Add two DECIMAL128 columns and produce a DECIMAL128 result rounded to the specified
   * scale with overflow detection. This method considers a precision greater than 38 as overflow
   * even if the number still fits in a 128-bit representation.
   *
   * NOTE: This is very specific to Spark 3.4. This method is incompatible with previous versions
   * of Spark. We don't need this for versions prior to Spark 3.4
   *
   * @param a input, must match row count of the other input
   * @param b input, must match row count of the other input
   * @param targetScale scale to use for the sum
   * @return table containing a boolean column and a DECIMAL128 sum column of the specified
   *         scale. The boolean value will be true if an overflow was detected for that row's
   *         DECIMAL128 result value. A null input row will result in a corresponding null output
   *         row.
   */
  public static Table add128(ColumnView a, ColumnView b, int targetScale) {
    if (java.lang.Math.abs(a.getType().getScale() - b.getType().getScale()) > 77) {
      throw new IllegalArgumentException("The intermediate scale for calculating the result " +
          "exceeds 256-bit representation");
    }
    return new Table(add128(a.getNativeView(), b.getNativeView(), targetScale));
  }

  private static native long[] multiply128(long viewA, long viewB, int productScale);

  private static native long[] divide128(long viewA, long viewB, int quotientScale, boolean isIntegerDivide);

  private static native long[] remainder128(long viewA, long viewB, int remainderScale);

  private static native long[] add128(long viewA, long viewB, int targetScale);

  private static native long[] subtract128(long viewA, long viewB, int targetScale);
}
