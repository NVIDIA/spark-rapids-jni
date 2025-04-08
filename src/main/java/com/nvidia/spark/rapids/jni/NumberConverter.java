/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

public class NumberConverter {

  /**
   *
   * Convert numbers(in string column) between different number bases. If toBase>0
   * the
   * result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertCvCvCv(
    ColumnVector input, ColumnVector fromBase, ColumnVector toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";

    return new ColumnVector(
      convertCvCvCv(input.getNativeView(), fromBase.getNativeView(), toBase.getNativeView()));
  }

  /**
   *
   * Convert numbers(in string column) between different number bases. If toBase>0
   * the
   * result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertCvCvS(ColumnVector input, ColumnVector fromBase, int toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";

    return new ColumnVector(convertCvCvS(input.getNativeView(), fromBase.getNativeView(), toBase));
  }

  /**
   *
   * Convert numbers(in string column) between different number bases. If toBase>0
   * the
   * result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertCvSCv(ColumnVector input, int fromBase, ColumnVector toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";

    return new ColumnVector(convertCvSCv(input.getNativeView(), fromBase, toBase.getNativeView()));
  }

  /**
   *
   * Convert numbers(in string column) between different number bases. If toBase>0
   * the
   * result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertCvSS(ColumnVector input, int fromBase, int toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";

    return new ColumnVector(convertCvSS(input.getNativeView(), fromBase, toBase));
  }


  /**
   *
   * Convert numbers(in string column) between different number bases. If toBase>0
   * the
   * result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertSCvCv(
    Scalar input, ColumnVector fromBase, ColumnVector toBase) {
    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";
    return new ColumnVector(
      convertSCvCv(input.getScalarHandle(), fromBase.getNativeView(), toBase.getNativeView()));
  }

  /**
   *
   * Convert numbers(in string column) between different number bases. If toBase>0
   * the
   * result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertSCvS(Scalar input, ColumnVector fromBase, int toBase) {
    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";
    return new ColumnVector(convertSCvS(input.getScalarHandle(), fromBase.getNativeView(), toBase));
  }

  /**
   *
   * Convert numbers(in string column) between different number bases. If toBase>0
   * the
   * result
   * is unsigned, otherwise it is signed.
   * First trim the space characters (ASCII 32).
   * Return null if len(trim_ascii_32(str)) == 0.
   * Return all nulls if `from_base` or `to_base` is not in range [2, 36].
   *
   * e.g.:
   * convert('11', 2, 10) = '3'
   * convert('F', 16, 10) = '15'
   * convert('17', 10, 16) = '11'
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return the string column contains numbers with `to_base` base
   */
  public static ColumnVector convertSSCv(Scalar input, int fromBase, ColumnVector toBase) {
    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";
    return new ColumnVector(convertSSCv(input.getScalarHandle(), fromBase, toBase.getNativeView()));
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between
   * different number bases.
   * This is for the checking when it's ANSI mode. For more details, please refer
   * to the convert function.
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowCvCvCv(
    ColumnVector input, ColumnVector fromBase, ColumnVector toBase) {
    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";
    return isConvertOverflowCvCvCv(
      input.getNativeView(), fromBase.getNativeView(), toBase.getNativeView());
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between
   * different number
   * bases. This is for the checking when it's ANSI mode. For more details, please
   * refer to the
   * convert function.
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowCvCvS(
    ColumnVector input, ColumnVector fromBase, int toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";

    return isConvertOverflowCvCvS(input.getNativeView(), fromBase.getNativeView(), toBase);
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between
   * different number
   * bases. This is for the checking when it's ANSI mode. For more details, please
   * refer to the
   * convert function.
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowCvSCv(
    ColumnVector input, int fromBase, ColumnVector toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";

    return isConvertOverflowCvSCv(input.getNativeView(), fromBase, toBase.getNativeView());
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between
   * different number
   * bases. This is for the checking when it's ANSI mode. For more details, please
   * refer to the
   * convert function.
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowCvSS(ColumnVector input, int fromBase, int toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";

    return isConvertOverflowCvSS(input.getNativeView(), fromBase, toBase);
  }


  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between
   * different number bases.
   * This is for the checking when it's ANSI mode. For more details, please refer
   * to the convert function.
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowSCvCv(
    Scalar input, ColumnVector fromBase, ColumnVector toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";

    return isConvertOverflowSCvCv(
      input.getScalarHandle(), fromBase.getNativeView(), toBase.getNativeView());
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between
   * different number
   * bases. This is for the checking when it's ANSI mode. For more details, please
   * refer to the
   * convert function.
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowSCvS(Scalar input, ColumnVector fromBase, int toBase) {

    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (fromBase.getType().equals(DType.INT32)) : "From base must be Integers";

    return isConvertOverflowSCvS(input.getScalarHandle(), fromBase.getNativeView(), toBase);
  }

  /**
   *
   * Check if overflow occurs for converting numbers(in string column) between
   * different number
   * bases. This is for the checking when it's ANSI mode. For more details, please
   * refer to the
   * convert function.
   *
   * @param input    the input string column contains numbers
   * @param fromBase the number base of input, valid range is [2, 36]
   * @param toBase   the number base of output, valid range is [2, 36]
   *
   * @return If overflow occurs, return true; otherwise, return false.
   */
  public static boolean isConvertOverflowSSCv(Scalar input, int fromBase, ColumnVector toBase) {
    assert (input.getType().equals(DType.STRING)) : "Input must be strings";
    assert (toBase.getType().equals(DType.INT32)) : "To base must be Integers";
    return isConvertOverflowSSCv(input.getScalarHandle(), fromBase, toBase.getNativeView());
  }

  private static native long convertCvCvCv(long input, long fromBase, long toBase);

  private static native long convertCvCvS(long input, long fromBase, int toBase);

  private static native long convertCvSCv(long input, int fromBase, long toBase);

  private static native long convertCvSS(long input, int fromBase, int toBase);

  private static native long convertSCvCv(long input, long fromBase, long toBase);

  private static native long convertSCvS(long input, long fromBase, int toBase);

  private static native long convertSSCv(long input, int fromBase, long toBase);

  private static native boolean isConvertOverflowCvCvCv(long input, long fromBase, long toBase);

  private static native boolean isConvertOverflowCvCvS(long input, long fromBase, int toBase);

  private static native boolean isConvertOverflowCvSCv(long input, int fromBase, long toBase);

  private static native boolean isConvertOverflowCvSS(long input, int fromBase, int toBase);

  private static native boolean isConvertOverflowSCvCv(long input, long fromBase, long toBase);

  private static native boolean isConvertOverflowSCvS(long input, long fromBase, int toBase);

  private static native boolean isConvertOverflowSSCv(long input, int fromBase, long toBase);

}
