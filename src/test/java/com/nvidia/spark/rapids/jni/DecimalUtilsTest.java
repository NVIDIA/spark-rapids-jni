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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

import static ai.rapids.cudf.AssertUtils.*;

public class DecimalUtilsTest {
  ColumnVector makeDec128Column(String ... values) {
    BigDecimal[] decVals = new BigDecimal[values.length];
    for (int i = 0; i < values.length; i++) {
      if (values[i] != null) {
        decVals[i] = new BigDecimal(values[i]);
      }
    }
    try (ColumnVector small = ColumnVector.fromDecimals(decVals)) {
      return small.castTo(DType.create(DType.DTypeEnum.DECIMAL128, small.getType().getScale()));
    }
  }

  @Test
  void simplePosMultiplyOneByZero() {
    try (ColumnVector lhs =
             makeDec128Column("1.0", "10.0", "1000000000000000000000000000000000000.0");
         ColumnVector rhs =
             makeDec128Column("1",   "1",    "1");
         ColumnVector expectedBasic =
             makeDec128Column("1.0", "10.0", "1000000000000000000000000000000000000.0");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simplePosMultiplyOneByOne() {
    try (ColumnVector lhs =
             makeDec128Column("1.0", "3.7");
         ColumnVector rhs =
             makeDec128Column("1.0", "1.5");
         ColumnVector expectedBasic =
             makeDec128Column("1.0", "5.6");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false, false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simplePosMultiplyZeroByNegOne() {
    try (ColumnVector lhs =
             makeDec128Column("1");
         ColumnVector rhs =
             makeDec128Column("1e1");
         ColumnVector expectedBasic =
             makeDec128Column("10.0");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void largePosMultiplyTenByTen() {
    try (ColumnVector lhs =
             makeDec128Column("577694940161436285811555447.3103121126");
         ColumnVector rhs =
             makeDec128Column("100.0000000000");
         ColumnVector expectedBasic =
             makeDec128Column("57769494016143628581155544731.031211");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void overflowMult() {
    try (ColumnVector lhs =
             makeDec128Column("577694938495380589068894346.7625198736");
         ColumnVector rhs =
             makeDec128Column("-1258508260891400005608241690.1564700995");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(true);
         Table found = DecimalUtils.multiply128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
    }

  }

  @Test
  void simpleNegMultiplyOneByZero() {
    try (ColumnVector lhs =
             makeDec128Column("1.0",  "-1.0", "10.0");
         ColumnVector rhs =
             makeDec128Column("-1",   "1",    "-1");
         ColumnVector expectedBasic =
             makeDec128Column("-1.0", "-1.0", "-10.0");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simpleNegMultiplyOneByOne() {
    try (ColumnVector lhs =
             makeDec128Column("1.0",  "-1.0", "3.7");
         ColumnVector rhs =
             makeDec128Column("-1.0", "-1.0", "-1.5");
         ColumnVector expectedBasic =
             makeDec128Column("-1.0",  "1.0", "-5.6");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simpleNegMultiplyTenByTenSparkCompat() {
    // many of the numbers listed here are *NOT* what BigDecimal would 
    // normally spit out. Spark has a bug https://issues.apache.org/jira/browse/SPARK-40129
    // which causes some of the rounding to be off, so these come directly from
    // Spark. It should be simple to fix this issue by deleteing code, or bypassing the
    // first divide step when/if Spark fixes it.
    try (ColumnVector lhs =
             makeDec128Column("3358377338823096511784947656.4650294583",
                 "7161021785186010157110137546.5940777916",
                 "9173594185998001607642838421.5479932913");
         ColumnVector rhs =
             makeDec128Column("-12.0000000000",
                 "-12.0000000000",
                 "-12.0000000000");
         ColumnVector expectedBasic =
             makeDec128Column("-40300528065877158141419371877.580354",
                 "-85932261422232121885321650559.128933",
                 "-110083130231976019291714061058.575920");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.multiply128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simplePosDivOneByZero() {
    try (ColumnVector lhs =
             makeDec128Column("1.0", "10.0", "1.0", "1000000000000000000000000000000000000.0");
         ColumnVector rhs =
             makeDec128Column("1",   "2",    "0",   "5");
         ColumnVector expectedBasic =
             makeDec128Column("1.0", "5.0",  "0",   "200000000000000000000000000000000000.0");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false, false, true, false);
         Table found = DecimalUtils.divide128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simplePosDivOneByOne() {
    try (ColumnVector lhs =
             makeDec128Column("1.0", "3.7", "99.9");
         ColumnVector rhs =
             makeDec128Column("1.0", "1.5", "4.5");
         ColumnVector expectedBasic =
             makeDec128Column("1.0", "2.5", "22.2");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.divide128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void simpleNegDivOneByOne() {
    try (ColumnVector lhs =
             makeDec128Column("1.0", "-3.7", "-99.9");
         ColumnVector rhs =
             makeDec128Column("-1.0", "1.5", "-4.5");
         ColumnVector expectedBasic =
             makeDec128Column("-1.0", "-2.5", "22.2");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.divide128(lhs, rhs, -1)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void divComplex() {
    try (ColumnVector lhs =
             makeDec128Column("100000000000000000000000000000000");
         ColumnVector rhs =
             makeDec128Column("3.0000000000000000000000000000000000000");
         ColumnVector expectedBasic =
             makeDec128Column("33333333333333333333333333333333.333333");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false);
         Table found = DecimalUtils.divide128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void div17() {
    try (ColumnVector lhs =
             makeDec128Column("1454.48287885760884146",
                 "3655.54438423288356646");
         ColumnVector rhs =
             makeDec128Column("100.00000000000000000",
                 "100.00000000000000000");
         ColumnVector expectedBasic =
             makeDec128Column("14.54482878857608841",
                 "36.55544384232883566");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false, false);
         Table found = DecimalUtils.divide128(lhs, rhs, -17)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void div17WithPosScale() {
    try (ColumnVector lhs =
             makeDec128Column("1454.48287885760884146");
         ColumnVector rhs =
             makeDec128Column("1e2");
         ColumnVector expectedBasic =
             makeDec128Column("14.54482878857608841");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false);
         Table found = DecimalUtils.divide128(lhs, rhs, -17)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void div21WithPosScale() {
    try (ColumnVector lhs =
             makeDec128Column("5776949401614362.858115554473103121126");
         ColumnVector rhs =
             makeDec128Column("1e2");
         ColumnVector expectedBasic =
             makeDec128Column("57769494016143.628581");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false);
         Table found = DecimalUtils.divide128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }

  @Test
  void div21() {
    try (ColumnVector lhs =
             makeDec128Column("60250054953505368.439892586764888491018",
                 "91910085134512953.335347579448489062875",
                 "51312633107598808.869351260608653423886");
         ColumnVector rhs =
             makeDec128Column("97982875273794447.385070145919990343867",
                 "94478503341597285.814104936062234698349",
                 "92266075543848323.800466593082956765923");
         ColumnVector expectedBasic =
             makeDec128Column("0.614904",
                 "0.972815",
                 "0.556138");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false, false, false);
         Table found = DecimalUtils.divide128(lhs, rhs, -6)) {
      assertColumnsAreEqual(expectedValid, found.getColumn(0));
      assertColumnsAreEqual(expectedBasic, found.getColumn(1));
    }
  }
}
