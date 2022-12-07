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
  void intDivide() {
    try (ColumnVector lhs =
             makeDec128Column("3396191716868766147341919609.06", "-6893798181986328848375556144.67");
         ColumnVector rhs =
             makeDec128Column("7317548469.64", "98565515088.44");
         ColumnVector expectedBasic =
             makeDec128Column("464116053478747633", "-69941278912819784");
         ColumnVector expectedValid =
             ColumnVector.fromBooleans(false, false);
         Table found = DecimalUtils.integerDivide128(lhs, rhs)) {
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

  @Test
  void addPrecision38ScaleNeg10WithOverflow() {
    try (
        ColumnVector lhs = makeDec128Column("9191008513307131620269245301.1615457290",
            "-9191008513307131620269245301.1615457290");
        ColumnVector rhs = makeDec128Column("9447850332473678680446404122.5624623187",
            "-9447850332473678680446404122.5624623187");
        ColumnVector expectedValid = ColumnVector.fromBooleans(true, true)) {
      Table result = DecimalUtils.add128(lhs, rhs, -10);
      assertColumnsAreEqual(expectedValid, result.getColumn(0));
    }
  }

  @Test
  void addPrecision38ScaleNeg10() {
    try (
        ColumnVector lhs = makeDec128Column("9191008513307131620269245301.1615457290",
            "-9191008513307131620269245301.1615457290",
            "577694938495380589068894346.7625198736",
            "-7949989536398283250841565918.6123449781",
            "-569260079419403643627836417.1451349695",
            "4268696962649098725873162852.3422176564",
            "948521076935839001259204571.1574829065",
            "-9299778357834801251892834048.0026057082",
            "8127384240098008972235509102.7063990819",
            "-1012433127481465711031073593.0625063701",
            "-3008128675386495592846447084.0906874636");
        ColumnVector rhs = makeDec128Column("9447850332473678680446404122.5624623187",
            "-9447850332473678680446404122.5624623187",
            "-1258508260891400005608241690.1564700995",
            "0E-10",
            "4506903505351346531188531230.8104179784",
            "8289592062844478064245294937.3714242072",
            "475827447078875704758652459.0564660621",
            "960510811873374359477931158.7077642783",
            "7213672086663445017824298126.4525607205",
            "2346189245818456940830953479.5847958897",
            "449885491907950809374133839.5150485453");
        ColumnVector expected = makeDec128Column("18638858845780810300715649423.724008048",
            "-18638858845780810300715649423.724008048",
            "-680813322396019416539347343.393950226",
            "-7949989536398283250841565918.612344978",
            "3937643425931942887560694813.665283009",
            "12558289025493576790118457789.713641864",
            "1424348524014714706017857030.213948969",
            "-8339267545961426892414902889.294841430",
            "15341056326761453990059807229.158959802",
            "1333756118336991229799879886.522289520",
            "-2558243183478544783472313244.575638918");
        ColumnVector expectedValid = ColumnVector.fromBooleans(false, false, false, false, false,
            false, false, false, false, false, false)) {
      Table result = DecimalUtils.add128(lhs, rhs, -9);
      assertColumnsAreEqual(expectedValid, result.getColumn(0));
      assertColumnsAreEqual(expected, result.getColumn(1));
    }
  }

  @Test
  void addPrecision38Scale5() {
    try (
        ColumnVector lhs = makeDec128Column(
            "4.2701861951571908374098848594277520E+39",
            "-9.51477182371612065851896242097995638E+40",
            "-2.0167866914929483784509827485383359E+39",
            "3.09186385410128070998385426348594484E+40",
            "7.1672663199631946247197119155144713E+39",
            "-9.32396355260007858810554960112006290E+40",
            "8.24190234828859904475261796305602287E+40",
            "6.10646349654220618869425418121505315E+40",
            "-5.4790787707639406411507823776332565E+39",
            null);
        ColumnVector rhs = makeDec128Column(
            "-7.4015414116488076297669800353634627E+39",
            "8.26223612055178995785348949126553327E+40",
            "3.27796298399180383738215644697505864E+40",
            "6.23318861108302118457923491160201752E+40",
            "1.2868445730284429449720988121912717E+39",
            "-9.89573762074541324330058371364880604E+40",
            "1.83583924726137822744760302018523424E+40",
            "5.39262612260712860406222466457256229E+40",
            "-1.0688816822936864401341690563696501E+39",
            "-1.0688816822936864401341690563696501E+39");
        ColumnVector expected = makeDec128Column(
            "-3.1313552164916167923570951759357107E+39",
            "-1.25253570316433070066547292971442311E+40",
            "3.07628431484250899953705817212122505E+40",
            "9.32505246518430189456308917508796236E+40",
            "8.4541108929916375696918107277057430E+39",
            "-1.921970117334549183140613331476886894E+41",
            "1.007774159554997727220022098324125711E+41",
            "1.149908961914933479275647884578761544E+41",
            "-6.5479604530576270812849514340029066E+39",
            null);
        ColumnVector expectedValid = ColumnVector.fromBoxedBooleans(false, false, false, false,
            false, false, false, false, false, null)) {
      Table result = DecimalUtils.add128(lhs, rhs, 5);
      assertColumnsAreEqual(expectedValid, result.getColumn(0));
      assertColumnsAreEqual(expected, result.getColumn(1));
    }
  }

  @Test
  void addDifferentScales() {
    try (
        ColumnVector lhs = makeDec128Column(
            "9191008513307131620269245301.1615457290",
            "-9191008513307131620269245301.1615457290",
            "577694938495380589068894346.7625198736",
            "-7949989536398283250841565918.6123449781",
            "-569260079419403643627836417.1451349695",
            "4268696962649098725873162852.3422176564",
            "948521076935839001259204571.1574829065",
            "-9299778357834801251892834048.0026057082",
            "8127384240098008972235509102.7063990819",
            "-1012433127481465711031073593.0625063701");
        ColumnVector rhs = makeDec128Column(
            "451635271134476686911387864.48",
            "-9037370400215680718822505020.06",
            "-200173438757934601210092407.67",
            "3022290197578200820919308997.64",
            "388221337108432989001879408.73",
            "-9119163961520067341639997328.82",
            "7732813484881363300406806463.83",
            "5941454871287785414686091453.79",
            "-357209139972312354271434821.33",
            "-857448828702886587693936536.21");

        ColumnVector expected = makeDec128Column(
            "9642643784441608307180633165.641545729",
            "-18228378913522812339091750321.221545729",
            "377521499737445987858801939.092519874",
            "-4927699338820082429922256920.972344978",
            "-181038742310970654625957008.415134970",
            "-4850466998870968615766834476.477782344",
            "8681334561817202301666011034.987482907",
            "-3358323486547015837206742594.212605708",
            "7770175100125696617964074281.376399082",
            "-1869881956184352298725010129.272506370");
        ColumnVector expectedValid = ColumnVector.fromBoxedBooleans(false, false, false, false,
            false, false, false, false, false, false)) {
      Table result = DecimalUtils.add128(lhs, rhs, -9);
      assertColumnsAreEqual(expectedValid, result.getColumn(0));
      assertColumnsAreEqual(expected, result.getColumn(1));
    }
  }

  @Test
  void mulTestOverflow() {
    try (
        ColumnVector lhs = makeDec128Column("50000000000000000000000000000000000000");
        ColumnVector rhs = makeDec128Column("2");
        ColumnVector expectedValid = ColumnVector.fromBooleans(true)) {
      Table result = DecimalUtils.multiply128(lhs, rhs, 0);
      assertColumnsAreEqual(expectedValid, result.getColumn(0));
    }
  }

  @Test
  void addTestOverflow() {
    try (
        ColumnVector lhs = makeDec128Column("99999999999999999999999999999999999999");
        ColumnVector rhs = makeDec128Column("1");
        ColumnVector expectedValid = ColumnVector.fromBooleans(true)) {
      Table result = DecimalUtils.add128(lhs, rhs, 0);
      assertColumnsAreEqual(expectedValid, result.getColumn(0));
    }
  }

  @Test
  void subTestOverflow() {
    try (
        ColumnVector lhs = makeDec128Column("-99999999999999999999999999999999999999");
        ColumnVector rhs = makeDec128Column("1");
        ColumnVector expectedValid = ColumnVector.fromBooleans(true)) {
      Table result = DecimalUtils.subtract128(lhs, rhs, 0);
      assertColumnsAreEqual(expectedValid, result.getColumn(0));
    }
  }

  @Test
  void subDifferentScales() {
    try (
        ColumnVector lhs = makeDec128Column(
            "9191008513307131620269245301.1615457290",
            "-9191008513307131620269245301.1615457290",
            "577694938495380589068894346.7625198736",
            "-7949989536398283250841565918.6123449781",
            "-569260079419403643627836417.1451349695",
            "4268696962649098725873162852.3422176564",
            "948521076935839001259204571.1574829065",
            "-9299778357834801251892834048.0026057082",
            "8127384240098008972235509102.7063990819",
            "-1012433127481465711031073593.0625063701");
        ColumnVector rhs = makeDec128Column(
            "451635271134476686911387864.48",
            "-9037370400215680718822505020.06",
            "-200173438757934601210092407.67",
            "3022290197578200820919308997.64",
            "388221337108432989001879408.73",
            "-9119163961520067341639997328.82",
            "7732813484881363300406806463.83",
            "5941454871287785414686091453.79",
            "-357209139972312354271434821.33",
            "-857448828702886587693936536.21");

        ColumnVector expected = makeDec128Column(
            "8739373242172654933357857436.681545729",
            "-153638113091450901446740281.101545729",
            "777868377253315190278986754.432519874",
            "-10972279733976484071760874916.252344978",
            "-957481416527836632629715825.875134970",
            "13387860924169166067513160181.162217656",
            "-6784292407945524299147601892.672517094",
            "-15241233229122586666578925501.792605708",
            "8484593380070321326506943924.036399082",
            "-154984298778579123337137056.852506370");
        ColumnVector expectedValid = ColumnVector.fromBoxedBooleans(false, false, false, false,
            false, false, false, false, false, false)) {
      Table result = DecimalUtils.subtract128(lhs, rhs, -9);
      assertColumnsAreEqual(expectedValid, result.getColumn(0));
      assertColumnsAreEqual(expected, result.getColumn(1));
    }
  }
}
