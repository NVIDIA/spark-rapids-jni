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

import com.nvidia.spark.rapids.jni.Hash;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVector.*;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.*;

public class HashTest {
// IEEE 754 NaN values
  static final float POSITIVE_FLOAT_NAN_LOWER_RANGE = Float.intBitsToFloat(0x7f800001);
  static final float POSITIVE_FLOAT_NAN_UPPER_RANGE = Float.intBitsToFloat(0x7fffffff);
  static final float NEGATIVE_FLOAT_NAN_LOWER_RANGE = Float.intBitsToFloat(0xff800001);
  static final float NEGATIVE_FLOAT_NAN_UPPER_RANGE = Float.intBitsToFloat(0xffffffff);

  static final double POSITIVE_DOUBLE_NAN_LOWER_RANGE = Double.longBitsToDouble(0x7ff0000000000001L);
  static final double POSITIVE_DOUBLE_NAN_UPPER_RANGE = Double.longBitsToDouble(0x7fffffffffffffffL);
  static final double NEGATIVE_DOUBLE_NAN_LOWER_RANGE = Double.longBitsToDouble(0xfff0000000000001L);
  static final double NEGATIVE_DOUBLE_NAN_UPPER_RANGE = Double.longBitsToDouble(0xffffffffffffffffL);

  @Test
  void testSpark32BitMurmur3HashStrings() {
    try (ColumnVector v0 = ColumnVector.fromStrings(
           "a", "B\nc",  "dE\"\u0100\t\u0101 \ud720\ud721\\Fg2\'",
           "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
           "in the MD5 hash function. This string needed to be longer.A 60 character string to " +
           "test MD5's message padding algorithm",
           "hiJ\ud720\ud721\ud720\ud721", null);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v0});
         ColumnVector expected = ColumnVector.fromBoxedInts(1485273170, 1709559900, 1423943036, 176121990, 1199621434, 42)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashInts() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(0, 100, null, null, Integer.MIN_VALUE, null);
         ColumnVector v1 = ColumnVector.fromBoxedInts(0, null, -100, null, null, Integer.MAX_VALUE);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v0, v1});
         ColumnVector expected = ColumnVector.fromBoxedInts(59727262, 751823303, -1080202046, 42, 723455942, 133916647)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashDoubles() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(
          0.0, null, 100.0, -100.0, Double.MIN_NORMAL, Double.MAX_VALUE,
          POSITIVE_DOUBLE_NAN_UPPER_RANGE, POSITIVE_DOUBLE_NAN_LOWER_RANGE,
          NEGATIVE_DOUBLE_NAN_UPPER_RANGE, NEGATIVE_DOUBLE_NAN_LOWER_RANGE,
          Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
         ColumnVector result = Hash.murmurHash32(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(1669671676, 0, -544903190, -1831674681, 150502665, 474144502, 1428788237, 1428788237, 1428788237, 1428788237, 420913893, 1915664072)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashTimestamps() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/rapidsai/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.timestampMicroSecondsFromBoxedLongs(
        0L, null, 100L, -100L, 0x123456789abcdefL, null, -0x123456789abcdefL);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(-1670924195, 42, 1114849490, 904948192, 657182333, 42, -57193045)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashDecimal64() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/rapidsai/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.decimalFromLongs(-7,
        0L, 100L, -100L, 0x123456789abcdefL, -0x123456789abcdefL);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(-1670924195, 1114849490, 904948192, 657182333, -57193045)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashDecimal32() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/rapidsai/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.decimalFromInts(-3,
        0, 100, -100, 0x12345678, -0x12345678);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(-1670924195, 1114849490, 904948192, -958054811, -1447702630)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashDates() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/rapidsai/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.timestampDaysFromBoxedInts(
        0, null, 100, -100, 0x12345678, null, -0x12345678);
         ColumnVector result = Hash.murmurHash32(42, new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(933211791, 42, 751823303, -1080202046, -1721170160, 42, 1852996993)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashFloats() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, Float.MIN_NORMAL, Float.MAX_VALUE, null,
          POSITIVE_FLOAT_NAN_LOWER_RANGE, POSITIVE_FLOAT_NAN_UPPER_RANGE,
          NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE,
          Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
         ColumnVector result = Hash.murmurHash32(411, new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedInts(-235179434, 1812056886, 2028471189, 1775092689, -1531511762, 411, -1053523253, -1053523253, -1053523253, -1053523253, -1526256646, 930080402)){
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashBools() {
    try (ColumnVector v0 = ColumnVector.fromBoxedBooleans(null, true, false, true, null, false);
         ColumnVector v1 = ColumnVector.fromBoxedBooleans(null, true, false, null, false, true);
         ColumnVector result = Hash.murmurHash32(0, new ColumnVector[]{v0, v1});
         ColumnVector expected = ColumnVector.fromBoxedInts(0, -1589400010, -239939054, -68075478, 593689054, -1194558265)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashMixed() {
    try (ColumnVector strings = ColumnVector.fromStrings(
          "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
          "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
          "in the MD5 hash function. This string needed to be longer.",
          null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
          0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(true, false, null, false, true, null);
         ColumnVector result = Hash.murmurHash32(1868, new ColumnVector[]{strings, integers, doubles, floats, bools});
         ColumnVector expected = ColumnVector.fromBoxedInts(1936985022, 720652989, 339312041, 1400354989, 769988643, 1868)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashStruct() {
    try (ColumnVector strings = ColumnVector.fromStrings(
        "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
        "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
            "in the MD5 hash function. This string needed to be longer.",
        null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
             0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
             0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(true, false, null, false, true, null);
         ColumnView structs = ColumnView.makeStructView(strings, integers, doubles, floats, bools);
         ColumnVector result = Hash.murmurHash32(1868, new ColumnView[]{structs});
         ColumnVector expected = Hash.murmurHash32(1868, new ColumnVector[]{strings, integers, doubles, floats, bools})) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashNestedStruct() {
    try (ColumnVector strings = ColumnVector.fromStrings(
        "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
        "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
            "in the MD5 hash function. This string needed to be longer.",
        null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
             0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
             0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(true, false, null, false, true, null);
         ColumnView structs1 = ColumnView.makeStructView(strings, integers);
         ColumnView structs2 = ColumnView.makeStructView(structs1, doubles);
         ColumnView structs3 = ColumnView.makeStructView(bools);
         ColumnView structs = ColumnView.makeStructView(structs2, floats, structs3);
         ColumnVector expected = Hash.murmurHash32(1868, new ColumnVector[]{strings, integers, doubles, floats, bools});
         ColumnVector result = Hash.murmurHash32(1868, new ColumnView[]{structs})) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testSpark32BitMurmur3HashListsAndNestedLists() {
    try (ColumnVector stringListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.STRING)),
             Arrays.asList(null, "a"),
             Arrays.asList("B\n", ""),
             Arrays.asList("dE\"\u0100\t\u0101", " \ud720\ud721"),
             Collections.singletonList("A very long (greater than 128 bytes/char string) to test a multi" +
             " hash-step data point in the Murmur3 hash function. This string needed to be longer."),
             Collections.singletonList(""),
             null);
         ColumnVector strings1 = ColumnVector.fromStrings(
             "a", "B\n", "dE\"\u0100\t\u0101",
             "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
             "in the Murmur3 hash function. This string needed to be longer.", null, null);
         ColumnVector strings2 = ColumnVector.fromStrings(
             null, "", " \ud720\ud721", null, "", null);
         ColumnView stringStruct = ColumnView.makeStructView(strings1, strings2);
         ColumnVector stringExpected = Hash.murmurHash32(1868, new ColumnView[]{stringStruct});
         ColumnVector stringResult = Hash.murmurHash32(1868, new ColumnView[]{stringListCV});
         ColumnVector intListCV = ColumnVector.fromLists(
             new ListType(true, new BasicType(true, DType.INT32)),
             null,
             Arrays.asList(0, -2, 3),
             Collections.singletonList(Integer.MAX_VALUE),
             Arrays.asList(5, -6, null),
             Collections.singletonList(Integer.MIN_VALUE),
             null);
         ColumnVector integers1 = ColumnVector.fromBoxedInts(null, 0, null, 5, Integer.MIN_VALUE, null);
         ColumnVector integers2 = ColumnVector.fromBoxedInts(null, -2, Integer.MAX_VALUE, null, null, null);
         ColumnVector integers3 = ColumnVector.fromBoxedInts(null, 3, null, -6, null, null);
         ColumnVector intExpected =
             Hash.murmurHash32(1868, new ColumnVector[]{integers1, integers2, integers3});
         ColumnVector intResult = Hash.murmurHash32(1868, new ColumnVector[]{intListCV});
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
          0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnView structCV = ColumnView.makeStructView(intListCV, stringListCV, doubles, floats);
         ColumnVector nestedExpected =
             Hash.murmurHash32(1868, new ColumnView[]{intListCV, strings1, strings2, doubles, floats});
         ColumnVector nestedResult =
             Hash.murmurHash32(1868, new ColumnView[]{structCV})) {
      assertColumnsAreEqual(stringExpected, stringResult);
      assertColumnsAreEqual(intExpected, intResult);
      assertColumnsAreEqual(nestedExpected, nestedResult);
    }
  }

  @Test
  void testXXHash64Strings() {
    try (ColumnVector v0 = ColumnVector.fromStrings(
           "a", "B\nc",  "dE\"\u0100\t\u0101 \ud720\ud721\\Fg2\'",
           "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
           "in the MD5 hash function. This string needed to be longer.A 60 character string to " +
           "test MD5's message padding algorithm",
           "hiJ\ud720\ud721\ud720\ud721", null);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v0});
         ColumnVector expected = ColumnVector.fromBoxedLongs(-8582455328737087284L, 2221214721321197934L, 5798966295358745941L, -4834097201550955483L, -3782648123388245694L, Hash.DEFAULT_XXHASH64_SEED)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64Ints() {
    try (ColumnVector v0 = ColumnVector.fromBoxedInts(0, 100, null, null, Integer.MIN_VALUE, null);
         ColumnVector v1 = ColumnVector.fromBoxedInts(0, null, -100, null, null, Integer.MAX_VALUE);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v0, v1});
         ColumnVector expected = ColumnVector.fromBoxedLongs(1151812168208346021L, -7987742665087449293L, 8990748234399402673L, Hash.DEFAULT_XXHASH64_SEED, 2073849959933241805L, 1508894993788531228L)) {
      assertColumnsAreEqual(expected, result);
    }
  }
  
  @Test
  void testXXHash64Doubles() {
    try (ColumnVector v = ColumnVector.fromBoxedDoubles(
          0.0, null, 100.0, -100.0, Double.MIN_NORMAL, Double.MAX_VALUE,
          POSITIVE_DOUBLE_NAN_UPPER_RANGE, POSITIVE_DOUBLE_NAN_LOWER_RANGE,
          NEGATIVE_DOUBLE_NAN_UPPER_RANGE, NEGATIVE_DOUBLE_NAN_LOWER_RANGE,
          Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(-5252525462095825812L, Hash.DEFAULT_XXHASH64_SEED, -7996023612001835843L, 5695175288042369293L, 6181148431538304986L, -4222314252576420879L, -3127944061524951246L, -3127944061524951246L, -3127944061524951246L, -3127944061524951246L, 5810986238603807492L, 5326262080505358431L)) {
      assertColumnsAreEqual(expected, result);
    }
  }
    
  @Test
  void testXXHash64Timestamps() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/rapidsai/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.timestampMicroSecondsFromBoxedLongs(
        0L, null, 100L, -100L, 0x123456789abcdefL, null, -0x123456789abcdefL);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(-5252525462095825812L, Hash.DEFAULT_XXHASH64_SEED, 8713583529807266080L, 5675770457807661948L, 1941233597257011502L, Hash.DEFAULT_XXHASH64_SEED, -1318946533059658749L)) {
      assertColumnsAreEqual(expected, result);
    }
  }
  
  @Test
  void testXXHash64Decimal64() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/rapidsai/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.decimalFromLongs(-7,
        0L, 100L, -100L, 0x123456789abcdefL, -0x123456789abcdefL);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(-5252525462095825812L, 8713583529807266080L, 5675770457807661948L, 1941233597257011502L, -1318946533059658749L)) {
      assertColumnsAreEqual(expected, result);
    }
  }
    
  @Test
  void testXXHash64Decimal32() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/rapidsai/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.decimalFromInts(-3,
        0, 100, -100, 0x12345678, -0x12345678);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(-5252525462095825812L, 8713583529807266080L, 5675770457807661948L, -7728554078125612835L, 3142315292375031143L)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64Dates() {
    // The hash values were derived from Apache Spark in a manner similar to the one documented at
    // https://github.com/rapidsai/cudf/blob/aa7ca46dcd9e/cpp/tests/hashing/hash_test.cpp#L281-L307
    try (ColumnVector v = ColumnVector.timestampDaysFromBoxedInts(
        0, null, 100, -100, 0x12345678, null, -0x12345678);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(3614696996920510707L, Hash.DEFAULT_XXHASH64_SEED, -7987742665087449293L, 8990748234399402673L, 6954428822481665164L, Hash.DEFAULT_XXHASH64_SEED, -4294222333805341278L)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64Floats() {
    try (ColumnVector v = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, Float.MIN_NORMAL, Float.MAX_VALUE, null,
          POSITIVE_FLOAT_NAN_LOWER_RANGE, POSITIVE_FLOAT_NAN_UPPER_RANGE,
          NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE,
          Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v});
         ColumnVector expected = ColumnVector.fromBoxedLongs(3614696996920510707L, -8232251799677946044L, -6625719127870404449L, -6699704595004115126L, -1065250890878313112L, Hash.DEFAULT_XXHASH64_SEED, 2692338816207849720L, 2692338816207849720L, 2692338816207849720L, 2692338816207849720L, -5940311692336719973L, -7580553461823983095L)){
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testXXHash64Bools() {
    try (ColumnVector v0 = ColumnVector.fromBoxedBooleans(null, true, false, true, null, false);
         ColumnVector v1 = ColumnVector.fromBoxedBooleans(null, true, false, null, false, true);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{v0, v1});
         ColumnVector expected = ColumnVector.fromBoxedLongs(Hash.DEFAULT_XXHASH64_SEED, 9083826852238114423L, 1151812168208346021L, -6698625589789238999L, 3614696996920510707L, 7945966957015589024L)) {
      assertColumnsAreEqual(expected, result);
    }
  }
  
  @Test
  void testXXHash64Mixed() {
    try (ColumnVector strings = ColumnVector.fromStrings(
          "a", "B\n", "dE\"\u0100\t\u0101 \ud720\ud721",
          "A very long (greater than 128 bytes/char string) to test a multi hash-step data point " +
          "in the MD5 hash function. This string needed to be longer.",
          null, null);
         ColumnVector integers = ColumnVector.fromBoxedInts(0, 100, -100, Integer.MIN_VALUE, Integer.MAX_VALUE, null);
         ColumnVector doubles = ColumnVector.fromBoxedDoubles(
          0.0, 100.0, -100.0, POSITIVE_DOUBLE_NAN_LOWER_RANGE, POSITIVE_DOUBLE_NAN_UPPER_RANGE, null);
         ColumnVector floats = ColumnVector.fromBoxedFloats(
          0f, 100f, -100f, NEGATIVE_FLOAT_NAN_LOWER_RANGE, NEGATIVE_FLOAT_NAN_UPPER_RANGE, null);
         ColumnVector bools = ColumnVector.fromBoxedBooleans(true, false, null, false, true, null);
         ColumnVector result = Hash.xxhash64(new ColumnVector[]{strings, integers, doubles, floats, bools});
         ColumnVector expected = ColumnVector.fromBoxedLongs(7451748878409563026L, 6024043102550151964L, 3380664624738534402L, 8444697026100086329L, -5888679192448042852L, Hash.DEFAULT_XXHASH64_SEED)) {
      assertColumnsAreEqual(expected, result);
    }
  }
}
