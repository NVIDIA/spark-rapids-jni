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

#include <cast_string.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/io/parquet.hpp>

#include <limits>
#include <rmm/device_uvector.hpp>

using namespace cudf;

template <typename T>
struct StringToIntegerTests : public test::BaseFixture {
};

struct StringToDecimalTests : public test::BaseFixture {};

TYPED_TEST_SUITE(StringToIntegerTests, cudf::test::IntegralTypesNotBool);

TYPED_TEST(StringToIntegerTests, Simple)
{
  auto const strings = test::strings_column_wrapper{"1", "0", "42"};
  strings_column_view scv{strings};

  auto const result = spark_rapids_jni::string_to_integer(
    data_type{type_to_id<TypeParam>()}, scv, false, rmm::cuda_stream_default);

  test::fixed_width_column_wrapper<TypeParam> expected({1, 0, 42}, {1, 1, 1});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TYPED_TEST(StringToIntegerTests, Ansi)
{
  auto const strings = test::strings_column_wrapper(
    {"",       "null",  "+1",      "-0",           "4.2",
     "asdf",   "98fe",  "  00012", ".--e-37602.n", "\r\r\t\n11.12380",
     "-.2",    ".3",    ".",       "+1.2",         "\n123\n456\n",
     "1 2",    "123",   "",        "1. 2",         "+    7.6",
     "  12  ", "7.6.2", "15  ",    "7  2  ",       " 8.2  ",
     "3..14",  "c0",    "\r\r",    "    ",         "+\n"},
    {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  strings_column_view scv{strings};

  constexpr bool is_signed_type = std::is_signed_v<TypeParam>;

  try {
    spark_rapids_jni::string_to_integer(
      data_type{type_to_id<TypeParam>()}, scv, true, rmm::cuda_stream_default);
  } catch (spark_rapids_jni::cast_error& e) {
    auto const row = [&]() {
      if constexpr (is_signed_type) {
        return 4;
      } else {
        return 2;
      }
    }();
    auto const first_error_string = [&]() {
      if constexpr (is_signed_type) {
        return "4.2";
      } else {
        return "+1";
      }
    }();

    EXPECT_EQ(e.get_row_number(), row);
    EXPECT_STREQ(e.get_string_with_error(), first_error_string);
  }

  auto const result = spark_rapids_jni::string_to_integer(
    data_type{type_to_id<TypeParam>()}, scv, false, rmm::cuda_stream_default);

  test::fixed_width_column_wrapper<TypeParam> expected = []() {
    if constexpr (is_signed_type) {
      return test::fixed_width_column_wrapper<TypeParam>(
        {0, 0,   1, 0, 4, 0,  0, 12, 0, 11, 0, 0, 0, 1, 0,
         0, 123, 0, 0, 0, 12, 0, 15, 0, 8,  0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0});
    } else {
      return test::fixed_width_column_wrapper<TypeParam>(
        {0, 0,   0, 0, 4, 0,  0, 12, 0, 11, 0, 0, 0, 0, 0,
         0, 123, 0, 0, 0, 12, 0, 15, 0, 8,  0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0});
    }
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TYPED_TEST(StringToIntegerTests, Overflow)
{
  auto const strings = test::strings_column_wrapper({"127",
                                                     "128",
                                                     "-128",
                                                     "-129",
                                                     "255",
                                                     "256",
                                                     "32767",
                                                     "32768",
                                                     "-32768",
                                                     "-32769",
                                                     "65525",
                                                     "65536",
                                                     "2147483647",
                                                     "2147483648",
                                                     "-2147483648",
                                                     "-2147483649",
                                                     "4294967295",
                                                     "4294967296",
                                                     "-9223372036854775808",
                                                     "-9223372036854775809",
                                                     "9223372036854775807",
                                                     "9223372036854775808",
                                                     "18446744073709551615",
                                                     "18446744073709551616"});
  strings_column_view scv{strings};

  auto result = spark_rapids_jni::string_to_integer(
    data_type{type_to_id<TypeParam>()}, scv, false, rmm::cuda_stream_default);

  auto const expected = [&]() {
    if constexpr (std::is_same_v<TypeParam, int8_t>) {
      return test::fixed_width_column_wrapper<int8_t>(
        {127, 0, -128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, uint8_t>) {
      return test::fixed_width_column_wrapper<uint8_t>(
        {127, 128, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, int16_t>) {
      return test::fixed_width_column_wrapper<int16_t>(
        {127, 128, -128, -129, 255, 256, 32767, 0, -32768, 0, 0, 0,
         0,   0,   0,    0,    0,   0,   0,     0, 0,      0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, uint16_t>) {
      return test::fixed_width_column_wrapper<uint16_t>(
        {127, 128, 0, 0, 255, 256, 32767, 32768, 0, 0, 65525, 0,
         0,   0,   0, 0, 0,   0,   0,     0,     0, 0, 0,     0},
        {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, int32_t>) {
      auto ret = test::fixed_width_column_wrapper<int32_t>(
        {127,   128,   -128,       -129,   255,
         256,   32767, 32768,      -32768, -32769,
         65525, 65536, 2147483647, 0,      std::numeric_limits<int32_t>::min(),
         0,     0,     0,          0,      0,
         0,     0,     0,          0},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
      return ret;
    } else if constexpr (std::is_same_v<TypeParam, uint32_t>) {
      return test::fixed_width_column_wrapper<uint32_t>(
        {127u,        128u, 0u,     0u,     255u,        256u,        32767u, 32768u,
         0u,          0u,   65525u, 65536u, 2147483647u, 2147483648u, 0u,     0u,
         4294967295u, 0u,   0u,     0u,     0u,          0u,          0u,     0u},
        {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, int64_t>) {
      return test::fixed_width_column_wrapper<int64_t>(
        {127L,
         128L,
         -128L,
         -129L,
         255L,
         256L,
         32767L,
         32768L,
         -32768L,
         -32769L,
         65525L,
         65536L,
         2147483647L,
         2147483648L,
         -2147483648L,
         -2147483649L,
         4294967295L,
         4294967296L,
         std::numeric_limits<int64_t>::min(),
         0L,
         9223372036854775807L,
         0L,
         0L,
         0L},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0});
    } else if constexpr (std::is_same_v<TypeParam, uint64_t>) {
      return test::fixed_width_column_wrapper<uint64_t>(
        {127UL,
         128UL,
         0UL,
         0UL,
         255UL,
         256UL,
         32767UL,
         32768UL,
         0UL,
         0UL,
         65525UL,
         65536UL,
         2147483647UL,
         2147483648UL,
         0UL,
         0UL,
         4294967295UL,
         4294967296UL,
         0UL,
         0UL,
         9223372036854775807UL,
         9223372036854775808UL,
         18446744073709551615UL,
         0UL},
        {1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0});
    }
  }();

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, Simple)
{
  auto const strings = test::strings_column_wrapper({"1", "0", "-1"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(1, 0, scv, false, rmm::cuda_stream_default);

  test::fixed_point_column_wrapper<int32_t> expected({1, 0, -1}, {1, 1, 1}, numeric::scale_type{0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, OverPrecise)
{
  auto const strings = test::strings_column_wrapper({"123456", "999999", "-123456", "-999999"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(5, 0, scv, false, rmm::cuda_stream_default);

  test::fixed_point_column_wrapper<int32_t> expected(
    {0, 0, 0, 0}, {0, 0, 0, 0}, numeric::scale_type{0});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, Rounding)
{
  auto const strings = test::strings_column_wrapper({"1.23456", "9.99999", "-1.23456", "-9.99999"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(5, -4, scv, false, rmm::cuda_stream_default);

  test::fixed_point_column_wrapper<int32_t> expected(
    {12346, 100000, -12346, -100000}, {1, 0, 1, 0}, numeric::scale_type{-4});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, DecimalValues)
{
  auto const strings =
    test::strings_column_wrapper({"1.234", "0.12345", "-1.034", "-0.001234567890123456"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(6, -5, scv, false, rmm::cuda_stream_default);

  test::fixed_point_column_wrapper<int32_t> expected(
    {123400, 12345, -103400, -123}, {1, 1, 1, 1}, numeric::scale_type{-5});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, ExponentalNotation)
{
  auto const strings =
    test::strings_column_wrapper({"1.234e-1", "0.12345e1", "-1.034e-2", "-0.001234567890123456e2"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(6, -5, scv, false, rmm::cuda_stream_default);

  test::fixed_point_column_wrapper<int32_t> expected(
    {12340, 123450, -1034, -12346}, {1, 1, 1, 1}, numeric::scale_type{-5});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, PositiveScale)
{
  auto const strings =
    test::strings_column_wrapper({"1234e-1", "12345e1", "-1234.5678", "-0.001234567890123456e6"});
  strings_column_view scv{strings};

  auto const result =
    spark_rapids_jni::string_to_decimal(6, 2, scv, false, rmm::cuda_stream_default);

  test::fixed_point_column_wrapper<int32_t> expected(
    {100, 123500, -1200, -1200}, {1, 1, 1, 1}, numeric::scale_type{2});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
}

TEST_F(StringToDecimalTests, Edges)
{
  {
    auto const str = test::strings_column_wrapper({"8.483315330475049E-4"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(15, -1, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int64_t> expected({0}, {1}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"8.483315330475049E-2"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(15, -1, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int64_t> expected({1}, {1}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"-1.0E14"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(15, -1, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int64_t> expected({0}, {0}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"-1.0E14"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(16, -1, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int64_t> expected(
      {-1'000'000'000'000'000}, {1}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"8.575859E8"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(15, -1, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int64_t> expected({8575859000}, {1}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"10.0"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(3, -1, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int32_t> expected({100}, {1}, numeric::scale_type{-1});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"1.7142857343"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(9, -8, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int32_t> expected({171428573}, {1}, numeric::scale_type{-8});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"1.71428573437482136712623"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(9, -8, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int32_t> expected({171428573}, {1}, numeric::scale_type{-8});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);

    auto const fail_result =
      spark_rapids_jni::string_to_decimal(9, -9, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int32_t> null_col({0}, {0}, numeric::scale_type{-9});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(fail_result->view(), null_col);
  }

  {
    auto const str = test::strings_column_wrapper({"12.345678901"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(9, -8, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int32_t> expected({0}, {0}, numeric::scale_type{-8});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"0.12345678901"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(6, -6, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int32_t> expected({123457}, {1}, numeric::scale_type{-6});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"1.2345678901"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(6, -6, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int32_t> expected({0}, {0}, numeric::scale_type{-6});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }

  {
    auto const str = test::strings_column_wrapper({"NaN", "inf", "-inf", "0"});
    strings_column_view scv{str};

    auto const result =
      spark_rapids_jni::string_to_decimal(6, 0, scv, false, rmm::cuda_stream_default);
    test::fixed_point_column_wrapper<int32_t> expected({0, 0, 0, 0}, {0, 0, 0, 1}, numeric::scale_type{0});

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->view(), expected);
  }
}
