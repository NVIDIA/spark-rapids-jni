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

#include "cast_string.hpp"
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"
#include "jni_utils.hpp"

constexpr char const* JNI_CAST_ERROR_CLASS = "com/nvidia/spark/rapids/jni/CastException";

#define CATCH_CAST_EXCEPTION(env, ret_val)                                                \
  catch (const spark_rapids_jni::cast_error& e)                                           \
  {                                                                                       \
    if (env->ExceptionOccurred()) { return ret_val; }                                     \
    jclass ex_class = env->FindClass(JNI_CAST_ERROR_CLASS);                               \
    if (ex_class != NULL) {                                                               \
      jmethodID ctor_id = env->GetMethodID(ex_class, "<init>", "(Ljava/lang/String;I)V"); \
      if (ctor_id != NULL) {                                                              \
        std::string n_msg = e.get_string_with_error();                                    \
        jstring j_msg     = env->NewStringUTF(n_msg.c_str());                             \
        if (j_msg != NULL) {                                                              \
          jint e_row         = static_cast<jint>(e.get_row_number());                     \
          jobject cuda_error = env->NewObject(ex_class, ctor_id, j_msg, e_row);           \
          if (cuda_error != NULL) { env->Throw((jthrowable)cuda_error); }                 \
        }                                                                                 \
      }                                                                                   \
    }                                                                                     \
    return ret_val;                                                                       \
  }                                                                                       \
  CATCH_STD(env, 0);

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CastStrings_toInteger(
  JNIEnv* env, jclass, jlong input_column, jboolean ansi_enabled, jboolean strip, jint j_dtype)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);

    cudf::strings_column_view scv{*reinterpret_cast<cudf::column_view const*>(input_column)};
    return cudf::jni::release_as_jlong(spark_rapids_jni::string_to_integer(
      cudf::jni::make_data_type(j_dtype, 0), scv, ansi_enabled, strip, cudf::get_default_stream()));
  }
  CATCH_CAST_EXCEPTION(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_CastStrings_toDecimal(JNIEnv* env,
                                                       jclass,
                                                       jlong input_column,
                                                       jboolean ansi_enabled,
                                                       jboolean strip,
                                                       jint precision,
                                                       jint scale)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);

    cudf::strings_column_view scv{*reinterpret_cast<cudf::column_view const*>(input_column)};
    return cudf::jni::release_as_jlong(spark_rapids_jni::string_to_decimal(
      precision, scale, scv, ansi_enabled, strip, cudf::get_default_stream()));
  }
  CATCH_CAST_EXCEPTION(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CastStrings_toFloat(
  JNIEnv* env, jclass, jlong input_column, jboolean ansi_enabled, jint j_dtype)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);

    cudf::strings_column_view scv{*reinterpret_cast<cudf::column_view const*>(input_column)};
    return cudf::jni::release_as_jlong(spark_rapids_jni::string_to_float(
      cudf::jni::make_data_type(j_dtype, 0), scv, ansi_enabled, cudf::get_default_stream()));
  }
  CATCH_CAST_EXCEPTION(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CastStrings_fromFloat(JNIEnv* env,
                                                                               jclass,
                                                                               jlong input_column)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);

    auto const& cv = *reinterpret_cast<cudf::column_view const*>(input_column);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::float_to_string(cv, cudf::get_default_stream()));
  }
  CATCH_CAST_EXCEPTION(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CastStrings_fromDecimal(JNIEnv* env,
                                                                                 jclass,
                                                                                 jlong input_column)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);

    auto const& cv = *reinterpret_cast<cudf::column_view const*>(input_column);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::decimal_to_non_ansi_string(cv, cudf::get_default_stream()));
  }
  CATCH_CAST_EXCEPTION(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CastStrings_toIntegersWithBase(
  JNIEnv* env, jclass, jlong input_column, jint base, jboolean ansi_enabled, jint j_dtype)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  using namespace cudf;
  try {
    if (base != 10 && base != 16) {
      auto const error_msg = "Bases supported 10, 16; Actual: " + std::to_string(base);
      throw spark_rapids_jni::cast_error(0, error_msg);
    }

    jni::auto_set_device(env);
    auto const zero_scalar   = numeric_scalar<uint64_t>(0);
    auto const res_data_type = jni::make_data_type(j_dtype, 0);
    auto const input_view{*reinterpret_cast<column_view const*>(input_column)};
    auto const validity_regex_str = [&] {
      switch (base) {
        case 10: return R"(^\s*(-?[0-9]+).*)"; break;
        case 16: return R"(^\s*(-?[0-9a-fA-F]+).*)"; break;
        default: throw spark_rapids_jni::cast_error(0, "INFEASIBLE"); break;
      }
    }();

    auto const validity_regex = strings::regex_program::create(validity_regex_str);
    auto const valid_rows     = strings::matches_re(input_view, *validity_regex);
    auto const int_col        = [&] {
      auto const prepped_table = strings::extract(input_view, *validity_regex);
      const strings_column_view prepped_view{prepped_table->get_column(0)};
      switch (base) {
        case 10: {
          return strings::to_integers(prepped_view, res_data_type);
        } break;
        case 16: {
          auto const is_negative = strings::starts_with(prepped_view, string_scalar("-"));
          auto const pos_vals    = strings::hex_to_integers(prepped_view, res_data_type);
          auto neg_vals =
            binary_operation(zero_scalar, *pos_vals, binary_operator::SUB, res_data_type);
          return copy_if_else(*neg_vals, *pos_vals, *is_negative);
        }
        default: {
          throw spark_rapids_jni::cast_error(0, "INFEASIBLE");
          break;
        }
      }
    }();

    auto unmatched_implies_zero = copy_if_else(*int_col, zero_scalar, *valid_rows);

    // output nulls: original + all rows matching \s*
    auto const space_only_regex = strings::regex_program::create(R"(^\s*$)");
    auto const new_mask         = [&] {
      auto const extra_null_rows = strings::matches_re(input_view, *space_only_regex);
      auto extra_mask            = unary_operation(*extra_null_rows, unary_operator::NOT);
      if (input_view.null_count() > 0) {
        return binary_operation(*mask_to_bools(input_view.null_mask(), 0, input_view.size()),
                                *extra_mask,
                                binary_operator::BITWISE_AND,
                                data_type(type_id::BOOL8));
      } else {
        return extra_mask;
      }
    }();

    auto const [null_mask, null_count] = bools_to_mask(*new_mask);
    unmatched_implies_zero->set_null_mask(*null_mask, null_count);
    return jni::release_as_jlong(unmatched_implies_zero);
  }
  CATCH_CAST_EXCEPTION(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CastStrings_fromIntegersWithBase(
  JNIEnv* env, jclass, jlong input_column, jint base)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  using namespace cudf;
  try {
    jni::auto_set_device(env);
    auto input_view{*reinterpret_cast<column_view const*>(input_column)};
    auto result = [&] {
      switch (base) {
        case 10: {
          return strings::from_integers(input_view);
        } break;
        case 16: {
          auto pre_res                = strings::integers_to_hex(input_view);
          auto const regex            = strings::regex_program::create("^0?([0-9a-fA-F]+)$");
          auto const wo_leading_zeros = strings::extract(strings_column_view(*pre_res), *regex);
          return std::move(wo_leading_zeros->release()[0]);
        }
        default: {
          auto const error_msg = "Bases supported 10, 16; Actual: " + std::to_string(base);
          throw spark_rapids_jni::cast_error(0, error_msg);
        }
      }
    }();
    return jni::release_as_jlong(result);
  }
  CATCH_CAST_EXCEPTION(env, 0);
}
}
