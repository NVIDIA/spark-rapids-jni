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

#include "cudf_jni_apis.hpp"
#include "jni_compiled_expr.hpp"
#include "mixed_sort_merge_join.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

namespace {

/**
 * @brief Convert pair of device vectors to Java long array
 * Returns a 5-element array: [size_in_bytes, left_ptr, left_handle, right_ptr, right_handle]
 */
jlongArray gather_maps_to_java(
  JNIEnv* env,
  std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>> gather_maps)
{
  // Both gather maps must have the same size for paired results
  CUDF_EXPECTS(gather_maps.first.size() == gather_maps.second.size(),
               "Left and right gather maps must have the same size");

  // Release the underlying device buffers to Java
  auto left_map_buffer  = std::make_unique<rmm::device_buffer>(gather_maps.first.release());
  auto right_map_buffer = std::make_unique<rmm::device_buffer>(gather_maps.second.release());

  cudf::jni::native_jlongArray result(env, 5);
  // Return size in bytes (as expected by DeviceMemoryBuffer.fromRmm)
  result[0] = static_cast<jlong>(left_map_buffer->size());
  result[1] = cudf::jni::ptr_as_jlong(left_map_buffer->data());
  result[2] = cudf::jni::release_as_jlong(left_map_buffer);
  result[3] = cudf::jni::ptr_as_jlong(right_map_buffer->data());
  result[4] = cudf::jni::release_as_jlong(right_map_buffer);
  return result.get_jArray();
}

/**
 * @brief Convert device vector to Java long array (single gather map)
 * Returns a 3-element array: [size_in_bytes, device_ptr, rmm_handle]
 */
jlongArray gather_single_map_to_java(JNIEnv* env, rmm::device_uvector<cudf::size_type> gather_map)
{
  cudf::jni::native_jlongArray result(env, 3);
  result[0]              = static_cast<jlong>(gather_map.size() * sizeof(cudf::size_type));
  auto gather_map_buffer = std::make_unique<rmm::device_buffer>(gather_map.release());
  result[1]              = cudf::jni::ptr_as_jlong(gather_map_buffer->data());
  result[2]              = cudf::jni::release_as_jlong(std::move(gather_map_buffer));
  return result.get_jArray();
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_MixedSortMergeJoin_mixedSortMergeInnerJoin(
  JNIEnv* env,
  jclass,
  jlong j_left_equality,
  jlong j_right_equality,
  jlong j_left_conditional,
  jlong j_right_conditional,
  jlong j_condition,
  jboolean j_is_left_sorted,
  jboolean j_is_right_sorted,
  jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_left_equality, "left equality table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_equality, "right equality table is null", nullptr);
  JNI_NULL_CHECK(env, j_left_conditional, "left conditional table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_conditional, "right conditional table is null", nullptr);
  JNI_NULL_CHECK(env, j_condition, "condition is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const left_equality     = reinterpret_cast<cudf::table_view const*>(j_left_equality);
    auto const right_equality    = reinterpret_cast<cudf::table_view const*>(j_right_equality);
    auto const left_conditional  = reinterpret_cast<cudf::table_view const*>(j_left_conditional);
    auto const right_conditional = reinterpret_cast<cudf::table_view const*>(j_right_conditional);
    auto const condition = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);

    auto const is_left_sorted  = j_is_left_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const is_right_sorted = j_is_right_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    auto result = spark_rapids_jni::mixed_sort_merge_inner_join(*left_equality,
                                                                *right_equality,
                                                                *left_conditional,
                                                                *right_conditional,
                                                                condition->get_top_expression(),
                                                                is_left_sorted,
                                                                is_right_sorted,
                                                                nulls_equal);

    return gather_maps_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_MixedSortMergeJoin_mixedSortMergeLeftJoin(
  JNIEnv* env,
  jclass,
  jlong j_left_equality,
  jlong j_right_equality,
  jlong j_left_conditional,
  jlong j_right_conditional,
  jlong j_condition,
  jboolean j_is_left_sorted,
  jboolean j_is_right_sorted,
  jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_left_equality, "left equality table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_equality, "right equality table is null", nullptr);
  JNI_NULL_CHECK(env, j_left_conditional, "left conditional table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_conditional, "right conditional table is null", nullptr);
  JNI_NULL_CHECK(env, j_condition, "condition is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const left_equality     = reinterpret_cast<cudf::table_view const*>(j_left_equality);
    auto const right_equality    = reinterpret_cast<cudf::table_view const*>(j_right_equality);
    auto const left_conditional  = reinterpret_cast<cudf::table_view const*>(j_left_conditional);
    auto const right_conditional = reinterpret_cast<cudf::table_view const*>(j_right_conditional);
    auto const condition = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);

    auto const is_left_sorted  = j_is_left_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const is_right_sorted = j_is_right_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    auto result = spark_rapids_jni::mixed_sort_merge_left_join(*left_equality,
                                                               *right_equality,
                                                               *left_conditional,
                                                               *right_conditional,
                                                               condition->get_top_expression(),
                                                               is_left_sorted,
                                                               is_right_sorted,
                                                               nulls_equal);

    return gather_maps_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_MixedSortMergeJoin_mixedSortMergeLeftSemiJoin(
  JNIEnv* env,
  jclass,
  jlong j_left_equality,
  jlong j_right_equality,
  jlong j_left_conditional,
  jlong j_right_conditional,
  jlong j_condition,
  jboolean j_is_left_sorted,
  jboolean j_is_right_sorted,
  jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_left_equality, "left equality table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_equality, "right equality table is null", nullptr);
  JNI_NULL_CHECK(env, j_left_conditional, "left conditional table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_conditional, "right conditional table is null", nullptr);
  JNI_NULL_CHECK(env, j_condition, "condition is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const left_equality     = reinterpret_cast<cudf::table_view const*>(j_left_equality);
    auto const right_equality    = reinterpret_cast<cudf::table_view const*>(j_right_equality);
    auto const left_conditional  = reinterpret_cast<cudf::table_view const*>(j_left_conditional);
    auto const right_conditional = reinterpret_cast<cudf::table_view const*>(j_right_conditional);
    auto const condition = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);

    auto const is_left_sorted  = j_is_left_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const is_right_sorted = j_is_right_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    auto result = spark_rapids_jni::mixed_sort_merge_left_semi_join(*left_equality,
                                                                    *right_equality,
                                                                    *left_conditional,
                                                                    *right_conditional,
                                                                    condition->get_top_expression(),
                                                                    is_left_sorted,
                                                                    is_right_sorted,
                                                                    nulls_equal);

    return gather_single_map_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_MixedSortMergeJoin_mixedSortMergeLeftAntiJoin(
  JNIEnv* env,
  jclass,
  jlong j_left_equality,
  jlong j_right_equality,
  jlong j_left_conditional,
  jlong j_right_conditional,
  jlong j_condition,
  jboolean j_is_left_sorted,
  jboolean j_is_right_sorted,
  jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_left_equality, "left equality table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_equality, "right equality table is null", nullptr);
  JNI_NULL_CHECK(env, j_left_conditional, "left conditional table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_conditional, "right conditional table is null", nullptr);
  JNI_NULL_CHECK(env, j_condition, "condition is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const left_equality     = reinterpret_cast<cudf::table_view const*>(j_left_equality);
    auto const right_equality    = reinterpret_cast<cudf::table_view const*>(j_right_equality);
    auto const left_conditional  = reinterpret_cast<cudf::table_view const*>(j_left_conditional);
    auto const right_conditional = reinterpret_cast<cudf::table_view const*>(j_right_conditional);
    auto const condition = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);

    auto const is_left_sorted  = j_is_left_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const is_right_sorted = j_is_right_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    auto result = spark_rapids_jni::mixed_sort_merge_left_anti_join(*left_equality,
                                                                    *right_equality,
                                                                    *left_conditional,
                                                                    *right_conditional,
                                                                    condition->get_top_expression(),
                                                                    is_left_sorted,
                                                                    is_right_sorted,
                                                                    nulls_equal);

    return gather_single_map_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_SortMergeJoin_sortMergeInnerJoin(JNIEnv* env,
                                                                  jclass,
                                                                  jlong j_left_keys,
                                                                  jlong j_right_keys,
                                                                  jboolean j_is_left_sorted,
                                                                  jboolean j_is_right_sorted,
                                                                  jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const left_keys  = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys = reinterpret_cast<cudf::table_view const*>(j_right_keys);

    auto const is_left_sorted  = j_is_left_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const is_right_sorted = j_is_right_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    auto result = spark_rapids_jni::sort_merge_inner_join(
      *left_keys, *right_keys, is_left_sorted, is_right_sorted, nulls_equal);

    return gather_maps_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_SortMergeJoin_sortMergeLeftJoin(JNIEnv* env,
                                                                 jclass,
                                                                 jlong j_left_keys,
                                                                 jlong j_right_keys,
                                                                 jboolean j_is_left_sorted,
                                                                 jboolean j_is_right_sorted,
                                                                 jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const left_keys  = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys = reinterpret_cast<cudf::table_view const*>(j_right_keys);

    auto const is_left_sorted  = j_is_left_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const is_right_sorted = j_is_right_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    auto result = spark_rapids_jni::sort_merge_left_join(
      *left_keys, *right_keys, is_left_sorted, is_right_sorted, nulls_equal);

    return gather_maps_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_SortMergeJoin_sortMergeLeftSemiJoin(JNIEnv* env,
                                                                     jclass,
                                                                     jlong j_left_keys,
                                                                     jlong j_right_keys,
                                                                     jboolean j_is_left_sorted,
                                                                     jboolean j_is_right_sorted,
                                                                     jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const left_keys  = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys = reinterpret_cast<cudf::table_view const*>(j_right_keys);

    auto const is_left_sorted  = j_is_left_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const is_right_sorted = j_is_right_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    auto result = spark_rapids_jni::sort_merge_left_semi_join(
      *left_keys, *right_keys, is_left_sorted, is_right_sorted, nulls_equal);

    return gather_single_map_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_SortMergeJoin_sortMergeLeftAntiJoin(JNIEnv* env,
                                                                     jclass,
                                                                     jlong j_left_keys,
                                                                     jlong j_right_keys,
                                                                     jboolean j_is_left_sorted,
                                                                     jboolean j_is_right_sorted,
                                                                     jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const left_keys  = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys = reinterpret_cast<cudf::table_view const*>(j_right_keys);

    auto const is_left_sorted  = j_is_left_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const is_right_sorted = j_is_right_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    auto result = spark_rapids_jni::sort_merge_left_anti_join(
      *left_keys, *right_keys, is_left_sorted, is_right_sorted, nulls_equal);

    return gather_single_map_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"
