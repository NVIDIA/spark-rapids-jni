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
#include "error.hpp"
#include "jni_utils.hpp"
#include "multiply.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_Arithmetic_multiply(JNIEnv* env,
                                                                             jclass,
                                                                             jlong left,
                                                                             jboolean is_left_cv,
                                                                             jlong right,
                                                                             jboolean is_right_cv,
                                                                             jboolean ansi_enabled)
{
  JNI_NULL_CHECK(env, left, "input column is null", 0);
  JNI_NULL_CHECK(env, right, "input column is null", 0);

  try {
    cudf::jni::auto_set_device(env);

    if (is_left_cv && is_right_cv) {
      auto left_cv  = *reinterpret_cast<cudf::column_view*>(left);
      auto right_cv = *reinterpret_cast<cudf::column_view*>(right);
      return cudf::jni::release_as_jlong(
        spark_rapids_jni::multiply(left_cv, right_cv, ansi_enabled));
    } else {
      throw cudf::logic_error("Both left and right must be column views");
    }
  }
  // throw ExceptionWithRowIndex if an exception occurs if in ansi mode
  // ExceptionWithRowIndex contains the row number that caused the exception
  CATCH_EXCEPTION_WITH_ROW_INDEX(env, 0);
}
}
