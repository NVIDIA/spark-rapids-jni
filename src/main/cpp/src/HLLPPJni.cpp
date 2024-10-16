/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "HLLPP.hpp"
#include "cudf_jni_apis.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_HLLPP_estimateDistinctValueFromSketches(
  JNIEnv* env, jclass, jlong sketches, jint precision)
{
  JNI_NULL_CHECK(env, sketches, "Sketch column is null", 0);
  try {
    cudf::jni::auto_set_device(env);
    auto const sketch_view = reinterpret_cast<cudf::column_view const*>(sketches);
    return cudf::jni::ptr_as_jlong(
      spark_rapids_jni::estimate_from_hll_sketches(*sketch_view, precision).release());
  }
  CATCH_STD(env, 0);
}
}
