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
#include "uuid.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_StringUtils_randomUUIDs(JNIEnv* env,
                                                                                 jclass,
                                                                                 jint row_count,
                                                                                 jlong seed)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    return cudf::jni::release_as_jlong(spark_rapids_jni::random_uuids(row_count, seed));
  }
  JNI_CATCH(env, 0);
}
}
