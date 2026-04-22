/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

#include "charset_decode.hpp"
#include "cudf_jni_apis.hpp"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CharsetDecode_decodeNative(
  JNIEnv* env, jclass, jlong input_column, jint charset)
{
  JNI_NULL_CHECK(env, input_column, "input column is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const input = *reinterpret_cast<cudf::column_view const*>(input_column);
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::decode_charset(input,
                                       static_cast<spark_rapids_jni::charset_type>(charset),
                                       cudf::get_default_stream(),
                                       cudf::get_current_device_resource_ref()));
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
