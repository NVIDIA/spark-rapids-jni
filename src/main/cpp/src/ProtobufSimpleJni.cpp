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
#include "dtype_utils.hpp"
#include "protobuf_simple.hpp"

#include <cudf/column/column_view.hpp>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_ProtobufSimple_decodeToStruct(
  JNIEnv* env, jclass, jlong binary_input_view, jintArray field_numbers, jintArray type_ids, jintArray type_scales)
{
  JNI_NULL_CHECK(env, binary_input_view, "binary_input_view is null", 0);
  JNI_NULL_CHECK(env, field_numbers, "field_numbers is null", 0);
  JNI_NULL_CHECK(env, type_ids, "type_ids is null", 0);
  JNI_NULL_CHECK(env, type_scales, "type_scales is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* input = reinterpret_cast<cudf::column_view const*>(binary_input_view);
    cudf::jni::native_jintArray n_field_numbers(env, field_numbers);
    cudf::jni::native_jintArray n_type_ids(env, type_ids);
    cudf::jni::native_jintArray n_type_scales(env, type_scales);
    if (n_field_numbers.size() != n_type_ids.size() || n_field_numbers.size() != n_type_scales.size()) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "fieldNumbers/typeIds/typeScales must be the same length",
                    0);
    }

    std::vector<int> field_nums(n_field_numbers.begin(), n_field_numbers.end());
    std::vector<cudf::data_type> out_types;
    out_types.reserve(n_type_ids.size());
    for (int i = 0; i < n_type_ids.size(); ++i) {
      out_types.emplace_back(cudf::jni::make_data_type(n_type_ids[i], n_type_scales[i]));
    }

    auto result =
      spark_rapids_jni::decode_protobuf_simple_to_struct(*input, field_nums, out_types);
    return cudf::jni::release_as_jlong(result);
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"



