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

#include "cudf_jni_apis.hpp"
#include "dtype_utils.hpp"
#include "protobuf.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/traits.hpp>

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_Protobuf_decodeToStruct(JNIEnv* env,
                                                         jclass,
                                                         jlong binary_input_view,
                                                         jint total_num_fields,
                                                         jintArray decoded_field_indices,
                                                         jintArray field_numbers,
                                                         jintArray all_type_ids,
                                                         jintArray encodings,
                                                         jboolean fail_on_errors)
{
  JNI_NULL_CHECK(env, binary_input_view, "binary_input_view is null", 0);
  JNI_NULL_CHECK(env, decoded_field_indices, "decoded_field_indices is null", 0);
  JNI_NULL_CHECK(env, field_numbers, "field_numbers is null", 0);
  JNI_NULL_CHECK(env, all_type_ids, "all_type_ids is null", 0);
  JNI_NULL_CHECK(env, encodings, "encodings is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* input = reinterpret_cast<cudf::column_view const*>(binary_input_view);

    cudf::jni::native_jintArray n_decoded_indices(env, decoded_field_indices);
    cudf::jni::native_jintArray n_field_numbers(env, field_numbers);
    cudf::jni::native_jintArray n_all_type_ids(env, all_type_ids);
    cudf::jni::native_jintArray n_encodings(env, encodings);

    // Validate array sizes
    if (n_decoded_indices.size() != n_field_numbers.size() ||
        n_decoded_indices.size() != n_encodings.size()) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "decoded_field_indices/field_numbers/encodings must be the same length",
                    0);
    }
    if (n_all_type_ids.size() != total_num_fields) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "all_type_ids size must equal total_num_fields",
                    0);
    }

    std::vector<int> decoded_indices(n_decoded_indices.begin(), n_decoded_indices.end());
    std::vector<int> field_nums(n_field_numbers.begin(), n_field_numbers.end());
    std::vector<int> encs(n_encodings.begin(), n_encodings.end());

    // Build all_types vector - types for ALL fields in the output struct
    std::vector<cudf::data_type> all_types;
    all_types.reserve(total_num_fields);
    for (int i = 0; i < total_num_fields; ++i) {
      // For non-decimal types, scale is always 0
      all_types.emplace_back(cudf::jni::make_data_type(n_all_type_ids[i], 0));
    }

    auto result = spark_rapids_jni::decode_protobuf_to_struct(
      *input, total_num_fields, decoded_indices, field_nums, all_types, encs, fail_on_errors);
    return cudf::jni::release_as_jlong(result);
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
