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
                                                         jbooleanArray is_required,
                                                         jbooleanArray has_default_value,
                                                         jlongArray default_ints,
                                                         jdoubleArray default_floats,
                                                         jbooleanArray default_bools,
                                                         jobjectArray default_strings,
                                                         jobjectArray enum_valid_values,
                                                         jboolean fail_on_errors)
{
  JNI_NULL_CHECK(env, binary_input_view, "binary_input_view is null", 0);
  JNI_NULL_CHECK(env, decoded_field_indices, "decoded_field_indices is null", 0);
  JNI_NULL_CHECK(env, field_numbers, "field_numbers is null", 0);
  JNI_NULL_CHECK(env, all_type_ids, "all_type_ids is null", 0);
  JNI_NULL_CHECK(env, encodings, "encodings is null", 0);
  JNI_NULL_CHECK(env, is_required, "is_required is null", 0);
  JNI_NULL_CHECK(env, has_default_value, "has_default_value is null", 0);
  JNI_NULL_CHECK(env, default_ints, "default_ints is null", 0);
  JNI_NULL_CHECK(env, default_floats, "default_floats is null", 0);
  JNI_NULL_CHECK(env, default_bools, "default_bools is null", 0);
  JNI_NULL_CHECK(env, default_strings, "default_strings is null", 0);
  JNI_NULL_CHECK(env, enum_valid_values, "enum_valid_values is null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* input = reinterpret_cast<cudf::column_view const*>(binary_input_view);

    cudf::jni::native_jintArray n_decoded_indices(env, decoded_field_indices);
    cudf::jni::native_jintArray n_field_numbers(env, field_numbers);
    cudf::jni::native_jintArray n_all_type_ids(env, all_type_ids);
    cudf::jni::native_jintArray n_encodings(env, encodings);
    cudf::jni::native_jbooleanArray n_is_required(env, is_required);
    cudf::jni::native_jbooleanArray n_has_default(env, has_default_value);
    cudf::jni::native_jlongArray n_default_ints(env, default_ints);
    cudf::jni::native_jdoubleArray n_default_floats(env, default_floats);
    cudf::jni::native_jbooleanArray n_default_bools(env, default_bools);

    int num_decoded_fields = n_decoded_indices.size();

    // Validate array sizes
    if (n_field_numbers.size() != num_decoded_fields || n_encodings.size() != num_decoded_fields ||
        n_is_required.size() != num_decoded_fields || n_has_default.size() != num_decoded_fields ||
        n_default_ints.size() != num_decoded_fields ||
        n_default_floats.size() != num_decoded_fields ||
        n_default_bools.size() != num_decoded_fields) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "All decoded field arrays must have the same length",
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

    // Convert jboolean arrays to std::vector<bool>
    std::vector<bool> required_flags;
    std::vector<bool> has_default_flags;
    std::vector<bool> default_bool_values;
    required_flags.reserve(num_decoded_fields);
    has_default_flags.reserve(num_decoded_fields);
    default_bool_values.reserve(num_decoded_fields);
    for (int i = 0; i < num_decoded_fields; ++i) {
      required_flags.push_back(n_is_required[i] != 0);
      has_default_flags.push_back(n_has_default[i] != 0);
      default_bool_values.push_back(n_default_bools[i] != 0);
    }

    // Convert default int/float values
    std::vector<int64_t> default_int_values(n_default_ints.begin(), n_default_ints.end());
    std::vector<double> default_float_values(n_default_floats.begin(), n_default_floats.end());

    // Convert default string values (byte[][] -> vector<vector<uint8_t>>)
    std::vector<std::vector<uint8_t>> default_string_values;
    default_string_values.reserve(num_decoded_fields);
    for (int i = 0; i < num_decoded_fields; ++i) {
      jbyteArray byte_arr = static_cast<jbyteArray>(env->GetObjectArrayElement(default_strings, i));
      if (byte_arr == nullptr) {
        default_string_values.emplace_back();  // empty vector for null
      } else {
        jsize len    = env->GetArrayLength(byte_arr);
        jbyte* bytes = env->GetByteArrayElements(byte_arr, nullptr);
        default_string_values.emplace_back(reinterpret_cast<uint8_t*>(bytes),
                                           reinterpret_cast<uint8_t*>(bytes) + len);
        env->ReleaseByteArrayElements(byte_arr, bytes, JNI_ABORT);
      }
    }

    // Convert enum valid values (int[][] -> vector<vector<int32_t>>)
    // Each element is either null (not an enum field) or an array of valid enum values
    std::vector<std::vector<int32_t>> enum_values;
    enum_values.reserve(num_decoded_fields);
    for (int i = 0; i < num_decoded_fields; ++i) {
      jintArray int_arr = static_cast<jintArray>(env->GetObjectArrayElement(enum_valid_values, i));
      if (int_arr == nullptr) {
        enum_values.emplace_back();  // empty vector for null (not an enum field)
      } else {
        jsize len  = env->GetArrayLength(int_arr);
        jint* ints = env->GetIntArrayElements(int_arr, nullptr);
        enum_values.emplace_back(ints, ints + len);
        env->ReleaseIntArrayElements(int_arr, ints, JNI_ABORT);
      }
    }

    // Build all_types vector - types for ALL fields in the output struct
    std::vector<cudf::data_type> all_types;
    all_types.reserve(total_num_fields);
    for (int i = 0; i < total_num_fields; ++i) {
      // For non-decimal types, scale is always 0
      all_types.emplace_back(cudf::jni::make_data_type(n_all_type_ids[i], 0));
    }

    auto result = spark_rapids_jni::decode_protobuf_to_struct(*input,
                                                              total_num_fields,
                                                              decoded_indices,
                                                              field_nums,
                                                              all_types,
                                                              encs,
                                                              required_flags,
                                                              has_default_flags,
                                                              default_int_values,
                                                              default_float_values,
                                                              default_bool_values,
                                                              default_string_values,
                                                              enum_values,
                                                              fail_on_errors);
    return cudf::jni::release_as_jlong(result);
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
