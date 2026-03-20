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
#include "protobuf/protobuf.hpp"

#include <cudf/utilities/default_stream.hpp>

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_Protobuf_decodeToStruct(JNIEnv* env,
                                                         jclass,
                                                         jlong binary_input_view,
                                                         jintArray field_numbers,
                                                         jintArray parent_indices,
                                                         jintArray depth_levels,
                                                         jintArray wire_types,
                                                         jintArray output_type_ids,
                                                         jintArray encodings,
                                                         jbooleanArray is_repeated,
                                                         jbooleanArray is_required,
                                                         jbooleanArray has_default_value,
                                                         jlongArray default_ints,
                                                         jdoubleArray default_floats,
                                                         jbooleanArray default_bools,
                                                         jobjectArray default_strings,
                                                         jobjectArray enum_valid_values,
                                                         jobjectArray enum_names,
                                                         jboolean fail_on_errors)
{
  auto const all_inputs_valid = binary_input_view && field_numbers && parent_indices &&
                                depth_levels && wire_types && output_type_ids && encodings &&
                                is_repeated && is_required && has_default_value && default_ints &&
                                default_floats && default_bools && default_strings &&
                                enum_valid_values && enum_names;
  JNI_NULL_CHECK(env, all_inputs_valid, "one or more input arrays are null", 0);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* input = reinterpret_cast<cudf::column_view const*>(binary_input_view);

    cudf::jni::native_jintArray n_field_numbers(env, field_numbers);
    cudf::jni::native_jintArray n_parent_indices(env, parent_indices);
    cudf::jni::native_jintArray n_depth_levels(env, depth_levels);
    cudf::jni::native_jintArray n_wire_types(env, wire_types);
    cudf::jni::native_jintArray n_output_type_ids(env, output_type_ids);
    cudf::jni::native_jintArray n_encodings(env, encodings);
    cudf::jni::native_jbooleanArray n_is_repeated(env, is_repeated);
    cudf::jni::native_jbooleanArray n_is_required(env, is_required);
    cudf::jni::native_jbooleanArray n_has_default(env, has_default_value);
    cudf::jni::native_jlongArray n_default_ints(env, default_ints);
    cudf::jni::native_jdoubleArray n_default_floats(env, default_floats);
    cudf::jni::native_jbooleanArray n_default_bools(env, default_bools);

    int num_fields = n_field_numbers.size();

    // Validate array sizes
    if (n_parent_indices.size() != num_fields || n_depth_levels.size() != num_fields ||
        n_wire_types.size() != num_fields || n_output_type_ids.size() != num_fields ||
        n_encodings.size() != num_fields || n_is_repeated.size() != num_fields ||
        n_is_required.size() != num_fields || n_has_default.size() != num_fields ||
        n_default_ints.size() != num_fields || n_default_floats.size() != num_fields ||
        n_default_bools.size() != num_fields ||
        env->GetArrayLength(default_strings) != num_fields ||
        env->GetArrayLength(enum_valid_values) != num_fields ||
        env->GetArrayLength(enum_names) != num_fields) {
      JNI_THROW_NEW(env,
                    cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                    "All field arrays must have the same length",
                    0);
    }

    // Build schema descriptors
    std::vector<spark_rapids_jni::protobuf::nested_field_descriptor> schema;
    schema.reserve(num_fields);
    for (int i = 0; i < num_fields; ++i) {
      schema.push_back({n_field_numbers[i],
                        n_parent_indices[i],
                        n_depth_levels[i],
                        static_cast<spark_rapids_jni::protobuf::proto_wire_type>(n_wire_types[i]),
                        static_cast<cudf::type_id>(n_output_type_ids[i]),
                        static_cast<spark_rapids_jni::protobuf::proto_encoding>(n_encodings[i]),
                        n_is_repeated[i] != 0,
                        n_is_required[i] != 0,
                        n_has_default[i] != 0});
    }

    // Build output types
    // Convert boolean arrays
    std::vector<bool> default_bool_values;
    default_bool_values.reserve(num_fields);
    for (int i = 0; i < num_fields; ++i) {
      default_bool_values.push_back(n_default_bools[i] != 0);
    }

    // Convert default values
    std::vector<int64_t> default_int_values(n_default_ints.begin(), n_default_ints.end());
    std::vector<double> default_float_values(n_default_floats.begin(), n_default_floats.end());

    // Convert default string values
    std::vector<std::vector<uint8_t>> default_string_values;
    default_string_values.reserve(num_fields);
    for (int i = 0; i < num_fields; ++i) {
      jbyteArray byte_arr = static_cast<jbyteArray>(env->GetObjectArrayElement(default_strings, i));
      if (env->ExceptionCheck()) { return 0; }
      if (byte_arr == nullptr) {
        default_string_values.emplace_back();
      } else {
        jsize len    = env->GetArrayLength(byte_arr);
        jbyte* bytes = env->GetByteArrayElements(byte_arr, nullptr);
        if (bytes == nullptr) {
          env->DeleteLocalRef(byte_arr);
          return 0;
        }
        default_string_values.emplace_back(reinterpret_cast<uint8_t*>(bytes),
                                           reinterpret_cast<uint8_t*>(bytes) + len);
        env->ReleaseByteArrayElements(byte_arr, bytes, JNI_ABORT);
        env->DeleteLocalRef(byte_arr);
      }
    }

    // Convert enum valid values
    std::vector<std::vector<int32_t>> enum_values;
    enum_values.reserve(num_fields);
    for (int i = 0; i < num_fields; ++i) {
      jintArray int_arr = static_cast<jintArray>(env->GetObjectArrayElement(enum_valid_values, i));
      if (env->ExceptionCheck()) { return 0; }
      if (int_arr == nullptr) {
        enum_values.emplace_back();
      } else {
        jsize len  = env->GetArrayLength(int_arr);
        jint* ints = env->GetIntArrayElements(int_arr, nullptr);
        if (ints == nullptr) {
          env->DeleteLocalRef(int_arr);
          return 0;
        }
        enum_values.emplace_back(ints, ints + len);
        env->ReleaseIntArrayElements(int_arr, ints, JNI_ABORT);
        env->DeleteLocalRef(int_arr);
      }
    }

    // Convert enum names (byte[][][]). For each field:
    // - null => not an enum-as-string field
    // - byte[][] where each byte[] is UTF-8 enum name, ordered with enum_values[field]
    std::vector<std::vector<std::vector<uint8_t>>> enum_name_values;
    enum_name_values.reserve(num_fields);
    for (int i = 0; i < num_fields; ++i) {
      jobjectArray names_arr = static_cast<jobjectArray>(env->GetObjectArrayElement(enum_names, i));
      if (env->ExceptionCheck()) { return 0; }
      if (names_arr == nullptr) {
        enum_name_values.emplace_back();
      } else {
        jsize num_names = env->GetArrayLength(names_arr);
        std::vector<std::vector<uint8_t>> names_for_field;
        names_for_field.reserve(num_names);
        for (jsize j = 0; j < num_names; ++j) {
          jbyteArray name_bytes = static_cast<jbyteArray>(env->GetObjectArrayElement(names_arr, j));
          if (env->ExceptionCheck()) {
            env->DeleteLocalRef(names_arr);
            return 0;
          }
          if (name_bytes == nullptr) {
            names_for_field.emplace_back();
          } else {
            jsize len    = env->GetArrayLength(name_bytes);
            jbyte* bytes = env->GetByteArrayElements(name_bytes, nullptr);
            if (bytes == nullptr) {
              env->DeleteLocalRef(name_bytes);
              env->DeleteLocalRef(names_arr);
              return 0;
            }
            names_for_field.emplace_back(reinterpret_cast<uint8_t*>(bytes),
                                         reinterpret_cast<uint8_t*>(bytes) + len);
            env->ReleaseByteArrayElements(name_bytes, bytes, JNI_ABORT);
            env->DeleteLocalRef(name_bytes);
          }
        }
        enum_name_values.push_back(std::move(names_for_field));
        env->DeleteLocalRef(names_arr);
      }
    }

    spark_rapids_jni::protobuf::ProtobufDecodeContext context{std::move(schema),
                                                              std::move(default_int_values),
                                                              std::move(default_float_values),
                                                              std::move(default_bool_values),
                                                              std::move(default_string_values),
                                                              std::move(enum_values),
                                                              std::move(enum_name_values),
                                                              static_cast<bool>(fail_on_errors)};

    auto result = spark_rapids_jni::protobuf::decode_protobuf_to_struct(
      *input, context, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

    return cudf::jni::release_as_jlong(result);
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
