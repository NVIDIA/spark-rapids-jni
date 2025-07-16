#include "cudf_jni_apis.hpp"
#include "map_zip_with_utils.hpp"

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_GpuMapZipWithUtils_mapZip(
  JNIEnv* env, jclass, jlong input_column1, jlong input_column2)
{
  JNI_NULL_CHECK(env, input_column1, "input column1 is null", 0);
  JNI_NULL_CHECK(env, input_column2, "input column2 is null", 0);
  try {
    cudf::jni::auto_set_device(env);

    // The following constructor expects that the type of the input_column is LIST.
    // If the type is not LIST, an exception will be thrown.
    cudf::lists_column_view col1{*reinterpret_cast<cudf::column_view const*>(input_column1)};
    cudf::lists_column_view col2{*reinterpret_cast<cudf::column_view const*>(input_column2)};
    return cudf::jni::release_as_jlong(spark_rapids_jni::map_zip(col1, col2));
  }
  CATCH_STD(env, 0);
}
}