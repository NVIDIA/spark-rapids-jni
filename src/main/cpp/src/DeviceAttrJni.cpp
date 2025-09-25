#include "cudf_jni_apis.hpp"

extern "C" {

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_DeviceAttr_isDeviceIntegrated(JNIEnv* env, jclass)
{
  try {
    int device{};
    CUDF_CUDA_TRY(cudaGetDevice(&device));

    int integrated{};
    CUDF_CUDA_TRY(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device));

    return integrated;
  }
  CATCH_STD(env, 0);
}

}