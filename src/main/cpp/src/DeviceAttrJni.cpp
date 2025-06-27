#include "jni_utils.hpp"

extern "C" {

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_DeviceAttr_isDeviceIntegrated(JNIEnv* env, jclass)
{
    int device{};
    cudaGetDevice(&device);

    int integrated{};
    cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device);

    return integrated;
}

}