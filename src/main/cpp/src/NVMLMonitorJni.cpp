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

#include <jni.h>
#include <nvml.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

// NVML JNI implementation with comprehensive GPU metrics for Spark Rapids

// Helper function to get NVML device handle from CUDA device ID
static nvmlReturn_t getNvmlDeviceFromCudaDevice(int cudaDeviceId, nvmlDevice_t* device) {
    char pciBuf[32];
    cudaError_t cerr = cudaDeviceGetPCIBusId(pciBuf, sizeof(pciBuf), cudaDeviceId);
    if (cerr != cudaSuccess) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    nvmlReturn_t nerr = nvmlDeviceGetHandleByPciBusId(pciBuf, device);
    return nerr;
}

// Helper function to populate GPUInfo object from NVML device handle
static jobject populateGPUInfoFromDevice(JNIEnv *env, nvmlDevice_t device, jint deviceIndex) {
    nvmlReturn_t result;

    // Get Java GPUInfo class and constructor
    jclass gpuInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/GPUInfo");
    if (gpuInfoClass == NULL) {
        return NULL;
    }

    jmethodID constructor = env->GetMethodID(gpuInfoClass, "<init>", "()V");
    if (constructor == NULL) {
        return NULL;
    }

    // Create GPUInfo object
    jobject gpuInfo = env->NewObject(gpuInfoClass, constructor);
    if (gpuInfo == NULL) {
        return NULL;
    }

    // Get field IDs
    jfieldID deviceIndexField = env->GetFieldID(gpuInfoClass, "deviceIndex", "I");
    jfieldID nameField = env->GetFieldID(gpuInfoClass, "name", "Ljava/lang/String;");
    jfieldID brandField = env->GetFieldID(gpuInfoClass, "brand", "Ljava/lang/String;");

    // Utilization fields
    jfieldID gpuUtilField = env->GetFieldID(gpuInfoClass, "gpuUtilization", "I");
    jfieldID memUtilField = env->GetFieldID(gpuInfoClass, "memoryUtilization", "I");

    // Memory fields
    jfieldID memUsedField = env->GetFieldID(gpuInfoClass, "memoryUsedMB", "J");
    jfieldID memTotalField = env->GetFieldID(gpuInfoClass, "memoryTotalMB", "J");
    jfieldID memFreeField = env->GetFieldID(gpuInfoClass, "memoryFreeMB", "J");

    // Temperature fields
    jfieldID tempGpuField = env->GetFieldID(gpuInfoClass, "temperatureGpu", "I");
    jfieldID tempMemoryField = env->GetFieldID(gpuInfoClass, "temperatureMemory", "I");

    // Power fields
    jfieldID powerUsageField = env->GetFieldID(gpuInfoClass, "powerUsageW", "I");
    jfieldID powerLimitField = env->GetFieldID(gpuInfoClass, "powerLimitW", "I");
    jfieldID powerDefaultLimitField = env->GetFieldID(gpuInfoClass, "powerDefaultLimitW", "I");

    // Clock fields
    jfieldID graphicsClockField = env->GetFieldID(gpuInfoClass, "graphicsClockMHz", "I");
    jfieldID memoryClockField = env->GetFieldID(gpuInfoClass, "memoryClockMHz", "I");
    jfieldID smClockField = env->GetFieldID(gpuInfoClass, "smClockMHz", "I");

    // Advanced fields
    jfieldID smCountField = env->GetFieldID(gpuInfoClass, "streamingMultiprocessors", "I");
    jfieldID performanceStateField = env->GetFieldID(gpuInfoClass, "performanceState", "I");
    jfieldID fanSpeedField = env->GetFieldID(gpuInfoClass, "fanSpeedPercent", "I");

    // PCIe fields
    jfieldID pcieLinkGenField = env->GetFieldID(gpuInfoClass, "pcieLinkGeneration", "I");
    jfieldID pcieLinkWidthField = env->GetFieldID(gpuInfoClass, "pcieLinkWidth", "I");

    // Error fields
    jfieldID eccSingleBitField = env->GetFieldID(gpuInfoClass, "eccSingleBitErrors", "J");
    jfieldID eccDoubleBitField = env->GetFieldID(gpuInfoClass, "eccDoubleBitErrors", "J");

    // Set device index
    env->SetIntField(gpuInfo, deviceIndexField, deviceIndex);

    // Get device name
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
        jstring jname = env->NewStringUTF(name);
        env->SetObjectField(gpuInfo, nameField, jname);
        env->DeleteLocalRef(jname);
    }

    // Get brand type (simplified)
    nvmlBrandType_t brandType;
    result = nvmlDeviceGetBrand(device, &brandType);
    if (result == NVML_SUCCESS) {
        char brand[50];
        snprintf(brand, sizeof(brand), "Brand_%d", (int)brandType);
        jstring jbrand = env->NewStringUTF(brand);
        env->SetObjectField(gpuInfo, brandField, jbrand);
        env->DeleteLocalRef(jbrand);
    }

    // Get utilization rates
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, gpuUtilField, (jint)utilization.gpu);
        env->SetIntField(gpuInfo, memUtilField, (jint)utilization.memory);
    }

    // Get memory info
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (result == NVML_SUCCESS) {
        env->SetLongField(gpuInfo, memUsedField, (jlong)(memory.used / (1024 * 1024)));
        env->SetLongField(gpuInfo, memTotalField, (jlong)(memory.total / (1024 * 1024)));
        env->SetLongField(gpuInfo, memFreeField, (jlong)(memory.free / (1024 * 1024)));
    }

    // Get GPU temperature
    unsigned int temp;
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, tempGpuField, (jint)temp);
    }

    // Get memory temperature (use GPU temperature as fallback since memory temp may not be available)
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, tempMemoryField, (jint)temp);
    }

    // Get power usage
    unsigned int power;
    result = nvmlDeviceGetPowerUsage(device, &power);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, powerUsageField, (jint)(power / 1000)); // mW to W
    }

    // Get power management limit
    result = nvmlDeviceGetPowerManagementLimit(device, &power);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, powerLimitField, (jint)(power / 1000)); // mW to W
    }

    // Get default power management limit (use current limit as fallback)
    result = nvmlDeviceGetPowerManagementLimit(device, &power);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, powerDefaultLimitField, (jint)(power / 1000)); // mW to W
    }

    // Get clock speeds
    unsigned int clock;
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, graphicsClockField, (jint)clock);
    }

    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, memoryClockField, (jint)clock);
    }

    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, smClockField, (jint)clock);
    }

    // Get SM count (GPU cores/multiprocessors)
    unsigned int smCount = 0;
    result = nvmlDeviceGetNumGpuCores(device, &smCount);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, smCountField, (jint)smCount);
    }

    // Get performance state
    nvmlPstates_t pState;
    result = nvmlDeviceGetPerformanceState(device, &pState);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, performanceStateField, (jint)pState);
    }

    // Get fan speed
    unsigned int fanSpeed;
    result = nvmlDeviceGetFanSpeed(device, &fanSpeed);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, fanSpeedField, (jint)fanSpeed);
    }

    // Get PCIe link info
    unsigned int linkGen;
    result = nvmlDeviceGetCurrPcieLinkGeneration(device, &linkGen);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, pcieLinkGenField, (jint)linkGen);
    }

    unsigned int linkWidth;
    result = nvmlDeviceGetCurrPcieLinkWidth(device, &linkWidth);
    if (result == NVML_SUCCESS) {
        env->SetIntField(gpuInfo, pcieLinkWidthField, (jint)linkWidth);
    }

    // Get ECC errors (if supported)
    unsigned long long eccCount;
    result = nvmlDeviceGetTotalEccErrors(device, NVML_SINGLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
    if (result == NVML_SUCCESS) {
        env->SetLongField(gpuInfo, eccSingleBitField, (jlong)eccCount);
    }

    result = nvmlDeviceGetTotalEccErrors(device, NVML_DOUBLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
    if (result == NVML_SUCCESS) {
        env->SetLongField(gpuInfo, eccDoubleBitField, (jlong)eccCount);
    }

    return gpuInfo;
}

extern "C" {

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_NVMLMonitor_nvmlInit(JNIEnv *env, jclass cls) {
    nvmlReturn_t result = nvmlInit();
    return (result == NVML_SUCCESS) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_NVMLMonitor_nvmlShutdown(JNIEnv *env, jclass cls) {
    nvmlShutdown();
}

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_NVMLMonitor_nvmlGetDeviceCount(JNIEnv *env, jclass cls) {
    unsigned int deviceCount = 0;
    nvmlReturn_t result = nvmlDeviceGetCount(&deviceCount);

    if (result != NVML_SUCCESS) {
        return -1;
    }

    return (jint)deviceCount;
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_NVMLMonitor_nvmlGetGPUInfo(JNIEnv *env, jclass cls, jint deviceIndex) {
    nvmlDevice_t device;
    nvmlReturn_t result;

    // Get device handle by index
    result = nvmlDeviceGetHandleByIndex((unsigned int)deviceIndex, &device);
    if (result != NVML_SUCCESS) {
        return NULL;
    }

    // Use common helper to populate GPUInfo object
    return populateGPUInfoFromDevice(env, device, deviceIndex);
}

JNIEXPORT jobjectArray JNICALL Java_com_nvidia_spark_rapids_jni_NVMLMonitor_nvmlGetAllGPUInfo(JNIEnv *env, jclass cls) {
    // Get device count
    unsigned int deviceCount = 0;
    nvmlReturn_t result = nvmlDeviceGetCount(&deviceCount);

    if (result != NVML_SUCCESS || deviceCount == 0) {
        // Return empty array
        jclass gpuInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/GPUInfo");
        if (gpuInfoClass == NULL) {
            return NULL;
        }
        return env->NewObjectArray(0, gpuInfoClass, NULL);
    }

    // Create array
    jclass gpuInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/GPUInfo");
    if (gpuInfoClass == NULL) {
        return NULL;
    }

    jobjectArray gpuInfoArray = env->NewObjectArray((jsize)deviceCount, gpuInfoClass, NULL);
    if (gpuInfoArray == NULL) {
        return NULL;
    }

    // Fill array with GPU info
    for (unsigned int i = 0; i < deviceCount; i++) {
        jobject gpuInfo = Java_com_nvidia_spark_rapids_jni_NVMLMonitor_nvmlGetGPUInfo(env, cls, (jint)i);
        if (gpuInfo != NULL) {
            env->SetObjectArrayElement(gpuInfoArray, (jsize)i, gpuInfo);
            env->DeleteLocalRef(gpuInfo);
        }
    }

    return gpuInfoArray;
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_NVMLMonitor_nvmlGetGPUInfoByCudaDevice(JNIEnv *env, jclass cls, jint cudaDeviceId) {
    nvmlDevice_t device;
    nvmlReturn_t result;

    // Get NVML device handle from CUDA device ID
    result = getNvmlDeviceFromCudaDevice((int)cudaDeviceId, &device);
    if (result != NVML_SUCCESS) {
        return NULL;
    }

    // Use common helper to populate GPUInfo object (pass CUDA device ID as the device index for reference)
    return populateGPUInfoFromDevice(env, device, cudaDeviceId);
}

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_NVMLMonitor_nvmlIsCudaDeviceValid(JNIEnv *env, jclass cls, jint cudaDeviceId) {
    nvmlDevice_t device;
    nvmlReturn_t result = getNvmlDeviceFromCudaDevice((int)cudaDeviceId, &device);
    return (result == NVML_SUCCESS) ? JNI_TRUE : JNI_FALSE;
}

} // extern "C"
