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

#include <cuda_runtime.h>

#include <jni.h>
#include <nvml.h>
#include <stdio.h>
#include <string.h>

#include <iostream>

// NVML JNI implementation with comprehensive GPU metrics for Spark Rapids

// Helper functions for individual NVML API groups

static jobject populateDeviceInfo(JNIEnv* env, nvmlDevice_t device)
{
  jclass deviceInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUDeviceInfo");
  if (deviceInfoClass == NULL) return NULL;

  jmethodID constructor = env->GetMethodID(deviceInfoClass, "<init>", "()V");
  if (constructor == NULL) return NULL;

  jobject deviceInfo = env->NewObject(deviceInfoClass, constructor);
  if (deviceInfo == NULL) return NULL;

  jfieldID nameField  = env->GetFieldID(deviceInfoClass, "name", "Ljava/lang/String;");
  jfieldID brandField = env->GetFieldID(deviceInfoClass, "brand", "Ljava/lang/String;");

  char name[NVML_DEVICE_NAME_BUFFER_SIZE];
  nvmlReturn_t result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
  if (result == NVML_SUCCESS) {
    jstring jname = env->NewStringUTF(name);
    env->SetObjectField(deviceInfo, nameField, jname);
    env->DeleteLocalRef(jname);
  }

  nvmlBrandType_t brandType;
  result = nvmlDeviceGetBrand(device, &brandType);
  if (result == NVML_SUCCESS) {
    char brand[50];
    snprintf(brand, sizeof(brand), "Brand_%d", (int)brandType);
    jstring jbrand = env->NewStringUTF(brand);
    env->SetObjectField(deviceInfo, brandField, jbrand);
    env->DeleteLocalRef(jbrand);
  }

  return deviceInfo;
}

static jobject populateUtilizationInfo(JNIEnv* env, nvmlDevice_t device)
{
  jclass utilizationInfoClass =
    env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUUtilizationInfo");
  if (utilizationInfoClass == NULL) return NULL;

  jmethodID constructor = env->GetMethodID(utilizationInfoClass, "<init>", "()V");
  if (constructor == NULL) return NULL;

  jobject utilizationInfo = env->NewObject(utilizationInfoClass, constructor);
  if (utilizationInfo == NULL) return NULL;

  jfieldID gpuUtilField = env->GetFieldID(utilizationInfoClass, "gpuUtilization", "I");
  jfieldID memUtilField = env->GetFieldID(utilizationInfoClass, "memoryUtilization", "I");

  nvmlUtilization_t utilization;
  nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);
  if (result == NVML_SUCCESS) {
    env->SetIntField(utilizationInfo, gpuUtilField, (jint)utilization.gpu);
    env->SetIntField(utilizationInfo, memUtilField, (jint)utilization.memory);
  }

  return utilizationInfo;
}

static jobject populateMemoryInfo(JNIEnv* env, nvmlDevice_t device)
{
  jclass memoryInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUMemoryInfo");
  if (memoryInfoClass == NULL) return NULL;

  jmethodID constructor = env->GetMethodID(memoryInfoClass, "<init>", "()V");
  if (constructor == NULL) return NULL;

  jobject memoryInfo = env->NewObject(memoryInfoClass, constructor);
  if (memoryInfo == NULL) return NULL;

  jfieldID memUsedField  = env->GetFieldID(memoryInfoClass, "memoryUsedMB", "J");
  jfieldID memTotalField = env->GetFieldID(memoryInfoClass, "memoryTotalMB", "J");
  jfieldID memFreeField  = env->GetFieldID(memoryInfoClass, "memoryFreeMB", "J");

  nvmlMemory_t memory;
  nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device, &memory);
  if (result == NVML_SUCCESS) {
    env->SetLongField(memoryInfo, memUsedField, (jlong)(memory.used / (1024 * 1024)));
    env->SetLongField(memoryInfo, memTotalField, (jlong)(memory.total / (1024 * 1024)));
    env->SetLongField(memoryInfo, memFreeField, (jlong)(memory.free / (1024 * 1024)));
  }

  return memoryInfo;
}

static jobject populateTemperatureInfo(JNIEnv* env, nvmlDevice_t device)
{
  jclass temperatureInfoClass =
    env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUTemperatureInfo");
  if (temperatureInfoClass == NULL) return NULL;

  jmethodID constructor = env->GetMethodID(temperatureInfoClass, "<init>", "()V");
  if (constructor == NULL) return NULL;

  jobject temperatureInfo = env->NewObject(temperatureInfoClass, constructor);
  if (temperatureInfo == NULL) return NULL;

  jfieldID tempGpuField = env->GetFieldID(temperatureInfoClass, "temperatureGpu", "I");

  unsigned int temp;
  nvmlReturn_t result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
  if (result == NVML_SUCCESS) { env->SetIntField(temperatureInfo, tempGpuField, (jint)temp); }

  return temperatureInfo;
}

static jobject populatePowerInfo(JNIEnv* env, nvmlDevice_t device)
{
  jclass powerInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUPowerInfo");
  if (powerInfoClass == NULL) return NULL;

  jmethodID constructor = env->GetMethodID(powerInfoClass, "<init>", "()V");
  if (constructor == NULL) return NULL;

  jobject powerInfo = env->NewObject(powerInfoClass, constructor);
  if (powerInfo == NULL) return NULL;

  jfieldID powerUsageField = env->GetFieldID(powerInfoClass, "powerUsageW", "I");
  jfieldID powerLimitField = env->GetFieldID(powerInfoClass, "powerLimitW", "I");

  unsigned int power;
  nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &power);
  if (result == NVML_SUCCESS) {
    env->SetIntField(powerInfo, powerUsageField, (jint)(power / 1000));  // mW to W
  }

  result = nvmlDeviceGetPowerManagementLimit(device, &power);
  if (result == NVML_SUCCESS) {
    env->SetIntField(powerInfo, powerLimitField, (jint)(power / 1000));  // mW to W
  }

  return powerInfo;
}

static jobject populateClockInfo(JNIEnv* env, nvmlDevice_t device)
{
  jclass clockInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUClockInfo");
  if (clockInfoClass == NULL) return NULL;

  jmethodID constructor = env->GetMethodID(clockInfoClass, "<init>", "()V");
  if (constructor == NULL) return NULL;

  jobject clockInfo = env->NewObject(clockInfoClass, constructor);
  if (clockInfo == NULL) return NULL;

  jfieldID graphicsClockField = env->GetFieldID(clockInfoClass, "graphicsClockMHz", "I");
  jfieldID memoryClockField   = env->GetFieldID(clockInfoClass, "memoryClockMHz", "I");
  jfieldID smClockField       = env->GetFieldID(clockInfoClass, "smClockMHz", "I");

  unsigned int clock;
  nvmlReturn_t result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
  if (result == NVML_SUCCESS) { env->SetIntField(clockInfo, graphicsClockField, (jint)clock); }

  result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock);
  if (result == NVML_SUCCESS) { env->SetIntField(clockInfo, memoryClockField, (jint)clock); }

  result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
  if (result == NVML_SUCCESS) { env->SetIntField(clockInfo, smClockField, (jint)clock); }

  return clockInfo;
}

static jobject populateHardwareInfo(JNIEnv* env, nvmlDevice_t device)
{
  jclass hardwareInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUHardwareInfo");
  if (hardwareInfoClass == NULL) return NULL;

  jmethodID constructor = env->GetMethodID(hardwareInfoClass, "<init>", "()V");
  if (constructor == NULL) return NULL;

  jobject hardwareInfo = env->NewObject(hardwareInfoClass, constructor);
  if (hardwareInfo == NULL) return NULL;

  jfieldID smCountField = env->GetFieldID(hardwareInfoClass, "streamingMultiprocessors", "I");
  jfieldID performanceStateField = env->GetFieldID(hardwareInfoClass, "performanceState", "I");
  jfieldID fanSpeedField         = env->GetFieldID(hardwareInfoClass, "fanSpeedPercent", "I");

  unsigned int smCount = 0;
  nvmlReturn_t result  = nvmlDeviceGetNumGpuCores(device, &smCount);
  if (result == NVML_SUCCESS) { env->SetIntField(hardwareInfo, smCountField, (jint)smCount); }

  nvmlPstates_t pState;
  result = nvmlDeviceGetPerformanceState(device, &pState);
  if (result == NVML_SUCCESS) {
    env->SetIntField(hardwareInfo, performanceStateField, (jint)pState);
  }

  unsigned int fanSpeed;
  result = nvmlDeviceGetFanSpeed(device, &fanSpeed);
  if (result == NVML_SUCCESS) { env->SetIntField(hardwareInfo, fanSpeedField, (jint)fanSpeed); }

  return hardwareInfo;
}

static jobject populatePCIeInfo(JNIEnv* env, nvmlDevice_t device)
{
  jclass pcieInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUPCIeInfo");
  if (pcieInfoClass == NULL) return NULL;

  jmethodID constructor = env->GetMethodID(pcieInfoClass, "<init>", "()V");
  if (constructor == NULL) return NULL;

  jobject pcieInfo = env->NewObject(pcieInfoClass, constructor);
  if (pcieInfo == NULL) return NULL;

  jfieldID pcieLinkGenField   = env->GetFieldID(pcieInfoClass, "pcieLinkGeneration", "I");
  jfieldID pcieLinkWidthField = env->GetFieldID(pcieInfoClass, "pcieLinkWidth", "I");

  unsigned int linkGen;
  nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkGeneration(device, &linkGen);
  if (result == NVML_SUCCESS) { env->SetIntField(pcieInfo, pcieLinkGenField, (jint)linkGen); }

  unsigned int linkWidth;
  result = nvmlDeviceGetCurrPcieLinkWidth(device, &linkWidth);
  if (result == NVML_SUCCESS) { env->SetIntField(pcieInfo, pcieLinkWidthField, (jint)linkWidth); }

  return pcieInfo;
}

static jobject populateECCInfo(JNIEnv* env, nvmlDevice_t device)
{
  jclass eccInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUECCInfo");
  if (eccInfoClass == NULL) return NULL;

  jmethodID constructor = env->GetMethodID(eccInfoClass, "<init>", "()V");
  if (constructor == NULL) return NULL;

  jobject eccInfo = env->NewObject(eccInfoClass, constructor);
  if (eccInfo == NULL) return NULL;

  jfieldID eccSingleBitField = env->GetFieldID(eccInfoClass, "eccSingleBitErrors", "J");
  jfieldID eccDoubleBitField = env->GetFieldID(eccInfoClass, "eccDoubleBitErrors", "J");

  unsigned long long eccCount;
  nvmlReturn_t result =
    nvmlDeviceGetTotalEccErrors(device, NVML_SINGLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  if (result == NVML_SUCCESS) { env->SetLongField(eccInfo, eccSingleBitField, (jlong)eccCount); }

  result = nvmlDeviceGetTotalEccErrors(device, NVML_DOUBLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  if (result == NVML_SUCCESS) { env->SetLongField(eccInfo, eccDoubleBitField, (jlong)eccCount); }

  return eccInfo;
}

// Helper function to populate GPUInfo object from NVML device handle using individual helpers
static jobject populateGPUInfoFromDevice(JNIEnv* env, nvmlDevice_t device)
{
  // Get Java GPUInfo class and constructor
  jclass gpuInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUInfo");
  if (gpuInfoClass == NULL) { return NULL; }

  jmethodID constructor = env->GetMethodID(gpuInfoClass, "<init>", "()V");
  if (constructor == NULL) { return NULL; }

  // Create GPUInfo object
  jobject gpuInfo = env->NewObject(gpuInfoClass, constructor);
  if (gpuInfo == NULL) { return NULL; }

  // Populate nested info objects using individual helpers
  jobject deviceInfo      = populateDeviceInfo(env, device);
  jobject utilizationInfo = populateUtilizationInfo(env, device);
  jobject memoryInfo      = populateMemoryInfo(env, device);
  jobject temperatureInfo = populateTemperatureInfo(env, device);
  jobject powerInfo       = populatePowerInfo(env, device);
  jobject clockInfo       = populateClockInfo(env, device);
  jobject hardwareInfo    = populateHardwareInfo(env, device);
  jobject pcieInfo        = populatePCIeInfo(env, device);
  jobject eccInfo         = populateECCInfo(env, device);

  // Set nested info objects in GPUInfo
  jfieldID deviceInfoField =
    env->GetFieldID(gpuInfoClass, "deviceInfo", "Lcom/nvidia/spark/rapids/jni/nvml/GPUDeviceInfo;");
  jfieldID utilizationInfoField = env->GetFieldID(
    gpuInfoClass, "utilizationInfo", "Lcom/nvidia/spark/rapids/jni/nvml/GPUUtilizationInfo;");
  jfieldID memoryInfoField =
    env->GetFieldID(gpuInfoClass, "memoryInfo", "Lcom/nvidia/spark/rapids/jni/nvml/GPUMemoryInfo;");
  jfieldID temperatureInfoField = env->GetFieldID(
    gpuInfoClass, "temperatureInfo", "Lcom/nvidia/spark/rapids/jni/nvml/GPUTemperatureInfo;");
  jfieldID powerInfoField =
    env->GetFieldID(gpuInfoClass, "powerInfo", "Lcom/nvidia/spark/rapids/jni/nvml/GPUPowerInfo;");
  jfieldID clockInfoField =
    env->GetFieldID(gpuInfoClass, "clockInfo", "Lcom/nvidia/spark/rapids/jni/nvml/GPUClockInfo;");
  jfieldID hardwareInfoField = env->GetFieldID(
    gpuInfoClass, "hardwareInfo", "Lcom/nvidia/spark/rapids/jni/nvml/GPUHardwareInfo;");
  jfieldID pcieInfoField =
    env->GetFieldID(gpuInfoClass, "pcieInfo", "Lcom/nvidia/spark/rapids/jni/nvml/GPUPCIeInfo;");
  jfieldID eccInfoField =
    env->GetFieldID(gpuInfoClass, "eccInfo", "Lcom/nvidia/spark/rapids/jni/nvml/GPUECCInfo;");

  if (deviceInfo != NULL) env->SetObjectField(gpuInfo, deviceInfoField, deviceInfo);
  if (utilizationInfo != NULL) env->SetObjectField(gpuInfo, utilizationInfoField, utilizationInfo);
  if (memoryInfo != NULL) env->SetObjectField(gpuInfo, memoryInfoField, memoryInfo);
  if (temperatureInfo != NULL) env->SetObjectField(gpuInfo, temperatureInfoField, temperatureInfo);
  if (powerInfo != NULL) env->SetObjectField(gpuInfo, powerInfoField, powerInfo);
  if (clockInfo != NULL) env->SetObjectField(gpuInfo, clockInfoField, clockInfo);
  if (hardwareInfo != NULL) env->SetObjectField(gpuInfo, hardwareInfoField, hardwareInfo);
  if (pcieInfo != NULL) env->SetObjectField(gpuInfo, pcieInfoField, pcieInfo);
  if (eccInfo != NULL) env->SetObjectField(gpuInfo, eccInfoField, eccInfo);

  return gpuInfo;
}

extern "C" {

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlInit(JNIEnv* env,
                                                                               jclass cls)
{
  nvmlReturn_t result = nvmlInit();
  return (result == NVML_SUCCESS) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlShutdown(JNIEnv* env,
                                                                               jclass cls)
{
  nvmlShutdown();
}

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetDeviceCount(JNIEnv* env,
                                                                                     jclass cls)
{
  unsigned int deviceCount = 0;
  nvmlReturn_t result      = nvmlDeviceGetCount(&deviceCount);

  if (result != NVML_SUCCESS) { return -1; }

  return (jint)deviceCount;
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetDeviceHandleFromCudaDevice(JNIEnv* env,
                                                                             jclass cls,
                                                                             jint cudaDeviceId)
{
  char pciBuf[32];
  cudaError_t cerr = cudaDeviceGetPCIBusId(pciBuf, sizeof(pciBuf), cudaDeviceId);
  if (cerr != cudaSuccess) { return 0; }

  nvmlDevice_t device;
  nvmlReturn_t nerr = nvmlDeviceGetHandleByPciBusId(pciBuf, &device);
  if (nerr != NVML_SUCCESS) { return 0; }

  return (jlong)device;  // Return the handle as a long
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetGPUInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = (nvmlDevice_t)deviceHandle;

  // Use common helper to populate GPUInfo object
  return populateGPUInfoFromDevice(env, device);
}

JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetAllGPUInfo(JNIEnv* env, jclass cls)
{
  // Get device count
  unsigned int deviceCount = 0;
  nvmlReturn_t result      = nvmlDeviceGetCount(&deviceCount);

  if (result != NVML_SUCCESS || deviceCount == 0) {
    // Return empty array
    jclass gpuInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUInfo");
    if (gpuInfoClass == NULL) { return NULL; }
    return env->NewObjectArray(0, gpuInfoClass, NULL);
  }

  // Create array
  jclass gpuInfoClass = env->FindClass("com/nvidia/spark/rapids/jni/nvml/GPUInfo");
  if (gpuInfoClass == NULL) { return NULL; }

  jobjectArray gpuInfoArray = env->NewObjectArray((jsize)deviceCount, gpuInfoClass, NULL);
  if (gpuInfoArray == NULL) { return NULL; }

  // Fill array with GPU info
  for (unsigned int i = 0; i < deviceCount; i++) {
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(i, &device);
    if (result == NVML_SUCCESS) {
      jobject gpuInfo = populateGPUInfoFromDevice(env, device);
      if (gpuInfo != NULL) {
        env->SetObjectArrayElement(gpuInfoArray, (jsize)i, gpuInfo);
        env->DeleteLocalRef(gpuInfo);
      }
    }
  }

  return gpuInfoArray;
}

// Individual info JNI functions using device handles
JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetDeviceInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = (nvmlDevice_t)deviceHandle;
  return populateDeviceInfo(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetUtilizationInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = (nvmlDevice_t)deviceHandle;
  return populateUtilizationInfo(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetMemoryInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = (nvmlDevice_t)deviceHandle;
  return populateMemoryInfo(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetTemperatureInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = (nvmlDevice_t)deviceHandle;
  return populateTemperatureInfo(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetPowerInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = (nvmlDevice_t)deviceHandle;
  return populatePowerInfo(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetClockInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = (nvmlDevice_t)deviceHandle;
  return populateClockInfo(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetHardwareInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = (nvmlDevice_t)deviceHandle;
  return populateHardwareInfo(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetPCIeInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = (nvmlDevice_t)deviceHandle;
  return populatePCIeInfo(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetECCInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = (nvmlDevice_t)deviceHandle;
  return populateECCInfo(env, device);
}

}  // extern "C"
