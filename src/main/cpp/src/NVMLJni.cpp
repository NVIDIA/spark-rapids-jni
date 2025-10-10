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

#include <cstdint>

// NVML JNI implementation with comprehensive GPU metrics for Spark Rapids

#define NVML_CLASS_PATH "com/nvidia/spark/rapids/jni/nvml/"

// Helper functions for individual NVML API groups
namespace {

jobject create_object(JNIEnv* env, char const* class_name, char const* ctor_sig)
{
  jclass j_class = env->FindClass(class_name);
  if (j_class == nullptr) { return nullptr; }

  jmethodID ctor = env->GetMethodID(j_class, "<init>", ctor_sig);
  if (ctor == nullptr) { return nullptr; }

  return env->NewObject(j_class, ctor);
}

jobject populate_device_info(JNIEnv* env, nvmlDevice_t device)
{
  jobject deviceInfo = create_object(env, NVML_CLASS_PATH "GPUDeviceInfo", "()V");
  if (deviceInfo == nullptr) return nullptr;

  jclass deviceInfoClass = env->GetObjectClass(deviceInfo);

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
    snprintf(brand, sizeof(brand), "Brand_%d", static_cast<int>(brandType));
    jstring jbrand = env->NewStringUTF(brand);
    env->SetObjectField(deviceInfo, brandField, jbrand);
    env->DeleteLocalRef(jbrand);
  }

  return deviceInfo;
}

jobject populate_utilization_info(JNIEnv* env, nvmlDevice_t device)
{
  jobject utilizationInfo = create_object(env, NVML_CLASS_PATH "GPUUtilizationInfo", "()V");
  if (utilizationInfo == nullptr) return nullptr;

  jclass utilizationInfoClass = env->GetObjectClass(utilizationInfo);

  jfieldID gpuUtilField = env->GetFieldID(utilizationInfoClass, "gpuUtilization", "I");
  jfieldID memUtilField = env->GetFieldID(utilizationInfoClass, "memoryUtilization", "I");

  nvmlUtilization_t utilization;
  nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);
  if (result == NVML_SUCCESS) {
    env->SetIntField(utilizationInfo, gpuUtilField, static_cast<jint>(utilization.gpu));
    env->SetIntField(utilizationInfo, memUtilField, static_cast<jint>(utilization.memory));
  }

  return utilizationInfo;
}

jobject populate_memory_info(JNIEnv* env, nvmlDevice_t device)
{
  jobject memoryInfo = create_object(env, NVML_CLASS_PATH "GPUMemoryInfo", "()V");
  if (memoryInfo == nullptr) return nullptr;

  jclass memoryInfoClass = env->GetObjectClass(memoryInfo);

  jfieldID memUsedField  = env->GetFieldID(memoryInfoClass, "memoryUsedMB", "J");
  jfieldID memTotalField = env->GetFieldID(memoryInfoClass, "memoryTotalMB", "J");
  jfieldID memFreeField  = env->GetFieldID(memoryInfoClass, "memoryFreeMB", "J");

  nvmlMemory_t memory;
  nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device, &memory);
  if (result == NVML_SUCCESS) {
    env->SetLongField(memoryInfo, memUsedField, static_cast<jlong>(memory.used / (1024 * 1024)));
    env->SetLongField(memoryInfo, memTotalField, static_cast<jlong>(memory.total / (1024 * 1024)));
    env->SetLongField(memoryInfo, memFreeField, static_cast<jlong>(memory.free / (1024 * 1024)));
  }

  return memoryInfo;
}

jobject populate_temperature_info(JNIEnv* env, nvmlDevice_t device)
{
  jobject temperatureInfo = create_object(env, NVML_CLASS_PATH "GPUTemperatureInfo", "()V");
  if (temperatureInfo == nullptr) return nullptr;

  jclass temperatureInfoClass = env->GetObjectClass(temperatureInfo);

  jfieldID tempGpuField = env->GetFieldID(temperatureInfoClass, "temperatureGpu", "I");

  unsigned int temp;
  nvmlReturn_t result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
  if (result == NVML_SUCCESS) {
    env->SetIntField(temperatureInfo, tempGpuField, static_cast<jint>(temp));
  }

  return temperatureInfo;
}

jobject populate_power_info(JNIEnv* env, nvmlDevice_t device)
{
  jobject powerInfo = create_object(env, NVML_CLASS_PATH "GPUPowerInfo", "()V");
  if (powerInfo == nullptr) return nullptr;

  jclass powerInfoClass = env->GetObjectClass(powerInfo);

  jfieldID powerUsageField = env->GetFieldID(powerInfoClass, "powerUsageW", "I");
  jfieldID powerLimitField = env->GetFieldID(powerInfoClass, "powerLimitW", "I");

  unsigned int power;
  nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &power);
  if (result == NVML_SUCCESS) {
    env->SetIntField(powerInfo, powerUsageField, static_cast<jint>(power / 1000));  // mW to W
  }

  result = nvmlDeviceGetPowerManagementLimit(device, &power);
  if (result == NVML_SUCCESS) {
    env->SetIntField(powerInfo, powerLimitField, static_cast<jint>(power / 1000));  // mW to W
  }

  return powerInfo;
}

jobject populate_clock_info(JNIEnv* env, nvmlDevice_t device)
{
  jobject clockInfo = create_object(env, NVML_CLASS_PATH "GPUClockInfo", "()V");
  if (clockInfo == nullptr) return nullptr;

  jclass clockInfoClass = env->GetObjectClass(clockInfo);

  jfieldID graphicsClockField = env->GetFieldID(clockInfoClass, "graphicsClockMHz", "I");
  jfieldID memoryClockField   = env->GetFieldID(clockInfoClass, "memoryClockMHz", "I");
  jfieldID smClockField       = env->GetFieldID(clockInfoClass, "smClockMHz", "I");

  unsigned int clock;
  nvmlReturn_t result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
  if (result == NVML_SUCCESS) {
    env->SetIntField(clockInfo, graphicsClockField, static_cast<jint>(clock));
  }

  result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock);
  if (result == NVML_SUCCESS) {
    env->SetIntField(clockInfo, memoryClockField, static_cast<jint>(clock));
  }

  result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
  if (result == NVML_SUCCESS) {
    env->SetIntField(clockInfo, smClockField, static_cast<jint>(clock));
  }

  return clockInfo;
}

jobject populate_hardware_info(JNIEnv* env, nvmlDevice_t device)
{
  jobject hardwareInfo = create_object(env, NVML_CLASS_PATH "GPUHardwareInfo", "()V");
  if (hardwareInfo == nullptr) return nullptr;

  jclass hardwareInfoClass = env->GetObjectClass(hardwareInfo);

  jfieldID smCountField = env->GetFieldID(hardwareInfoClass, "streamingMultiprocessors", "I");
  jfieldID performanceStateField = env->GetFieldID(hardwareInfoClass, "performanceState", "I");
  jfieldID fanSpeedField         = env->GetFieldID(hardwareInfoClass, "fanSpeedPercent", "I");

  unsigned int smCount = 0;
  nvmlReturn_t result  = nvmlDeviceGetNumGpuCores(device, &smCount);
  if (result == NVML_SUCCESS) {
    env->SetIntField(hardwareInfo, smCountField, static_cast<jint>(smCount));
  }

  nvmlPstates_t pState;
  result = nvmlDeviceGetPerformanceState(device, &pState);
  if (result == NVML_SUCCESS) {
    env->SetIntField(hardwareInfo, performanceStateField, static_cast<jint>(pState));
  }

  unsigned int fanSpeed;
  result = nvmlDeviceGetFanSpeed(device, &fanSpeed);
  if (result == NVML_SUCCESS) {
    env->SetIntField(hardwareInfo, fanSpeedField, static_cast<jint>(fanSpeed));
  }

  return hardwareInfo;
}

jobject populate_pcie_info(JNIEnv* env, nvmlDevice_t device)
{
  jobject pcieInfo = create_object(env, NVML_CLASS_PATH "GPUPCIeInfo", "()V");
  if (pcieInfo == nullptr) return nullptr;

  jclass pcieInfoClass = env->GetObjectClass(pcieInfo);

  jfieldID pcieLinkGenField   = env->GetFieldID(pcieInfoClass, "pcieLinkGeneration", "I");
  jfieldID pcieLinkWidthField = env->GetFieldID(pcieInfoClass, "pcieLinkWidth", "I");

  unsigned int linkGen;
  nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkGeneration(device, &linkGen);
  if (result == NVML_SUCCESS) {
    env->SetIntField(pcieInfo, pcieLinkGenField, static_cast<jint>(linkGen));
  }

  unsigned int linkWidth;
  result = nvmlDeviceGetCurrPcieLinkWidth(device, &linkWidth);
  if (result == NVML_SUCCESS) {
    env->SetIntField(pcieInfo, pcieLinkWidthField, static_cast<jint>(linkWidth));
  }

  return pcieInfo;
}

jobject populate_ecc_info(JNIEnv* env, nvmlDevice_t device)
{
  jobject eccInfo = create_object(env, NVML_CLASS_PATH "GPUECCInfo", "()V");
  if (eccInfo == nullptr) return nullptr;

  jclass eccInfoClass = env->GetObjectClass(eccInfo);

  jfieldID eccSingleBitField = env->GetFieldID(eccInfoClass, "eccSingleBitErrors", "J");
  jfieldID eccDoubleBitField = env->GetFieldID(eccInfoClass, "eccDoubleBitErrors", "J");

  unsigned long long eccCount;
  nvmlReturn_t result =
    nvmlDeviceGetTotalEccErrors(device, NVML_SINGLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  if (result == NVML_SUCCESS) {
    env->SetLongField(eccInfo, eccSingleBitField, static_cast<jlong>(eccCount));
  }

  result = nvmlDeviceGetTotalEccErrors(device, NVML_DOUBLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  if (result == NVML_SUCCESS) {
    env->SetLongField(eccInfo, eccDoubleBitField, static_cast<jlong>(eccCount));
  }

  return eccInfo;
}

// Helper function to populate GPUInfo object from NVML device handle using individual helpers
jobject populate_gpu_info_from_device(JNIEnv* env, nvmlDevice_t device)
{
  // Create GPUInfo object
  jobject gpuInfo = create_object(env, NVML_CLASS_PATH "GPUInfo", "()V");
  if (gpuInfo == nullptr) { return nullptr; }

  jclass gpuInfoClass = env->GetObjectClass(gpuInfo);

  // Populate nested info objects using individual helpers
  jobject deviceInfo      = populate_device_info(env, device);
  jobject utilizationInfo = populate_utilization_info(env, device);
  jobject memoryInfo      = populate_memory_info(env, device);
  jobject temperatureInfo = populate_temperature_info(env, device);
  jobject powerInfo       = populate_power_info(env, device);
  jobject clockInfo       = populate_clock_info(env, device);
  jobject hardwareInfo    = populate_hardware_info(env, device);
  jobject pcieInfo        = populate_pcie_info(env, device);
  jobject eccInfo         = populate_ecc_info(env, device);

  // Set nested info objects in GPUInfo
  jfieldID deviceInfoField =
    env->GetFieldID(gpuInfoClass, "deviceInfo", "L" NVML_CLASS_PATH "GPUDeviceInfo;");
  jfieldID utilizationInfoField =
    env->GetFieldID(gpuInfoClass, "utilizationInfo", "L" NVML_CLASS_PATH "GPUUtilizationInfo;");
  jfieldID memoryInfoField =
    env->GetFieldID(gpuInfoClass, "memoryInfo", "L" NVML_CLASS_PATH "GPUMemoryInfo;");
  jfieldID temperatureInfoField =
    env->GetFieldID(gpuInfoClass, "temperatureInfo", "L" NVML_CLASS_PATH "GPUTemperatureInfo;");
  jfieldID powerInfoField =
    env->GetFieldID(gpuInfoClass, "powerInfo", "L" NVML_CLASS_PATH "GPUPowerInfo;");
  jfieldID clockInfoField =
    env->GetFieldID(gpuInfoClass, "clockInfo", "L" NVML_CLASS_PATH "GPUClockInfo;");
  jfieldID hardwareInfoField =
    env->GetFieldID(gpuInfoClass, "hardwareInfo", "L" NVML_CLASS_PATH "GPUHardwareInfo;");
  jfieldID pcieInfoField =
    env->GetFieldID(gpuInfoClass, "pcieInfo", "L" NVML_CLASS_PATH "GPUPCIeInfo;");
  jfieldID eccInfoField =
    env->GetFieldID(gpuInfoClass, "eccInfo", "L" NVML_CLASS_PATH "GPUECCInfo;");

  if (deviceInfo != nullptr) env->SetObjectField(gpuInfo, deviceInfoField, deviceInfo);
  if (utilizationInfo != nullptr)
    env->SetObjectField(gpuInfo, utilizationInfoField, utilizationInfo);
  if (memoryInfo != nullptr) env->SetObjectField(gpuInfo, memoryInfoField, memoryInfo);
  if (temperatureInfo != nullptr)
    env->SetObjectField(gpuInfo, temperatureInfoField, temperatureInfo);
  if (powerInfo != nullptr) env->SetObjectField(gpuInfo, powerInfoField, powerInfo);
  if (clockInfo != nullptr) env->SetObjectField(gpuInfo, clockInfoField, clockInfo);
  if (hardwareInfo != nullptr) env->SetObjectField(gpuInfo, hardwareInfoField, hardwareInfo);
  if (pcieInfo != nullptr) env->SetObjectField(gpuInfo, pcieInfoField, pcieInfo);
  if (eccInfo != nullptr) env->SetObjectField(gpuInfo, eccInfoField, eccInfo);

  return gpuInfo;
}

}  // namespace

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

  return static_cast<jint>(deviceCount);
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

  return static_cast<jlong>(
    reinterpret_cast<std::intptr_t>(device));  // Return the handle as a long
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetGPUInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));

  // Use common helper to populate GPUInfo object
  return populate_gpu_info_from_device(env, device);
}

JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetAllGPUInfo(JNIEnv* env, jclass cls)
{
  // Get device count
  unsigned int deviceCount = 0;
  nvmlReturn_t result      = nvmlDeviceGetCount(&deviceCount);

  if (result != NVML_SUCCESS || deviceCount == 0) {
    // Return empty array
    jclass gpuInfoClass = env->FindClass(NVML_CLASS_PATH "GPUInfo");
    if (gpuInfoClass == nullptr) { return nullptr; }
    return env->NewObjectArray(0, gpuInfoClass, nullptr);
  }

  // Create array
  jclass gpuInfoClass = env->FindClass(NVML_CLASS_PATH "GPUInfo");
  if (gpuInfoClass == nullptr) { return nullptr; }

  jobjectArray gpuInfoArray =
    env->NewObjectArray(static_cast<jsize>(deviceCount), gpuInfoClass, nullptr);
  if (gpuInfoArray == nullptr) { return nullptr; }

  // Fill array with GPU info
  for (unsigned int i = 0; i < deviceCount; i++) {
    nvmlDevice_t device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex(i, &device);
    if (result == NVML_SUCCESS) {
      jobject gpuInfo = populate_gpu_info_from_device(env, device);
      if (gpuInfo != nullptr) {
        env->SetObjectArrayElement(gpuInfoArray, static_cast<jsize>(i), gpuInfo);
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
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  return populate_device_info(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetUtilizationInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  return populate_utilization_info(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetMemoryInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  return populate_memory_info(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetTemperatureInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  return populate_temperature_info(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetPowerInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  return populate_power_info(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetClockInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  return populate_clock_info(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetHardwareInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  return populate_hardware_info(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetPCIeInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  return populate_pcie_info(env, device);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetECCInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  return populate_ecc_info(env, device);
}

}  // extern "C"
