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
#include <stdio.h>

#include <cstdint>

// NVML JNI implementation with comprehensive GPU metrics for Spark Rapids

#define NVML_CLASS_PATH "com/nvidia/spark/rapids/jni/nvml/"

// C++ struct to hold NVML operation results
struct NVMLResult {
  nvmlReturn_t returnCode;
  jobject data;
};

// Helper functions for individual NVML API groups
namespace {

// Helper function to create NVMLResult Java object from C++ NVMLResult
jobject create_nvml_result(JNIEnv* env, NVMLResult cppResult)
{
  // Create NVMLResult object using default constructor
  jclass nvmlResultClass = env->FindClass(NVML_CLASS_PATH "NVMLResult");
  if (nvmlResultClass == nullptr) return nullptr;

  jmethodID constructor = env->GetMethodID(nvmlResultClass, "<init>", "()V");
  if (constructor == nullptr) return nullptr;

  jobject nvmlResult = env->NewObject(nvmlResultClass, constructor);
  if (nvmlResult == nullptr) return nullptr;

  // Set the return code and data fields directly
  jfieldID returnCodeField = env->GetFieldID(nvmlResultClass, "returnCode", "I");
  jfieldID dataField       = env->GetFieldID(nvmlResultClass, "data", "Ljava/lang/Object;");

  env->SetIntField(nvmlResult, returnCodeField, static_cast<jint>(cppResult.returnCode));
  env->SetObjectField(nvmlResult, dataField, cppResult.data);

  return nvmlResult;
}

jobject create_object(JNIEnv* env, char const* class_name, char const* ctor_sig)
{
  jclass j_class = env->FindClass(class_name);
  if (j_class == nullptr) { return nullptr; }

  jmethodID ctor = env->GetMethodID(j_class, "<init>", ctor_sig);
  if (ctor == nullptr) { return nullptr; }

  return env->NewObject(j_class, ctor);
}

NVMLResult populate_device_info(JNIEnv* env, nvmlDevice_t device)
{
  NVMLResult result;
  result.returnCode = NVML_ERROR_UNKNOWN;
  result.data       = nullptr;

  jobject deviceInfo = create_object(env, NVML_CLASS_PATH "GPUDeviceInfo", "()V");
  if (deviceInfo == nullptr) {
    result.returnCode = NVML_ERROR_UNKNOWN;
    return result;
  }

  jclass deviceInfoClass = env->GetObjectClass(deviceInfo);

  jfieldID nameField  = env->GetFieldID(deviceInfoClass, "name", "Ljava/lang/String;");
  jfieldID brandField = env->GetFieldID(deviceInfoClass, "brand", "Ljava/lang/String;");

  char name[NVML_DEVICE_NAME_BUFFER_SIZE];
  nvmlReturn_t nvmlResult = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
  result.returnCode       = nvmlResult;
  result.data             = deviceInfo;

  if (nvmlResult == NVML_SUCCESS) {
    jstring jname = env->NewStringUTF(name);
    env->SetObjectField(deviceInfo, nameField, jname);
    env->DeleteLocalRef(jname);
  }

  nvmlBrandType_t brandType;
  nvmlResult = nvmlDeviceGetBrand(device, &brandType);
  if (nvmlResult == NVML_SUCCESS) {
    char brand[50];
    snprintf(brand, sizeof(brand), "Brand_%d", static_cast<int>(brandType));
    jstring jbrand = env->NewStringUTF(brand);
    env->SetObjectField(deviceInfo, brandField, jbrand);
    env->DeleteLocalRef(jbrand);
  }

  return result;
}

NVMLResult populate_utilization_info(JNIEnv* env, nvmlDevice_t device)
{
  NVMLResult result;
  result.returnCode = NVML_ERROR_UNKNOWN;
  result.data       = nullptr;

  jobject utilizationInfo = create_object(env, NVML_CLASS_PATH "GPUUtilizationInfo", "()V");
  if (utilizationInfo == nullptr) {
    result.returnCode = NVML_ERROR_UNKNOWN;
    return result;
  }

  jclass utilizationInfoClass = env->GetObjectClass(utilizationInfo);

  jfieldID gpuUtilField = env->GetFieldID(utilizationInfoClass, "gpuUtilization", "I");
  jfieldID memUtilField = env->GetFieldID(utilizationInfoClass, "memoryUtilization", "I");

  nvmlUtilization_t utilization;
  nvmlReturn_t nvmlResult = nvmlDeviceGetUtilizationRates(device, &utilization);
  result.returnCode       = nvmlResult;
  result.data             = utilizationInfo;

  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(utilizationInfo, gpuUtilField, static_cast<jint>(utilization.gpu));
    env->SetIntField(utilizationInfo, memUtilField, static_cast<jint>(utilization.memory));
  }

  return result;
}

NVMLResult populate_memory_info(JNIEnv* env, nvmlDevice_t device)
{
  NVMLResult result;
  result.returnCode = NVML_ERROR_UNKNOWN;
  result.data       = nullptr;

  jobject memoryInfo = create_object(env, NVML_CLASS_PATH "GPUMemoryInfo", "()V");
  if (memoryInfo == nullptr) {
    result.returnCode = NVML_ERROR_UNKNOWN;
    return result;
  }

  jclass memoryInfoClass = env->GetObjectClass(memoryInfo);

  jfieldID memUsedField  = env->GetFieldID(memoryInfoClass, "memoryUsedMB", "J");
  jfieldID memTotalField = env->GetFieldID(memoryInfoClass, "memoryTotalMB", "J");
  jfieldID memFreeField  = env->GetFieldID(memoryInfoClass, "memoryFreeMB", "J");

  nvmlMemory_t memory;
  nvmlReturn_t nvmlResult = nvmlDeviceGetMemoryInfo(device, &memory);
  result.returnCode       = nvmlResult;
  result.data             = memoryInfo;

  if (nvmlResult == NVML_SUCCESS) {
    env->SetLongField(memoryInfo, memUsedField, static_cast<jlong>(memory.used / (1024 * 1024)));
    env->SetLongField(memoryInfo, memTotalField, static_cast<jlong>(memory.total / (1024 * 1024)));
    env->SetLongField(memoryInfo, memFreeField, static_cast<jlong>(memory.free / (1024 * 1024)));
  }

  return result;
}

NVMLResult populate_temperature_info(JNIEnv* env, nvmlDevice_t device)
{
  NVMLResult result;
  result.returnCode = NVML_ERROR_UNKNOWN;
  result.data       = nullptr;

  jobject temperatureInfo = create_object(env, NVML_CLASS_PATH "GPUTemperatureInfo", "()V");
  if (temperatureInfo == nullptr) {
    result.returnCode = NVML_ERROR_UNKNOWN;
    return result;
  }

  jclass temperatureInfoClass = env->GetObjectClass(temperatureInfo);

  jfieldID tempGpuField = env->GetFieldID(temperatureInfoClass, "temperatureGpu", "I");

  unsigned int temp;
  nvmlReturn_t nvmlResult = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
  result.returnCode       = nvmlResult;
  result.data             = temperatureInfo;

  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(temperatureInfo, tempGpuField, static_cast<jint>(temp));
  }

  return result;
}

NVMLResult populate_power_info(JNIEnv* env, nvmlDevice_t device)
{
  NVMLResult result;
  result.returnCode = NVML_ERROR_UNKNOWN;
  result.data       = nullptr;

  jobject powerInfo = create_object(env, NVML_CLASS_PATH "GPUPowerInfo", "()V");
  if (powerInfo == nullptr) {
    result.returnCode = NVML_ERROR_UNKNOWN;
    return result;
  }

  jclass powerInfoClass = env->GetObjectClass(powerInfo);

  jfieldID powerUsageField = env->GetFieldID(powerInfoClass, "powerUsageW", "I");
  jfieldID powerLimitField = env->GetFieldID(powerInfoClass, "powerLimitW", "I");

  unsigned int power;
  nvmlReturn_t nvmlResult = nvmlDeviceGetPowerUsage(device, &power);
  result.returnCode       = nvmlResult;
  result.data             = powerInfo;

  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(powerInfo, powerUsageField, static_cast<jint>(power / 1000));  // mW to W
  }

  nvmlResult = nvmlDeviceGetPowerManagementLimit(device, &power);
  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(powerInfo, powerLimitField, static_cast<jint>(power / 1000));  // mW to W
  }

  return result;
}

NVMLResult populate_clock_info(JNIEnv* env, nvmlDevice_t device)
{
  NVMLResult result;
  result.returnCode = NVML_ERROR_UNKNOWN;
  result.data       = nullptr;

  jobject clockInfo = create_object(env, NVML_CLASS_PATH "GPUClockInfo", "()V");
  if (clockInfo == nullptr) {
    result.returnCode = NVML_ERROR_UNKNOWN;
    return result;
  }

  jclass clockInfoClass = env->GetObjectClass(clockInfo);

  jfieldID graphicsClockField = env->GetFieldID(clockInfoClass, "graphicsClockMHz", "I");
  jfieldID memoryClockField   = env->GetFieldID(clockInfoClass, "memoryClockMHz", "I");
  jfieldID smClockField       = env->GetFieldID(clockInfoClass, "smClockMHz", "I");

  unsigned int clock;
  nvmlReturn_t nvmlResult = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
  result.returnCode       = nvmlResult;
  result.data             = clockInfo;

  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(clockInfo, graphicsClockField, static_cast<jint>(clock));
  }

  nvmlResult = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock);
  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(clockInfo, memoryClockField, static_cast<jint>(clock));
  }

  nvmlResult = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(clockInfo, smClockField, static_cast<jint>(clock));
  }

  return result;
}

NVMLResult populate_hardware_info(JNIEnv* env, nvmlDevice_t device)
{
  NVMLResult result;
  result.returnCode = NVML_ERROR_UNKNOWN;
  result.data       = nullptr;

  jobject hardwareInfo = create_object(env, NVML_CLASS_PATH "GPUHardwareInfo", "()V");
  if (hardwareInfo == nullptr) {
    result.returnCode = NVML_ERROR_UNKNOWN;
    return result;
  }

  jclass hardwareInfoClass = env->GetObjectClass(hardwareInfo);

  jfieldID smCountField = env->GetFieldID(hardwareInfoClass, "streamingMultiprocessors", "I");
  jfieldID performanceStateField = env->GetFieldID(hardwareInfoClass, "performanceState", "I");
  jfieldID fanSpeedField         = env->GetFieldID(hardwareInfoClass, "fanSpeedPercent", "I");

  unsigned int smCount    = 0;
  nvmlReturn_t nvmlResult = nvmlDeviceGetNumGpuCores(device, &smCount);
  result.returnCode       = nvmlResult;
  result.data             = hardwareInfo;

  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(hardwareInfo, smCountField, static_cast<jint>(smCount));
  }

  nvmlPstates_t pState;
  nvmlResult = nvmlDeviceGetPerformanceState(device, &pState);
  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(hardwareInfo, performanceStateField, static_cast<jint>(pState));
  }

  unsigned int fanSpeed;
  nvmlResult = nvmlDeviceGetFanSpeed(device, &fanSpeed);
  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(hardwareInfo, fanSpeedField, static_cast<jint>(fanSpeed));
  }

  return result;
}

NVMLResult populate_pcie_info(JNIEnv* env, nvmlDevice_t device)
{
  NVMLResult result;
  result.returnCode = NVML_ERROR_UNKNOWN;
  result.data       = nullptr;

  jobject pcieInfo = create_object(env, NVML_CLASS_PATH "GPUPCIeInfo", "()V");
  if (pcieInfo == nullptr) {
    result.returnCode = NVML_ERROR_UNKNOWN;
    return result;
  }

  jclass pcieInfoClass = env->GetObjectClass(pcieInfo);

  jfieldID pcieLinkGenField   = env->GetFieldID(pcieInfoClass, "pcieLinkGeneration", "I");
  jfieldID pcieLinkWidthField = env->GetFieldID(pcieInfoClass, "pcieLinkWidth", "I");

  unsigned int linkGen;
  nvmlReturn_t nvmlResult = nvmlDeviceGetCurrPcieLinkGeneration(device, &linkGen);
  result.returnCode       = nvmlResult;
  result.data             = pcieInfo;

  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(pcieInfo, pcieLinkGenField, static_cast<jint>(linkGen));
  }

  unsigned int linkWidth;
  nvmlResult = nvmlDeviceGetCurrPcieLinkWidth(device, &linkWidth);
  if (nvmlResult == NVML_SUCCESS) {
    env->SetIntField(pcieInfo, pcieLinkWidthField, static_cast<jint>(linkWidth));
  }

  return result;
}

NVMLResult populate_ecc_info(JNIEnv* env, nvmlDevice_t device)
{
  NVMLResult result;
  result.returnCode = NVML_ERROR_UNKNOWN;
  result.data       = nullptr;

  jobject eccInfo = create_object(env, NVML_CLASS_PATH "GPUECCInfo", "()V");
  if (eccInfo == nullptr) {
    result.returnCode = NVML_ERROR_UNKNOWN;
    return result;
  }

  jclass eccInfoClass = env->GetObjectClass(eccInfo);

  jfieldID eccSingleBitField = env->GetFieldID(eccInfoClass, "eccSingleBitErrors", "J");
  jfieldID eccDoubleBitField = env->GetFieldID(eccInfoClass, "eccDoubleBitErrors", "J");

  unsigned long long eccCount;
  nvmlReturn_t nvmlResult =
    nvmlDeviceGetTotalEccErrors(device, NVML_SINGLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  result.returnCode = nvmlResult;
  result.data       = eccInfo;

  if (nvmlResult == NVML_SUCCESS) {
    env->SetLongField(eccInfo, eccSingleBitField, static_cast<jlong>(eccCount));
  }

  nvmlResult =
    nvmlDeviceGetTotalEccErrors(device, NVML_DOUBLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  if (nvmlResult == NVML_SUCCESS) {
    env->SetLongField(eccInfo, eccDoubleBitField, static_cast<jlong>(eccCount));
  }

  return result;
}

// Helper function to populate GPUInfo object from NVML device handle using individual helpers
NVMLResult populate_gpu_info_from_device(JNIEnv* env, nvmlDevice_t device)
{
  NVMLResult result;
  result.returnCode = NVML_SUCCESS;  // Start with success, track first error
  result.data       = nullptr;

  // Create GPUInfo object
  jobject gpuInfo = create_object(env, NVML_CLASS_PATH "GPUInfo", "()V");
  if (gpuInfo == nullptr) {
    result.returnCode = NVML_ERROR_UNKNOWN;
    return result;
  }

  result.data         = gpuInfo;
  jclass gpuInfoClass = env->GetObjectClass(gpuInfo);

  // Populate nested info objects using individual helpers
  NVMLResult deviceResult      = populate_device_info(env, device);
  NVMLResult utilizationResult = populate_utilization_info(env, device);
  NVMLResult memoryResult      = populate_memory_info(env, device);
  NVMLResult temperatureResult = populate_temperature_info(env, device);
  NVMLResult powerResult       = populate_power_info(env, device);
  NVMLResult clockResult       = populate_clock_info(env, device);
  NVMLResult hardwareResult    = populate_hardware_info(env, device);
  NVMLResult pcieResult        = populate_pcie_info(env, device);
  NVMLResult eccResult         = populate_ecc_info(env, device);

  // Track the first error encountered
  if (result.returnCode == NVML_SUCCESS && deviceResult.returnCode != NVML_SUCCESS)
    result.returnCode = deviceResult.returnCode;
  if (result.returnCode == NVML_SUCCESS && utilizationResult.returnCode != NVML_SUCCESS)
    result.returnCode = utilizationResult.returnCode;
  if (result.returnCode == NVML_SUCCESS && memoryResult.returnCode != NVML_SUCCESS)
    result.returnCode = memoryResult.returnCode;
  if (result.returnCode == NVML_SUCCESS && temperatureResult.returnCode != NVML_SUCCESS)
    result.returnCode = temperatureResult.returnCode;
  if (result.returnCode == NVML_SUCCESS && powerResult.returnCode != NVML_SUCCESS)
    result.returnCode = powerResult.returnCode;
  if (result.returnCode == NVML_SUCCESS && clockResult.returnCode != NVML_SUCCESS)
    result.returnCode = clockResult.returnCode;
  if (result.returnCode == NVML_SUCCESS && hardwareResult.returnCode != NVML_SUCCESS)
    result.returnCode = hardwareResult.returnCode;
  if (result.returnCode == NVML_SUCCESS && pcieResult.returnCode != NVML_SUCCESS)
    result.returnCode = pcieResult.returnCode;
  if (result.returnCode == NVML_SUCCESS && eccResult.returnCode != NVML_SUCCESS)
    result.returnCode = eccResult.returnCode;

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

  if (deviceResult.data != nullptr)
    env->SetObjectField(gpuInfo, deviceInfoField, deviceResult.data);
  if (utilizationResult.data != nullptr)
    env->SetObjectField(gpuInfo, utilizationInfoField, utilizationResult.data);
  if (memoryResult.data != nullptr)
    env->SetObjectField(gpuInfo, memoryInfoField, memoryResult.data);
  if (temperatureResult.data != nullptr)
    env->SetObjectField(gpuInfo, temperatureInfoField, temperatureResult.data);
  if (powerResult.data != nullptr) env->SetObjectField(gpuInfo, powerInfoField, powerResult.data);
  if (clockResult.data != nullptr) env->SetObjectField(gpuInfo, clockInfoField, clockResult.data);
  if (hardwareResult.data != nullptr)
    env->SetObjectField(gpuInfo, hardwareInfoField, hardwareResult.data);
  if (pcieResult.data != nullptr) env->SetObjectField(gpuInfo, pcieInfoField, pcieResult.data);
  if (eccResult.data != nullptr) env->SetObjectField(gpuInfo, eccInfoField, eccResult.data);

  return result;
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

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetDeviceHandleFromUUID(
  JNIEnv* env, jclass cls, jbyteArray uuid)
{
  if (uuid == nullptr) { return 0; }

  jsize uuidLen = env->GetArrayLength(uuid);
  if (uuidLen != 16) { return 0; }  // UUID should be 16 bytes

  // Get the UUID bytes from Java (raw binary format from cudaDeviceProp.uuid)
  jbyte* uuidBytes = env->GetByteArrayElements(uuid, nullptr);
  if (uuidBytes == nullptr) { return 0; }

  // Convert binary UUID to string format: "GPU-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
  char uuidStr[64];
  snprintf(uuidStr,
           sizeof(uuidStr),
           "GPU-%02hhx%02hhx%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx%"
           "02hhx%02hhx%02hhx%02hhx",
           uuidBytes[0],
           uuidBytes[1],
           uuidBytes[2],
           uuidBytes[3],
           uuidBytes[4],
           uuidBytes[5],
           uuidBytes[6],
           uuidBytes[7],
           uuidBytes[8],
           uuidBytes[9],
           uuidBytes[10],
           uuidBytes[11],
           uuidBytes[12],
           uuidBytes[13],
           uuidBytes[14],
           uuidBytes[15]);

  env->ReleaseByteArrayElements(uuid, uuidBytes, JNI_ABORT);

  // Get device handle by UUID string
  nvmlDevice_t device;
  nvmlReturn_t nerr = nvmlDeviceGetHandleByUUID(uuidStr, &device);

  if (nerr != NVML_SUCCESS) { return 0; }

  return static_cast<jlong>(
    reinterpret_cast<std::intptr_t>(device));  // Return the handle as a long
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetGPUInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));

  // Use common helper to populate GPUInfo object
  NVMLResult result = populate_gpu_info_from_device(env, device);
  return create_nvml_result(env, result);
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
      NVMLResult gpuInfoResult = populate_gpu_info_from_device(env, device);
      if (gpuInfoResult.data != nullptr) {
        env->SetObjectArrayElement(gpuInfoArray, static_cast<jsize>(i), gpuInfoResult.data);
        env->DeleteLocalRef(gpuInfoResult.data);
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
  NVMLResult result   = populate_device_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetUtilizationInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  NVMLResult result   = populate_utilization_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetMemoryInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  NVMLResult result   = populate_memory_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetTemperatureInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  NVMLResult result   = populate_temperature_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetPowerInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  NVMLResult result   = populate_power_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetClockInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  NVMLResult result   = populate_clock_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetHardwareInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  NVMLResult result   = populate_hardware_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetPCIeInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  NVMLResult result   = populate_pcie_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetECCInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  NVMLResult result   = populate_ecc_info(env, device);
  return create_nvml_result(env, result);
}

}  // extern "C"
