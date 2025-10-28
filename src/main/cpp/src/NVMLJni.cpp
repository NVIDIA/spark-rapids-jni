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
#include <dlfcn.h>

#include <cstdint>

// NVML JNI implementation with comprehensive GPU metrics for Spark Rapids

#define NVML_CLASS_PATH "com/nvidia/spark/rapids/jni/nvml/"

// Dynamic loading of NVML library
namespace {
void* nvml_handle = nullptr;

// Function pointer types for NVML functions
typedef nvmlReturn_t (*nvmlInit_func)();
typedef nvmlReturn_t (*nvmlShutdown_func)();
typedef nvmlReturn_t (*nvmlDeviceGetCount_func)(unsigned int* deviceCount);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_func)(unsigned int index, nvmlDevice_t* device);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByUUID_func)(const char* uuid, nvmlDevice_t* device);
typedef nvmlReturn_t (*nvmlDeviceGetName_func)(nvmlDevice_t device, char* name, unsigned int length);
typedef nvmlReturn_t (*nvmlDeviceGetBrand_func)(nvmlDevice_t device, nvmlBrandType_t* type);
typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_func)(nvmlDevice_t device, nvmlUtilization_t* utilization);
typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_func)(nvmlDevice_t device, nvmlMemory_t* memory);
typedef nvmlReturn_t (*nvmlDeviceGetTemperature_func)(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int* temp);
typedef nvmlReturn_t (*nvmlDeviceGetPowerUsage_func)(nvmlDevice_t device, unsigned int* power);
typedef nvmlReturn_t (*nvmlDeviceGetPowerManagementLimit_func)(nvmlDevice_t device, unsigned int* limit);
typedef nvmlReturn_t (*nvmlDeviceGetClockInfo_func)(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock);
typedef nvmlReturn_t (*nvmlDeviceGetNumGpuCores_func)(nvmlDevice_t device, unsigned int* numCores);
typedef nvmlReturn_t (*nvmlDeviceGetPerformanceState_func)(nvmlDevice_t device, nvmlPstates_t* pState);
typedef nvmlReturn_t (*nvmlDeviceGetFanSpeed_func)(nvmlDevice_t device, unsigned int* speed);
typedef nvmlReturn_t (*nvmlDeviceGetCurrPcieLinkGeneration_func)(nvmlDevice_t device, unsigned int* linkGen);
typedef nvmlReturn_t (*nvmlDeviceGetCurrPcieLinkWidth_func)(nvmlDevice_t device, unsigned int* linkWidth);
typedef nvmlReturn_t (*nvmlDeviceGetTotalEccErrors_func)(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts);

// Function pointers
nvmlInit_func _nvmlInit = nullptr;
nvmlShutdown_func _nvmlShutdown = nullptr;
nvmlDeviceGetCount_func _nvmlDeviceGetCount = nullptr;
nvmlDeviceGetHandleByIndex_func _nvmlDeviceGetHandleByIndex = nullptr;
nvmlDeviceGetHandleByUUID_func _nvmlDeviceGetHandleByUUID = nullptr;
nvmlDeviceGetName_func _nvmlDeviceGetName = nullptr;
nvmlDeviceGetBrand_func _nvmlDeviceGetBrand = nullptr;
nvmlDeviceGetUtilizationRates_func _nvmlDeviceGetUtilizationRates = nullptr;
nvmlDeviceGetMemoryInfo_func _nvmlDeviceGetMemoryInfo = nullptr;
nvmlDeviceGetTemperature_func _nvmlDeviceGetTemperature = nullptr;
nvmlDeviceGetPowerUsage_func _nvmlDeviceGetPowerUsage = nullptr;
nvmlDeviceGetPowerManagementLimit_func _nvmlDeviceGetPowerManagementLimit = nullptr;
nvmlDeviceGetClockInfo_func _nvmlDeviceGetClockInfo = nullptr;
nvmlDeviceGetNumGpuCores_func _nvmlDeviceGetNumGpuCores = nullptr;
nvmlDeviceGetPerformanceState_func _nvmlDeviceGetPerformanceState = nullptr;
nvmlDeviceGetFanSpeed_func _nvmlDeviceGetFanSpeed = nullptr;
nvmlDeviceGetCurrPcieLinkGeneration_func _nvmlDeviceGetCurrPcieLinkGeneration = nullptr;
nvmlDeviceGetCurrPcieLinkWidth_func _nvmlDeviceGetCurrPcieLinkWidth = nullptr;
nvmlDeviceGetTotalEccErrors_func _nvmlDeviceGetTotalEccErrors = nullptr;

// Initialize NVML library dynamically
bool initialize_nvml() {
  if (nvml_handle != nullptr) {
    return true; // Already initialized
  }

  // Try to load the NVML library
  nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  if (nvml_handle == nullptr) {
    fprintf(stderr, "Failed to load NVML library: %s\n", dlerror());
    return false;
  }

  // Load function pointers
  _nvmlInit = (nvmlInit_func)dlsym(nvml_handle, "nvmlInit");
  _nvmlShutdown = (nvmlShutdown_func)dlsym(nvml_handle, "nvmlShutdown");
  _nvmlDeviceGetCount = (nvmlDeviceGetCount_func)dlsym(nvml_handle, "nvmlDeviceGetCount");
  _nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_func)dlsym(nvml_handle, "nvmlDeviceGetHandleByIndex");
  _nvmlDeviceGetHandleByUUID = (nvmlDeviceGetHandleByUUID_func)dlsym(nvml_handle, "nvmlDeviceGetHandleByUUID");
  _nvmlDeviceGetName = (nvmlDeviceGetName_func)dlsym(nvml_handle, "nvmlDeviceGetName");
  _nvmlDeviceGetBrand = (nvmlDeviceGetBrand_func)dlsym(nvml_handle, "nvmlDeviceGetBrand");
  _nvmlDeviceGetUtilizationRates = (nvmlDeviceGetUtilizationRates_func)dlsym(nvml_handle, "nvmlDeviceGetUtilizationRates");
  _nvmlDeviceGetMemoryInfo = (nvmlDeviceGetMemoryInfo_func)dlsym(nvml_handle, "nvmlDeviceGetMemoryInfo");
  _nvmlDeviceGetTemperature = (nvmlDeviceGetTemperature_func)dlsym(nvml_handle, "nvmlDeviceGetTemperature");
  _nvmlDeviceGetPowerUsage = (nvmlDeviceGetPowerUsage_func)dlsym(nvml_handle, "nvmlDeviceGetPowerUsage");
  _nvmlDeviceGetPowerManagementLimit = (nvmlDeviceGetPowerManagementLimit_func)dlsym(nvml_handle, "nvmlDeviceGetPowerManagementLimit");
  _nvmlDeviceGetClockInfo = (nvmlDeviceGetClockInfo_func)dlsym(nvml_handle, "nvmlDeviceGetClockInfo");
  _nvmlDeviceGetNumGpuCores = (nvmlDeviceGetNumGpuCores_func)dlsym(nvml_handle, "nvmlDeviceGetNumGpuCores");
  _nvmlDeviceGetPerformanceState = (nvmlDeviceGetPerformanceState_func)dlsym(nvml_handle, "nvmlDeviceGetPerformanceState");
  _nvmlDeviceGetFanSpeed = (nvmlDeviceGetFanSpeed_func)dlsym(nvml_handle, "nvmlDeviceGetFanSpeed");
  _nvmlDeviceGetCurrPcieLinkGeneration = (nvmlDeviceGetCurrPcieLinkGeneration_func)dlsym(nvml_handle, "nvmlDeviceGetCurrPcieLinkGeneration");
  _nvmlDeviceGetCurrPcieLinkWidth = (nvmlDeviceGetCurrPcieLinkWidth_func)dlsym(nvml_handle, "nvmlDeviceGetCurrPcieLinkWidth");
  _nvmlDeviceGetTotalEccErrors = (nvmlDeviceGetTotalEccErrors_func)dlsym(nvml_handle, "nvmlDeviceGetTotalEccErrors");

  // Check if all functions were loaded successfully
  if (!_nvmlInit || !_nvmlShutdown || !_nvmlDeviceGetCount || !_nvmlDeviceGetHandleByIndex ||
      !_nvmlDeviceGetHandleByUUID || !_nvmlDeviceGetName || !_nvmlDeviceGetBrand ||
      !_nvmlDeviceGetUtilizationRates || !_nvmlDeviceGetMemoryInfo || !_nvmlDeviceGetTemperature ||
      !_nvmlDeviceGetPowerUsage || !_nvmlDeviceGetPowerManagementLimit || !_nvmlDeviceGetClockInfo ||
      !_nvmlDeviceGetNumGpuCores || !_nvmlDeviceGetPerformanceState || !_nvmlDeviceGetFanSpeed ||
      !_nvmlDeviceGetCurrPcieLinkGeneration || !_nvmlDeviceGetCurrPcieLinkWidth || !_nvmlDeviceGetTotalEccErrors) {
    fprintf(stderr, "Failed to load one or more NVML functions\n");
    dlclose(nvml_handle);
    nvml_handle = nullptr;
    return false;
  }

  return true;
}

// Cleanup NVML library
void cleanup_nvml() {
  if (nvml_handle != nullptr) {
    dlclose(nvml_handle);
    nvml_handle = nullptr;
    _nvmlInit = nullptr;
    _nvmlShutdown = nullptr;
    _nvmlDeviceGetCount = nullptr;
    _nvmlDeviceGetHandleByIndex = nullptr;
    _nvmlDeviceGetHandleByUUID = nullptr;
    _nvmlDeviceGetName = nullptr;
    _nvmlDeviceGetBrand = nullptr;
    _nvmlDeviceGetUtilizationRates = nullptr;
    _nvmlDeviceGetMemoryInfo = nullptr;
    _nvmlDeviceGetTemperature = nullptr;
    _nvmlDeviceGetPowerUsage = nullptr;
    _nvmlDeviceGetPowerManagementLimit = nullptr;
    _nvmlDeviceGetClockInfo = nullptr;
    _nvmlDeviceGetNumGpuCores = nullptr;
    _nvmlDeviceGetPerformanceState = nullptr;
    _nvmlDeviceGetFanSpeed = nullptr;
    _nvmlDeviceGetCurrPcieLinkGeneration = nullptr;
    _nvmlDeviceGetCurrPcieLinkWidth = nullptr;
    _nvmlDeviceGetTotalEccErrors = nullptr;
  }
}

} // anonymous namespace

// Helper functions for individual NVML API groups
// namespace {

// C++ struct to hold NVML operation results
struct nvml_result {
  nvmlReturn_t return_code;
  jobject data;

  nvml_result(nvmlReturn_t return_code_, jobject data_) : return_code{return_code_}, data{data_} {}
  nvml_result() : return_code{NVML_ERROR_UNKNOWN}, data{nullptr} {}
};

// Helper function to create nvml_result Java object from C++ nvml_result
jobject create_nvml_result(JNIEnv* env, nvml_result const& cpp_result)
{
  // Create nvml_result object using default constructor
  jclass nvml_result_class = env->FindClass(NVML_CLASS_PATH "NVMLResult");
  if (nvml_result_class == nullptr) return nullptr;

  jmethodID constructor = env->GetMethodID(nvml_result_class, "<init>", "()V");
  if (constructor == nullptr) return nullptr;

  jobject java_nvml_result = env->NewObject(nvml_result_class, constructor);
  if (java_nvml_result == nullptr) return nullptr;

  // Set the return code and data fields directly
  jfieldID return_code_field = env->GetFieldID(nvml_result_class, "returnCode", "I");
  jfieldID data_field        = env->GetFieldID(nvml_result_class, "data", "Ljava/lang/Object;");

  env->SetIntField(java_nvml_result, return_code_field, static_cast<jint>(cpp_result.return_code));
  env->SetObjectField(java_nvml_result, data_field, cpp_result.data);

  return java_nvml_result;
}

jobject create_object(JNIEnv* env, char const* class_name, char const* constructor_signature)
{
  jclass j_class = env->FindClass(class_name);
  if (j_class == nullptr) { return nullptr; }

  jmethodID constructor = env->GetMethodID(j_class, "<init>", constructor_signature);
  if (constructor == nullptr) { return nullptr; }

  return env->NewObject(j_class, constructor);
}

nvml_result populate_device_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result;

  jobject device_info = create_object(env, NVML_CLASS_PATH "GPUDeviceInfo", "()V");
  if (device_info == nullptr) { return result; }

  jclass device_info_class = env->GetObjectClass(device_info);

  jfieldID name_field  = env->GetFieldID(device_info_class, "name", "Ljava/lang/String;");
  jfieldID brand_field = env->GetFieldID(device_info_class, "brand", "Ljava/lang/String;");

  char name[NVML_DEVICE_NAME_BUFFER_SIZE];
  auto return_code = _nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
  result           = nvml_result(return_code, device_info);

  if (return_code == NVML_SUCCESS) {
    jstring j_name = env->NewStringUTF(name);
    env->SetObjectField(device_info, name_field, j_name);
    env->DeleteLocalRef(j_name);
  }

  nvmlBrandType_t brand_type;
  auto brand_return_code = _nvmlDeviceGetBrand(device, &brand_type);
  if (brand_return_code == NVML_SUCCESS) {
    char brand[50];
    snprintf(brand, sizeof(brand), "Brand_%d", static_cast<int>(brand_type));
    jstring j_brand = env->NewStringUTF(brand);
    env->SetObjectField(device_info, brand_field, j_brand);
    env->DeleteLocalRef(j_brand);
  }

  return result;
}

nvml_result populate_utilization_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result;

  jobject utilization_info = create_object(env, NVML_CLASS_PATH "GPUUtilizationInfo", "()V");
  if (utilization_info == nullptr) { return result; }

  jclass utilization_info_class = env->GetObjectClass(utilization_info);

  jfieldID gpu_util_field = env->GetFieldID(utilization_info_class, "gpuUtilization", "I");
  jfieldID mem_util_field = env->GetFieldID(utilization_info_class, "memoryUtilization", "I");

  nvmlUtilization_t utilization;
  auto return_code = _nvmlDeviceGetUtilizationRates(device, &utilization);
  result           = nvml_result(return_code, utilization_info);

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(utilization_info, gpu_util_field, static_cast<jint>(utilization.gpu));
    env->SetIntField(utilization_info, mem_util_field, static_cast<jint>(utilization.memory));
  }

  return result;
}

nvml_result populate_memory_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result;

  jobject memory_info = create_object(env, NVML_CLASS_PATH "GPUMemoryInfo", "()V");
  if (memory_info == nullptr) { return result; }

  jclass memory_info_class = env->GetObjectClass(memory_info);

  jfieldID mem_used_field  = env->GetFieldID(memory_info_class, "memoryUsedMB", "J");
  jfieldID mem_total_field = env->GetFieldID(memory_info_class, "memoryTotalMB", "J");
  jfieldID mem_free_field  = env->GetFieldID(memory_info_class, "memoryFreeMB", "J");

  nvmlMemory_t memory;
  auto return_code = _nvmlDeviceGetMemoryInfo(device, &memory);
  result           = nvml_result(return_code, memory_info);

  if (return_code == NVML_SUCCESS) {
    env->SetLongField(memory_info, mem_used_field, static_cast<jlong>(memory.used / (1024 * 1024)));
    env->SetLongField(
      memory_info, mem_total_field, static_cast<jlong>(memory.total / (1024 * 1024)));
    env->SetLongField(memory_info, mem_free_field, static_cast<jlong>(memory.free / (1024 * 1024)));
  }

  return result;
}

nvml_result populate_temperature_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result;

  jobject temperature_info = create_object(env, NVML_CLASS_PATH "GPUTemperatureInfo", "()V");
  if (temperature_info == nullptr) { return result; }

  jclass temperature_info_class = env->GetObjectClass(temperature_info);

  jfieldID temp_gpu_field = env->GetFieldID(temperature_info_class, "temperatureGpu", "I");

  unsigned int temp;
  auto return_code = _nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
  result           = nvml_result(return_code, temperature_info);

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(temperature_info, temp_gpu_field, static_cast<jint>(temp));
  }

  return result;
}

nvml_result populate_power_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result;

  jobject power_info = create_object(env, NVML_CLASS_PATH "GPUPowerInfo", "()V");
  if (power_info == nullptr) { return result; }

  jclass power_info_class = env->GetObjectClass(power_info);

  jfieldID power_usage_field = env->GetFieldID(power_info_class, "powerUsageW", "I");
  jfieldID power_limit_field = env->GetFieldID(power_info_class, "powerLimitW", "I");

  unsigned int power;
  auto return_code = _nvmlDeviceGetPowerUsage(device, &power);
  result           = nvml_result(return_code, power_info);

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(power_info, power_usage_field, static_cast<jint>(power / 1000));  // mW to W
  }

  auto limit_return_code = _nvmlDeviceGetPowerManagementLimit(device, &power);
  if (limit_return_code == NVML_SUCCESS) {
    env->SetIntField(power_info, power_limit_field, static_cast<jint>(power / 1000));  // mW to W
  }

  return result;
}

nvml_result populate_clock_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result;

  jobject clock_info = create_object(env, NVML_CLASS_PATH "GPUClockInfo", "()V");
  if (clock_info == nullptr) { return result; }

  jclass clock_info_class = env->GetObjectClass(clock_info);

  jfieldID graphics_clock_field = env->GetFieldID(clock_info_class, "graphicsClockMHz", "I");
  jfieldID memory_clock_field   = env->GetFieldID(clock_info_class, "memoryClockMHz", "I");
  jfieldID sm_clock_field       = env->GetFieldID(clock_info_class, "smClockMHz", "I");

  unsigned int clock;
  auto return_code = _nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
  result           = nvml_result(return_code, clock_info);

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(clock_info, graphics_clock_field, static_cast<jint>(clock));
  }

  auto mem_return_code = _nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock);
  if (mem_return_code == NVML_SUCCESS) {
    env->SetIntField(clock_info, memory_clock_field, static_cast<jint>(clock));
  }

  auto sm_return_code = _nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
  if (sm_return_code == NVML_SUCCESS) {
    env->SetIntField(clock_info, sm_clock_field, static_cast<jint>(clock));
  }

  return result;
}

nvml_result populate_hardware_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result;

  jobject hardware_info = create_object(env, NVML_CLASS_PATH "GPUHardwareInfo", "()V");
  if (hardware_info == nullptr) { return result; }

  jclass hardware_info_class = env->GetObjectClass(hardware_info);

  jfieldID sm_count_field = env->GetFieldID(hardware_info_class, "streamingMultiprocessors", "I");
  jfieldID performance_state_field = env->GetFieldID(hardware_info_class, "performanceState", "I");
  jfieldID fan_speed_field         = env->GetFieldID(hardware_info_class, "fanSpeedPercent", "I");

  unsigned int sm_count = 0;
  auto return_code      = _nvmlDeviceGetNumGpuCores(device, &sm_count);
  result                = nvml_result(return_code, hardware_info);

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(hardware_info, sm_count_field, static_cast<jint>(sm_count));
  }

  nvmlPstates_t p_state;
  auto pstate_return_code = _nvmlDeviceGetPerformanceState(device, &p_state);
  if (pstate_return_code == NVML_SUCCESS) {
    env->SetIntField(hardware_info, performance_state_field, static_cast<jint>(p_state));
  }

  unsigned int fan_speed;
  auto fan_return_code = _nvmlDeviceGetFanSpeed(device, &fan_speed);
  if (fan_return_code == NVML_SUCCESS) {
    env->SetIntField(hardware_info, fan_speed_field, static_cast<jint>(fan_speed));
  }

  return result;
}

nvml_result populate_pcie_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result;

  jobject pcie_info = create_object(env, NVML_CLASS_PATH "GPUPCIeInfo", "()V");
  if (pcie_info == nullptr) { return result; }

  jclass pcie_info_class = env->GetObjectClass(pcie_info);

  jfieldID pcie_link_gen_field   = env->GetFieldID(pcie_info_class, "pcieLinkGeneration", "I");
  jfieldID pcie_link_width_field = env->GetFieldID(pcie_info_class, "pcieLinkWidth", "I");

  unsigned int link_gen;
  auto return_code = _nvmlDeviceGetCurrPcieLinkGeneration(device, &link_gen);
  result           = nvml_result(return_code, pcie_info);

  if (return_code == NVML_SUCCESS) {
    env->SetIntField(pcie_info, pcie_link_gen_field, static_cast<jint>(link_gen));
  }

  unsigned int link_width;
  auto width_return_code = _nvmlDeviceGetCurrPcieLinkWidth(device, &link_width);
  if (width_return_code == NVML_SUCCESS) {
    env->SetIntField(pcie_info, pcie_link_width_field, static_cast<jint>(link_width));
  }

  return result;
}

nvml_result populate_ecc_info(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result;

  jobject ecc_info = create_object(env, NVML_CLASS_PATH "GPUECCInfo", "()V");
  if (ecc_info == nullptr) { return result; }

  jclass ecc_info_class = env->GetObjectClass(ecc_info);

  jfieldID ecc_single_bit_field = env->GetFieldID(ecc_info_class, "eccSingleBitErrors", "J");
  jfieldID ecc_double_bit_field = env->GetFieldID(ecc_info_class, "eccDoubleBitErrors", "J");

  unsigned long long eccCount;
  auto return_code =
    _nvmlDeviceGetTotalEccErrors(device, NVML_SINGLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  result = nvml_result(return_code, ecc_info);

  if (return_code == NVML_SUCCESS) {
    env->SetLongField(ecc_info, ecc_single_bit_field, static_cast<jlong>(eccCount));
  }

  auto double_bit_return_code =
    _nvmlDeviceGetTotalEccErrors(device, NVML_DOUBLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  if (double_bit_return_code == NVML_SUCCESS) {
    env->SetLongField(ecc_info, ecc_double_bit_field, static_cast<jlong>(eccCount));
  }

  return result;
}

// Helper function to populate GPUInfo object from NVML device handle using individual helpers
nvml_result populate_gpu_info_from_device(JNIEnv* env, nvmlDevice_t device)
{
  nvml_result result;
  result.return_code = NVML_SUCCESS;  // Start with success, track first error

  // Create GPUInfo object
  jobject gpu_info = create_object(env, NVML_CLASS_PATH "GPUInfo", "()V");
  if (gpu_info == nullptr) { return result; }

  result.data           = gpu_info;
  jclass gpu_info_class = env->GetObjectClass(gpu_info);

  // Populate nested info objects using individual helpers
  nvml_result device_result      = populate_device_info(env, device);
  nvml_result utilization_result = populate_utilization_info(env, device);
  nvml_result memory_result      = populate_memory_info(env, device);
  nvml_result temperature_result = populate_temperature_info(env, device);
  nvml_result power_result       = populate_power_info(env, device);
  nvml_result clock_result       = populate_clock_info(env, device);
  nvml_result hardware_result    = populate_hardware_info(env, device);
  nvml_result pcie_result        = populate_pcie_info(env, device);
  nvml_result ecc_result         = populate_ecc_info(env, device);

  // Track the first error encountered
  if (result.return_code == NVML_SUCCESS && device_result.return_code != NVML_SUCCESS) {
    result.return_code = device_result.return_code;
  }
  if (result.return_code == NVML_SUCCESS && utilization_result.return_code != NVML_SUCCESS) {
    result.return_code = utilization_result.return_code;
  }
  if (result.return_code == NVML_SUCCESS && memory_result.return_code != NVML_SUCCESS) {
    result.return_code = memory_result.return_code;
  }
  if (result.return_code == NVML_SUCCESS && temperature_result.return_code != NVML_SUCCESS) {
    result.return_code = temperature_result.return_code;
  }
  if (result.return_code == NVML_SUCCESS && power_result.return_code != NVML_SUCCESS) {
    result.return_code = power_result.return_code;
  }
  if (result.return_code == NVML_SUCCESS && clock_result.return_code != NVML_SUCCESS) {
    result.return_code = clock_result.return_code;
  }
  if (result.return_code == NVML_SUCCESS && hardware_result.return_code != NVML_SUCCESS) {
    result.return_code = hardware_result.return_code;
  }
  if (result.return_code == NVML_SUCCESS && pcie_result.return_code != NVML_SUCCESS) {
    result.return_code = pcie_result.return_code;
  }
  if (result.return_code == NVML_SUCCESS && ecc_result.return_code != NVML_SUCCESS) {
    result.return_code = ecc_result.return_code;
  }

  // Set nested info objects in GPUInfo
  jfieldID device_info_field =
    env->GetFieldID(gpu_info_class, "deviceInfo", "L" NVML_CLASS_PATH "GPUDeviceInfo;");
  jfieldID utilization_info_field =
    env->GetFieldID(gpu_info_class, "utilizationInfo", "L" NVML_CLASS_PATH "GPUUtilizationInfo;");
  jfieldID memory_info_field =
    env->GetFieldID(gpu_info_class, "memoryInfo", "L" NVML_CLASS_PATH "GPUMemoryInfo;");
  jfieldID temperature_info_field =
    env->GetFieldID(gpu_info_class, "temperatureInfo", "L" NVML_CLASS_PATH "GPUTemperatureInfo;");
  jfieldID power_info_field =
    env->GetFieldID(gpu_info_class, "powerInfo", "L" NVML_CLASS_PATH "GPUPowerInfo;");
  jfieldID clock_info_field =
    env->GetFieldID(gpu_info_class, "clockInfo", "L" NVML_CLASS_PATH "GPUClockInfo;");
  jfieldID hardware_info_field =
    env->GetFieldID(gpu_info_class, "hardwareInfo", "L" NVML_CLASS_PATH "GPUHardwareInfo;");
  jfieldID pcie_info_field =
    env->GetFieldID(gpu_info_class, "pcieInfo", "L" NVML_CLASS_PATH "GPUPCIeInfo;");
  jfieldID ecc_info_field =
    env->GetFieldID(gpu_info_class, "eccInfo", "L" NVML_CLASS_PATH "GPUECCInfo;");

  if (device_result.return_code == NVML_SUCCESS) {
    env->SetObjectField(gpu_info, device_info_field, device_result.data);
  }
  if (utilization_result.return_code == NVML_SUCCESS) {
    env->SetObjectField(gpu_info, utilization_info_field, utilization_result.data);
  }
  if (memory_result.return_code == NVML_SUCCESS) {
    env->SetObjectField(gpu_info, memory_info_field, memory_result.data);
  }
  if (temperature_result.return_code == NVML_SUCCESS) {
    env->SetObjectField(gpu_info, temperature_info_field, temperature_result.data);
  }
  if (power_result.return_code == NVML_SUCCESS) {
    env->SetObjectField(gpu_info, power_info_field, power_result.data);
  }
  if (clock_result.return_code == NVML_SUCCESS) {
    env->SetObjectField(gpu_info, clock_info_field, clock_result.data);
  }
  if (hardware_result.return_code == NVML_SUCCESS) {
    env->SetObjectField(gpu_info, hardware_info_field, hardware_result.data);
  }
  if (pcie_result.return_code == NVML_SUCCESS) {
    env->SetObjectField(gpu_info, pcie_info_field, pcie_result.data);
  }
  if (ecc_result.return_code == NVML_SUCCESS) {
    env->SetObjectField(gpu_info, ecc_info_field, ecc_result.data);
  }

  env->DeleteLocalRef(device_result.data);
  env->DeleteLocalRef(utilization_result.data);
  env->DeleteLocalRef(memory_result.data);
  env->DeleteLocalRef(temperature_result.data);
  env->DeleteLocalRef(power_result.data);
  env->DeleteLocalRef(clock_result.data);
  env->DeleteLocalRef(hardware_result.data);
  env->DeleteLocalRef(pcie_result.data);
  env->DeleteLocalRef(ecc_result.data);

  return result;
}

// }  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlInit(JNIEnv* env,
                                                                               jclass cls)
{
  // Initialize dynamic loading of NVML library
  if (!initialize_nvml()) {
    return JNI_FALSE;
  }

  nvmlReturn_t result = _nvmlInit();
  return (result == NVML_SUCCESS) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlShutdown(JNIEnv* env,
                                                                               jclass cls)
{
  if (_nvmlShutdown) {
    _nvmlShutdown();
  }
  cleanup_nvml();
}

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetDeviceCount(JNIEnv* env,
                                                                                     jclass cls)
{
  unsigned int device_count = 0;
  nvmlReturn_t result       = _nvmlDeviceGetCount(&device_count);

  if (result != NVML_SUCCESS) { return -1; }

  return static_cast<jint>(device_count);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetDeviceHandleFromUUID(
  JNIEnv* env, jclass cls, jbyteArray uuid)
{
  if (uuid == nullptr) { return 0; }

  jsize uuid_len = env->GetArrayLength(uuid);
  if (uuid_len != 16) { return 0; }  // UUID should be 16 bytes

  // Get the UUID bytes from Java (raw binary format from cudaDeviceProp.uuid)
  jbyte* uuid_bytes = env->GetByteArrayElements(uuid, nullptr);
  if (uuid_bytes == nullptr) { return 0; }

  // Convert binary UUID to string format: "GPU-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"
  char uuid_str[64];
  snprintf(uuid_str,
           sizeof(uuid_str),
           "GPU-%02hhx%02hhx%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx%"
           "02hhx%02hhx%02hhx%02hhx",
           uuid_bytes[0],
           uuid_bytes[1],
           uuid_bytes[2],
           uuid_bytes[3],
           uuid_bytes[4],
           uuid_bytes[5],
           uuid_bytes[6],
           uuid_bytes[7],
           uuid_bytes[8],
           uuid_bytes[9],
           uuid_bytes[10],
           uuid_bytes[11],
           uuid_bytes[12],
           uuid_bytes[13],
           uuid_bytes[14],
           uuid_bytes[15]);

  env->ReleaseByteArrayElements(uuid, uuid_bytes, JNI_ABORT);

  // Get device handle by UUID string
  nvmlDevice_t device;
  nvmlReturn_t nvml_error = _nvmlDeviceGetHandleByUUID(uuid_str, &device);

  if (nvml_error != NVML_SUCCESS) { return 0; }

  return static_cast<jlong>(
    reinterpret_cast<std::intptr_t>(device));  // Return the handle as a long
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetGPUInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));

  // Use common helper to populate GPUInfo object
  nvml_result result = populate_gpu_info_from_device(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobjectArray JNICALL
Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetAllGPUInfo(JNIEnv* env, jclass cls)
{
  // Get device count
  unsigned int device_count = 0;
  nvmlReturn_t result       = _nvmlDeviceGetCount(&device_count);

  if (result != NVML_SUCCESS) {
    // Return empty array on error
    jclass nvml_result_class = env->FindClass(NVML_CLASS_PATH "NVMLResult");
    if (nvml_result_class == nullptr) { return nullptr; }
    return env->NewObjectArray(0, nvml_result_class, nullptr);
  }

  // Create array of nvml_result objects
  jclass nvml_result_class = env->FindClass(NVML_CLASS_PATH "NVMLResult");
  if (nvml_result_class == nullptr) { return nullptr; }

  jobjectArray result_array =
    env->NewObjectArray(static_cast<jsize>(device_count), nvml_result_class, nullptr);
  if (result_array == nullptr) { return nullptr; }

  // Fill array with individual nvml_result objects for each GPU
  for (unsigned int i = 0; i < device_count; i++) {
    nvmlDevice_t device;
    nvmlReturn_t device_result = _nvmlDeviceGetHandleByIndex(i, &device);

    nvml_result cpp_result;
    if (device_result == NVML_SUCCESS) {
      // Successfully got device handle, try to populate GPU info
      cpp_result = populate_gpu_info_from_device(env, device);
    } else {
      // Failed to get device handle
      cpp_result = nvml_result(device_result, nullptr);
    }

    // Create Java nvml_result object and add to array
    jobject java_result = create_nvml_result(env, cpp_result);
    if (java_result != nullptr) {
      env->SetObjectArrayElement(result_array, static_cast<jsize>(i), java_result);
      env->DeleteLocalRef(java_result);
    }
  }

  return result_array;
}

// Individual info JNI functions using device handles
JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetDeviceInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_device_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetUtilizationInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_utilization_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetMemoryInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_memory_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetTemperatureInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_temperature_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetPowerInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_power_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetClockInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_clock_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetHardwareInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_hardware_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetPCIeInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_pcie_info(env, device);
  return create_nvml_result(env, result);
}

JNIEXPORT jobject JNICALL Java_com_nvidia_spark_rapids_jni_nvml_NVML_nvmlGetECCInfo(
  JNIEnv* env, jclass cls, jlong deviceHandle)
{
  nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(static_cast<std::intptr_t>(deviceHandle));
  nvml_result result  = populate_ecc_info(env, device);
  return create_nvml_result(env, result);
}

}  // extern "C"
