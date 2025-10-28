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

#include <gtest/gtest.h>
#include <nvml.h>
#include <dlfcn.h>

// Dynamic loading of NVML library for testing
namespace {
void* nvml_handle = nullptr;

typedef nvmlReturn_t (*nvmlInit_func)();
typedef nvmlReturn_t (*nvmlShutdown_func)();
typedef nvmlReturn_t (*nvmlDeviceGetCount_func)(unsigned int* deviceCount);
typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_func)(unsigned int index, nvmlDevice_t* device);
typedef nvmlReturn_t (*_nvmlDeviceGetName_func)(nvmlDevice_t device, char* name, unsigned int length);
typedef nvmlReturn_t (*_nvmlDeviceGetBrand_func)(nvmlDevice_t device, nvmlBrandType_t* type);
typedef nvmlReturn_t (*_nvmlDeviceGetUtilizationRates_func)(nvmlDevice_t device, nvmlUtilization_t* utilization);
typedef nvmlReturn_t (*_nvmlDeviceGetMemoryInfo_func)(nvmlDevice_t device, nvmlMemory_t* memory);
typedef nvmlReturn_t (*_nvmlDeviceGetTemperature_func)(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int* temp);
typedef nvmlReturn_t (*_nvmlDeviceGetPowerUsage_func)(nvmlDevice_t device, unsigned int* power);
typedef nvmlReturn_t (*_nvmlDeviceGetPowerManagementLimit_func)(nvmlDevice_t device, unsigned int* limit);
typedef nvmlReturn_t (*_nvmlDeviceGetClockInfo_func)(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock);
typedef nvmlReturn_t (*_nvmlDeviceGetNumGpuCores_func)(nvmlDevice_t device, unsigned int* numCores);
typedef nvmlReturn_t (*_nvmlDeviceGetPerformanceState_func)(nvmlDevice_t device, nvmlPstates_t* pState);
typedef nvmlReturn_t (*_nvmlDeviceGetFanSpeed_func)(nvmlDevice_t device, unsigned int* speed);
typedef nvmlReturn_t (*_nvmlDeviceGetCurrPcieLinkGeneration_func)(nvmlDevice_t device, unsigned int* linkGen);
typedef nvmlReturn_t (*_nvmlDeviceGetCurrPcieLinkWidth_func)(nvmlDevice_t device, unsigned int* linkWidth);
typedef nvmlReturn_t (*_nvmlDeviceGetTotalEccErrors_func)(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts);

nvmlInit_func _nvmlInit = nullptr;
nvmlShutdown_func _nvmlShutdown = nullptr;
nvmlDeviceGetCount_func _nvmlDeviceGetCount = nullptr;
nvmlDeviceGetHandleByIndex_func _nvmlDeviceGetHandleByIndex = nullptr;
_nvmlDeviceGetName_func _nvmlDeviceGetName = nullptr;
_nvmlDeviceGetBrand_func _nvmlDeviceGetBrand = nullptr;
_nvmlDeviceGetUtilizationRates_func _nvmlDeviceGetUtilizationRates = nullptr;
_nvmlDeviceGetMemoryInfo_func _nvmlDeviceGetMemoryInfo = nullptr;
_nvmlDeviceGetTemperature_func _nvmlDeviceGetTemperature = nullptr;
_nvmlDeviceGetPowerUsage_func _nvmlDeviceGetPowerUsage = nullptr;
_nvmlDeviceGetPowerManagementLimit_func _nvmlDeviceGetPowerManagementLimit = nullptr;
_nvmlDeviceGetClockInfo_func _nvmlDeviceGetClockInfo = nullptr;
_nvmlDeviceGetNumGpuCores_func _nvmlDeviceGetNumGpuCores = nullptr;
_nvmlDeviceGetPerformanceState_func _nvmlDeviceGetPerformanceState = nullptr;
_nvmlDeviceGetFanSpeed_func _nvmlDeviceGetFanSpeed = nullptr;
_nvmlDeviceGetCurrPcieLinkGeneration_func _nvmlDeviceGetCurrPcieLinkGeneration = nullptr;
_nvmlDeviceGetCurrPcieLinkWidth_func _nvmlDeviceGetCurrPcieLinkWidth = nullptr;
_nvmlDeviceGetTotalEccErrors_func _nvmlDeviceGetTotalEccErrors = nullptr;

bool initialize_nvml_test() {
  if (nvml_handle != nullptr) {
    return true;
  }

  nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  if (nvml_handle == nullptr) {
    return false;
  }
  _nvmlInit = (nvmlInit_func)dlsym(nvml_handle, "nvmlInit");
  _nvmlShutdown = (nvmlShutdown_func)dlsym(nvml_handle, "nvmlShutdown");
  _nvmlDeviceGetCount = (nvmlDeviceGetCount_func)dlsym(nvml_handle, "nvmlDeviceGetCount");
  _nvmlDeviceGetHandleByIndex = (nvmlDeviceGetHandleByIndex_func)dlsym(nvml_handle, "nvmlDeviceGetHandleByIndex");
  _nvmlDeviceGetName = (_nvmlDeviceGetName_func)dlsym(nvml_handle, "nvmlDeviceGetName");
  _nvmlDeviceGetBrand = (_nvmlDeviceGetBrand_func)dlsym(nvml_handle, "nvmlDeviceGetBrand");
  _nvmlDeviceGetUtilizationRates = (_nvmlDeviceGetUtilizationRates_func)dlsym(nvml_handle, "nvmlDeviceGetUtilizationRates");
  _nvmlDeviceGetMemoryInfo = (_nvmlDeviceGetMemoryInfo_func)dlsym(nvml_handle, "nvmlDeviceGetMemoryInfo");
  _nvmlDeviceGetTemperature = (_nvmlDeviceGetTemperature_func)dlsym(nvml_handle, "nvmlDeviceGetTemperature");
  _nvmlDeviceGetPowerUsage = (_nvmlDeviceGetPowerUsage_func)dlsym(nvml_handle, "nvmlDeviceGetPowerUsage");
  _nvmlDeviceGetPowerManagementLimit = (_nvmlDeviceGetPowerManagementLimit_func)dlsym(nvml_handle, "nvmlDeviceGetPowerManagementLimit");
  _nvmlDeviceGetClockInfo = (_nvmlDeviceGetClockInfo_func)dlsym(nvml_handle, "nvmlDeviceGetClockInfo");
  _nvmlDeviceGetNumGpuCores = (_nvmlDeviceGetNumGpuCores_func)dlsym(nvml_handle, "nvmlDeviceGetNumGpuCores");
  _nvmlDeviceGetPerformanceState = (_nvmlDeviceGetPerformanceState_func)dlsym(nvml_handle, "nvmlDeviceGetPerformanceState");
  _nvmlDeviceGetFanSpeed = (_nvmlDeviceGetFanSpeed_func)dlsym(nvml_handle, "nvmlDeviceGetFanSpeed");
  _nvmlDeviceGetCurrPcieLinkGeneration = (_nvmlDeviceGetCurrPcieLinkGeneration_func)dlsym(nvml_handle, "nvmlDeviceGetCurrPcieLinkGeneration");
  _nvmlDeviceGetCurrPcieLinkWidth = (_nvmlDeviceGetCurrPcieLinkWidth_func)dlsym(nvml_handle, "nvmlDeviceGetCurrPcieLinkWidth");
  _nvmlDeviceGetTotalEccErrors = (_nvmlDeviceGetTotalEccErrors_func)dlsym(nvml_handle, "nvmlDeviceGetTotalEccErrors");

  return _nvmlInit && _nvmlShutdown && _nvmlDeviceGetCount && _nvmlDeviceGetHandleByIndex &&
         _nvmlDeviceGetName && _nvmlDeviceGetBrand && _nvmlDeviceGetUtilizationRates &&
         _nvmlDeviceGetMemoryInfo && _nvmlDeviceGetTemperature && _nvmlDeviceGetPowerUsage &&
         _nvmlDeviceGetPowerManagementLimit && _nvmlDeviceGetClockInfo && _nvmlDeviceGetNumGpuCores &&
         _nvmlDeviceGetPerformanceState && _nvmlDeviceGetFanSpeed &&
         _nvmlDeviceGetCurrPcieLinkGeneration && _nvmlDeviceGetCurrPcieLinkWidth && _nvmlDeviceGetTotalEccErrors;
}

void cleanup_nvml_test() {
  if (nvml_handle != nullptr) {
    dlclose(nvml_handle);
    nvml_handle = nullptr;
    _nvmlInit = nullptr;
    _nvmlShutdown = nullptr;
    _nvmlDeviceGetCount = nullptr;
    _nvmlDeviceGetHandleByIndex = nullptr;
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

class NVMLTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Initialize NVML dynamically
    if (!initialize_nvml_test()) {
      nvml_result = NVML_ERROR_LIBRARY_NOT_FOUND;
      return;
    }
    nvml_result = _nvmlInit();
  }

  void TearDown() override
  {
    // Shutdown NVML if it was initialized
    if (nvml_result == NVML_SUCCESS && _nvmlShutdown) {
      _nvmlShutdown();
    }
    cleanup_nvml_test();
  }

  nvmlReturn_t nvml_result = NVML_ERROR_UNKNOWN;

  // Helper to get first available device, returns nullptr if none
  nvmlDevice_t getFirstDevice()
  {
    if (nvml_result != NVML_SUCCESS || !_nvmlDeviceGetCount || !_nvmlDeviceGetHandleByIndex) {
      return nullptr;
    }

    unsigned int deviceCount = 0;
    nvmlReturn_t result      = _nvmlDeviceGetCount(&deviceCount);
    if (result != NVML_SUCCESS || deviceCount == 0) {
      return nullptr;
    }

    nvmlDevice_t device;
    result = _nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
      return nullptr;
    }

    return device;
  }
};

TEST_F(NVMLTest, NVMLDeviceGetName_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();

  char name[NVML_DEVICE_NAME_BUFFER_SIZE];
  nvmlReturn_t result = _nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetBrand_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  nvmlBrandType_t brandType;
  nvmlReturn_t result = _nvmlDeviceGetBrand(device, &brandType);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetUtilizationRates_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  nvmlUtilization_t utilization;
  nvmlReturn_t result = _nvmlDeviceGetUtilizationRates(device, &utilization);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetMemoryInfo_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  nvmlMemory_t memory;
  nvmlReturn_t result = _nvmlDeviceGetMemoryInfo(device, &memory);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetTemperature_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int temp;
  nvmlReturn_t result = _nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetPowerUsage_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int power;
  nvmlReturn_t result = _nvmlDeviceGetPowerUsage(device, &power);
  // Power usage may not be supported on all GPUs
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetPowerManagementLimit_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int limit;
  nvmlReturn_t result = _nvmlDeviceGetPowerManagementLimit(device, &limit);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetClockInfo_Graphics_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int clock;
  nvmlReturn_t result = _nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetClockInfo_Memory_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int clock;
  nvmlReturn_t result = _nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetClockInfo_SM_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int clock;
  nvmlReturn_t result = _nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetNumGpuCores_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int num_cores;
  nvmlReturn_t result = _nvmlDeviceGetNumGpuCores(device, &num_cores);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetPerformanceState_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  nvmlPstates_t p_state;
  nvmlReturn_t result = _nvmlDeviceGetPerformanceState(device, &p_state);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetFanSpeed_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int speed;
  nvmlReturn_t result = _nvmlDeviceGetFanSpeed(device, &speed);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetCurrPcieLinkGeneration_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int gen;
  nvmlReturn_t result = _nvmlDeviceGetCurrPcieLinkGeneration(device, &gen);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetCurrPcieLinkWidth_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int width;
  nvmlReturn_t result = _nvmlDeviceGetCurrPcieLinkWidth(device, &width);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetTotalEccErrors_SingleBit_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned long long ecc_count;
  nvmlReturn_t result =
    _nvmlDeviceGetTotalEccErrors(device, NVML_SINGLE_BIT_ECC, NVML_VOLATILE_ECC, &ecc_count);
  // ECC error reporting may not be supported on all GPUs
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetTotalEccErrors_DoubleBit_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned long long eccCount;
  nvmlReturn_t result =
    _nvmlDeviceGetTotalEccErrors(device, NVML_DOUBLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  // ECC error reporting may not be supported on all GPUs
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}