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

class NVMLTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Initialize NVML
    nvml_result = nvmlInit();
  }

  void TearDown() override
  {
    // Shutdown NVML if it was initialized
    if (nvml_result == NVML_SUCCESS) { nvmlShutdown(); }
  }

  nvmlReturn_t nvml_result = NVML_ERROR_UNKNOWN;

  // Helper to get first available device, returns nullptr if none
  nvmlDevice_t getFirstDevice()
  {
    if (nvml_result != NVML_SUCCESS) { return nullptr; }

    unsigned int deviceCount = 0;
    nvmlReturn_t result      = nvmlDeviceGetCount(&deviceCount);
    if (result != NVML_SUCCESS || deviceCount == 0) { return nullptr; }

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) { return nullptr; }

    return device;
  }
};

TEST_F(NVMLTest, NVMLDeviceGetName_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  char name[NVML_DEVICE_NAME_BUFFER_SIZE];
  nvmlReturn_t result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetBrand_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  nvmlBrandType_t brandType;
  nvmlReturn_t result = nvmlDeviceGetBrand(device, &brandType);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetUtilizationRates_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  nvmlUtilization_t utilization;
  nvmlReturn_t result = nvmlDeviceGetUtilizationRates(device, &utilization);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetMemoryInfo_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  nvmlMemory_t memory;
  nvmlReturn_t result = nvmlDeviceGetMemoryInfo(device, &memory);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetTemperature_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int temp;
  nvmlReturn_t result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetPowerUsage_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int power;
  nvmlReturn_t result = nvmlDeviceGetPowerUsage(device, &power);
  // Power usage may not be supported on all GPUs
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetPowerManagementLimit_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int limit;
  nvmlReturn_t result = nvmlDeviceGetPowerManagementLimit(device, &limit);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetClockInfo_Graphics_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int clock;
  nvmlReturn_t result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetClockInfo_Memory_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int clock;
  nvmlReturn_t result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetClockInfo_SM_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int clock;
  nvmlReturn_t result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetNumGpuCores_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int num_cores;
  nvmlReturn_t result = nvmlDeviceGetNumGpuCores(device, &num_cores);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetPerformanceState_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  nvmlPstates_t p_state;
  nvmlReturn_t result = nvmlDeviceGetPerformanceState(device, &p_state);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetFanSpeed_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int speed;
  nvmlReturn_t result = nvmlDeviceGetFanSpeed(device, &speed);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetCurrPcieLinkGeneration_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int gen;
  nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkGeneration(device, &gen);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetCurrPcieLinkWidth_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned int width;
  nvmlReturn_t result = nvmlDeviceGetCurrPcieLinkWidth(device, &width);
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetTotalEccErrors_SingleBit_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned long long ecc_count;
  nvmlReturn_t result =
    nvmlDeviceGetTotalEccErrors(device, NVML_SINGLE_BIT_ECC, NVML_VOLATILE_ECC, &ecc_count);
  // ECC error reporting may not be supported on all GPUs
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}

TEST_F(NVMLTest, NVMLDeviceGetTotalEccErrors_DoubleBit_Succeeds)
{
  nvmlDevice_t device = getFirstDevice();
  ASSERT_NE(device, nullptr) << "No NVML devices available";

  unsigned long long eccCount;
  nvmlReturn_t result =
    nvmlDeviceGetTotalEccErrors(device, NVML_DOUBLE_BIT_ECC, NVML_VOLATILE_ECC, &eccCount);
  // ECC error reporting may not be supported on all GPUs
  EXPECT_TRUE(result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED);
}