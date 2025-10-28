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

#pragma once

#include <dlfcn.h>
#include <nvml.h>
#include <cstdint>

// Dynamic NVML loader utility
class NVMLDynamicLoader {
 public:
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
  nvmlInit_func init = nullptr;
  nvmlShutdown_func shutdown = nullptr;
  nvmlDeviceGetCount_func device_get_count = nullptr;
  nvmlDeviceGetHandleByIndex_func device_get_handle_by_index = nullptr;
  nvmlDeviceGetHandleByUUID_func device_get_handle_by_UUID = nullptr;
  nvmlDeviceGetName_func device_get_name = nullptr;
  nvmlDeviceGetBrand_func device_get_brand = nullptr;
  nvmlDeviceGetUtilizationRates_func device_get_utilization_rates = nullptr;
  nvmlDeviceGetMemoryInfo_func device_get_memory_info = nullptr;
  nvmlDeviceGetTemperature_func device_get_temperature = nullptr;
  nvmlDeviceGetPowerUsage_func device_get_power_usage = nullptr;
  nvmlDeviceGetPowerManagementLimit_func device_get_power_management_limit = nullptr;
  nvmlDeviceGetClockInfo_func device_get_clock_info = nullptr;
  nvmlDeviceGetNumGpuCores_func device_get_num_gpu_cores = nullptr;
  nvmlDeviceGetPerformanceState_func device_get_performance_state = nullptr;
  nvmlDeviceGetFanSpeed_func device_get_fan_speed = nullptr;
  nvmlDeviceGetCurrPcieLinkGeneration_func device_get_curr_pcie_link_generation = nullptr;
  nvmlDeviceGetCurrPcieLinkWidth_func device_get_curr_pcie_link_width = nullptr;
  nvmlDeviceGetTotalEccErrors_func device_get_total_ecc_errors = nullptr;

  // Initialize NVML library dynamically
  bool initialize();

  // Cleanup NVML library
  void cleanup();

  // Check if NVML is loaded and available
  bool isLoaded() const { return nvml_handle != nullptr; }

 private:
  void* nvml_handle = nullptr;
};

