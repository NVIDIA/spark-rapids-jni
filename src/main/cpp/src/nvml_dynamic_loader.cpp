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

#include "nvml_dynamic_loader.hpp"
#include <stdio.h>

// Initialize NVML library dynamically
bool NVMLDynamicLoader::initialize() {
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
  init = (nvmlInit_func)dlsym(nvml_handle, "nvmlInit");
  shutdown = (nvmlShutdown_func)dlsym(nvml_handle, "nvmlShutdown");
  device_get_count = (nvmlDeviceGetCount_func)dlsym(nvml_handle, "nvmlDeviceGetCount");
  device_get_handle_by_index = (nvmlDeviceGetHandleByIndex_func)dlsym(nvml_handle, "nvmlDeviceGetHandleByIndex");
  device_get_handle_by_UUID = (nvmlDeviceGetHandleByUUID_func)dlsym(nvml_handle, "nvmlDeviceGetHandleByUUID");
  device_get_name = (nvmlDeviceGetName_func)dlsym(nvml_handle, "nvmlDeviceGetName");
  device_get_brand = (nvmlDeviceGetBrand_func)dlsym(nvml_handle, "nvmlDeviceGetBrand");
  device_get_utilization_rates = (nvmlDeviceGetUtilizationRates_func)dlsym(nvml_handle, "nvmlDeviceGetUtilizationRates");
  device_get_memory_info = (nvmlDeviceGetMemoryInfo_func)dlsym(nvml_handle, "nvmlDeviceGetMemoryInfo");
  device_get_temperature = (nvmlDeviceGetTemperature_func)dlsym(nvml_handle, "nvmlDeviceGetTemperature");
  device_get_power_usage = (nvmlDeviceGetPowerUsage_func)dlsym(nvml_handle, "nvmlDeviceGetPowerUsage");
  device_get_power_management_limit = (nvmlDeviceGetPowerManagementLimit_func)dlsym(nvml_handle, "nvmlDeviceGetPowerManagementLimit");
  device_get_clock_info = (nvmlDeviceGetClockInfo_func)dlsym(nvml_handle, "nvmlDeviceGetClockInfo");
  device_get_num_gpu_cores = (nvmlDeviceGetNumGpuCores_func)dlsym(nvml_handle, "nvmlDeviceGetNumGpuCores");
  device_get_performance_state = (nvmlDeviceGetPerformanceState_func)dlsym(nvml_handle, "nvmlDeviceGetPerformanceState");
  device_get_fan_speed = (nvmlDeviceGetFanSpeed_func)dlsym(nvml_handle, "nvmlDeviceGetFanSpeed");
  device_get_curr_pcie_link_generation = (nvmlDeviceGetCurrPcieLinkGeneration_func)dlsym(nvml_handle, "nvmlDeviceGetCurrPcieLinkGeneration");
  device_get_curr_pcie_link_width = (nvmlDeviceGetCurrPcieLinkWidth_func)dlsym(nvml_handle, "nvmlDeviceGetCurrPcieLinkWidth");
  device_get_total_ecc_errors = (nvmlDeviceGetTotalEccErrors_func)dlsym(nvml_handle, "nvmlDeviceGetTotalEccErrors");

  // Check if all functions were loaded successfully
  if (!init || !shutdown || !device_get_count || !device_get_handle_by_index ||
      !device_get_handle_by_UUID || !device_get_name || !device_get_brand ||
      !device_get_utilization_rates || !device_get_memory_info || !device_get_temperature ||
      !device_get_power_usage || !device_get_power_management_limit || !device_get_clock_info ||
      !device_get_num_gpu_cores || !device_get_performance_state || !device_get_fan_speed ||
      !device_get_curr_pcie_link_generation || !device_get_curr_pcie_link_width || !device_get_total_ecc_errors) {
    fprintf(stderr, "Failed to load one or more NVML functions\n");
    dlclose(nvml_handle);
    nvml_handle = nullptr;
    return false;
  }

  return true;
}

// Cleanup NVML library
void NVMLDynamicLoader::cleanup() {
  if (nvml_handle != nullptr) {
    dlclose(nvml_handle);
    nvml_handle = nullptr;
    init = nullptr;
    shutdown = nullptr;
    device_get_count = nullptr;
    device_get_handle_by_index = nullptr;
    device_get_handle_by_UUID = nullptr;
    device_get_name = nullptr;
    device_get_brand = nullptr;
    device_get_utilization_rates = nullptr;
    device_get_memory_info = nullptr;
    device_get_temperature = nullptr;
    device_get_power_usage = nullptr;
    device_get_power_management_limit = nullptr;
    device_get_clock_info = nullptr;
    device_get_num_gpu_cores = nullptr;
    device_get_performance_state = nullptr;
    device_get_fan_speed = nullptr;
    device_get_curr_pcie_link_generation = nullptr;
    device_get_curr_pcie_link_width = nullptr;
    device_get_total_ecc_errors = nullptr;
  }
}

