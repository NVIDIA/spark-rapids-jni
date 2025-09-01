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
package com.nvidia.spark.rapids.jni;

/**
 * Comprehensive GPU information collected via NVML
 * Contains all major GPU metrics for detailed monitoring and analysis
 */
public class GPUInfo {
    
    // Basic device info
    public int deviceIndex;
    public String name;
    public String brand;
    
    // Utilization metrics
    public int gpuUtilization;          // GPU utilization percentage (0-100)
    public int memoryUtilization;       // Memory utilization percentage (0-100)
    
    // Memory metrics
    public long memoryUsedMB;           // Used memory in MB
    public long memoryTotalMB;          // Total memory in MB  
    public long memoryFreeMB;           // Free memory in MB
    
    // Temperature metrics
    public int temperatureGpu;          // GPU temperature in Celsius
    public int temperatureMemory;       // Memory temperature in Celsius
    
    // Power metrics
    public int powerUsageW;             // Current power usage in Watts
    public int powerLimitW;             // Current power limit in Watts
    public int powerDefaultLimitW;      // Default power limit in Watts
    
    // Clock speeds
    public int graphicsClockMHz;        // Graphics clock in MHz
    public int memoryClockMHz;          // Memory clock in MHz
    public int smClockMHz;              // SM clock in MHz
    
    // Hardware info
    public int streamingMultiprocessors; // Number of SMs
    public int performanceState;        // Performance state (P-state)
    public int fanSpeedPercent;         // Fan speed percentage
    
    // PCIe info
    public int pcieLinkGeneration;      // PCIe generation (1, 2, 3, 4, etc.)
    public int pcieLinkWidth;           // PCIe width (x1, x4, x8, x16)
    
    // Error counters
    public long eccSingleBitErrors;     // Single bit ECC errors
    public long eccDoubleBitErrors;     // Double bit ECC errors
    
    // Timestamp when this info was collected
    public long timestampMs;
    
    public GPUInfo() {
        this.timestampMs = System.currentTimeMillis();
    }
    
    /**
     * Copy constructor for creating snapshots
     */
    public GPUInfo(GPUInfo other) {
        this.deviceIndex = other.deviceIndex;
        this.name = other.name;
        this.brand = other.brand;
        this.gpuUtilization = other.gpuUtilization;
        this.memoryUtilization = other.memoryUtilization;
        this.memoryUsedMB = other.memoryUsedMB;
        this.memoryTotalMB = other.memoryTotalMB;
        this.memoryFreeMB = other.memoryFreeMB;
        this.temperatureGpu = other.temperatureGpu;
        this.temperatureMemory = other.temperatureMemory;
        this.powerUsageW = other.powerUsageW;
        this.powerLimitW = other.powerLimitW;
        this.powerDefaultLimitW = other.powerDefaultLimitW;
        this.graphicsClockMHz = other.graphicsClockMHz;
        this.memoryClockMHz = other.memoryClockMHz;
        this.smClockMHz = other.smClockMHz;
        this.streamingMultiprocessors = other.streamingMultiprocessors;
        this.performanceState = other.performanceState;
        this.fanSpeedPercent = other.fanSpeedPercent;
        this.pcieLinkGeneration = other.pcieLinkGeneration;
        this.pcieLinkWidth = other.pcieLinkWidth;
        this.eccSingleBitErrors = other.eccSingleBitErrors;
        this.eccDoubleBitErrors = other.eccDoubleBitErrors;
        this.timestampMs = System.currentTimeMillis();
    }
    
    /**
     * Get memory utilization as percentage based on used/total
     */
    public double getMemoryUtilizationPercent() {
        if (memoryTotalMB == 0) return 0.0;
        return (double) memoryUsedMB / memoryTotalMB * 100.0;
    }
    
    /**
     * Get PCIe generation description
     */
    public String getPCIeDescription() {
        return "PCIe Gen" + pcieLinkGeneration + " x" + pcieLinkWidth;
    }
    
    /**
     * Get performance state description
     */
    public String getPerformanceStateDescription() {
        switch (performanceState) {
            case 0: return "P0 (Maximum Performance)";
            case 1: return "P1 (High Performance)";
            case 2: return "P2 (Moderate Performance)";
            case 3: return "P3 (Battery Optimized)";
            case 8: return "P8 (Minimum Performance)";
            case 12: return "P12 (Lowest Performance)";
            default: return "P" + performanceState;
        }
    }
    
    /**
     * Check if ECC is supported and has errors
     */
    public boolean hasECCErrors() {
        return eccSingleBitErrors > 0 || eccDoubleBitErrors > 0;
    }
    
    /**
     * Get total ECC errors
     */
    public long getTotalECCErrors() {
        return eccSingleBitErrors + eccDoubleBitErrors;
    }
    
    /**
     * Format basic GPU info for display
     */
    @Override
    public String toString() {
        return String.format("GPU_%d (%s): Util: %d%%, Mem: %d%% (%dMB/%dMB), " +
                           "Temp: %d°C, Power: %dW/%dW, Clocks: %d/%d MHz",
                           deviceIndex, name, gpuUtilization, memoryUtilization,
                           memoryUsedMB, memoryTotalMB, temperatureGpu, powerUsageW, powerLimitW,
                           graphicsClockMHz, memoryClockMHz);
    }
    
    /**
     * Get detailed multi-line format with NVML API annotations
     */
    public String toDetailedString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("GPU_%d: %s (%s)\n", deviceIndex, 
                  name, // nvmlDeviceGetName()
                  brand)); // nvmlDeviceGetBrand()
        sb.append(String.format("  Utilization: GPU %d%% [nvmlDeviceGetUtilizationRates().gpu], Memory %d%% [nvmlDeviceGetUtilizationRates().memory]\n", 
                  gpuUtilization, memoryUtilization));
        sb.append(String.format("  Memory: %d/%d MB (%.1f%% used) [nvmlDeviceGetMemoryInfo()]\n", 
                  memoryUsedMB, memoryTotalMB, getMemoryUtilizationPercent()));
        sb.append(String.format("  Temperature: GPU %d°C [nvmlDeviceGetTemperature(NVML_TEMPERATURE_GPU)], Memory %d°C [fallback]\n", 
                  temperatureGpu, temperatureMemory));
        sb.append(String.format("  Power: %d/%d W [nvmlDeviceGetPowerUsage/nvmlDeviceGetPowerManagementLimit] (default: %d W)\n", 
                  powerUsageW, powerLimitW, powerDefaultLimitW));
        sb.append(String.format("  Clocks: Graphics %d MHz [nvmlDeviceGetClockInfo(NVML_CLOCK_GRAPHICS)], Memory %d MHz [NVML_CLOCK_MEM], SM %d MHz [NVML_CLOCK_SM]\n", 
                  graphicsClockMHz, memoryClockMHz, smClockMHz));
        sb.append(String.format("  Hardware: %d SMs [nvmlDeviceGetNumGpuCores], %s [nvmlDeviceGetCurrPcieLinkGeneration/Width], Fan %d%% [nvmlDeviceGetFanSpeed]\n", 
                  streamingMultiprocessors, getPCIeDescription(), fanSpeedPercent));
        sb.append(String.format("  State: %s [nvmlDeviceGetPerformanceState]\n", getPerformanceStateDescription()));
        if (hasECCErrors()) {
            sb.append(String.format("  ECC Errors: %d single-bit, %d double-bit [nvmlDeviceGetTotalEccErrors]\n", 
                      eccSingleBitErrors, eccDoubleBitErrors));
        }
        return sb.toString();
    }
    
    /**
     * Get compact single-line format for logs
     */
    public String toCompactString() {
        return String.format("GPU_%d: %d%%/%d%% %dMB %d°C %dW %dMHz",
                           deviceIndex, gpuUtilization, memoryUtilization, 
                           memoryUsedMB, temperatureGpu, powerUsageW, graphicsClockMHz);
    }
}
