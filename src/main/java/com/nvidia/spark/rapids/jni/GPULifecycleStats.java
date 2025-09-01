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

import java.util.ArrayList;
import java.util.List;

/**
 * Lifecycle statistics for GPU metrics across the entire program execution.
 * Tracks min, max, average, and other aggregate statistics for each GPU metric.
 */
public class GPULifecycleStats {
    
    private final String gpuName;
    private final int deviceIndex;
    private final List<GPUInfo> samples;
    private final long startTimeMs;
    private long lastUpdateTimeMs;
    
    // Utilization stats
    private IntegerStats gpuUtilizationStats;
    private IntegerStats memoryUtilizationStats;
    
    // Memory stats (in MB)
    private LongStats memoryUsedStats;
    private LongStats memoryFreeStats;
    
    // Temperature stats
    private IntegerStats temperatureGpuStats;
    private IntegerStats temperatureMemoryStats;
    
    // Power stats
    private IntegerStats powerUsageStats;
    
    // Clock stats
    private IntegerStats graphicsClockStats;
    private IntegerStats memoryClockStats;
    private IntegerStats smClockStats;
    
    // Other stats
    private IntegerStats fanSpeedStats;
    private IntegerStats performanceStateStats;
    
    // Error counters (cumulative)
    private long maxEccSingleBitErrors;
    private long maxEccDoubleBitErrors;
    
    public GPULifecycleStats(String gpuName, int deviceIndex) {
        this.gpuName = gpuName;
        this.deviceIndex = deviceIndex;
        this.samples = new ArrayList<>();
        this.startTimeMs = System.currentTimeMillis();
        this.lastUpdateTimeMs = startTimeMs;
        
        // Initialize stats objects
        this.gpuUtilizationStats = new IntegerStats();
        this.memoryUtilizationStats = new IntegerStats();
        this.memoryUsedStats = new LongStats();
        this.memoryFreeStats = new LongStats();
        this.temperatureGpuStats = new IntegerStats();
        this.temperatureMemoryStats = new IntegerStats();
        this.powerUsageStats = new IntegerStats();
        this.graphicsClockStats = new IntegerStats();
        this.memoryClockStats = new IntegerStats();
        this.smClockStats = new IntegerStats();
        this.fanSpeedStats = new IntegerStats();
        this.performanceStateStats = new IntegerStats();
        
        this.maxEccSingleBitErrors = 0;
        this.maxEccDoubleBitErrors = 0;
    }
    
    /**
     * Add a new GPU info sample to the lifecycle statistics
     */
    public synchronized void addSample(GPUInfo info) {
        if (info.deviceIndex != this.deviceIndex) {
            return; // Wrong GPU
        }
        
        samples.add(new GPUInfo(info)); // Store a copy
        lastUpdateTimeMs = System.currentTimeMillis();
        
        // Update all statistics
        gpuUtilizationStats.addValue(info.gpuUtilization);
        memoryUtilizationStats.addValue(info.memoryUtilization);
        memoryUsedStats.addValue(info.memoryUsedMB);
        memoryFreeStats.addValue(info.memoryFreeMB);
        temperatureGpuStats.addValue(info.temperatureGpu);
        temperatureMemoryStats.addValue(info.temperatureMemory);
        powerUsageStats.addValue(info.powerUsageW);
        graphicsClockStats.addValue(info.graphicsClockMHz);
        memoryClockStats.addValue(info.memoryClockMHz);
        smClockStats.addValue(info.smClockMHz);
        fanSpeedStats.addValue(info.fanSpeedPercent);
        performanceStateStats.addValue(info.performanceState);
        
        // Update error counters (they are cumulative)
        maxEccSingleBitErrors = Math.max(maxEccSingleBitErrors, info.eccSingleBitErrors);
        maxEccDoubleBitErrors = Math.max(maxEccDoubleBitErrors, info.eccDoubleBitErrors);
    }
    
    /**
     * Get the number of samples collected
     */
    public int getSampleCount() {
        return samples.size();
    }
    
    /**
     * Get the total monitoring duration in milliseconds
     */
    public long getMonitoringDurationMs() {
        return lastUpdateTimeMs - startTimeMs;
    }
    
    /**
     * Get the total monitoring duration in seconds
     */
    public double getMonitoringDurationSeconds() {
        return getMonitoringDurationMs() / 1000.0;
    }
    
    /**
     * Get average sampling rate (samples per second)
     */
    public double getAverageSamplingRate() {
        double durationSec = getMonitoringDurationSeconds();
        return durationSec > 0 ? getSampleCount() / durationSec : 0.0;
    }
    
    /**
     * Generate comprehensive lifecycle statistics report
     */
    public String generateReport() {
        if (samples.isEmpty()) {
            return String.format("No samples collected for GPU_%d (%s)", deviceIndex, gpuName);
        }
        
        StringBuilder sb = new StringBuilder();
        
        // Header
        String separator60 = "============================================================";
        sb.append(separator60).append("\n");
        sb.append(String.format("GPU_%d Lifecycle Statistics: %s\n", deviceIndex, gpuName));
        sb.append(separator60).append("\n");
        
        // Basic info
        sb.append(String.format("Monitoring Duration: %.2f seconds\n", getMonitoringDurationSeconds()));
        sb.append(String.format("Total Samples: %d (avg %.1f samples/sec)\n", 
                  getSampleCount(), getAverageSamplingRate()));
        
        // Hardware info (from latest sample)
        GPUInfo latest = samples.get(samples.size() - 1);
        sb.append(String.format("Hardware: %d SMs, %s\n", 
                  latest.streamingMultiprocessors, latest.getPCIeDescription()));
        sb.append("\n");
        
        // Utilization statistics (with NVML API annotations)
        sb.append("Utilization Statistics [nvmlDeviceGetUtilizationRates]:\n");
        sb.append(String.format("  GPU Utilization:    %s%% [nvmlUtilization_t.gpu]\n", gpuUtilizationStats.toString()));
        sb.append(String.format("  Memory Utilization: %s%% [nvmlUtilization_t.memory]\n", memoryUtilizationStats.toString()));
        sb.append("\n");
        
        // Memory statistics
        sb.append("Memory Statistics [nvmlDeviceGetMemoryInfo]:\n");
        sb.append(String.format("  Memory Used (MB):   %s [nvmlMemory_t.used]\n", memoryUsedStats.toString()));
        sb.append(String.format("  Memory Free (MB):   %s [nvmlMemory_t.free]\n", memoryFreeStats.toString()));
        sb.append(String.format("  Total Memory:       %d MB [nvmlMemory_t.total]\n", latest.memoryTotalMB));
        sb.append("\n");
        
        // Temperature statistics
        sb.append("Temperature Statistics [nvmlDeviceGetTemperature]:\n");
        sb.append(String.format("  GPU Temperature:    %s°C [NVML_TEMPERATURE_GPU]\n", temperatureGpuStats.toString()));
        sb.append(String.format("  Memory Temperature: %s°C [fallback to GPU temp]\n", temperatureMemoryStats.toString()));
        sb.append("\n");
        
        // Power statistics
        sb.append("Power Statistics [nvmlDeviceGetPowerUsage/nvmlDeviceGetPowerManagementLimit]:\n");
        sb.append(String.format("  Power Usage (W):    %s [nvmlDeviceGetPowerUsage]\n", powerUsageStats.toString()));
        sb.append(String.format("  Power Limit:        %d W [nvmlDeviceGetPowerManagementLimit]\n", latest.powerLimitW));
        sb.append("\n");
        
        // Clock statistics
        sb.append("Clock Statistics [nvmlDeviceGetClockInfo]:\n");
        sb.append(String.format("  Graphics Clock:     %s MHz [NVML_CLOCK_GRAPHICS]\n", graphicsClockStats.toString()));
        sb.append(String.format("  Memory Clock:       %s MHz [NVML_CLOCK_MEM]\n", memoryClockStats.toString()));
        sb.append(String.format("  SM Clock:           %s MHz [NVML_CLOCK_SM]\n", smClockStats.toString()));
        sb.append("\n");
        
        // Other statistics
        sb.append("Other Statistics:\n");
        sb.append(String.format("  SM Count:           %d [nvmlDeviceGetNumGpuCores]\n", latest.streamingMultiprocessors));
        sb.append(String.format("  Fan Speed:          %s%% [nvmlDeviceGetFanSpeed]\n", fanSpeedStats.toString()));
        sb.append(String.format("  Performance State:  %s (avg P%d) [nvmlDeviceGetPerformanceState]\n", 
                  performanceStateStats.toString(), performanceStateStats.getAverage()));
        sb.append("\n");
        
        // Error statistics
        if (maxEccSingleBitErrors > 0 || maxEccDoubleBitErrors > 0) {
            sb.append("ECC Error Statistics:\n");
            sb.append(String.format("  Single-bit Errors:  %d (max during monitoring)\n", maxEccSingleBitErrors));
            sb.append(String.format("  Double-bit Errors:  %d (max during monitoring)\n", maxEccDoubleBitErrors));
            sb.append("\n");
        }
        
        return sb.toString();
    }
    
    /**
     * Get compact summary for quick overview
     */
    public String getSummary() {
        if (samples.isEmpty()) {
            return String.format("GPU_%d: No data", deviceIndex);
        }
        
        return String.format("GPU_%d (%s): %.1fs, %d samples, GPU %d%% (avg), Mem %d%% (avg), %d°C (avg), %dW (avg)",
                           deviceIndex, gpuName, getMonitoringDurationSeconds(), getSampleCount(),
                           gpuUtilizationStats.getAverage(), memoryUtilizationStats.getAverage(),
                           temperatureGpuStats.getAverage(), powerUsageStats.getAverage());
    }
    
    // Getters for individual stats
    public IntegerStats getGpuUtilizationStats() { return gpuUtilizationStats; }
    public IntegerStats getMemoryUtilizationStats() { return memoryUtilizationStats; }
    public LongStats getMemoryUsedStats() { return memoryUsedStats; }
    public IntegerStats getTemperatureGpuStats() { return temperatureGpuStats; }
    public IntegerStats getPowerUsageStats() { return powerUsageStats; }
    public IntegerStats getGraphicsClockStats() { return graphicsClockStats; }
    
    /**
     * Helper class for integer statistics
     */
    public static class IntegerStats {
        private int min = Integer.MAX_VALUE;
        private int max = Integer.MIN_VALUE;
        private long sum = 0;
        private int count = 0;
        
        public void addValue(int value) {
            min = Math.min(min, value);
            max = Math.max(max, value);
            sum += value;
            count++;
        }
        
        public int getMin() { return count > 0 ? min : 0; }
        public int getMax() { return count > 0 ? max : 0; }
        public int getAverage() { return count > 0 ? (int)(sum / count) : 0; }
        public int getCount() { return count; }
        
        @Override
        public String toString() {
            if (count == 0) return "No data";
            return String.format("Min: %3d, Max: %3d, Avg: %3d", getMin(), getMax(), getAverage());
        }
    }
    
    /**
     * Helper class for long statistics  
     */
    public static class LongStats {
        private long min = Long.MAX_VALUE;
        private long max = Long.MIN_VALUE;
        private long sum = 0;
        private int count = 0;
        
        public void addValue(long value) {
            min = Math.min(min, value);
            max = Math.max(max, value);
            sum += value;
            count++;
        }
        
        public long getMin() { return count > 0 ? min : 0; }
        public long getMax() { return count > 0 ? max : 0; }
        public long getAverage() { return count > 0 ? sum / count : 0; }
        public int getCount() { return count; }
        
        @Override
        public String toString() {
            if (count == 0) return "No data";
            return String.format("Min: %4d, Max: %4d, Avg: %4d", getMin(), getMax(), getAverage());
        }
    }
}
