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
        
        GPUInfo latest = samples.get(samples.size() - 1);
        StringBuilder sb = new StringBuilder();
        
        // Header
        sb.append(String.format(">>> GPU_%d Detailed Statistics: %s\n", deviceIndex, gpuName));
        
        // Basic monitoring info
        sb.append(String.format("Duration: %.2fs, Samples: %d (%.1f/s), Hardware: %d SMs, %s\n", 
                  getMonitoringDurationSeconds(), getSampleCount(), getAverageSamplingRate(),
                  latest.streamingMultiprocessors, latest.getPCIeDescription()));
        
        // Utilization stats
        sb.append(String.format("Utilization - GPU: %s%%, Memory: %s%%\n", 
                  gpuUtilizationStats.toString(), memoryUtilizationStats.toString()));
        
        // Memory stats
        sb.append(String.format("Memory - Used: %s MB, Free: %s MB, Total: %d MB\n", 
                  memoryUsedStats.toString(), memoryFreeStats.toString(), latest.memoryTotalMB));
        
        // Temperature and power
        sb.append(String.format("Thermal/Power - Temp: %s°C, Power: %s W (limit: %d W)\n", 
                  temperatureGpuStats.toString(), powerUsageStats.toString(), latest.powerLimitW));
        
        // Clock frequencies
        sb.append(String.format("Clocks - Graphics: %s MHz, Memory: %s MHz, SM: %s MHz\n", 
                  graphicsClockStats.toString(), memoryClockStats.toString(), smClockStats.toString()));
        
        // Other stats
        sb.append(String.format("Other - Fan: %s%%, Performance State: %s (avg P%d)", 
                  fanSpeedStats.toString(), performanceStateStats.toString(), 
                  performanceStateStats.getAverage()));
        
        // ECC errors if any
        if (maxEccSingleBitErrors > 0 || maxEccDoubleBitErrors > 0) {
            sb.append(String.format("\nECC Errors - Single-bit: %d, Double-bit: %d", 
                      maxEccSingleBitErrors, maxEccDoubleBitErrors));
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
