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

package com.nvidia.spark.rapids.jni.nvml;

import ai.rapids.cudf.Cuda;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for NVML JNI wrapper functionality.
 * Tests require a CUDA-capable GPU to run.
 */
@EnabledIfSystemProperty(named = "ai.rapids.cudf.nvml.test.enabled", matches = "true",
    disabledReason = "NVML tests require GPU and NVML library")
public class NVMLTest {

  @BeforeAll
  public static void setup() {
    // Set CUDA device to 0 before running tests
    Cuda.setDevice(0);
    
    // Initialize NVML before running tests
    assertTrue(NVML.initialize(), "NVML initialization should succeed");
    assertTrue(NVML.isAvailable(), "NVML should be available after initialization");
  }

  @AfterAll
  public static void teardown() {
    // Shutdown NVML after all tests
    NVML.shutdown();
  }

  @Test
  public void testGetDeviceCount() {
    int deviceCount = NVML.getDeviceCount();
    assertTrue(deviceCount > 0, "Device count should be greater than 0");
    System.out.println("Found " + deviceCount + " GPU device(s)");
  }

  @Test
  public void testGetDeviceInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    GPUDeviceInfo deviceInfo = NVML.getDeviceInfo(uuid);
    
    assertNotNull(deviceInfo, "Device info should not be null");
    assertNotNull(deviceInfo.name, "Device name should not be null");
    assertNotNull(deviceInfo.brand, "Device brand should not be null");
    
    System.out.println("Device name: " + deviceInfo.name);
    System.out.println("Device brand: " + deviceInfo.brand);
  }

  @Test
  public void testGetUtilizationInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    GPUUtilizationInfo utilizationInfo = NVML.getUtilizationInfo(uuid);
    
    assertNotNull(utilizationInfo, "Utilization info should not be null");
    assertTrue(utilizationInfo.gpuUtilization >= 0 && utilizationInfo.gpuUtilization <= 100,
        "GPU utilization should be between 0 and 100");
    assertTrue(utilizationInfo.memoryUtilization >= 0 && utilizationInfo.memoryUtilization <= 100,
        "Memory utilization should be between 0 and 100");
    
    System.out.println("GPU utilization: " + utilizationInfo.gpuUtilization + "%");
    System.out.println("Memory utilization: " + utilizationInfo.memoryUtilization + "%");
  }

  @Test
  public void testGetMemoryInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    GPUMemoryInfo memoryInfo = NVML.getMemoryInfo(uuid);
    
    assertNotNull(memoryInfo, "Memory info should not be null");
    assertTrue(memoryInfo.memoryTotalMB > 0, "Total memory should be greater than 0");
    assertTrue(memoryInfo.memoryUsedMB >= 0, "Used memory should be non-negative");
    assertTrue(memoryInfo.memoryFreeMB >= 0, "Free memory should be non-negative");
    assertTrue(memoryInfo.memoryUsedMB + memoryInfo.memoryFreeMB <= memoryInfo.memoryTotalMB,
        "Used + Free should not exceed Total memory");
    
    System.out.println("Total memory: " + memoryInfo.memoryTotalMB + " MB");
    System.out.println("Used memory: " + memoryInfo.memoryUsedMB + " MB");
    System.out.println("Free memory: " + memoryInfo.memoryFreeMB + " MB");
  }

  @Test
  public void testGetTemperatureInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    GPUTemperatureInfo temperatureInfo = NVML.getTemperatureInfo(uuid);
    
    assertNotNull(temperatureInfo, "Temperature info should not be null");
    assertTrue(temperatureInfo.temperatureGpu > 0 && temperatureInfo.temperatureGpu < 150,
        "GPU temperature should be in reasonable range (0-150°C)");
    
    System.out.println("GPU temperature: " + temperatureInfo.temperatureGpu + "°C");
  }

  @Test
  public void testGetPowerInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    GPUPowerInfo powerInfo = NVML.getPowerInfo(uuid);
    
    assertNotNull(powerInfo, "Power info should not be null");
    assertTrue(powerInfo.powerUsageW >= 0, "Power usage should be non-negative");
    assertTrue(powerInfo.powerLimitW > 0, "Power limit should be greater than 0");
    assertTrue(powerInfo.powerUsageW <= powerInfo.powerLimitW * 2,
        "Power usage should be within reasonable range of power limit");
    
    System.out.println("Power usage: " + powerInfo.powerUsageW + " W");
    System.out.println("Power limit: " + powerInfo.powerLimitW + " W");
  }

  @Test
  public void testGetClockInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    GPUClockInfo clockInfo = NVML.getClockInfo(uuid);
    
    assertNotNull(clockInfo, "Clock info should not be null");
    assertTrue(clockInfo.graphicsClockMHz >= 0, "Graphics clock should be non-negative");
    assertTrue(clockInfo.memoryClockMHz >= 0, "Memory clock should be non-negative");
    assertTrue(clockInfo.smClockMHz >= 0, "SM clock should be non-negative");
    
    System.out.println("Graphics clock: " + clockInfo.graphicsClockMHz + " MHz");
    System.out.println("Memory clock: " + clockInfo.memoryClockMHz + " MHz");
    System.out.println("SM clock: " + clockInfo.smClockMHz + " MHz");
  }

  @Test
  public void testGetHardwareInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    GPUHardwareInfo hardwareInfo = NVML.getHardwareInfo(uuid);
    
    assertNotNull(hardwareInfo, "Hardware info should not be null");
    assertTrue(hardwareInfo.streamingMultiprocessors > 0, "SM count should be greater than 0");
    assertTrue(hardwareInfo.performanceState >= 0 && hardwareInfo.performanceState <= 32,
        "Performance state should be in valid range");
    assertTrue(hardwareInfo.fanSpeedPercent >= 0 && hardwareInfo.fanSpeedPercent <= 100,
        "Fan speed should be between 0 and 100");
    
    System.out.println("Streaming Multiprocessors: " + hardwareInfo.streamingMultiprocessors);
    System.out.println("Performance state: P" + hardwareInfo.performanceState);
    System.out.println("Fan speed: " + hardwareInfo.fanSpeedPercent + "%");
  }

  @Test
  public void testGetPCIeInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    GPUPCIeInfo pcieInfo = NVML.getPCIeInfo(uuid);
    
    assertNotNull(pcieInfo, "PCIe info should not be null");
    assertTrue(pcieInfo.pcieLinkGeneration > 0, "PCIe link generation should be greater than 0");
    assertTrue(pcieInfo.pcieLinkWidth > 0, "PCIe link width should be greater than 0");
    
    System.out.println("PCIe link generation: " + pcieInfo.pcieLinkGeneration);
    System.out.println("PCIe link width: x" + pcieInfo.pcieLinkWidth);
  }

  @Test
  public void testGetECCInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    GPUECCInfo eccInfo = NVML.getECCInfo(uuid);
    
    assertNotNull(eccInfo, "ECC info should not be null");
    assertTrue(eccInfo.eccSingleBitErrors >= 0, "Single-bit ECC errors should be non-negative");
    assertTrue(eccInfo.eccDoubleBitErrors >= 0, "Double-bit ECC errors should be non-negative");
    
    System.out.println("Single-bit ECC errors: " + eccInfo.eccSingleBitErrors);
    System.out.println("Double-bit ECC errors: " + eccInfo.eccDoubleBitErrors);
  }

  @Test
  public void testGetGPUInfo() {
    byte[] uuid = Cuda.getGpuUuid();
    GPUInfo gpuInfo = NVML.getGPUInfo(uuid);
    
    assertNotNull(gpuInfo, "GPU info should not be null");
    assertNotNull(gpuInfo.deviceInfo, "Device info should not be null");
    assertNotNull(gpuInfo.utilizationInfo, "Utilization info should not be null");
    assertNotNull(gpuInfo.memoryInfo, "Memory info should not be null");
    assertNotNull(gpuInfo.temperatureInfo, "Temperature info should not be null");
    assertNotNull(gpuInfo.powerInfo, "Power info should not be null");
    assertNotNull(gpuInfo.clockInfo, "Clock info should not be null");
    assertNotNull(gpuInfo.hardwareInfo, "Hardware info should not be null");
    assertNotNull(gpuInfo.pcieInfo, "PCIe info should not be null");
    assertNotNull(gpuInfo.eccInfo, "ECC info should not be null");
    
    System.out.println("===== Complete GPU Info =====");
    System.out.println("Device: " + gpuInfo.deviceInfo.name);
    System.out.println("Memory: " + gpuInfo.memoryInfo.memoryUsedMB + "/" + 
                       gpuInfo.memoryInfo.memoryTotalMB + " MB");
    System.out.println("Temperature: " + gpuInfo.temperatureInfo.temperatureGpu + "°C");
    System.out.println("Power: " + gpuInfo.powerInfo.powerUsageW + "/" + 
                       gpuInfo.powerInfo.powerLimitW + " W");
  }

  @Test
  public void testGetAllGPUInfo() {
    GPUInfo[] allGPUInfo = NVML.getAllGPUInfo();
    
    assertNotNull(allGPUInfo, "All GPU info array should not be null");
    assertTrue(allGPUInfo.length > 0, "Should have at least one GPU");
    
    System.out.println("===== All GPUs Info =====");
    for (int i = 0; i < allGPUInfo.length; i++) {
      GPUInfo info = allGPUInfo[i];
      assertNotNull(info, "GPU info at index " + i + " should not be null");
      assertNotNull(info.deviceInfo, "Device info at index " + i + " should not be null");
      
      System.out.println("GPU " + i + ": " + info.deviceInfo.name);
      System.out.println("  Memory: " + info.memoryInfo.memoryUsedMB + "/" + 
                         info.memoryInfo.memoryTotalMB + " MB");
      System.out.println("  Temperature: " + info.temperatureInfo.temperatureGpu + "°C");
    }
  }

  @Test
  public void testUUIDBasedMethods() {
    // Test UUID-based methods using current CUDA device UUID
    byte[] uuid = Cuda.getGpuUuid();
    assertNotNull(uuid, "GPU UUID should not be null");
    
    GPUDeviceInfo deviceInfo = NVML.getDeviceInfo(uuid);
    assertNotNull(deviceInfo, "Device info should not be null");
    
    GPUUtilizationInfo utilizationInfo = NVML.getUtilizationInfo(uuid);
    assertNotNull(utilizationInfo, "Utilization info should not be null");
    
    GPUMemoryInfo memoryInfo = NVML.getMemoryInfo(uuid);
    assertNotNull(memoryInfo, "Memory info should not be null");
    
    GPUTemperatureInfo temperatureInfo = NVML.getTemperatureInfo(uuid);
    assertNotNull(temperatureInfo, "Temperature info should not be null");
    
    GPUPowerInfo powerInfo = NVML.getPowerInfo(uuid);
    assertNotNull(powerInfo, "Power info should not be null");
    
    GPUClockInfo clockInfo = NVML.getClockInfo(uuid);
    assertNotNull(clockInfo, "Clock info should not be null");
    
    GPUHardwareInfo hardwareInfo = NVML.getHardwareInfo(uuid);
    assertNotNull(hardwareInfo, "Hardware info should not be null");
    
    GPUPCIeInfo pcieInfo = NVML.getPCIeInfo(uuid);
    assertNotNull(pcieInfo, "PCIe info should not be null");
    
    GPUECCInfo eccInfo = NVML.getECCInfo(uuid);
    assertNotNull(eccInfo, "ECC info should not be null");
    
    GPUInfo gpuInfo = NVML.getGPUInfo(uuid);
    assertNotNull(gpuInfo, "GPU info should not be null");
    
    System.out.println("All UUID-based methods executed successfully");
    System.out.println("GPU UUID: " + new String(uuid));
  }
}

