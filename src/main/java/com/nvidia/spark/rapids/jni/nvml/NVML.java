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
import ai.rapids.cudf.NativeDepsLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Low-level JNI wrapper for NVML native library calls.
 * This class provides direct access to NVML functionality without any monitoring logic.
 */
public class NVML {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    private static final Logger logger = LoggerFactory.getLogger(NVML.class);

    private static native boolean nvmlInit();
    private static native void nvmlShutdown();

    private static native int nvmlGetDeviceCount();
    
    private static native long nvmlGetDeviceHandleFromCudaDevice(int cudaDeviceId);
    
    // Coarse-grained native methods
    private static native GPUInfo nvmlGetGPUInfo(long deviceHandle);
    private static native GPUInfo[] nvmlGetAllGPUInfo();

    // Fine-grained native methods
    private static native GPUDeviceInfo nvmlGetDeviceInfo(long deviceHandle);
    private static native GPUUtilizationInfo nvmlGetUtilizationInfo(long deviceHandle);
    private static native GPUMemoryInfo nvmlGetMemoryInfo(long deviceHandle);
    private static native GPUTemperatureInfo nvmlGetTemperatureInfo(long deviceHandle);
    private static native GPUPowerInfo nvmlGetPowerInfo(long deviceHandle);
    private static native GPUClockInfo nvmlGetClockInfo(long deviceHandle);
    private static native GPUHardwareInfo nvmlGetHardwareInfo(long deviceHandle);
    private static native GPUPCIeInfo nvmlGetPCIeInfo(long deviceHandle);
    private static native GPUECCInfo nvmlGetECCInfo(long deviceHandle);

    // Initialization state
    private static boolean nvmlInitialized = false;
    private static boolean nativeLibraryLoaded = false;

    /**
     * Initialize NVML library
     */
    public static synchronized boolean initialize() {
        if (nvmlInitialized) {
            return true;
        }

        try {
            if (nvmlInit()) {
                nvmlInitialized = true;
                nativeLibraryLoaded = true;
                return true;
            } else {
                logger.error("Failed to initialize NVML");
                return false;
            }
        } catch (UnsatisfiedLinkError e) {
            logger.error("NVML JNI not available: " + e.getMessage());
            nativeLibraryLoaded = false;
            return false;
        }
    }

    /**
     * Shutdown NVML library
     */
    public static synchronized void shutdown() {
        if (nvmlInitialized && nativeLibraryLoaded) {
            try {
                nvmlShutdown();
                nvmlInitialized = false;
            } catch (UnsatisfiedLinkError e) {
                logger.error("Error during NVML shutdown: " + e.getMessage());
            }
        }
    }

    /**
     * Check if NVML is available and initialized
     */
    public static boolean isAvailable() {
        return nvmlInitialized && nativeLibraryLoaded;
    }

    /**
     * Get number of GPU devices
     */
    public static int getDeviceCount() {
        if (!isAvailable()) {
            return 0;
        }

        try {
            return nvmlGetDeviceCount();
        } catch (Exception e) {
            logger.error("Error getting device count: " + e.getMessage());
            return 0;
        }
    }

    /**
     * Get NVML device handle for specified CUDA device
     */
    public static long getDeviceHandle(int cudaDevice) {
        if (!isAvailable()) {
            return 0;
        }

        try {
            return nvmlGetDeviceHandleFromCudaDevice(cudaDevice);
        } catch (Exception e) {
            logger.error("Error getting device handle for CUDA device {}: {}", cudaDevice, e.getMessage());
            return 0;
        }
    }

    /**
     * Get NVML device handle for current CUDA device  
     */
    public static long getDeviceHandle() {
        try {
            int cudaDevice = Cuda.getDevice();
            return getDeviceHandle(cudaDevice);
        } catch (Exception e) {
            logger.error("Error getting current CUDA device: {}", e.getMessage());
            return 0;
        }
    }

    /**
     * Get GPU information for all devices
     */
    public static GPUInfo[] getAllGPUInfo() {
        if (!isAvailable()) {
            return new GPUInfo[0];
        }

        try {
            return nvmlGetAllGPUInfo();
        } catch (Exception e) {
            logger.error("Error getting GPU info for all devices: " + e.getMessage());
            return new GPUInfo[0];
        }
    }

    /**
     * Get GPU information using device handle
     */
    public static GPUInfo getGPUInfo(long deviceHandle) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetGPUInfo(deviceHandle);
        } catch (Exception e) {
            logger.error("Error getting GPU info for device handle {}: {}", deviceHandle, e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU information for current CUDA device
     */
    public static GPUInfo getGPUInfo() {
        long handle = getDeviceHandle();
        if (handle == 0) {
            return null;
        }

        return getGPUInfo(handle);
    }

    /**
     * Get device info using device handle
     */
    public static GPUDeviceInfo getDeviceInfo(long deviceHandle) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetDeviceInfo(deviceHandle);
        } catch (Exception e) {
            logger.error("Error getting device info for device handle {}: {}", deviceHandle, e.getMessage());
            return null;
        }
    }

    /**
     * Get device info for current CUDA device
     */
    public static GPUDeviceInfo getDeviceInfo() {
        long handle = getDeviceHandle();
        if (handle == 0) {
            return null;
        }

        return getDeviceInfo(handle);
    }

    /**
     * Get utilization info using device handle
     */
    public static GPUUtilizationInfo getUtilizationInfo(long deviceHandle) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetUtilizationInfo(deviceHandle);
        } catch (Exception e) {
            logger.error("Error getting utilization info for device handle {}: {}", deviceHandle, e.getMessage());
            return null;
        }
    }

    /**
     * Get utilization info for current CUDA device
     */
    public static GPUUtilizationInfo getUtilizationInfo() {
        long handle = getDeviceHandle();
        if (handle == 0) {
            return null;
        }

        return getUtilizationInfo(handle);
    }

    /**
     * Get memory info using device handle
     */
    public static GPUMemoryInfo getMemoryInfo(long deviceHandle) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetMemoryInfo(deviceHandle);
        } catch (Exception e) {
            logger.error("Error getting memory info for device handle {}: {}", deviceHandle, e.getMessage());
            return null;
        }
    }

    /**
     * Get memory info for current CUDA device
     */
    public static GPUMemoryInfo getMemoryInfo() {
        long handle = getDeviceHandle();
        if (handle == 0) {
            return null;
        }

        return getMemoryInfo(handle);
    }

    /**
     * Get temperature info using device handle
     */
    public static GPUTemperatureInfo getTemperatureInfo(long deviceHandle) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetTemperatureInfo(deviceHandle);
        } catch (Exception e) {
            logger.error("Error getting temperature info for device handle {}: {}", deviceHandle, e.getMessage());
            return null;
        }
    }

    /**
     * Get temperature info for current CUDA device
     */
    public static GPUTemperatureInfo getTemperatureInfo() {
        long handle = getDeviceHandle();
        if (handle == 0) {
            return null;
        }

        return getTemperatureInfo(handle);
    }

    /**
     * Get power info using device handle
     */
    public static GPUPowerInfo getPowerInfo(long deviceHandle) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetPowerInfo(deviceHandle);
        } catch (Exception e) {
            logger.error("Error getting power info for device handle {}: {}", deviceHandle, e.getMessage());
            return null;
        }
    }

    /**
     * Get power info for current CUDA device
     */
    public static GPUPowerInfo getPowerInfo() {
        long handle = getDeviceHandle();
        if (handle == 0) {
            return null;
        }
        return getPowerInfo(handle);
    }

    /**
     * Get clock info using device handle
     */
    public static GPUClockInfo getClockInfo(long deviceHandle) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetClockInfo(deviceHandle);
        } catch (Exception e) {
            logger.error("Error getting clock info for device handle {}: {}", deviceHandle, e.getMessage());
            return null;
        }
    }

    /**
     * Get clock info for current CUDA device
     */
    public static GPUClockInfo getClockInfo() {
        long handle = getDeviceHandle();
        if (handle == 0) {
            return null;
        }
        return getClockInfo(handle);
    }

    /**
     * Get hardware info using device handle
     */
    public static GPUHardwareInfo getHardwareInfo(long deviceHandle) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetHardwareInfo(deviceHandle);
        } catch (Exception e) {
            logger.error("Error getting hardware info for device handle {}: {}", deviceHandle, e.getMessage());
            return null;
        }
    }

    /**
     * Get hardware info for current CUDA device
     */
    public static GPUHardwareInfo getHardwareInfo() {
        long handle = getDeviceHandle();
        if (handle == 0) {
            return null;
        }
        return getHardwareInfo(handle);
    }

    /**
     * Get PCIe info using device handle
     */
    public static GPUPCIeInfo getPCIeInfo(long deviceHandle) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetPCIeInfo(deviceHandle);
        } catch (Exception e) {
            logger.error("Error getting PCIe info for device handle {}: {}", deviceHandle, e.getMessage());
            return null;
        }
    }

    /**
     * Get PCIe info for current CUDA device
     */
    public static GPUPCIeInfo getPCIeInfo() {
        long handle = getDeviceHandle();
        if (handle == 0) {
            return null;
        }
        return getPCIeInfo(handle);
    }

    /**
     * Get ECC info using device handle
     */
    public static GPUECCInfo getECCInfo(long deviceHandle) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetECCInfo(deviceHandle);
        } catch (Exception e) {
            logger.error("Error getting ECC info for device handle {}: {}", deviceHandle, e.getMessage());
            return null;
        }
    }

    /**
     * Get ECC info for current CUDA device
     */
    public static GPUECCInfo getECCInfo() {
        long handle = getDeviceHandle();
        if (handle == 0) {
            return null;
        }
        return getECCInfo(handle);
    }
}