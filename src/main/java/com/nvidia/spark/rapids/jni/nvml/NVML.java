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
    
    private static native long nvmlGetDeviceHandleFromUUID(byte[] uuid);
    
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
     * Get NVML device handle for specified GPU UUID
     * @param uuid GPU UUID as byte array (same format as Cuda.getGpuUuid())
     * @return device handle, or 0 on error
     */
    private static long getDeviceHandle(byte[] uuid) {
        if (!isAvailable()) {
            return 0;
        }

        if (uuid == null) {
            logger.error("UUID cannot be null");
            return 0;
        }

        try {
            return nvmlGetDeviceHandleFromUUID(uuid);
        } catch (Exception e) {
            logger.error("Error getting device handle for UUID: {}", e.getMessage());
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
    private static GPUInfo getGPUInfo(long deviceHandle) {
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
     * Get GPU information for specified GPU UUID
     * @param uuid GPU UUID as byte array (same format as Cuda.getGpuUuid())
     */
    public static GPUInfo getGPUInfo(byte[] uuid) {
        long handle = getDeviceHandle(uuid);
        if (handle == 0) {
            return null;
        }

        return getGPUInfo(handle);
    }

    /**
     * Get device info using device handle
     */
    private static GPUDeviceInfo getDeviceInfo(long deviceHandle) {
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
     * Get device info for specified GPU UUID
     * @param uuid GPU UUID as byte array (same format as Cuda.getGpuUuid())
     */
    public static GPUDeviceInfo getDeviceInfo(byte[] uuid) {
        long handle = getDeviceHandle(uuid);
        if (handle == 0) {
            return null;
        }

        return getDeviceInfo(handle);
    }

    /**
     * Get utilization info using device handle
     */
    private static GPUUtilizationInfo getUtilizationInfo(long deviceHandle) {
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
     * Get utilization info for specified GPU UUID
     * @param uuid GPU UUID as byte array (same format as Cuda.getGpuUuid())
     */
    public static GPUUtilizationInfo getUtilizationInfo(byte[] uuid) {
        long handle = getDeviceHandle(uuid);
        if (handle == 0) {
            return null;
        }

        return getUtilizationInfo(handle);
    }

    /**
     * Get memory info using device handle
     */
    private static GPUMemoryInfo getMemoryInfo(long deviceHandle) {
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
     * Get memory info for specified GPU UUID
     * @param uuid GPU UUID as byte array (same format as Cuda.getGpuUuid())
     */
    public static GPUMemoryInfo getMemoryInfo(byte[] uuid) {
        long handle = getDeviceHandle(uuid);
        if (handle == 0) {
            return null;
        }

        return getMemoryInfo(handle);
    }

    /**
     * Get temperature info using device handle
     */
    private static GPUTemperatureInfo getTemperatureInfo(long deviceHandle) {
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
     * Get temperature info for specified GPU UUID
     * @param uuid GPU UUID as byte array (same format as Cuda.getGpuUuid())
     */
    public static GPUTemperatureInfo getTemperatureInfo(byte[] uuid) {
        long handle = getDeviceHandle(uuid);
        if (handle == 0) {
            return null;
        }

        return getTemperatureInfo(handle);
    }

    /**
     * Get power info using device handle
     */
    private static GPUPowerInfo getPowerInfo(long deviceHandle) {
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
     * Get power info for specified GPU UUID
     * @param uuid GPU UUID as byte array (same format as Cuda.getGpuUuid())
     */
    public static GPUPowerInfo getPowerInfo(byte[] uuid) {
        long handle = getDeviceHandle(uuid);
        if (handle == 0) {
            return null;
        }
        return getPowerInfo(handle);
    }

    /**
     * Get clock info using device handle
     */
    private static GPUClockInfo getClockInfo(long deviceHandle) {
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
     * Get clock info for specified GPU UUID
     * @param uuid GPU UUID as byte array (same format as Cuda.getGpuUuid())
     */
    public static GPUClockInfo getClockInfo(byte[] uuid) {
        long handle = getDeviceHandle(uuid);
        if (handle == 0) {
            return null;
        }
        return getClockInfo(handle);
    }

    /**
     * Get hardware info using device handle
     */
    private static GPUHardwareInfo getHardwareInfo(long deviceHandle) {
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
     * Get hardware info for specified GPU UUID
     * @param uuid GPU UUID as byte array (same format as Cuda.getGpuUuid())
     */
    public static GPUHardwareInfo getHardwareInfo(byte[] uuid) {
        long handle = getDeviceHandle(uuid);
        if (handle == 0) {
            return null;
        }
        return getHardwareInfo(handle);
    }

    /**
     * Get PCIe info using device handle
     */
    private static GPUPCIeInfo getPCIeInfo(long deviceHandle) {
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
     * Get PCIe info for specified GPU UUID
     * @param uuid GPU UUID as byte array (same format as Cuda.getGpuUuid())
     */
    public static GPUPCIeInfo getPCIeInfo(byte[] uuid) {
        long handle = getDeviceHandle(uuid);
        if (handle == 0) {
            return null;
        }
        return getPCIeInfo(handle);
    }

    /**
     * Get ECC info using device handle
     */
    private static GPUECCInfo getECCInfo(long deviceHandle) {
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
     * Get ECC info for specified GPU UUID
     * @param uuid GPU UUID as byte array (same format as Cuda.getGpuUuid())
     */
    public static GPUECCInfo getECCInfo(byte[] uuid) {
        long handle = getDeviceHandle(uuid);
        if (handle == 0) {
            return null;
        }
        return getECCInfo(handle);
    }
}