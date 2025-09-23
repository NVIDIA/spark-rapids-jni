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

    // Native method declarations
    public static native boolean nvmlInit();
    public static native void nvmlShutdown();
    public static native int nvmlGetDeviceCount();
    public static native GPUInfo nvmlGetGPUInfo(int deviceIndex);
    public static native GPUInfo[] nvmlGetAllGPUInfo();

    // CUDA device ID support - new native methods
    public static native GPUInfo nvmlGetGPUInfoByCudaDevice(int cudaDeviceId);
    public static native boolean nvmlIsCudaDeviceValid(int cudaDeviceId);

    // Fine-grained native methods for individual info groups
    public static native GPUDeviceInfo nvmlGetDeviceInfo(int deviceIndex);
    public static native GPUUtilizationInfo nvmlGetUtilizationInfo(int deviceIndex);
    public static native GPUMemoryInfo nvmlGetMemoryInfo(int deviceIndex);
    public static native GPUTemperatureInfo nvmlGetTemperatureInfo(int deviceIndex);
    public static native GPUPowerInfo nvmlGetPowerInfo(int deviceIndex);
    public static native GPUClockInfo nvmlGetClockInfo(int deviceIndex);
    public static native GPUHardwareInfo nvmlGetHardwareInfo(int deviceIndex);
    public static native GPUPCIeInfo nvmlGetPCIeInfo(int deviceIndex);
    public static native GPUECCInfo nvmlGetECCInfo(int deviceIndex);

    // Fine-grained native methods using CUDA device IDs
    public static native GPUDeviceInfo nvmlGetDeviceInfoByCudaDevice(int cudaDeviceId);
    public static native GPUUtilizationInfo nvmlGetUtilizationInfoByCudaDevice(int cudaDeviceId);
    public static native GPUMemoryInfo nvmlGetMemoryInfoByCudaDevice(int cudaDeviceId);
    public static native GPUTemperatureInfo nvmlGetTemperatureInfoByCudaDevice(int cudaDeviceId);
    public static native GPUPowerInfo nvmlGetPowerInfoByCudaDevice(int cudaDeviceId);
    public static native GPUClockInfo nvmlGetClockInfoByCudaDevice(int cudaDeviceId);
    public static native GPUHardwareInfo nvmlGetHardwareInfoByCudaDevice(int cudaDeviceId);
    public static native GPUPCIeInfo nvmlGetPCIeInfoByCudaDevice(int cudaDeviceId);
    public static native GPUECCInfo nvmlGetECCInfoByCudaDevice(int cudaDeviceId);

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
     * Get current GPU information for all devices
     */
    public static GPUInfo[] getCurrentGPUInfo() {
        if (!isAvailable()) {
            return new GPUInfo[0];
        }

        try {
            return nvmlGetAllGPUInfo();
        } catch (Exception e) {
            logger.error("Error getting GPU info: " + e.getMessage());
            return new GPUInfo[0];
        }
    }

    /**
     * Get GPU information for specific device using NVML device index
     */
    public static GPUInfo getGPUInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetGPUInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting GPU info for device {}: {}", deviceIndex, e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU information for specific device using CUDA device ID
     */
    public static GPUInfo getGPUInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetGPUInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting GPU info for CUDA device {}: {}", cudaDeviceId, e.getMessage());
            return null;
        }
    }

    /**
     * Check if a CUDA device ID is valid
     */
    public static boolean isCudaDeviceValid(int cudaDeviceId) {
        if (!isAvailable()) {
            return false;
        }

        try {
            return nvmlIsCudaDeviceValid(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error checking CUDA device {}: {}", cudaDeviceId, e.getMessage());
            return false;
        }
    }

    /**
     * Get device info for specific device using NVML device index
     */
    public static GPUDeviceInfo getDeviceInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetDeviceInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting device info for device {}: {}", deviceIndex, e.getMessage());
            return null;
        }
    }

    /**
     * Get utilization info for specific device using NVML device index
     */
    public static GPUUtilizationInfo getUtilizationInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetUtilizationInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting utilization info for device {}: {}", deviceIndex, e.getMessage());
            return null;
        }
    }

    /**
     * Get memory info for specific device using NVML device index
     */
    public static GPUMemoryInfo getMemoryInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetMemoryInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting memory info for device {}: {}", deviceIndex, e.getMessage());
            return null;
        }
    }

    /**
     * Get temperature info for specific device using NVML device index
     */
    public static GPUTemperatureInfo getTemperatureInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetTemperatureInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting temperature info for device {}: {}", deviceIndex, e.getMessage());
            return null;
        }
    }

    /**
     * Get power info for specific device using NVML device index
     */
    public static GPUPowerInfo getPowerInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetPowerInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting power info for device {}: {}", deviceIndex, e.getMessage());
            return null;
        }
    }

    /**
     * Get clock info for specific device using NVML device index
     */
    public static GPUClockInfo getClockInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetClockInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting clock info for device {}: {}", deviceIndex, e.getMessage());
            return null;
        }
    }

    /**
     * Get hardware info for specific device using NVML device index
     */
    public static GPUHardwareInfo getHardwareInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetHardwareInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting hardware info for device {}: {}", deviceIndex, e.getMessage());
            return null;
        }
    }

    /**
     * Get PCIe info for specific device using NVML device index
     */
    public static GPUPCIeInfo getPCIeInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetPCIeInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting PCIe info for device {}: {}", deviceIndex, e.getMessage());
            return null;
        }
    }

    /**
     * Get ECC info for specific device using NVML device index
     */
    public static GPUECCInfo getECCInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetECCInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting ECC info for device {}: {}", deviceIndex, e.getMessage());
            return null;
        }
    }

    /**
     * Get device info for specific device using CUDA device ID
     */
    public static GPUDeviceInfo getDeviceInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetDeviceInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting device info for CUDA device {}: {}", cudaDeviceId, e.getMessage());
            return null;
        }
    }

    /**
     * Get utilization info for specific device using CUDA device ID
     */
    public static GPUUtilizationInfo getUtilizationInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetUtilizationInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting utilization info for CUDA device {}: {}", cudaDeviceId, e.getMessage());
            return null;
        }
    }

    /**
     * Get memory info for specific device using CUDA device ID
     */
    public static GPUMemoryInfo getMemoryInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetMemoryInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting memory info for CUDA device {}: {}", cudaDeviceId, e.getMessage());
            return null;
        }
    }

    /**
     * Get temperature info for specific device using CUDA device ID
     */
    public static GPUTemperatureInfo getTemperatureInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetTemperatureInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting temperature info for CUDA device {}: {}", cudaDeviceId, e.getMessage());
            return null;
        }
    }

    /**
     * Get power info for specific device using CUDA device ID
     */
    public static GPUPowerInfo getPowerInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetPowerInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting power info for CUDA device {}: {}", cudaDeviceId, e.getMessage());
            return null;
        }
    }

    /**
     * Get clock info for specific device using CUDA device ID
     */
    public static GPUClockInfo getClockInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetClockInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting clock info for CUDA device {}: {}", cudaDeviceId, e.getMessage());
            return null;
        }
    }

    /**
     * Get hardware info for specific device using CUDA device ID
     */
    public static GPUHardwareInfo getHardwareInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetHardwareInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting hardware info for CUDA device {}: {}", cudaDeviceId, e.getMessage());
            return null;
        }
    }

    /**
     * Get PCIe info for specific device using CUDA device ID
     */
    public static GPUPCIeInfo getPCIeInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetPCIeInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting PCIe info for CUDA device {}: {}", cudaDeviceId, e.getMessage());
            return null;
        }
    }

    /**
     * Get ECC info for specific device using CUDA device ID
     */
    public static GPUECCInfo getECCInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }

        try {
            return nvmlGetECCInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting ECC info for CUDA device {}: {}", cudaDeviceId, e.getMessage());
            return null;
        }
    }
}
