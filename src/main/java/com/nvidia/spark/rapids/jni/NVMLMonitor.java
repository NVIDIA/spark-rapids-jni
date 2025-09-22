package com.nvidia.spark.rapids.jni;

import java.util.*;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.ConcurrentHashMap;

import ai.rapids.cudf.NativeDepsLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * NVML-based GPU monitoring system with comprehensive metrics collection
 * and lifecycle statistics tracking for Spark Rapids JNI.
 */
public class NVMLMonitor {
    static {
        NativeDepsLoader.loadNativeDeps();
    }

    private static final Logger logger = LoggerFactory.getLogger(NVMLMonitor.class);

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

    // Instance variables
    private static boolean nvmlInitialized = false;
    private static boolean nativeLibraryLoaded = false;
    private final AtomicBoolean monitoring = new AtomicBoolean(false);
    private Thread monitoringThread;
    private final int intervalMs;
    private final boolean verbose;

    // Lifecycle statistics for each GPU
    private final java.util.Map<Integer, GPULifecycleStats> lifecycleStats = new ConcurrentHashMap<>();

    // Callback for real-time monitoring
    private volatile MonitoringCallback callback;

    /**
     * Callback interface for real-time GPU monitoring events
     */
    public interface MonitoringCallback {
        void onGPUUpdate(GPUInfo[] gpuInfos, long timestamp);
        void onMonitoringStarted();
        void onMonitoringStopped();
        void onError(String error);
    }

    public NVMLMonitor(int intervalMs, boolean verbose) {
        this.intervalMs = intervalMs;
        this.verbose = verbose;
    }

    public NVMLMonitor() {
        this(1000, false); // Default: 1 second interval, not verbose
    }

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
            logger.error("Error getting GPU info for device " + deviceIndex + ": " + e.getMessage());
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
            logger.error("Error getting GPU info for CUDA device " + cudaDeviceId + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Check if a CUDA device ID is valid and accessible
     */
    public static boolean isCudaDeviceValid(int cudaDeviceId) {
        if (!isAvailable()) {
            return false;
        }

        try {
            return nvmlIsCudaDeviceValid(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error checking CUDA device validity for device " + cudaDeviceId + ": " + e.getMessage());
            return false;
        }
    }

    // Fine-grained API methods using NVML device index

    /**
     * Get basic device information using NVML device index
     */
    public static GPUDeviceInfo getDeviceInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetDeviceInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting device info for device " + deviceIndex + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU utilization information using NVML device index
     */
    public static GPUUtilizationInfo getUtilizationInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetUtilizationInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting utilization info for device " + deviceIndex + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU memory information using NVML device index
     */
    public static GPUMemoryInfo getMemoryInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetMemoryInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting memory info for device " + deviceIndex + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU temperature information using NVML device index
     */
    public static GPUTemperatureInfo getTemperatureInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetTemperatureInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting temperature info for device " + deviceIndex + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU power information using NVML device index
     */
    public static GPUPowerInfo getPowerInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetPowerInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting power info for device " + deviceIndex + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU clock information using NVML device index
     */
    public static GPUClockInfo getClockInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetClockInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting clock info for device " + deviceIndex + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU hardware information using NVML device index
     */
    public static GPUHardwareInfo getHardwareInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetHardwareInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting hardware info for device " + deviceIndex + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU PCIe information using NVML device index
     */
    public static GPUPCIeInfo getPCIeInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetPCIeInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting PCIe info for device " + deviceIndex + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU ECC error information using NVML device index
     */
    public static GPUECCInfo getECCInfo(int deviceIndex) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetECCInfo(deviceIndex);
        } catch (Exception e) {
            logger.error("Error getting ECC info for device " + deviceIndex + ": " + e.getMessage());
            return null;
        }
    }

    // Fine-grained API methods using CUDA device ID

    /**
     * Get basic device information using CUDA device ID
     */
    public static GPUDeviceInfo getDeviceInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetDeviceInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting device info for CUDA device " + cudaDeviceId + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU utilization information using CUDA device ID
     */
    public static GPUUtilizationInfo getUtilizationInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetUtilizationInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting utilization info for CUDA device " + cudaDeviceId + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU memory information using CUDA device ID
     */
    public static GPUMemoryInfo getMemoryInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetMemoryInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting memory info for CUDA device " + cudaDeviceId + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU temperature information using CUDA device ID
     */
    public static GPUTemperatureInfo getTemperatureInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetTemperatureInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting temperature info for CUDA device " + cudaDeviceId + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU power information using CUDA device ID
     */
    public static GPUPowerInfo getPowerInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetPowerInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting power info for CUDA device " + cudaDeviceId + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU clock information using CUDA device ID
     */
    public static GPUClockInfo getClockInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetClockInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting clock info for CUDA device " + cudaDeviceId + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU hardware information using CUDA device ID
     */
    public static GPUHardwareInfo getHardwareInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetHardwareInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting hardware info for CUDA device " + cudaDeviceId + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU PCIe information using CUDA device ID
     */
    public static GPUPCIeInfo getPCIeInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetPCIeInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting PCIe info for CUDA device " + cudaDeviceId + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Get GPU ECC error information using CUDA device ID
     */
    public static GPUECCInfo getECCInfoByCudaDevice(int cudaDeviceId) {
        if (!isAvailable()) {
            return null;
        }
        try {
            return nvmlGetECCInfoByCudaDevice(cudaDeviceId);
        } catch (Exception e) {
            logger.error("Error getting ECC info for CUDA device " + cudaDeviceId + ": " + e.getMessage());
            return null;
        }
    }

    /**
     * Set callback for real-time monitoring events
     */
    public void setCallback(MonitoringCallback callback) {
        this.callback = callback;
    }

    /**
     * Start continuous GPU monitoring
     */
    public synchronized void startMonitoring() {
        if (monitoring.get()) {
            return; // Already monitoring
        }

        if (!isAvailable()) {
            logger.error("NVML not available, cannot start monitoring");
            return;
        }

        // Initialize lifecycle stats for all GPUs
        GPUInfo[] initialInfo = getCurrentGPUInfo();
        for (GPUInfo info : initialInfo) {
            if (info.deviceInfo != null) {
                lifecycleStats.put(info.deviceInfo.deviceIndex,
                                  new GPULifecycleStats(info.deviceInfo.name, info.deviceInfo.deviceIndex));
            }
        }

        monitoring.set(true);
        monitoringThread = new Thread(this::monitoringLoop, "NVMLMonitor");
        monitoringThread.setDaemon(true);
        monitoringThread.start();

        if (verbose) {
            logger.info("Started NVML GPU monitoring (interval: " + intervalMs + "ms)");
        }

        if (callback != null) {
            callback.onMonitoringStarted();
        }
    }

    /**
     * Stop continuous GPU monitoring
     */
    public synchronized void stopMonitoring() {
        if (!monitoring.get()) {
            return; // Not monitoring
        }

        monitoring.set(false);
        if (monitoringThread != null) {
            try {
                monitoringThread.join(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        if (verbose) {
            logger.info("Stopped NVML GPU monitoring");
        }

        if (callback != null) {
            callback.onMonitoringStopped();
        }
    }

    /**
     * Main monitoring loop (runs in background thread)
     */
    private void monitoringLoop() {
        while (monitoring.get()) {
            try {
                GPUInfo[] gpuInfos = getCurrentGPUInfo();

                if (gpuInfos != null && gpuInfos.length > 0) {
                    long timestamp = System.currentTimeMillis();

                    // Update lifecycle statistics
                    for (GPUInfo info : gpuInfos) {
                        if (info.deviceInfo != null) {
                            GPULifecycleStats stats = lifecycleStats.get(info.deviceInfo.deviceIndex);
                            if (stats != null) {
                                stats.addSample(info);
                            }
                        }
                    }

                    // Verbose output
                    if (verbose) {
                        logger.info("=== GPU Status Update ===");
                        for (GPUInfo info : gpuInfos) {
                            logger.info(info.toString());
                        }
                        logger.info("");
                    }

                    // Callback notification
                    if (callback != null) {
                        callback.onGPUUpdate(gpuInfos, timestamp);
                    }
                }

                Thread.sleep(intervalMs);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                String error = "Error in monitoring loop: " + e.getMessage();
                logger.error(error);
                if (callback != null) {
                    callback.onError(error);
                }
            }
        }
    }

    /**
     * Get lifecycle statistics for all GPUs
     */
    public Map<Integer, GPULifecycleStats> getLifecycleStats() {
        return new HashMap<>(lifecycleStats);
    }

    /**
     * Get lifecycle statistics for specific GPU
     */
    public GPULifecycleStats getLifecycleStats(int deviceIndex) {
        return lifecycleStats.get(deviceIndex);
    }

    /**
     * Print comprehensive lifecycle statistics report
     */
    public void printLifecycleReport() {
        printLifecycleReport("NVML GPU MONITORING");
    }

    /**
     * Print comprehensive lifecycle statistics report with custom lifecycle name
     * @param lifecycleName the name of the lifecycle to display in the report
     */
    public void printLifecycleReport(String lifecycleName) {
        if (lifecycleStats.isEmpty()) {
            logger.info("No lifecycle statistics available for: " + lifecycleName);
            return;
        }

        StringBuilder report = new StringBuilder();

        // Header with summary in same line
        List<String> summaries = new ArrayList<>();
        for (GPULifecycleStats stats : lifecycleStats.values()) {
            summaries.add(stats.getSummary());
        }
        report.append(lifecycleName).append(" - LIFECYCLE REPORT: ")
              .append(String.join(" | ", summaries)).append("\n");

        // Detailed report for each GPU
        for (GPULifecycleStats stats : lifecycleStats.values()) {
            report.append("\n").append(stats.generateReport());
        }

        report.append("\n");

        // Output the entire report as one log message to avoid multiple logger prefixes
        logger.info(report.toString());
    }

    /**
     * Get monitoring duration in seconds
     */
    public double getMonitoringDurationSeconds() {
        if (lifecycleStats.isEmpty()) {
            return 0.0;
        }

        return lifecycleStats.values().iterator().next().getMonitoringDurationSeconds();
    }
}