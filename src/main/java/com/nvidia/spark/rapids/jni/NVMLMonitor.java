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
            RmmSpark.getCurrentThreadId();
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
     * Get GPU information for specific device
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
            lifecycleStats.put(info.deviceIndex, 
                              new GPULifecycleStats(info.name, info.deviceIndex));
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
                        GPULifecycleStats stats = lifecycleStats.get(info.deviceIndex);
                        if (stats != null) {
                            stats.addSample(info);
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
        
        String separator80 = "================================================================================";
        logger.info("\n" + separator80);
        logger.info(lifecycleName + " - LIFECYCLE REPORT");
        logger.info(separator80);
        
        // Summary for all GPUs
        logger.info("\nSUMMARY:");
        for (GPULifecycleStats stats : lifecycleStats.values()) {
            logger.info("  " + stats.getSummary());
        }
        
        // Detailed report for each GPU
        for (GPULifecycleStats stats : lifecycleStats.values()) {
            logger.info("\n" + stats.generateReport());
        }
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