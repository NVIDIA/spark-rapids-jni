package com.nvidia.spark.rapids.jni;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Example demonstrating the NVMLMonitor system for Spark Rapids JNI
 */
public class NVMLMonitorExample {
    
    private static final Logger logger = LoggerFactory.getLogger(NVMLMonitorExample.class);
    
    static {
        try {
            // The native library is loaded automatically by the main Spark Rapids JNI system
            logger.info("Loading native libraries via Spark Rapids JNI...");
        } catch (Exception e) {
            logger.error("Failed to load native libraries: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        logger.info("NVML GPU Monitor Example for Spark Rapids JNI");
        logger.info("===========================================");
        
        // Initialize NVML
        if (!NVMLMonitor.initialize()) {
            logger.error("Failed to initialize NVML");
            return;
        }
        
        int deviceCount = NVMLMonitor.getDeviceCount();
        logger.info("Detected " + deviceCount + " GPU device(s)");
        
        if (deviceCount == 0) {
            logger.info("No GPUs found");
            NVMLMonitor.shutdown();
            return;
        }
        
        // Show initial GPU information
        GPUInfo[] initialInfo = NVMLMonitor.getCurrentGPUInfo();
        logger.info("\nInitial GPU Information:");
        for (GPUInfo info : initialInfo) {
            logger.info(info.toDetailedString());
        }
        
        // Create monitor and start monitoring
        NVMLMonitor monitor = new NVMLMonitor(1000, true);
        
        // Set up callback for real-time monitoring
        monitor.setCallback(new NVMLMonitor.MonitoringCallback() {
            private int updateCount = 0;
            
            @Override
            public void onGPUUpdate(GPUInfo[] gpuInfos, long timestamp) {
                updateCount++;
                logger.info("=== Update #" + updateCount + " ===");
                for (GPUInfo info : gpuInfos) {
                    logger.info("  " + info.toCompactString());
                }
                logger.info("");
            }
            
            @Override
            public void onMonitoringStarted() {
                logger.info("🚀 NVML monitoring started");
            }
            
            @Override
            public void onMonitoringStopped() {
                logger.info("⏹️  NVML monitoring stopped");
            }
            
            @Override
            public void onError(String error) {
                logger.error("❌ Monitoring error: " + error);
            }
        });
        
        monitor.startMonitoring();
        
        // Monitor for 10 seconds
        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // Stop monitoring and show results
        monitor.stopMonitoring();
        monitor.printLifecycleReport();
        
        NVMLMonitor.shutdown();
        logger.info("\nNVML monitoring example completed successfully!");
    }
}
