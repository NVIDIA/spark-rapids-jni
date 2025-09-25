package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.NativeDepsLoader;

public class DeviceAttr {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static int isIntegratedGPU() {
    return isDeviceIntegrated();
  }

  private static native int isDeviceIntegrated();
}
