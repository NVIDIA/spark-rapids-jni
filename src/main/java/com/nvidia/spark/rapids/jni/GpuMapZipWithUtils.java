package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.*;

import java.util.Map;

public class GpuMapZipWithUtils {
    static{
        NativeDepsLoader.loadNativeDeps();
    }
    public static ColumnVector mapZip(ColumnView input1, ColumnView input2) {
        return new ColumnVector(mapZip(input1.getNativeView(), input2.getNativeView()));
    }

    private static native long mapZip(long input1, long input2);
}