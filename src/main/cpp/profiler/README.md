# Spark Rapids Profile Converter Tool

This directory contains the Spark Rapids Profile Converter, a tool for converting NVTX profiling data from the Spark Rapids JNI library.

## Building the Tool

The profiler converter is built as part of the regular Maven build when `BUILD_PROFILER=ON` (default).

### Using Docker Build

From the repository root:

```bash
./build/build-in-docker clean package
```

This will build the entire project including the profiler converter.

The built executable will be located at:
```
target/jni/cmake-build/profiler/spark_rapids_profile_converter
```

### Native Build

You can also build natively if you have the required dependencies:

```bash
mvn clean package
```

To disable profiler building:
```bash
mvn clean package -DBUILD_PROFILER=OFF
```

## Usage

1. NvtxwEvents.h, NvtxwEvents.cpp are copied from Nsight Systems source code. They need to be kept in sync between this project and Nsight Systems.

2. Need to set the `NVTXW_BACKEND` environment variable pointing to the `libNvtxwBackend.so` library in the host machine running Nsight Systems. For example:
``(one more `)
> export NVTXW_BACKEND=/opt/nvidia/nsight-systems/2024.6.0/host-linux-x64/libNvtxwBackend.so
``(one more `)

3. Run like this:
``(one more `)
> ./target/jni/cmake-build/profiler/spark_rapids_profile_converter  -w -o <output_file>.nsys-rep <input_file>.bin
``(one more `)
The output will looks similar to this:
``(one more`)
      Backend implementation loaded!  Applying config string...
      Loader config key/value pairs not provided
      Creating report: "file3021460.nsys-rep"
      - Created session: file3021460
      Session config key/value pairs not provided
      - Created stream: Stream1
         Domain: SparkRAPIDS
         Scope: 
      - Destroyed stream: Stream1
      3946 events imported
      - Destroyed session: file3021460
      Backend implementation prepared for unload.
``(one more`)
   
4. Load into Nsight Systems UI: `nsys-ui <output_file>.nsys-rep`.