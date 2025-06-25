# Spark Rapids Profile Converter Tool

This directory contains the Spark Rapids Profile Converter, a tool for converting NVTX profiling data from the Spark Rapids JNI library.

## Building the Tool

### Using the Docker Build Script

From the repository root:

```bash
./build/build-profiler-in-docker
```

This script will:
1. Use the existing devtoolset Docker container
2. Build the necessary dependencies (cudf) if not already built
3. Build just the `spark_rapids_profile_converter` executable

The built executable will be located at:
```
target/jni/cmake-build/profiler/spark_rapids_profile_converter
```

### Integration with Maven Build

The profiler is also built as part of the regular Maven build when `BUILD_PROFILER=ON` (default):

```bash
mvn clean package
```

To disable profiler building:
```bash
mvn clean package -DBUILD_PROFILER=OFF
```

## Usage

1. NvtxwEvents.h, NvtxwEvents.cpp are copied from Nsight Systems source code. They need to be kept in sync between this project and Nsight Systems.

2. Need to set the NVTXW_BACKEND environment variable for the libNvtxwBackend.so library in the host directory a current build of Nsight Systems. For example:
   > export NVTXW_BACKEND=/opt/nvidia/nsight-systems/2024.6.0/host-linux-x64/libNvtxwBackend.so

3. Run like this:
      > ./target/jni/cmake-build/profiler/spark_rapids_profile_converter  -w -o file3021460.nsys-rep rapids-profile-3021460@jlowe-lcedt-driver.bin
   and get output similar to this:
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
   
4. Load into nsight systems UI: nsys-ui file3021460.nsys-rep