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

1. Files mirrored from Nsight Systems that must be kept in sync: `NvtxwEvents.h`, `NvtxwEvents.cpp`, `nvtxw3.h`, `nvtxw3.cpp`. Project integration files (owned here) are `nvtxw_events.h` and `initialize_nvtxw.cpp`.

2. Provide the NVTXW backend library path either via CLI or environment variable (CLI takes precedence over env):
   - CLI: `--nvtxw-backend=/path/to/libNvtxwBackend.so`
   - Env: set `NVTXW_BACKEND` to the same path. For example:
```bash
export NVTXW_BACKEND=/opt/nvidia/nsight-systems/2024.6.0/host-linux-x64/libNvtxwBackend.so
```

3. Run like this:
```bash
./target/jni/cmake-build/profiler/spark_rapids_profile_converter -w -o <output_file>.nsys-rep <input_file>.bin
```
The output will look similar to this:
```text
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
```
   
4. Load into Nsight Systems UI: `nsys-ui <output_file>.nsys-rep`.