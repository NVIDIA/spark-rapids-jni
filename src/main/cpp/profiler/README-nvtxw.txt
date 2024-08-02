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