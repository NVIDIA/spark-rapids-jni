# Build for ARM architecture

This document describes how to build the project in the ARM64 environment.
The following build process are based on `Amazon Linux 2 AMI (HVM) - Kernel 4.14, SSD Volume Type - arm64`
AMI number: `ami-0fadfbbdddef5a4fd`.
User needs to launch an EC2 instance with this AMI before following the steps below.

Note: the instance must contains GPU device, so the type must be `g5g.*xlarge` family.

## Prepare build environment

### Install devtoolset-9
```bash
sudo yum-config-manager --add-repo http://mirror.centos.org/altarch/7/sclo/aarch64/rh/
sudo yum install glibc
wget http://mirror.centos.org/altarch/7/os/aarch64/Packages/libgfortran5-8.3.1-2.1.1.el7.aarch64.rpm
sudo yum install libgfortran5-8.3.1-2.1.1.el7.aarch64.rpm -y
sudo yum install -y devtoolset-9.aarch64 --nogpgcheck
```

### Install CUDA Toolkit
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.5.2/local_installers/cuda_11.5.2_495.29.05_linux_sbsa.run
sudo sh cuda_11.5.2_495.29.05_linux_sbsa.run
# Then accept the license agreement and install the toolkit
# Using default options is recommended. Please do not modify the installation path.

# You can check the Nvidia GPU status by `nvidia-smi` to confirm the installation
nvidia-smi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
```

### Install necessary build tools
```bash
# maven
wget https://dlcdn.apache.org/maven/maven-3/3.8.6/binaries/apache-maven-3.8.6-bin.tar.gz
tar -xf apache-maven-3.8.6-bin.tar.gz
export PATH=$PWD/apache-maven-3.8.6/bin:$PATH

sudo yum install -y zlib-devel tar wget patch ninja-build git  tzdata-java java-1.8.0-openjdk-devel

# use Java 8
JAVA_8=$(alternatives --display java | grep 'family java-1.8.0-openjdk' | cut -d' ' -f1)
sudo alternatives --set java $JAVA_8
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.342.b07-1.amzn2.0.1.aarch64

# cmake
cd /usr/local/
export CMAKE_VERSION=3.23.3
sudo wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz
sudo tar zxf cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz
sudo rm cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz
sudo yum groupinstall -y "Development Tools"
export PATH=/usr/local/cmake-${CMAKE_VERSION}-linux-aarch64/bin:$PATH

# boost library
scl enable devtoolset-9 bash
sudo wget -q https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
sudo tar -xzf boost_1_79_0.tar.gz
sudo rm boost_1_79_0.tar.gz
cd boost_1_79_0/
sudo ./bootstrap.sh --prefix=/usr/local
sudo ./b2 install --prefix=/usr/local --with-filesystem --with-system
sudo rm -rf /usr/local/boost_1_79_0
```

### Build spark-rapids-jni target jar
```bash
git clone https://github.com/NVIDIA/spark-rapids-jni.git
cd spark-rapids-jni
git submodule update --init --recursive
```
The CUDA cupti library path is different from the one in x86 installation. So we need to modify the 
`spark-rapids-jni/src/main/cpp/faultinj/CMakeLists.txt` file to make it work.
```bash
diff --git a/src/main/cpp/faultinj/CMakeLists.txt b/src/main/cpp/faultinj/CMakeLists.txt
index a40a76e..3fc681b 100644
--- a/src/main/cpp/faultinj/CMakeLists.txt
+++ b/src/main/cpp/faultinj/CMakeLists.txt
@@ -32,10 +32,15 @@ add_library(
 find_path(SPDLOG_INCLUDE "spdlog"
     HINTS "$ENV{RMM_ROOT}/_deps/spdlog-src/include")
 
+find_library(CUPTI_LIB "cupti" REQUIRED NO_DEFAULT_PATH
+    HINTS "/usr/local/cuda/extras/CUPTI/lib64"
+)
+
 include_directories(
   "${SPDLOG_INCLUDE}"
+  "/usr/local/cuda/extras/CUPTI/include"
 )
 
 target_link_libraries(
-  cufaultinj CUDA::cupti_static
+       cufaultinj ${CUPTI_LIB}
 )
```
Then we can build the project.
```bash
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
scl enable devtoolset-9 "mvn clean package \
  -Psource-javadoc \
  -DCPP_PARALLEL_LEVEL=${PARALLEL_LEVEL} \
  -Dlibcudf.build.configure=true \
  -DUSE_GDS=OFF -Dtest=*,!CuFileTest,!CudaFatalTest" \
  -DBUILD_TESTS=ON -DskipTests
```

Then you can find the target jar file under `spark-rapids-jni/target/` directory.
