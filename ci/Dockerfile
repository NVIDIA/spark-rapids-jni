#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

###
# Build the image for spark-rapids-jni development environment.
#
# Arguments: CUDA_VERSION=11.5.0
#
###
ARG CUDA_VERSION=11.5.0
FROM gpuci/cuda:$CUDA_VERSION-devel-centos7

### Install basic requirements
RUN yum install -y centos-release-scl
RUN yum install -y devtoolset-9 rh-python38 epel-release
RUN yum install -y zlib-devel maven tar wget patch ninja-build
RUN yum -y install https://packages.endpoint.com/rhel/7/os/x86_64/endpoint-repo-1.7-1.x86_64.rpm && yum install -y git
RUN scl enable rh-python38 "pip install requests"

## pre-create the CMAKE_INSTALL_PREFIX folder, set writable by any user for Jenkins
RUN mkdir /usr/local/rapids && mkdir /rapids && chmod 777 /usr/local/rapids && chmod 777 /rapids

ARG CMAKE_VERSION=3.20.5
RUN cd /usr/local/ && wget --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
   tar zxf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
   rm cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz
ENV PATH /usr/local/cmake-${CMAKE_VERSION}-linux-x86_64/bin:$PATH
