#!/bin/bash
#
# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

set -ex

nvidia-smi

git submodule update --init --recursive

MVN="mvn -Dmaven.wagon.http.retryHandler.count=3 -B"
# cuda11 or cuda12
CUDA_VER=${CUDA_VER:-cuda`nvcc --version | sed -n 's/^.*release \([0-9]\+\)\..*$/\1/p'`}
PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
USE_GDS=${USE_GDS:-ON}
USE_SANITIZER=${USE_SANITIZER:-ON}
BUILD_FAULTINJ=${BUILD_FAULTINJ:-ON}
ARM64=${ARM64:-false}

profiles="source-javadoc"
if [ "${ARM64}" == "true" ]; then
  profiles="${profiles},arm64"
  USE_GDS="OFF"
  USE_SANITIZER="ON"
  BUILD_FAULTINJ="OFF"
fi

${MVN} clean package ${MVN_MIRROR}  \
  -P${profiles} \
  -DCPP_PARALLEL_LEVEL=${PARALLEL_LEVEL} \
  -Dlibcudf.build.configure=true \
  -DUSE_GDS=${USE_GDS} -Dtest=*,!CuFileTest,!CudaFatalTest,!ColumnViewNonEmptyNullsTest \
  -DBUILD_TESTS=ON -DBUILD_FAULTINJ=${BUILD_FAULTINJ} -Dcuda.version=$CUDA_VER \
  -DUSE_SANITIZER=${USE_SANITIZER}
