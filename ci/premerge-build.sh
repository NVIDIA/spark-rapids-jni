#!/bin/bash
#
# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
${MVN} verify ${MVN_MIRROR} \
  -DCPP_PARALLEL_LEVEL=${PARALLEL_LEVEL} \
  -Dlibcudf.build.configure=true \
  -DUSE_GDS=ON -Dtest=*,!CuFileTest,!CudaFatalTest,!ColumnViewNonEmptyNullsTest \
  -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON

build_name=$(${MVN} help:evaluate -Dexpression=project.build.finalName -q -DforceStdout)
cuda_version=$(${MVN} help:evaluate -Dexpression=cuda.version -q -DforceStdout)
. ci/check-cuda-dependencies.sh "target/${build_name}-${cuda_version}.jar"
