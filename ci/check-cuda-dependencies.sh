#!/bin/bash
#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

# common script to help check if libcudf.so has dynamical link to cuda libs

set -exo pipefail

jar_path=$1
tmp_path=/tmp/"jni-$(date "+%Y%m%d%H%M%S")"
unzip -j "${jar_path}" "*64/Linux/libcudf.so" -d "${tmp_path}"

if objdump -p "${tmp_path}/libcudf.so" | grep NEEDED | grep -q cuda; then
    echo "dynamical link to CUDA lib found in libcudf.so..."
    ldd "${tmp_path}/libcudf.so"
    exit 1
else
    echo "no dynamical link to CUDA lib found in libcudf.so"
fi
