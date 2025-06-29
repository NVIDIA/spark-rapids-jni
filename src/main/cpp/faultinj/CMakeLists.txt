#=============================================================================
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#=============================================================================

project(
  CUDA_FAULT_INJECTION
  VERSION 22.08.00
  LANGUAGES C CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)

add_library(
  cufaultinj SHARED
  faultinj.cu
)

target_link_libraries(
  cufaultinj PRIVATE spdlog::spdlog_header_only
)

target_link_libraries(
  cufaultinj PRIVATE CUDA::cupti_static
)
