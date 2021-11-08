# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

CPMAddPackage(
  NAME cudf
  GIT_TAG 49d1cc28648fe676dbddcf27c01939f87122ed8e
  GIT_REPOSITORY https://github.com/rapidsai/cudf.git
  OPTIONS
    "BUILD_TESTS OFF"
    "CUDF_ENABLE_ARROW_S3 OFF"
    "CUDF_USE_ARROW_STATIC ON"
    "PER_THREAD_DEFAULT_STREAM ${PER_THREAD_DEFAULT_STREAM}"
    "RMM_LOGGING_LEVEL ${RMM_LOGGING_LEVEL}"
  SOURCE_SUBDIR cpp
)
