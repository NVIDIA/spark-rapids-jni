# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

# Use CPM to find or clone speedlog
function(find_and_configure_spdlog)

  set(CPM_DOWNLOAD_spdlog ON)
  include(${rapids-cmake-dir}/cpm/spdlog.cmake)
  rapids_cpm_spdlog(
    FMT_OPTION "EXTERNAL_FMT_HO"
  )
  set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)

endfunction()

find_and_configure_spdlog()
