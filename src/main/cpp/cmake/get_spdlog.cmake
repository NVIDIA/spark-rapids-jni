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

# Get spdlog
function(get_spdlog)
  set(options)
  set(one_value BUILD_EXPORT_SET INSTALL_EXPORT_SET)
  set(multi_value)
  cmake_parse_arguments(_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  # Fix up _RAPIDS_UNPARSED_ARGUMENTS to have EXPORT_SETS as this is need for rapids_cpm_find. Also
  # propagate the user provided build and install export sets.
  if(_RAPIDS_INSTALL_EXPORT_SET)
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS INSTALL_EXPORT_SET ${_RAPIDS_INSTALL_EXPORT_SET})
  endif()
  if(_RAPIDS_BUILD_EXPORT_SET)
    list(APPEND _RAPIDS_UNPARSED_ARGUMENTS BUILD_EXPORT_SET ${_RAPIDS_BUILD_EXPORT_SET})
  endif()

  set(to_install OFF)
  if(_RAPIDS_INSTALL_EXPORT_SET)
    set(to_install ON)
  endif()

  set(spdlog_version 1.14.1)

  include("${rapids-cmake-dir}/cpm/find.cmake")
  rapids_cpm_find(
    spdlog "${spdlog_version}" ${_RAPIDS_UNPARSED_ARGUMENTS}
    GLOBAL_TARGETS spdlog::spdlog spdlog::spdlog_header_only
    CPM_ARGS
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG "v${spdlog_version}"
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL ON
    OPTIONS "SPDLOG_INSTALL ${to_install}" "SPDLOG_USE_STD_FORMAT ON"
  )
endfunction()


# Use CPM to find or clone speedlog
function(find_and_configure_spdlog)
  set(CPM_DOWNLOAD_spdlog ON)
  get_spdlog(CPM_ARGS OPTIONS "BUILD_SHARED_LIBS OFF" "SPDLOG_BUILD_SHARED OFF")
  set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

find_and_configure_spdlog()
