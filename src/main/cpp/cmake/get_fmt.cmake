# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
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

# Fetch fmt
function(get_fmt)
  set(to_install OFF)
  if(INSTALL_EXPORT_SET IN_LIST ARGN)
    set(to_install ON)
  endif()

  include("${rapids-cmake-dir}/cpm/find.cmake")
  set(version 11.0.2)
  rapids_cpm_find(
    fmt "${version}" ${ARGN}
    GLOBAL_TARGETS fmt::fmt fmt::fmt-header-only
    CPM_ARGS
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG "${version}"
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL ON
    OPTIONS "FMT_INSTALL ${to_install}" "CMAKE_POSITION_INDEPENDENT_CODE ON"
  )

  # Propagate up variables that CPMFindPackage provide
  set(fmt_SOURCE_DIR
      "${fmt_SOURCE_DIR}"
      PARENT_SCOPE
  )
  set(fmt_BINARY_DIR
      "${fmt_BINARY_DIR}"
      PARENT_SCOPE
  )
  set(fmt_ADDED
      "${fmt_ADDED}"
      PARENT_SCOPE
  )
  set(fmt_VERSION
      ${version}
      PARENT_SCOPE
  )
endfunction()

if(_RAPIDS_FMT_OPTION STREQUAL "EXTERNAL_FMT" OR _RAPIDS_FMT_OPTION STREQUAL "EXTERNAL_FMT_HO")
    get_fmt(${_RAPIDS_UNPARSED_ARGUMENTS})
endif()