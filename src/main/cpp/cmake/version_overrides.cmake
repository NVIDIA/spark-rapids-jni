include_guard(GLOBAL)
include(rapids-cpm)
include(${rapids-cmake-dir}/cpm/package_override.cmake)
rapids_cpm_package_override("${CMAKE_CURRENT_LIST_DIR}/version_overrides/versions.json")
