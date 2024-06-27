#pragma once

#include "nvtxw3.h"
#include "NvtxwEvents.h"
#include <string>

extern bool createNvtxwStream(const nvtxwInterfaceCore_t *nvtxwInterface,
  const nvtxwSessionHandle_t& session, 
  const std::string & name,
  const std::string & domain, 
  nvtxwStreamHandle_t & stream);

extern int initialize_nvtxw(std::ifstream& in, const std::string& outName, 
  void *& nvtxwModuleHandle,
  nvtxwInterfaceCore_t *&nvtxwInterface,
  nvtxwSessionHandle_t &session,
  nvtxwStreamHandle_t &stream);
