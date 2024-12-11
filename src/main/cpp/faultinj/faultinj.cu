/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/logger.hpp>

#include <cuda.h>

#include <assert.h>
#include <cupti.h>
#include <pthread.h>

#include <exception>
#include <iostream>
#include <map>

// thread-safe ptree
#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <sys/inotify.h>
#include <sys/time.h>

// Format enums for logging
// auto format_as(CUpti_CallbackDomain domain) { return fmt::underlying(domain); }

namespace {

#define CUPTI_CALL(call)                                                 \
  do {                                                                   \
    CUptiResult _status = call;                                          \
    if (_status != CUPTI_SUCCESS) {                                      \
      const char* errstr;                                                \
      cuptiGetResultString(_status, &errstr);                            \
      CUDF_LOG_ERROR("function {} failed with error {}", #call, errstr); \
    }                                                                    \
  } while (0)

#define PTHREAD_CALL(call)                                                                         \
  do {                                                                                             \
    int _status = call;                                                                            \
    if (_status != 0) { CUDF_LOG_ERROR("function {} failed with error code {}", #call, _status); } \
  } while (0)

typedef enum { FI_TRAP, FI_ASSERT, FI_RETURN_VALUE } FaultInjectionType;

typedef struct {
  volatile uint32_t initialized;
  CUpti_SubscriberHandle subscriber;

  std::string configFilePath = std::string();

  bool dynamic;
  int terminateThread;
  pthread_t dynamicThread;
  // TODO change to the RAII idiom
  pthread_rwlock_t configLock = PTHREAD_RWLOCK_INITIALIZER;

  boost::property_tree::ptree configRoot;
  boost::optional<boost::property_tree::ptree&> driverFaultConfigs  = boost::none;
  boost::optional<boost::property_tree::ptree&> runtimeFaultConfigs = boost::none;
} injGlobalControl;
injGlobalControl globalControl;

// Function Declarations
void atExitHandler(void);

void CUPTIAPI faultInjectionCallbackHandler(void* userdata,
                                            CUpti_CallbackDomain domain,
                                            CUpti_CallbackId cbid,
                                            void* cbInfo);

const std::string configFilePathEnv = "FAULT_INJECTOR_CONFIG_PATH";
CUptiResult cuptiInitialize(void);
void readFaultInjectorConfig(void);
void traceConfig(boost::property_tree::ptree const& pTree);
void* dynamicReconfig(void* args);

void globalControlInit(void)
{
  CUDF_LOG_DEBUG("globalControlInit of fault injection");
  globalControl.initialized     = 0;
  globalControl.subscriber      = 0;
  globalControl.terminateThread = 0;
  CUDF_LOG_TRACE("checking environment {}", configFilePathEnv);
  const char* configFilePath = std::getenv(configFilePathEnv.c_str());
  CUDF_LOG_DEBUG("{} is {}", configFilePathEnv, configFilePath);
  if (configFilePath) {
    globalControl.configFilePath = std::string(configFilePath);
    CUDF_LOG_DEBUG("will init config from {}", globalControl.configFilePath);
  }
  readFaultInjectorConfig();
  globalControl.initialized = 1;
}

void registerAtExitHandler(void)
{
  // Register atExitHandler
  atexit(&atExitHandler);
}

void atExitHandler(void)
{
  if (globalControl.dynamic) {
    globalControl.terminateThread = 1;
    PTHREAD_CALL(pthread_join(globalControl.dynamicThread, nullptr));
    CUDF_LOG_INFO("reconfig thread shut down ... exiting");
  }

  CUDF_LOG_DEBUG("atExitHandler: cuptiFinalize");
  CUPTI_CALL(cuptiFinalize());
}

CUptiResult cuptiInitialize(void)
{
  CUptiResult status = CUPTI_SUCCESS;

  CUPTI_CALL(cuptiSubscribe(
    &globalControl.subscriber, (CUpti_CallbackFunc)faultInjectionCallbackHandler, nullptr));

  // TODO do this dynamically based on config?
  CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
  CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

  return status;
}

__global__ static void faultInjectorKernelAssert(void)
{
  assert(0 && "faultInjectorKernelAssert triggered");
}

__global__ static void faultInjectorKernelTrap(void) { asm("trap;"); }

boost::optional<boost::property_tree::ptree&> lookupConfig(
  boost::optional<boost::property_tree::ptree&> domainConfigs,
  const char* key,
  CUpti_CallbackId cbid)
{
  boost::optional<boost::property_tree::ptree&> faultConfig =
    (*domainConfigs).get_child_optional(std::to_string(cbid));
  if (!faultConfig) { faultConfig = (*domainConfigs).get_child_optional(key); }
  if (!faultConfig) { faultConfig = (*domainConfigs).get_child_optional("*"); }
  return faultConfig;
}

void CUPTIAPI faultInjectionCallbackHandler(void*,
                                            CUpti_CallbackDomain domain,
                                            CUpti_CallbackId cbid,
                                            void* cbdata)
{
  const std::string faultInjectorKernelPrefix = std::string("faultInjectorKernel");

  CUpti_CallbackData* cbInfo = static_cast<CUpti_CallbackData*>(cbdata);

  // TODO maybe allow it in the config but right now CUPTI_API_EXIT is generally
  // preferrable for interception. However, it means that the execution has
  // happened.
  //
  if (cbInfo->callbackSite != CUPTI_API_EXIT) { return; }

  // Check last error
  CUPTI_CALL(cuptiGetLastError());
  boost::optional<boost::property_tree::ptree&> matchedFaultConfig = boost::none;
  PTHREAD_CALL(pthread_rwlock_rdlock(&globalControl.configLock));

  // check if we are processing the result of our own launch.
  // symbolName is only valid for launches
  //
  // https://gitlab.com/nvidia/headers/cuda-individual/cupti/-/blob/main/cupti_driver_cbid.h
  // https://gitlab.com/nvidia/headers/cuda-individual/cupti/-/blob/main/cupti_runtime_cbid.h
  //
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API && globalControl.driverFaultConfigs) {
    switch (cbid) {
      case CUPTI_DRIVER_TRACE_CBID_cuLaunch:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice:
      case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch:
      case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc:
      case CUPTI_DRIVER_TRACE_CBID_cuLaunchHostFunc_ptsz:
        // 11.7
        // case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
        // case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz:
        if (std::string(cbInfo->symbolName)
              .compare(0, faultInjectorKernelPrefix.size(), faultInjectorKernelPrefix) == 0) {
          CUDF_LOG_DEBUG("rejecting fake launch functionName={} symbol={}",
                         cbInfo->functionName,
                         cbInfo->symbolName);
          break;
        }
        // intentional fallthrough
      default:
        matchedFaultConfig =
          lookupConfig(globalControl.driverFaultConfigs, cbInfo->functionName, cbid);
    }
  } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API && globalControl.runtimeFaultConfigs) {
    switch (cbid) {
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernelMultiDevice_v9000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_v10000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchHostFunc_ptsz_v10000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000:
        if (std::string(cbInfo->symbolName)
              .compare(0, faultInjectorKernelPrefix.size(), faultInjectorKernelPrefix) == 0) {
          CUDF_LOG_DEBUG("rejecting fake launch functionName={} symbol={}",
                         cbInfo->functionName,
                         cbInfo->symbolName);
          break;
        }
        // intentional fallthrough
      default:
        matchedFaultConfig =
          lookupConfig(globalControl.runtimeFaultConfigs, cbInfo->functionName, cbid);
    }
  }
  // unlock we have a copy from the prior parse
  PTHREAD_CALL(pthread_rwlock_unlock(&globalControl.configLock));

  if (!matchedFaultConfig) { return; }

  // numeric value of FaultInjectionType: 0 (PTX trap), 1 (device assert), 2
  // (return code)
  const std::string injectionTypeKey        = "injectionType";
  const std::string substituteReturnCodeKey = "substituteReturnCode";
  const std::string percentKey              = "percent";
  const std::string interceptionCountKey    = "interceptionCount";

  const int injectionType = (*matchedFaultConfig)
                              .get_optional<int>(injectionTypeKey)
                              .value_or(static_cast<int>(FI_RETURN_VALUE));

  const int substituteReturnCode = (*matchedFaultConfig)
                                     .get_optional<int>(substituteReturnCodeKey)
                                     .value_or(static_cast<int>(CUDA_SUCCESS));

  const int injectionProbability = (*matchedFaultConfig).get_optional<int>(percentKey).value_or(0);

  const int interceptionCount =
    (*matchedFaultConfig).get_optional<int>(interceptionCountKey).value_or(INT_MAX);

  CUDF_LOG_TRACE(
    "considered config domain={} function={} injectionType={} probability={} "
    "interceptionCount={}",
    domain,
    cbInfo->functionName,
    injectionType,
    injectionProbability,
    interceptionCount);

  if (interceptionCount <= 0) {
    CUDF_LOG_TRACE(
      "skipping interception because hit count reached 0, "
      "domain={} function={} injectionType={} probability={} "
      "interceptionCount={}",
      domain,
      cbInfo->functionName,
      injectionType,
      injectionProbability,
      interceptionCount);
    return;
  }

  if (injectionProbability < 100) {
    if (injectionProbability <= 0) { return; }
    const int rand10000     = std::rand() % 10000;
    const int skipThreshold = injectionProbability * 10000 / 100;
    CUDF_LOG_TRACE("rand1000={} skipThreshold={}", rand10000, skipThreshold);
    if (rand10000 >= skipThreshold) { return; }
    CUDF_LOG_DEBUG(
      "matched config based on rand10000={} skipThreshold={} "
      "domain={} function={} injectionType={} probability={}",
      rand10000,
      skipThreshold,
      domain,
      cbInfo->functionName,
      injectionType,
      injectionProbability);
  } else {
    CUDF_LOG_DEBUG(
      "matched 100% config domain={} function={} injectionType={} "
      "probability={}",
      domain,
      cbInfo->functionName,
      injectionType,
      injectionProbability);
  }

  // update counter if not unlimited
  if (interceptionCount != INT_MAX) {
    CUDF_LOG_DEBUG("updating interception count {}: before locking", interceptionCount);
    // TODO the lock is too coarse-grained.
    PTHREAD_CALL(pthread_rwlock_wrlock(&globalControl.configLock));
    const int interceptionCount = (*matchedFaultConfig).get<int>("interceptionCount");
    (*matchedFaultConfig).put("interceptionCount", interceptionCount - 1);
    PTHREAD_CALL(pthread_rwlock_unlock(&globalControl.configLock));
  }

  switch (injectionType) {
    case FI_TRAP:
      faultInjectorKernelTrap<<<1, 1>>>();
      cudaStreamSynchronize(nullptr);
      break;

    case FI_ASSERT:
      faultInjectorKernelAssert<<<1, 1>>>();
      cudaStreamSynchronize(nullptr);
      break;

    case FI_RETURN_VALUE:
      if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        CUresult* cuResPtr = static_cast<CUresult*>(cbInfo->functionReturnValue);
        *cuResPtr          = static_cast<CUresult>(substituteReturnCode);
      } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        cudaError_t* cudaErrPtr = static_cast<cudaError_t*>(cbInfo->functionReturnValue);
        CUDF_LOG_ERROR("updating runtime return value DOES NOT WORK, use trap or assert");
        *cudaErrPtr = static_cast<cudaError_t>(substituteReturnCode);
        break;
      }

    default:;
  }
}

/**
 * Parse and apply new config
 */
void readFaultInjectorConfig(void)
{
  if (globalControl.configFilePath.empty()) {
    CUDF_LOG_ERROR("specify convig via environment {}", configFilePathEnv);
    return;
  }
  std::ifstream jsonStream(globalControl.configFilePath);
  if (!jsonStream.good()) {
    CUDF_LOG_ERROR("check file exists {}", globalControl.configFilePath);
    return;
  }

  // The numeric value of level_enum is retrieved from
  // https://github.com/rapidsai/rapids-logger/blob/main/logger.hpp.in#L40
  const std::string logLevelKey = "logLevel";

  // A Boolean flag as to whether to watch for config file modifications
  // and apply changes after initial boot config.
  // Currently only file-based config is supported,
  // TODO add a socket for remote config
  //
  const std::string dynamicConfigKey = "dynamic";

  // An unsigned int to seed the random number generator to deterministically
  // recreate a fault sequence
  //
  const std::string seedKey = "seed";

  // To retrieve a map of driver/runtime fault configs
  //   "functionName" -> fault config
  //   "CUPT callback id" -> fault config
  //   "*" -> fault config
  const std::string driverFaultsKey  = "cudaDriverFaults";
  const std::string runtimeFaultsKey = "cudaRuntimeFaults";

  PTHREAD_CALL(pthread_rwlock_wrlock(&globalControl.configLock));
  try {
    boost::property_tree::read_json(jsonStream, globalControl.configRoot);
    const int logLevel = globalControl.configRoot.get_optional<int>(logLevelKey).value_or(0);

    globalControl.dynamic =
      globalControl.configRoot.get_optional<bool>(dynamicConfigKey).value_or(false);

    const unsigned seed =
      globalControl.configRoot.get_optional<unsigned>(seedKey).value_or(std::time(0));
    CUDF_LOG_INFO("Seeding std::srand with {}", seed);
    std::srand(seed);

    CUDF_LOG_INFO("changed log level to {}", logLevel);
    cudf::default_logger().set_level(static_cast<cudf::level_enum>(logLevel));
    traceConfig(globalControl.configRoot);

    globalControl.driverFaultConfigs = globalControl.configRoot.get_child_optional(driverFaultsKey);
    globalControl.runtimeFaultConfigs =
      globalControl.configRoot.get_child_optional(runtimeFaultsKey);
  } catch (boost::property_tree::json_parser::json_parser_error& error) {
    CUDF_LOG_ERROR("error parsing fault injector config, still editing? {}", error.what());
  }
  PTHREAD_CALL(pthread_rwlock_unlock(&globalControl.configLock));
  jsonStream.close();
  CUDF_LOG_DEBUG("readFaultInjectorConfig from {} DONE", globalControl.configFilePath);
}

void traceConfig(boost::property_tree::ptree const& pTree)
{
  for (auto it = pTree.begin(); it != pTree.end(); ++it) {
    CUDF_LOG_TRACE("congig key={} value={}", it->first, it->second.get_value<std::string>());
    traceConfig(it->second);
  }
}

int eventCheck(int fd)
{
  fd_set rfds;
  FD_ZERO(&rfds);
  FD_SET(fd, &rfds);
  struct timeval tv;
  tv.tv_sec  = 5;
  tv.tv_usec = 0;
  return select(FD_SETSIZE, &rfds, nullptr, nullptr, &tv);
}

void* dynamicReconfig(void*)
{
  CUDF_LOG_DEBUG("config watcher thread: inotify_init()");
  const int inotifyFd = inotify_init();
  if (inotifyFd < 0) {
    CUDF_LOG_ERROR("inotify_init() failed");
    return nullptr;
  }
  CUDF_LOG_DEBUG("config watcher thread: inotify_add_watch {}", globalControl.configFilePath);
  const int watchFd = inotify_add_watch(inotifyFd, globalControl.configFilePath.c_str(), IN_MODIFY);
  if (watchFd < 0) {
    CUDF_LOG_ERROR("config watcher thread: inotify_add_watch {} failed",
                   globalControl.configFilePath);
    return nullptr;
  }

  constexpr auto MAX_EVENTS = 1024;
  constexpr auto EVENT_SIZE = sizeof(struct inotify_event);

  const auto configFilePathStr = std::string(globalControl.configFilePath);
  const auto BUF_LEN           = MAX_EVENTS * (EVENT_SIZE + configFilePathStr.length());
  char eventBuffer[BUF_LEN];

  while (!globalControl.terminateThread) {
    CUDF_LOG_TRACE("about to call eventCheck");
    const int eventCheckRes = eventCheck(inotifyFd);
    CUDF_LOG_TRACE("eventCheck returned {}", eventCheckRes);
    if (eventCheckRes > 0) {
      const int length = read(inotifyFd, eventBuffer, BUF_LEN);
      CUDF_LOG_DEBUG("config watcher thread: read {} bytes", length);
      if (length < EVENT_SIZE) { continue; }
      for (int i = 0; i < length;) {
        struct inotify_event* event = (struct inotify_event*)&eventBuffer[i];
        CUDF_LOG_DEBUG("modfiled file detected: {}", event->name);
        i += EVENT_SIZE + event->len;
      }
      readFaultInjectorConfig();
    }
  }

  if (watchFd >= 0) {
    CUDF_LOG_DEBUG("config watcher thread: inotify_rm_watch {} {}", inotifyFd, watchFd);
    inotify_rm_watch(inotifyFd, watchFd);
  }
  if (inotifyFd >= 0) {
    CUDF_LOG_DEBUG("config watcher thread: close {}", inotifyFd);
    close(inotifyFd);
  }
  CUDF_LOG_INFO("exiting dynamic reconfig thread: terminateThread={}",
                globalControl.terminateThread);
  return nullptr;
}

}  // end anonymous namespace

/**
 * cuInit hook entry point
 */
extern "C" int InitializeInjection(void)
{
  CUDF_LOG_INFO("cuInit entry point for libcufaultinj InitializeInjection");
  // intial log level is trace until the config is read
  cudf::default_logger().set_level(cudf::level_enum::trace);

  if (globalControl.initialized) { return 1; }
  // Init globalControl
  globalControlInit();

  registerAtExitHandler();

  if (globalControl.dynamic) {
    CUDF_LOG_DEBUG("creating a thread to watch the fault injector config interactively");
    PTHREAD_CALL(pthread_create(&globalControl.dynamicThread, nullptr, dynamicReconfig, nullptr));
  }

  // Initialize CUPTI
  CUPTI_CALL(cuptiInitialize());

  return 1;
}
