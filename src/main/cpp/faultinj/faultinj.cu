/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <assert.h>
#include <cuda.h>
#include <cupti.h>
#include <iostream>
#include <map>
#include <pthread.h>
#include <exception>

#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <spdlog/spdlog.h>
#include <sys/inotify.h>
#include <sys/time.h>

#define STDCALL
inline const std::string FAULT_INJECTOR_CONFIG_PATH{"FAULT_INJECTOR_CONFIG_PATH"};
const char *configFilePath = std::getenv(FAULT_INJECTOR_CONFIG_PATH.c_str());


#if defined(__cplusplus)
extern "C" {
#endif

#define CUPTI_CALL(call)                                                       \
do {                                                                           \
    CUptiResult _status = call;                                                \
    if (_status != CUPTI_SUCCESS) {                                            \
        const char *errstr;                                                    \
        cuptiGetResultString(_status, &errstr);                                \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #call, errstr);                            \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define PTHREAD_CALL(call)                                                         \
do {                                                                               \
    int _status = call;                                                            \
    if (_status != 0) {                                                            \
        fprintf(stderr, "%s:%d: error: function %s failed with error code %d.\n",  \
                __FILE__, __LINE__, #call, _status);                               \
        exit(EXIT_FAILURE);                                                        \
    }                                                                              \
} while (0)

typedef enum {
    FI_TRAP,
    FI_ASSERT,
    FI_RETURN_VALUE
} FaultInjectionType;

typedef struct {
    volatile uint32_t initialized;
    CUpti_SubscriberHandle  subscriber;

    int terminateThread;
    pthread_t dynamicThread;
    pthread_rwlock_t configLock = PTHREAD_RWLOCK_INITIALIZER;

    boost::property_tree::ptree configRoot;
    boost::optional<boost::property_tree::ptree&> driverFaultConfigs = boost::none;
    boost::optional<boost::property_tree::ptree&> runtimeFaultConfigs = boost::none;
} injGlobalControl;
injGlobalControl globalControl;

// Function Declarations
static void atExitHandler(void);

static void CUPTIAPI faultInjectionCallbackHandler(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void *cbInfo);

extern int STDCALL InitializeInjection(void);


#if defined(__cplusplus)
}
#endif

static CUptiResult cuptiInitialize(void);
static void readFaultInjectorConfig(void);
static void parseConfig(boost::property_tree::ptree const& pTree);
static void *dynamicReconfig(void *args);

// Function Definitions

static void
globalControlInit(void) {
    spdlog::trace("globalControlInit of fault injection");
    globalControl.initialized = 0;
    globalControl.subscriber = 0;
    globalControl.terminateThread = 0;
    readFaultInjectorConfig();
}

static void
registerAtExitHandler(void) {
    // Register atExitHandler
    atexit(&atExitHandler);
}


static void
atExitHandler(void) {
    globalControl.terminateThread = 1;
    PTHREAD_CALL(pthread_join(globalControl.dynamicThread, NULL));
    spdlog::info("reconfig thread shut down ... exiting");
}


static CUptiResult
cuptiInitialize(void) {
    CUptiResult status = CUPTI_SUCCESS;

    CUPTI_CALL(cuptiSubscribe(&globalControl.subscriber, (CUpti_CallbackFunc)faultInjectionCallbackHandler, NULL));

    // TODO do this dynamically based on config?
    CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

    return status;
}


__global__ void
faultInjectorKernelAssert(void) {
    assert(0 && "faultInjectorKernelAssert triggered");
}

static void
deviceAssertAndSync(bool isSync) {
    faultInjectorKernelAssert<<<1,1>>>();
    if (isSync) {
        cudaDeviceSynchronize();
    }
}


__global__ void
faultInjectorKernelTrap(void) {
    asm("trap;");
}

static void
deviceAsmTrapAndSync(bool isSync) {
    faultInjectorKernelTrap<<<1,1>>>();
    if (isSync) {
        cudaDeviceSynchronize();
    }
}


static boost::optional<boost::property_tree::ptree&>
lookupConfig(
    boost::optional<boost::property_tree::ptree&> domainConfigs,
    const char *key,
    CUpti_CallbackId cbid
) {
    boost::optional<boost::property_tree::ptree&> faultConfig =
        (*domainConfigs).get_child_optional(std::to_string(cbid));
    if (!faultConfig) {
        faultConfig = (*domainConfigs).get_child_optional(key);
    }
    if (!faultConfig) {
        faultConfig = (*domainConfigs).get_child_optional("*");
    }
    return faultConfig;
}


static void CUPTIAPI
faultInjectionCallbackHandler(
    void *userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    void *cbdata
) {
    const std::string faultInjectorKernelPrefix = std::string("faultInjectorKernel");

    CUpti_CallbackData *cbInfo = static_cast<CUpti_CallbackData*>(cbdata);

    // TODO maybe allow it in the config but right now CUPTI_API_EXIT is generally preferrable
    // for inteception, however, it means that the execution has happened.
    //
    if (cbInfo->callbackSite != CUPTI_API_EXIT) {
        return;
    }

    // Check last error
    CUPTI_CALL(cuptiGetLastError());
    boost::optional<const boost::property_tree::ptree&> matchedFaultConfig = boost::none;

    // TODO make a function, switch to read lock after debugging
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
            if (std::string(cbInfo->symbolName).compare(0, faultInjectorKernelPrefix.size(), faultInjectorKernelPrefix) == 0) {
                spdlog::debug("rejecting fake launch functionName={} symbol={}",
                    cbInfo->functionName, cbInfo->symbolName);
                return;
            }
        }
        matchedFaultConfig = lookupConfig(globalControl.driverFaultConfigs, cbInfo->functionName, cbid);
    }
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API && globalControl.runtimeFaultConfigs) {
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
            if (std::string(cbInfo->symbolName).compare(0, faultInjectorKernelPrefix.size(), faultInjectorKernelPrefix) == 0) {
                spdlog::debug("rejecting fake launch functionName={} symbol={}",
                    cbInfo->functionName, cbInfo->symbolName);
                return;
            }
        }
        matchedFaultConfig = lookupConfig(globalControl.runtimeFaultConfigs, cbInfo->functionName, cbid);
    }
    PTHREAD_CALL(pthread_rwlock_unlock(&globalControl.configLock));

    if (!matchedFaultConfig) {
        return;
    }

    const int injectionType = (*matchedFaultConfig)
        .get_optional<int>("injectionType")
        .value_or(static_cast<int>(FI_RETURN_VALUE));

    const int substituteReturnCode = (*matchedFaultConfig)
        .get_optional<int>("substituteReturnCode")
        .value_or(static_cast<int>(CUDA_SUCCESS));

    const int injectionProbability = (*matchedFaultConfig)
        .get_optional<int>("percent")
        .value_or(0);

    spdlog::trace("considered config domain={} function={} injectionType={} probability={}",
        domain, cbInfo->functionName, injectionType, injectionProbability);
    if (injectionProbability < 100) {
        if (injectionProbability <= 0 || rand() % 10000 >= injectionProbability * 10000.0 / 100.0) {
            return;
        }
    }
    spdlog::trace("matched config domain={} function={} injectionType={} probability={}",
        domain, cbInfo->functionName, injectionType, injectionProbability);

    switch (injectionType)
    {
    case FI_TRAP:
        deviceAsmTrapAndSync(false);
        break;

    case FI_ASSERT:
        deviceAssertAndSync(false);
        break;

    case FI_RETURN_VALUE:
        if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
            CUresult *cuResPtr = static_cast<CUresult*>(cbInfo->functionReturnValue);
            *cuResPtr = static_cast<CUresult>(substituteReturnCode);
        } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            cudaError_t *cudaErrPtr = static_cast<cudaError_t*>(cbInfo->functionReturnValue);
            spdlog::error("updating runtime return value DOES NOT WORK, use trap or assert");
            *cudaErrPtr = static_cast<cudaError_t>(substituteReturnCode);
            break;
        }

    default:
        break;
    }
}


int STDCALL
InitializeInjection(void) {
    spdlog::info("InitializeInjection");
    if (globalControl.initialized) {
        return 1;
    }
    // Init globalControl
    globalControlInit();
    globalControl.initialized = 1;

    registerAtExitHandler();

    PTHREAD_CALL(pthread_create(&globalControl.dynamicThread, NULL, dynamicReconfig, NULL));

    // Initialize CUPTI
    CUPTI_CALL(cuptiInitialize());

    return 1;
}


static void
readFaultInjectorConfig(void) {
    if (!configFilePath) {
        spdlog::error("specify convig via environment {}", FAULT_INJECTOR_CONFIG_PATH);
        return;
    }
    std::ifstream jsonStream(configFilePath);
    if (!jsonStream.good()){
        spdlog::error("check file exists {}", configFilePath);
        return;
    }

    PTHREAD_CALL(pthread_rwlock_wrlock(&globalControl.configLock));
    try {
        boost::property_tree::read_json(jsonStream, globalControl.configRoot);
        const int logLevel = globalControl.configRoot
            .get_optional<int>("logLevel")
            .value_or(static_cast<int>(0));
        const spdlog::level::level_enum logLevelEnum = static_cast<spdlog::level::level_enum>(logLevel);
        spdlog::info("changed log level to {}", logLevelEnum);
        spdlog::set_level(logLevelEnum);
        parseConfig(globalControl.configRoot);
        globalControl.driverFaultConfigs = globalControl.configRoot.get_child_optional("cudaDriverFaults");
        globalControl.runtimeFaultConfigs = globalControl.configRoot.get_child_optional("cudaRuntimeFaults");
    } catch (boost::property_tree::json_parser::json_parser_error& error) {
        spdlog::error("error parsing fault injector config, still editing? {}", error.what());
    }
    PTHREAD_CALL(pthread_rwlock_unlock(&globalControl.configLock));
    jsonStream.close();
    spdlog::debug("readFaultInjectorConfig from {} DONE", configFilePath);
}

static void
parseConfig(boost::property_tree::ptree const& pTree) {
    boost::property_tree::ptree::const_iterator end = pTree.end();
    for (boost::property_tree::ptree::const_iterator it = pTree.begin(); it != end; ++it) {
        spdlog::trace("congig key={} value={}",  it->first, it->second.get_value<std::string>());
        parseConfig(it->second);
    }
}

static int
eventCheck(int fd) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);
    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    return select(FD_SETSIZE, &rfds, NULL, NULL, &tv);
}


static void *
dynamicReconfig(void *args) {
    spdlog::debug("config watcher thread: inotify_init()");
    const int inotifyFd = inotify_init();
    if (inotifyFd < 0) {
        spdlog::error("inotify_init() failed");
        return NULL;
    }
    spdlog::debug("config watcher thread: inotify_add_watch {}", configFilePath);
    const int watchFd = inotify_add_watch(inotifyFd, configFilePath, IN_MODIFY);
    if (watchFd < 0) {
        spdlog::error("config watcher thread: inotify_add_watch {} failed", configFilePath);
        return NULL;
    }

    constexpr auto MAX_EVENTS = 1024;
    const auto configFilePathStr = std::string(configFilePath);
    constexpr auto EVENT_SIZE = sizeof(struct inotify_event);
    const auto BUF_LEN = MAX_EVENTS * (EVENT_SIZE + configFilePathStr.length());
    char eventBuffer[BUF_LEN];

    while (!globalControl.terminateThread) {
        spdlog::debug("about to call eventCheck");
        const int eventCheckRes = eventCheck(inotifyFd);
        spdlog::debug("eventCheck returned {}", eventCheckRes);
        if (eventCheckRes > 0) {
            const int length = read(inotifyFd, eventBuffer, BUF_LEN);
            spdlog::debug("config watcher thread: read {} bytes", length);
            if (length < EVENT_SIZE) {
                continue;
            }
            for (int i = 0; i < length; ) {
                struct inotify_event *event = (struct inotify_event *)&eventBuffer[i];
                spdlog::debug("modfiled file detected: {}", event->name);
                i += EVENT_SIZE + event->len;
            }
            readFaultInjectorConfig();
        }
    }

    if (watchFd >= 0) {
        spdlog::debug("config watcher thread: inotify_rm_watch {} {}", inotifyFd, watchFd);
        inotify_rm_watch(inotifyFd, watchFd);
    }
    if (inotifyFd >= 0) {
        spdlog::debug("config watcher thread: close {}", inotifyFd);
        close(inotifyFd);
    }
    spdlog::info("exiting dynamic reconfig thread: terminateThread={}", globalControl.terminateThread);
    return NULL;
}
