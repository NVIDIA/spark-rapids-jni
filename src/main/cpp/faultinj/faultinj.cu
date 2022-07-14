/*
 * Copyright 2020 NVIDIA Corporation. All rights reserved.
 *
 * Sample CUPTI based injection to attach and detach CUPTI
 * For detaching, it uses CUPTI API cuptiFinalize().
 *
 * It is recommended to invoke API cuptiFinalize() in the
 * exit callsite of any CUDA Driver/Runtime API.
 *
 * API cuptiFinalize() destroys and cleans up all the
 * resources associated with CUPTI in the current process.
 * After CUPTI detaches from the process, the process will
 * keep on running with no CUPTI attached to it.
 *
 * CUPTI can be attached by calling any CUPTI API as CUPTI
 * supports lazy initialization. Any subsequent CUPTI API
 * call will reinitialize the CUPTI.
 *
 * You can attach and detach CUPTI any number of times.
 *
 * After building the sample, set the following environment variable
 * export CUDA_INJECTION64_PATH=<full_path>/libCuptiFinalize.so
 * Add CUPTI library in LD_LIBRARY_PATH and run any CUDA sample
 * with runtime more than 10 sec for demonstration of the
 * CUPTI sample
 */

#include <cuda.h>
#include <cupti.h>

#include <iostream>
#include <map>
#include <assert.h>
#include <pthread.h>

#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#define STDCALL
inline const std::string FAULT_INJECTOR_CONFIG_PATH{"FAULT_INJECTOR_CONFIG_PATH"};

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

    int frequency;
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
    std::cerr << "#### globalControlInit of fault injection" << std::endl ;
    globalControl.initialized = 0;
    globalControl.subscriber = 0;
    globalControl.frequency = 10; // poll config file every 10 seconds
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
    std::cerr << "Reconfig thread shut down ... exiting" << std::endl;
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
deviceAssertAndSync(void) {
    faultInjectorKernelAssert<<<1,1>>>();
    cudaDeviceSynchronize();
}


__global__ void
faultInjectorKernelTrap(void) {
    asm("trap;");
}

static void
deviceAsmTrapAndSync(void) {
    faultInjectorKernelTrap<<<1,1>>>();
    cudaDeviceSynchronize();
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
                std::cerr << "#### GERA_DEBUG rejecting fake launch functionName= " << cbInfo->functionName
                    << " symbol=" << cbInfo->symbolName
                    << std::endl;
                return;
            }
        }
        matchedFaultConfig = (*globalControl.driverFaultConfigs)
            .get_child_optional(cbInfo->functionName);
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
                std::cerr << "#### GERA_DEBUG rejecting fake launch functionName= " << cbInfo->functionName
                    << " symbol=" << cbInfo->symbolName
                    << std::endl;
                return;
            }
        }
        matchedFaultConfig = (*globalControl.runtimeFaultConfigs)
            .get_child_optional(cbInfo->functionName);
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

    const double injectionProbability = (*matchedFaultConfig)
        .get_optional<double>("prob")
        .value_or(0.0);

    // std::cerr << "#### matched config domain=" << domain
    //           << " function=" << cbInfo->functionName
    //           << " injectionType=" << injectionType
    //           << " injectionProbability=" << injectionProbability
    //           << std::endl;
    if (injectionProbability < 1.0) {
        if (injectionProbability == 0.0 || rand() % 10000 >= injectionProbability * 10000) {
            return;
        }
    }

    switch (injectionType)
    {
    case FI_TRAP:
        deviceAsmTrapAndSync();
        break;

    case FI_ASSERT:
        deviceAssertAndSync();
        break;

    case FI_RETURN_VALUE:
        if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
            CUresult *cuResPtr = static_cast<CUresult*>(cbInfo->functionReturnValue);
            std::cerr << "'s CUresult return value: " << *cuResPtr << std::endl;
            *cuResPtr = static_cast<CUresult>(substituteReturnCode);
        } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            cudaError_t *cudaErrPtr = static_cast<cudaError_t*>(cbInfo->functionReturnValue);
            std::cerr  <<  "'s cudaError_t return value: " << *cudaErrPtr << " DOES NOT WORK" << std::endl;
            *cudaErrPtr = static_cast<cudaError_t>(substituteReturnCode);
            break;
        }

    default:
        break;
    }
}


int STDCALL
InitializeInjection(void) {
    std::cerr << "##### InitializeInjection logs " << std::endl;
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
    const auto configFilePath = std::getenv(FAULT_INJECTOR_CONFIG_PATH.c_str());
    if (!configFilePath) {
        std::cerr << "specify convig via environment " << FAULT_INJECTOR_CONFIG_PATH << std::endl;
        return;
    }
    std::ifstream jsonStream(configFilePath);
    if (!jsonStream.good()){
        std::cerr <<  "check file exists " << configFilePath << std::endl;
        return;
    }

    PTHREAD_CALL(pthread_rwlock_wrlock(&globalControl.configLock));
    boost::property_tree::read_json(jsonStream, globalControl.configRoot);

    globalControl.frequency = globalControl.configRoot
        .get_optional<int>("reloadAfterSeconds")
        .value_or(static_cast<int>(10));
    // parseConfig(globalControl.configRoot);
    globalControl.driverFaultConfigs = globalControl.configRoot.get_child_optional("cudaDriverFaults");
    globalControl.runtimeFaultConfigs = globalControl.configRoot.get_child_optional("cudaRuntimeFaults");
    PTHREAD_CALL(pthread_rwlock_unlock(&globalControl.configLock));
    jsonStream.close();
    std::cerr << "#### readFaultInjectorConfig " << configFilePath << " DONE" << std::endl ;
}

static void
parseConfig(boost::property_tree::ptree const& pTree) {
    boost::property_tree::ptree::const_iterator end = pTree.end();
    for (boost::property_tree::ptree::const_iterator it = pTree.begin(); it != end; ++it) {
        std::cerr <<  it->first << ": " << it->second.get_value<std::string>() << std::endl;
        parseConfig(it->second);
    }
}

static void *
dynamicReconfig(void *args) {
    while (!globalControl.terminateThread && globalControl.frequency > 0) {
        sleep(globalControl.frequency);
        readFaultInjectorConfig();
    }
    std::cerr << "#### Exiting dynamic reconfig thread" << std::endl;
    return NULL;
}
