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

typedef enum {
    FI_ASSERT,
    FI_TRAP,
    FI_RETURN_VALUE
} FaultInjection_Mode;

typedef struct {

} FaultInjectionConfig;

std::map<std::string,FaultInjectionConfig*> driverFaultConfigs;
std::map<std::string,FaultInjectionConfig*> runtimeFaultConfigs;


typedef struct {
    volatile uint32_t initialized;
    CUpti_SubscriberHandle  subscriber;
    int frequency;
    int terminateThread;
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


// Function Definitions

static void
globalControlInit(void) {
    std::cerr << "BOOST_LOG_TRIVIAL(info) globalControlInit of fault injection" << std::endl ;
    globalControl.initialized = 0;
    globalControl.subscriber = 0;
    globalControl.frequency = 2; // in seconds
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
}


static CUptiResult
cuptiInitialize(void) {
    CUptiResult status = CUPTI_SUCCESS;

    CUPTI_CALL(cuptiSubscribe(&globalControl.subscriber, (CUpti_CallbackFunc)faultInjectionCallbackHandler, NULL));

    // Subscribe Driver and Runtime callbacks to call cuptiFinalize in the entry/exit callback of these APIs
    CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    // CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

    return status;
}

static bool
prefix(const char *pre, const char *str) {
    return strncmp(pre, str, strlen(pre)) == 0;
}

static bool
faultInjectionMatch(CUpti_CallbackData *cbd, CUpti_CallbackId cbId) {
    if (cbd->callbackSite == CUPTI_API_ENTER) {
        std::cerr << "#### faultInjectionMatch function=" << cbd->functionName << std::endl;
    }

    // switch (globalControl.faultInjectionMode) {
    // case FI_ASSERT:
    // case FI_TRAP:
    //     if (cbd->callbackSite != CUPTI_API_EXIT)
    //         return false;

    //     if (   cbId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000
    //         || cbId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
    //           // std::cerr  <<  "#### symbolName=" << cbd->symbolName << std::endl;
    //           return   !strstr(cbd->symbolName, "faultInjectorKernel")
    //                 && prefix(globalControl.functionName, cbd->functionName);
    //         }
    //     return prefix(globalControl.functionName, cbd->functionName);

    // case FI_RETURN_VALUE:
    // default:
    //     return cbd->callbackSite == CUPTI_API_EXIT
    //         && prefix(globalControl.functionName, cbd->functionName);
    // }
    return false;
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
    CUpti_CallbackData *cbInfo = (CUpti_CallbackData *)cbdata;
    // Check last error
    CUPTI_CALL(cuptiGetLastError());

    if (faultInjectionMatch(cbInfo, cbid)) {
        // switch (globalControl.faultInjectionMode)
        // {
        // case FI_ASSERT:
        //     deviceAssertAndSync();
        //     break;

        // case FI_TRAP:
        //     deviceAsmTrapAndSync();
        //     break;

        // case FI_RETURN_VALUE:
        // default:
            std::cerr << "#### modifying " << cbInfo->functionName;
            switch (domain) {

            case CUPTI_CB_DOMAIN_DRIVER_API: {
                CUresult *cuResPtr = (CUresult *)cbInfo->functionReturnValue;
                std::cerr << "'s CUresult return value: " << *cuResPtr << std::endl;
                *cuResPtr = CUDA_ERROR_NO_DEVICE;
                break;
            }

            case CUPTI_CB_DOMAIN_RUNTIME_API:
            default:
                cudaError_t *cudaErrPtr = (cudaError_t *)cbInfo->functionReturnValue;
                std::cerr  <<  "'s cudaError_t return value: " << *cudaErrPtr << " DOES NOT WORK" << std::endl;
                *cudaErrPtr = cudaErrorInvalidValue;
                break;
            }
        // }
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

    boost::property_tree::ptree root;
    boost::property_tree::read_json(jsonStream, root);
    jsonStream.close();
    parseConfig(root);
}

static void
parseConfig(boost::property_tree::ptree const& pTree) {
    boost::property_tree::ptree::const_iterator end = pTree.end();
    for (boost::property_tree::ptree::const_iterator it = pTree.begin(); it != end; ++it) {
        std::cerr <<  it->first << ": " << it->second.get_value<std::string>() << std::endl;
        parseConfig(it->second);
    }
}


