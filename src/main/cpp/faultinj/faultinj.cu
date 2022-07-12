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

#include <assert.h>
#include <stdbool.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <cuda.h>
#include <cupti.h>

#define STDCALL

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
    volatile uint32_t initialized;
    CUpti_SubscriberHandle  subscriber;
    volatile uint32_t detachCupti;
    int frequency;
    int tracingEnabled;
    int terminateThread;
    uint64_t kernelsTraced;
    FaultInjection_Mode faultInjectionMode;
    const char *functionName;
} injGlobalControl;
injGlobalControl globalControl;

// Function Declarations

static CUptiResult cuptiInitialize(void);

static void atExitHandler(void);

static void CUPTIAPI faultInjectionCallbackHandler(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void *cbInfo);

extern int STDCALL InitializeInjection(void);

#if defined(__cplusplus)
}
#endif

// Function Definitions

static void
globalControlInit(void) {
    std::cerr << "GERA_DEBUG: globalControlInit of fault injection" << std::endl;
    globalControl.initialized = 0;
    globalControl.subscriber = 0;
    globalControl.detachCupti = 0;
    globalControl.frequency = 2; // in seconds
    globalControl.tracingEnabled = 0;
    globalControl.terminateThread = 0;
    globalControl.kernelsTraced = 0;
    globalControl.faultInjectionMode = FI_ASSERT;
    globalControl.functionName = "cudaLaunchKernel";
}

static void
registerAtExitHandler(void) {
    // Register atExitHandler
    atexit(&atExitHandler);
}


static void
atExitHandler(void) {
    globalControl.terminateThread = 1;
    std::cerr <<  "GERA_DEBUG CUPTI atExitHandler" << std::endl;
}


static CUptiResult
cuptiInitialize(void) {
    CUptiResult status = CUPTI_SUCCESS;

    CUPTI_CALL(cuptiSubscribe(&globalControl.subscriber, (CUpti_CallbackFunc)faultInjectionCallbackHandler, NULL));

    // Subscribe Driver and Runtime callbacks to call cuptiFinalize in the entry/exit callback of these APIs
    CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

    return status;
}

static bool
prefix(const char *pre, const char *str) {
    return strncmp(pre, str, strlen(pre)) == 0;
}

static bool
faultInjectionMatch(CUpti_CallbackData *cbd, CUpti_CallbackId cbId) {
    switch (globalControl.faultInjectionMode) {
    case FI_ASSERT:
    case FI_TRAP:
        if (cbd->callbackSite != CUPTI_API_EXIT)
            return false;

        if (   cbId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000
            || cbId == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
              std::cerr << "symbolName=" << cbd->symbolName << std::endl;
              return   !strstr(cbd->symbolName, "faultInjectorKernel")
                    && prefix(globalControl.functionName, cbd->functionName);
            }
        return prefix(globalControl.functionName, cbd->functionName);

    case FI_RETURN_VALUE:
    default:
        return cbd->callbackSite == CUPTI_API_EXIT
            && prefix(globalControl.functionName, cbd->functionName);
    }
}

__global__ void
faultInjectorKernelAssert(void) {
    assert(0 && "GERA_DEBUG kernelAssert triggered");
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
        switch (globalControl.faultInjectionMode)
        {
        case FI_ASSERT:
            deviceAssertAndSync();
            break;

        case FI_TRAP:
            deviceAsmTrapAndSync();
            break;

        case FI_RETURN_VALUE:
        default:
            switch (domain) {

            case CUPTI_CB_DOMAIN_DRIVER_API: {
                CUresult *cuResPtr = (CUresult *)cbInfo->functionReturnValue;
                std::cerr << "GERA_DEBUG: modifying function CUresult return value: " << *cuResPtr << std::endl;
                *cuResPtr = CUDA_ERROR_INVALID_VALUE;
                break;
            }

            case CUPTI_CB_DOMAIN_RUNTIME_API:
            default:
                cudaError_t *cudaErrPtr = (cudaError_t *)cbInfo->functionReturnValue;
                std::cerr << "GERA_DEBUG: modifying function cudaError_t return value: " << *cudaErrPtr << " DOES NOT WORK" << std::endl;
                *cudaErrPtr = cudaErrorInvalidValue;
                break;
            }
        }
    }
}


int STDCALL
InitializeInjection(void) {

    if (globalControl.initialized) {
        return 1;
    }
    // Init globalControl
    globalControlInit();
    globalControl.initialized = 1;
    globalControl.tracingEnabled = 1;


    registerAtExitHandler();

    // Initialize CUPTI
    CUPTI_CALL(cuptiInitialize());

    return 1;
}
