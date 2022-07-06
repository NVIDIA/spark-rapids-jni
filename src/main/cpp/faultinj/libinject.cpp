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

#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>

#include <cuda.h>
#include <cupti.h>

#define STDCALL

#if defined(__cplusplus)
extern "C" {
#endif

// MAcros

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

#define BUF_SIZE (8 * 1024 * 1024)  // 8MB
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// Global Structure

typedef struct {
    volatile uint32_t initialized;
    CUpti_SubscriberHandle  subscriber;
    volatile uint32_t detachCupti;
    int frequency;
    int tracingEnabled;
    int terminateThread;
    uint64_t kernelsTraced;
    pthread_t dynamicThread;
    pthread_mutex_t mutexFinalize;
    pthread_cond_t mutexCondition;
} injGlobalControl;
injGlobalControl globalControl;

// Function Declarations

static CUptiResult cuptiInitialize(void);

static void atExitHandler(void);

void CUPTIAPI callbackHandler(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, void *cbInfo);

extern int STDCALL InitializeInjection(void);

#if defined(__cplusplus)
}
#endif

// Function Definitions

static void
globalControlInit(void) {
    globalControl.initialized = 0;
    globalControl.subscriber = 0;
    globalControl.detachCupti = 0;
    globalControl.frequency = 2; // in seconds
    globalControl.tracingEnabled = 0;
    globalControl.terminateThread = 0;
    globalControl.kernelsTraced = 0;
    globalControl.mutexFinalize = PTHREAD_MUTEX_INITIALIZER;
    globalControl.mutexCondition = PTHREAD_COND_INITIALIZER;
}

void registerAtExitHandler(void) {
    // Register atExitHandler
    atexit(&atExitHandler);
}

static void
printSummary(void) {
    printf("\n-------------------------------------------------------------------\n");
    printf("\tKernels traced : %llu", (unsigned long long)globalControl.kernelsTraced);
    printf("\n-------------------------------------------------------------------\n");
}

static void
atExitHandler(void) {
    globalControl.terminateThread = 1;

    // Force flush
    if(globalControl.tracingEnabled) {
        CUPTI_CALL(cuptiActivityFlushAll(1));
    }

    // PTHREAD_CALL(pthread_join(globalControl.dynamicThread, NULL));
    printSummary();
}

static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    uint8_t *rawBuffer;

    *size = BUF_SIZE;
    rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

    *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
    *maxNumRecords = 0;

    if (*buffer == NULL) {
        printf("Error: Out of memory.\n");
        exit(EXIT_FAILURE);
    }
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    CUptiResult status;
    CUpti_Activity *record = NULL;
    size_t dropped;

    do {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            CUpti_ActivityKind kind = record->kind;
            switch (kind) {
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
                globalControl.kernelsTraced++;
                break;
            default:
                break;
            }
        }
        else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        }
        else {
            CUPTI_CALL(status);
        }
    } while (1);

    // Report any records dropped from the queue
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
        printf("Dropped %u activity records.\n", (unsigned int)dropped);
    }
    free(buffer);
}

static CUptiResult
cuptiInitialize(void) {
    CUptiResult status = CUPTI_SUCCESS;

    CUPTI_CALL(cuptiSubscribe(&globalControl.subscriber, (CUpti_CallbackFunc)callbackHandler, NULL));

    // Subscribe Driver and Runtime callbacks to call cuptiFinalize in the entry/exit callback of these APIs
    CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CALL(cuptiEnableDomain(1, globalControl.subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

    // Enable CUPTI activities
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    // Register buffer callbacks
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    return status;
}

void CUPTIAPI
callbackHandler(void *userdata, CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid, void *cbdata) {
    const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *)cbdata;
    // Check last error
    CUPTI_CALL(cuptiGetLastError());
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
        fprintf(stderr, "GERA_DEBUG callbackHandler: cbid=%d domain=%d function=%s\n", cbid, domain, cbInfo->functionName);
    }

    // This code path is taken only when we wish to perform the CUPTI teardown
    if (globalControl.detachCupti) {
        switch(domain) {
        case CUPTI_CB_DOMAIN_RUNTIME_API:
        case CUPTI_CB_DOMAIN_DRIVER_API:
            if (cbInfo->callbackSite == CUPTI_API_EXIT) {
                // Detach CUPTI calling cuptiFinalize() API
                CUPTI_CALL(cuptiFinalize());
                PTHREAD_CALL(pthread_cond_broadcast(&globalControl.mutexCondition));
            }
            break;
        default:
            break;
        }
    }
}

void *dynamicAttachDetach(void *arg) {
    while (!globalControl.terminateThread) {
        sleep(globalControl.frequency);

        // Check the condition again after sleep
        if (globalControl.terminateThread) {
            break;
        }

        // Turn on/off CUPTI at a regular interval
        if (globalControl.tracingEnabled) {
            printf("\nCUPTI detach starting ...\n");

            // Force flush
            CUPTI_CALL(cuptiActivityFlushAll(1));

            globalControl.detachCupti = 1;

            // Lock and wait for callbackHandler() to perform CUPTI teardown
            PTHREAD_CALL(pthread_mutex_lock(&globalControl.mutexFinalize));
            PTHREAD_CALL(pthread_cond_wait(&globalControl.mutexCondition, &globalControl.mutexFinalize));
            PTHREAD_CALL(pthread_mutex_unlock(&globalControl.mutexFinalize));

            printf("CUPTI detach completed.\n");

            globalControl.detachCupti = 0;
            globalControl.tracingEnabled = 0;
            globalControl.subscriber = 0;
        }
        else {
            printf("\nCUPTI attach starting ...\n");

            CUPTI_CALL(cuptiInitialize());
            globalControl.tracingEnabled = 1;

            printf("CUPTI attach completed.\n");
        }
    }
    return NULL;
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


    //Initialize Mutex
    // PTHREAD_CALL(pthread_mutex_init(&globalControl.mutexFinalize, 0));

    registerAtExitHandler();

    // Initialize CUPTI
    CUPTI_CALL(cuptiInitialize());

    // Launch the thread
    // PTHREAD_CALL(pthread_create(&globalControl.dynamicThread, NULL, dynamicAttachDetach, NULL));


    return 1;
}
