/*
 * SPDX-FileCopyrightText: Copyright (c) 2009-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://nvidia.github.io/NVTX/LICENSE.txt for license information.
 */

#ifndef NVTX_IMPL_GUARD
#error Never include this file directly -- it is automatically included by nvToolsExt.h (except when NVTX_NO_IMPL is defined).
#endif

#if defined(NVTX_AS_SYSTEM_HEADER)
#if defined(__clang__)
#pragma clang system_header
#elif defined(__GNUC__) || defined(__NVCOMPILER)
#pragma GCC system_header
#elif defined(_MSC_VER)
#pragma system_header
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

/* ---- Include required platform headers ---- */

#if defined(_WIN32)

#include <windows.h>

#else
#include <unistd.h>

#if defined(__ANDROID__)
#include <android/api-level.h>
#endif

#if defined(__linux__) || defined(__CYGWIN__)
#include <sched.h>
#endif

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <sys/types.h>

#endif

/* ---- Define macros used in this file ---- */

#define NVTX_INIT_STATE_FRESH    0
#define NVTX_INIT_STATE_STARTED  1
#define NVTX_INIT_STATE_COMPLETE 2

#ifdef NVTX_DEBUG_PRINT
#ifdef __ANDROID__
#include <android/log.h>
#define NVTX_ERR(...)  __android_log_print(ANDROID_LOG_ERROR, "NVTOOLSEXT", __VA_ARGS__);
#define NVTX_INFO(...) __android_log_print(ANDROID_LOG_INFO, "NVTOOLSEXT", __VA_ARGS__);
#else
#include <stdio.h>
#define NVTX_ERR(...)  fprintf(stderr, "NVTX_ERROR: " __VA_ARGS__)
#define NVTX_INFO(...) fprintf(stderr, "NVTX_INFO: " __VA_ARGS__)
#endif
#else /* !defined(NVTX_DEBUG_PRINT) */
#define NVTX_ERR(...)
#define NVTX_INFO(...)
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/* ---- Forward declare all functions referenced in globals ---- */

NVTX_LINKONCE_FWDDECL_FUNCTION void NVTX_VERSIONED_IDENTIFIER(nvtxInitOnce)(void);
NVTX_LINKONCE_FWDDECL_FUNCTION int NVTX_API
  NVTX_VERSIONED_IDENTIFIER(nvtxEtiGetModuleFunctionTable)(NvtxCallbackModule module,
                                                           NvtxFunctionTable* out_table,
                                                           unsigned int* out_size);
NVTX_LINKONCE_FWDDECL_FUNCTION void NVTX_API
  NVTX_VERSIONED_IDENTIFIER(nvtxEtiSetInjectionNvtxVersion)(uint32_t version);
NVTX_LINKONCE_FWDDECL_FUNCTION const void* NVTX_API
  NVTX_VERSIONED_IDENTIFIER(nvtxGetExportTable)(uint32_t exportTableId);

#include "nvtxInitDecls.h"

/* ---- Define all globals ---- */

typedef struct nvtxGlobals_t {
  volatile unsigned int initState;
  NvtxExportTableCallbacks etblCallbacks;
  NvtxExportTableVersionInfo etblVersionInfo;

  /* Implementation function pointers */
  nvtxMarkEx_impl_fntype nvtxMarkEx_impl_fnptr;
  nvtxMarkA_impl_fntype nvtxMarkA_impl_fnptr;
  nvtxMarkW_impl_fntype nvtxMarkW_impl_fnptr;
  nvtxRangeStartEx_impl_fntype nvtxRangeStartEx_impl_fnptr;
  nvtxRangeStartA_impl_fntype nvtxRangeStartA_impl_fnptr;
  nvtxRangeStartW_impl_fntype nvtxRangeStartW_impl_fnptr;
  nvtxRangeEnd_impl_fntype nvtxRangeEnd_impl_fnptr;
  nvtxRangePushEx_impl_fntype nvtxRangePushEx_impl_fnptr;
  nvtxRangePushA_impl_fntype nvtxRangePushA_impl_fnptr;
  nvtxRangePushW_impl_fntype nvtxRangePushW_impl_fnptr;
  nvtxRangePop_impl_fntype nvtxRangePop_impl_fnptr;
  nvtxNameCategoryA_impl_fntype nvtxNameCategoryA_impl_fnptr;
  nvtxNameCategoryW_impl_fntype nvtxNameCategoryW_impl_fnptr;
  nvtxNameOsThreadA_impl_fntype nvtxNameOsThreadA_impl_fnptr;
  nvtxNameOsThreadW_impl_fntype nvtxNameOsThreadW_impl_fnptr;

  nvtxNameCuDeviceA_fakeimpl_fntype nvtxNameCuDeviceA_impl_fnptr;
  nvtxNameCuDeviceW_fakeimpl_fntype nvtxNameCuDeviceW_impl_fnptr;
  nvtxNameCuContextA_fakeimpl_fntype nvtxNameCuContextA_impl_fnptr;
  nvtxNameCuContextW_fakeimpl_fntype nvtxNameCuContextW_impl_fnptr;
  nvtxNameCuStreamA_fakeimpl_fntype nvtxNameCuStreamA_impl_fnptr;
  nvtxNameCuStreamW_fakeimpl_fntype nvtxNameCuStreamW_impl_fnptr;
  nvtxNameCuEventA_fakeimpl_fntype nvtxNameCuEventA_impl_fnptr;
  nvtxNameCuEventW_fakeimpl_fntype nvtxNameCuEventW_impl_fnptr;

  nvtxNameClDeviceA_fakeimpl_fntype nvtxNameClDeviceA_impl_fnptr;
  nvtxNameClDeviceW_fakeimpl_fntype nvtxNameClDeviceW_impl_fnptr;
  nvtxNameClContextA_fakeimpl_fntype nvtxNameClContextA_impl_fnptr;
  nvtxNameClContextW_fakeimpl_fntype nvtxNameClContextW_impl_fnptr;
  nvtxNameClCommandQueueA_fakeimpl_fntype nvtxNameClCommandQueueA_impl_fnptr;
  nvtxNameClCommandQueueW_fakeimpl_fntype nvtxNameClCommandQueueW_impl_fnptr;
  nvtxNameClMemObjectA_fakeimpl_fntype nvtxNameClMemObjectA_impl_fnptr;
  nvtxNameClMemObjectW_fakeimpl_fntype nvtxNameClMemObjectW_impl_fnptr;
  nvtxNameClSamplerA_fakeimpl_fntype nvtxNameClSamplerA_impl_fnptr;
  nvtxNameClSamplerW_fakeimpl_fntype nvtxNameClSamplerW_impl_fnptr;
  nvtxNameClProgramA_fakeimpl_fntype nvtxNameClProgramA_impl_fnptr;
  nvtxNameClProgramW_fakeimpl_fntype nvtxNameClProgramW_impl_fnptr;
  nvtxNameClEventA_fakeimpl_fntype nvtxNameClEventA_impl_fnptr;
  nvtxNameClEventW_fakeimpl_fntype nvtxNameClEventW_impl_fnptr;

  nvtxNameCudaDeviceA_fakeimpl_fntype nvtxNameCudaDeviceA_impl_fnptr;
  nvtxNameCudaDeviceW_fakeimpl_fntype nvtxNameCudaDeviceW_impl_fnptr;
  nvtxNameCudaStreamA_fakeimpl_fntype nvtxNameCudaStreamA_impl_fnptr;
  nvtxNameCudaStreamW_fakeimpl_fntype nvtxNameCudaStreamW_impl_fnptr;
  nvtxNameCudaEventA_fakeimpl_fntype nvtxNameCudaEventA_impl_fnptr;
  nvtxNameCudaEventW_fakeimpl_fntype nvtxNameCudaEventW_impl_fnptr;

  nvtxDomainMarkEx_impl_fntype nvtxDomainMarkEx_impl_fnptr;
  nvtxDomainRangeStartEx_impl_fntype nvtxDomainRangeStartEx_impl_fnptr;
  nvtxDomainRangeEnd_impl_fntype nvtxDomainRangeEnd_impl_fnptr;
  nvtxDomainRangePushEx_impl_fntype nvtxDomainRangePushEx_impl_fnptr;
  nvtxDomainRangePop_impl_fntype nvtxDomainRangePop_impl_fnptr;
  nvtxDomainResourceCreate_impl_fntype nvtxDomainResourceCreate_impl_fnptr;
  nvtxDomainResourceDestroy_impl_fntype nvtxDomainResourceDestroy_impl_fnptr;
  nvtxDomainNameCategoryA_impl_fntype nvtxDomainNameCategoryA_impl_fnptr;
  nvtxDomainNameCategoryW_impl_fntype nvtxDomainNameCategoryW_impl_fnptr;
  nvtxDomainRegisterStringA_impl_fntype nvtxDomainRegisterStringA_impl_fnptr;
  nvtxDomainRegisterStringW_impl_fntype nvtxDomainRegisterStringW_impl_fnptr;
  nvtxDomainCreateA_impl_fntype nvtxDomainCreateA_impl_fnptr;
  nvtxDomainCreateW_impl_fntype nvtxDomainCreateW_impl_fnptr;
  nvtxDomainDestroy_impl_fntype nvtxDomainDestroy_impl_fnptr;
  nvtxInitialize_impl_fntype nvtxInitialize_impl_fnptr;

  nvtxDomainSyncUserCreate_fakeimpl_fntype nvtxDomainSyncUserCreate_impl_fnptr;
  nvtxDomainSyncUserDestroy_fakeimpl_fntype nvtxDomainSyncUserDestroy_impl_fnptr;
  nvtxDomainSyncUserAcquireStart_fakeimpl_fntype nvtxDomainSyncUserAcquireStart_impl_fnptr;
  nvtxDomainSyncUserAcquireFailed_fakeimpl_fntype nvtxDomainSyncUserAcquireFailed_impl_fnptr;
  nvtxDomainSyncUserAcquireSuccess_fakeimpl_fntype nvtxDomainSyncUserAcquireSuccess_impl_fnptr;
  nvtxDomainSyncUserReleasing_fakeimpl_fntype nvtxDomainSyncUserReleasing_impl_fnptr;

  /* Tables of function pointers -- Extra null added to the end to ensure
   *  a crash instead of silent corruption if a tool reads off the end. */
  NvtxFunctionPointer* functionTable_CORE[NVTX_CBID_CORE_SIZE + 1];
  NvtxFunctionPointer* functionTable_CUDA[NVTX_CBID_CUDA_SIZE + 1];
  NvtxFunctionPointer* functionTable_OPENCL[NVTX_CBID_OPENCL_SIZE + 1];
  NvtxFunctionPointer* functionTable_CUDART[NVTX_CBID_CUDART_SIZE + 1];
  NvtxFunctionPointer* functionTable_CORE2[NVTX_CBID_CORE2_SIZE + 1];
  NvtxFunctionPointer* functionTable_SYNC[NVTX_CBID_SYNC_SIZE + 1];
} nvtxGlobals_t;

#define NVTX_GLOBAL_TABLE_ENTRY(name)          \
  (NVTX_REINTERPRET_CAST(NvtxFunctionPointer*, \
                         &NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).name##_impl_fnptr))

NVTX_LINKONCE_DEFINE_GLOBAL nvtxGlobals_t NVTX_VERSIONED_IDENTIFIER(nvtxGlobals) = {
  NVTX_INIT_STATE_FRESH,

  {sizeof(NvtxExportTableCallbacks), NVTX_VERSIONED_IDENTIFIER(nvtxEtiGetModuleFunctionTable)},
  {sizeof(NvtxExportTableVersionInfo),
   NVTX_VERSION,
   0,
   NVTX_VERSIONED_IDENTIFIER(nvtxEtiSetInjectionNvtxVersion)},

  /* Implementation function pointers */
  NVTX_VERSIONED_IDENTIFIER(nvtxMarkEx_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxMarkA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxMarkW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxRangeStartEx_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxRangeStartA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxRangeStartW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxRangeEnd_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxRangePushEx_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxRangePushA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxRangePushW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxRangePop_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCategoryA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCategoryW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameOsThreadA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameOsThreadW_impl_init),

  NVTX_VERSIONED_IDENTIFIER(nvtxNameCuDeviceA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCuDeviceW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCuContextA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCuContextW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCuStreamA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCuStreamW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCuEventA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCuEventW_impl_init),

  NVTX_VERSIONED_IDENTIFIER(nvtxNameClDeviceA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClDeviceW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClContextA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClContextW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClCommandQueueA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClCommandQueueW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClMemObjectA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClMemObjectW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClSamplerA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClSamplerW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClProgramA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClProgramW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClEventA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameClEventW_impl_init),

  NVTX_VERSIONED_IDENTIFIER(nvtxNameCudaDeviceA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCudaDeviceW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCudaStreamA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCudaStreamW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCudaEventA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxNameCudaEventW_impl_init),

  NVTX_VERSIONED_IDENTIFIER(nvtxDomainMarkEx_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainRangeStartEx_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainRangeEnd_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainRangePushEx_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainRangePop_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainResourceCreate_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainResourceDestroy_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainNameCategoryA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainNameCategoryW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainRegisterStringA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainRegisterStringW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainCreateA_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainCreateW_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainDestroy_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxInitialize_impl_init),

  NVTX_VERSIONED_IDENTIFIER(nvtxDomainSyncUserCreate_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainSyncUserDestroy_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainSyncUserAcquireStart_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainSyncUserAcquireFailed_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainSyncUserAcquireSuccess_impl_init),
  NVTX_VERSIONED_IDENTIFIER(nvtxDomainSyncUserReleasing_impl_init),

  /* Tables of function pointers */
  {NVTX_NULLPTR,
   NVTX_GLOBAL_TABLE_ENTRY(nvtxMarkEx),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxMarkA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxMarkW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxRangeStartEx),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxRangeStartA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxRangeStartW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxRangeEnd),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxRangePushEx),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxRangePushA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxRangePushW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxRangePop),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCategoryA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCategoryW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameOsThreadA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameOsThreadW),
   NVTX_NULLPTR},
  {NVTX_NULLPTR,
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCuDeviceA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCuDeviceW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCuContextA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCuContextW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCuStreamA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCuStreamW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCuEventA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCuEventW),
   NVTX_NULLPTR},
  {NVTX_NULLPTR,
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClDeviceA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClDeviceW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClContextA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClContextW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClCommandQueueA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClCommandQueueW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClMemObjectA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClMemObjectW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClSamplerA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClSamplerW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClProgramA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClProgramW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClEventA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameClEventW),
   NVTX_NULLPTR},
  {NVTX_NULLPTR,
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCudaDeviceA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCudaDeviceW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCudaStreamA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCudaStreamW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCudaEventA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxNameCudaEventW),
   NVTX_NULLPTR},
  {NVTX_NULLPTR,
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainMarkEx),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainRangeStartEx),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainRangeEnd),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainRangePushEx),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainRangePop),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainResourceCreate),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainResourceDestroy),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainNameCategoryA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainNameCategoryW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainRegisterStringA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainRegisterStringW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainCreateA),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainCreateW),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainDestroy),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxInitialize),
   NVTX_NULLPTR},
  {NVTX_NULLPTR,
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainSyncUserCreate),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainSyncUserDestroy),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainSyncUserAcquireStart),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainSyncUserAcquireFailed),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainSyncUserAcquireSuccess),
   NVTX_GLOBAL_TABLE_ENTRY(nvtxDomainSyncUserReleasing),
   NVTX_NULLPTR}};

#undef NVTX_GLOBAL_TABLE_ENTRY

/* ---- Define static inline implementations of core API functions ---- */

#include "nvtxImplCore.h"

/* ---- Define implementations of export table functions ---- */

NVTX_LINKONCE_DEFINE_FUNCTION int NVTX_API NVTX_VERSIONED_IDENTIFIER(nvtxEtiGetModuleFunctionTable)(
  NvtxCallbackModule module, NvtxFunctionTable* out_table, unsigned int* out_size)
{
  unsigned int bytes      = 0;
  NvtxFunctionTable table = NVTX_NULLPTR;

  switch (module) {
    case NVTX_CB_MODULE_CORE:
      table = NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_CORE;
      bytes = NVTX_STATIC_CAST(unsigned int,
                               sizeof(NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_CORE));
      break;
    case NVTX_CB_MODULE_CUDA:
      table = NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_CUDA;
      bytes = NVTX_STATIC_CAST(unsigned int,
                               sizeof(NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_CUDA));
      break;
    case NVTX_CB_MODULE_OPENCL:
      table = NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_OPENCL;
      bytes = NVTX_STATIC_CAST(unsigned int,
                               sizeof(NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_OPENCL));
      break;
    case NVTX_CB_MODULE_CUDART:
      table = NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_CUDART;
      bytes = NVTX_STATIC_CAST(unsigned int,
                               sizeof(NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_CUDART));
      break;
    case NVTX_CB_MODULE_CORE2:
      table = NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_CORE2;
      bytes = NVTX_STATIC_CAST(unsigned int,
                               sizeof(NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_CORE2));
      break;
    case NVTX_CB_MODULE_SYNC:
      table = NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_SYNC;
      bytes = NVTX_STATIC_CAST(unsigned int,
                               sizeof(NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).functionTable_SYNC));
      break;
    case NVTX_CB_MODULE_INVALID:
    case NVTX_CB_MODULE_SIZE:
    case NVTX_CB_MODULE_FORCE_INT:
    default: return 0;
  }

  if (out_size)
    *out_size = (bytes / NVTX_STATIC_CAST(unsigned int, sizeof(NvtxFunctionPointer*))) - 1;

  if (out_table) *out_table = table;

  return 1;
}

NVTX_LINKONCE_DEFINE_FUNCTION const void* NVTX_API
NVTX_VERSIONED_IDENTIFIER(nvtxGetExportTable)(uint32_t exportTableId)
{
  switch (exportTableId) {
    case NVTX_ETID_CALLBACKS: return &NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).etblCallbacks;
    case NVTX_ETID_VERSIONINFO: return &NVTX_VERSIONED_IDENTIFIER(nvtxGlobals).etblVersionInfo;
    default: return NVTX_NULLPTR;
  }
}

NVTX_LINKONCE_DEFINE_FUNCTION void NVTX_API
NVTX_VERSIONED_IDENTIFIER(nvtxEtiSetInjectionNvtxVersion)(uint32_t version)
{
  /* Reserved for custom implementations to resolve problems with tools */
  (void)version;
}

/* ---- Define implementations of init versions of all API functions ---- */

#include "nvtxInitDefs.h"

/* ---- Define implementations of initialization functions ---- */

#include "nvtxInit.h"

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
