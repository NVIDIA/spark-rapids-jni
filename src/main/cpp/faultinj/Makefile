#
# Copyright 2020-2021 NVIDIA Corporation. All rights reserved
# This sample is for Linux desktop platforms. (Power, x86_64, ARM Server)
#

ifndef OS
OS   := $(shell uname)
HOST_ARCH := $(shell uname -m)
endif

CUDA_INSTALL_PATH ?= ../../../..
NVCC := "$(CUDA_INSTALL_PATH)/bin/nvcc"

TARGET_ARCH ?= $(HOST_ARCH)
TARGET_OS ?= $(shell uname | tr A-Z a-z)

ifneq ($(TARGET_ARCH), $(HOST_ARCH))
    INCLUDES = -I$(CUDA_INSTALL_PATH)/targets/$(HOST_ARCH)-$(shell uname | tr A-Z a-z)/include
    INCLUDES += -I$(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/include
    LIB_PATH ?= $(CUDA_INSTALL_PATH)/targets/$(TARGET_ARCH)-$(TARGET_OS)/lib
    TARGET_CUDA_PATH = -L $(LIB_PATH)/stubs
else
    INCLUDES := -I"$(CUDA_INSTALL_PATH)/include" -I../../include
    EXTRAS_LIB_PATH := ../../lib64
    LIB_PATH ?= "$(CUDA_INSTALL_PATH)/lib64"
endif

LIBS :=
ifeq ($(HOST_ARCH), $(TARGET_ARCH))
    LIBS = -L $(EXTRAS_LIB_PATH)
endif
LIBS += $(TARGET_CUDA_PATH) -lcuda -L $(LIB_PATH) -lcupti

NVCCFLAGS := --debug
ifneq ($(TARGET_ARCH), $(HOST_ARCH))
    ifeq ($(TARGET_ARCH), aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        endif
    endif

    ifdef HOST_COMPILER
        NVCC_COMPILER := -ccbin $(HOST_COMPILER)
    endif
endif

FAULT_INJ_LIB = libcuFaultInj.so

fault_inj_lib: faultinj.cu
	$(NVCC) $(NVCC_COMPILER) $(NVCCFLAGS) $(INCLUDES) -o $(FAULT_INJ_LIB) -Xcompiler -fPIC -shared $< $(LIBS)
	$(info export CUDA_INJECTION64_PATH=<full_path>/$(FAULT_INJ_LIB) and run any CUDA application)
clean:
	rm -f $(FAULT_INJ_LIB) faultinj.o
