### CONFIG
# CUDA_DIR := /usr/local/cuda-8.0
CUDA_DIR      ?= /usr/local/cuda
CUDA_UTIL_DIR ?= ~/projs/cuda-utils
PR_STM_DIR    ?= ~/projs/pr-stm
#/extra/cuda-8.0

NB_ACCOUNTS ?= 2621440

INCLUDES := -I $(CUDA_UTIL_DIR)/include -I $(PR_STM_DIR)/include
DEFINES  := -DarraySize=$(NB_ACCOUNTS)
LIBS     := -L$(CUDA_DIR)/lib64

CFLAGS   := -g -O0
CXXFLAGS := -g -O0
NVFLAGS  := -c -g -O0
LDFLAGS  := -lcuda -lcudart

SRCS_C   := $(shell ls -rt -d -1 *.c *.cpp)
SRCS_CU  := $(shell ls -rt -d -1 *.cu)

APP      := ./bank
OBJS_C   := $(addsuffix .o, $(basename $(SRCS_C)))
OBJS_CU  := $(addsuffix .o, $(basename $(SRCS_CU)))

CFLAGS   += $(INCLUDES) $(DEFINES)
CXXFLAGS += $(INCLUDES) $(DEFINES)
NVFLAGS  += $(INCLUDES) $(DEFINES)
LDFLAGS  += $(LIBS)

NVCC     := nvcc
CC       := gcc
CXX      := g++

### RULES
all: $(APP)

%.o: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

$(APP): $(OBJS_C) $(OBJS_CU)
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(APP) $(OBJS_C) $(OBJS_CU)
