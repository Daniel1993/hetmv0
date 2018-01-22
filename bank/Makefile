CURR_DIR      ?= ~/projs/HeTM_V0/bank
TINY_ROOT     ?= $(CURR_DIR)/../tinyMOD
HTM_SGL       ?= ~/projs/htm_alg
PR_STM_DIR    ?= ~/projs/pr-stm
CUDA_PATH     ?= /usr/local/cuda

CUDA_UTIL_DIR ?= ~/projs/cuda-utils
HTM_ARCH_DEPS ?= ~/projs/arch_dep

CFG           ?= prod
include $(TINY_ROOT)/Makefile.common

INCLUDES := -I $(CUDA_UTIL_DIR)/include \
	-I $(PR_STM_DIR)/include \
	-I $(CUDA_PATH)/include \
	-I $(CURR_DIR)/include \
	-I $(TINY_ROOT)/include \
	-I $(TINY_ROOT)/src
#
DEFINES  :=
#
LIBS     := -L $(CUDA_PATH)/lib64 -lcudart -lcuda \
	-L $(TINY_ROOT)/lib -lstm -lpthread \
	-L $(CUDA_UTIL_DIR) -lcuda-util
#
APP  = bank
BENCH= memcd
NVCC = nvcc
CC   = gcc
CXX  = g++

include $(CURR_DIR)/Makefile.defines

# the log only works with TM_STATISTICS3 on
DEFINES    += -DTM_STATISTICS3

### Add this if debug needed (GPU run much slower)
DEBUG_FLAGS  := -g -lineinfo -DHETM_DEB=1
NV_DEB_FLAGS := -G -g -lineinfo -DHETM_DEB=1

ifeq ($(USE_TSX_IMPL),1)
INCLUDES  += 	-I $(HTM_ARCH_DEPS)/include \
	-I $(HTM_SGL)/include
LIBS      += -L $(HTM_SGL)/bin -lhtm_sgl
endif

CFLAGS     := -c $(DEFINES) $(INCLUDES)
CXXFLAGS   := -c $(DEFINES) $(INCLUDES) -std=c++11
NVFLAGS    := -c $(DEFINES) $(INCLUDES) -std=c++11 \
	--default-stream per-thread \
	-arch sm_30
LDFLAGS    := $(LIBS) \
--default-stream per-thread \
-arch sm_30

ifeq ($(USE_TSX_IMPL),1)
TSX_FLAGS += -DUSE_TSX_IMPL
CFLAGS    += $(TSX_FLAGS) -mrtm
CXXFLAGS  += $(TSX_FLAGS) -mrtm
NVFLAGS   += $(TSX_FLAGS) --compiler-options='-mrtm'
endif

ifeq ($(CFG),debug)
CFLAGS   += $(DEBUG_FLAGS)
CXXFLAGS += $(DEBUG_FLAGS)
NVFLAGS  += $(NV_DEB_FLAGS)
endif

SRC        := $(CURR_DIR)/src
BANK_OBJS  := \
	$(SRC)/bank.o \
	$(SRC)/bankKernel.o \
	$(SRC)/cmp_kernels.o \
	$(SRC)/tsx_impl.o \
	$(SRC)/shared.o \
	$(SRC)/setupKernels.o \
	$(SRC)/hetm.o \
	$(SRC)/kernel.o

MEMCD_OBJS  := \
	$(SRC)/memcd.o \
	$(SRC)/bankKernel.o \
	$(SRC)/cmp_kernels.o \
	$(SRC)/tsx_impl.o \
	$(SRC)/shared.o \
	$(SRC)/hetm.o \
	$(SRC)/setupKernels.o \
	$(SRC)/tx_queue.o \
	$(SRC)/memcdKernel.o \
	$(SRC)/kernel.o

OBJS := $(BANK_OBJS)

ifeq ($(BENCH),MEMCD)
# use memcached instead
OBJS := $(MEMCD_OBJS)
endif

.PHONY:	all clean

all: $(APP)
	# bank bank.o tsx_impl.o kernel.o pr-stm.o cmp_kernels.o shared.o thread.o

$(APP): $(OBJS)
	$(NVCC) $(LDFLAGS) -o $@ $(OBJS)

%.o:	%.c
	@echo ""
	$(CC) $(CFLAGS) -o $@ $<

%.o:	%.cpp
	@echo ""
	$(CXX) $(CXXFLAGS) -o $@ $<

%.o:	%.cu
	@echo ""
	$(NVCC) $(NVFLAGS) -o $@ $<

run:
	make all
	nvprof ./bank -n 4 -r 5000 -w 5000 -a 200000

ptx:
	$(NVCC) $(NVFLAGS) $(NV_DEB_FLAGS) --ptx cmp_kernels.cu

clean:
	rm -f $(APP) *.o $(SRC)/*.o
