CURR_DIR      ?= ~/projs/HeTM_V0/hetm
CUDA_PATH     ?= /usr/local/cuda
TINY_ROOT     ?= $(CURR_DIR)/../tinyMOD
TSX_ROOT      ?= $(CURR_DIR)/../tsxMOD
HTM_SGL       ?= ~/projs/htm_alg

CUDA_UTIL_DIR ?= ~/projs/cuda-utils
HTM_ARCH_DEPS ?= ~/projs/arch_dep
PR_STM_DIR    ?= ~/projs/pr-stm

CFG           ?= prod

INCLUDES := -I $(CUDA_UTIL_DIR)/include \
	-I $(PR_STM_DIR)/include \
	-I $(TINY_ROOT)/include \
	-I $(TINY_ROOT)/src \
	-I $(TSX_ROOT)/include \
	-I $(CUDA_PATH)/include \
	-I $(CURR_DIR)/include \
#
DEFINES  :=
#
LIBS     := \
	-L $(CUDA_PATH)/lib64 -lcudart -lcuda -lcuda-util \
	-L $(TINY_ROOT)/lib -lstm -lpthread \
#
LIB  := libhetm.a
NVCC := nvcc
CC   := gcc
CXX  := g++
AR   := ar rcs

# Parameters for HeTM
include $(CURR_DIR)/Makefile.defines

### Add this if debug needed (GPU run much slower)
DEBUG_FLAGS  := -g
NV_DEB_FLAGS := -G -g

DEBUG_PRINT_FLAGS  := -DHETM_DEB=1
NV_DEB_PRINT_FLAGS := -DHETM_DEB=1

# this is needed for the barrier implementation
INCLUDES  += 	-I $(HTM_ARCH_DEPS)/include
ifeq ($(USE_TSX_IMPL),1)
INCLUDES  += -I $(HTM_SGL)/include
LIBS      += -L $(HTM_SGL)/bin -lhtm_sgl -L $(TSX_ROOT) -ltsxMOD
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
ifeq ($(CFG),deb_cpu_only)
CFLAGS   += $(DEBUG_FLAGS)
CXXFLAGS += $(DEBUG_FLAGS)
NVFLAGS  += $(DEBUG_FLAGS)
endif
ifeq ($(PRINT_DEB),1)
CFLAGS   += $(DEBUG_PRINT_FLAGS)
CXXFLAGS += $(DEBUG_PRINT_FLAGS)
NVFLAGS  += $(NV_DEB_PRINT_FLAGS)
endif

SRC      := $(CURR_DIR)/src
OBJS     := \
	$(SRC)/hetm.o \
	$(SRC)/hetm-threading.o \
	$(SRC)/hetm-threading-cpu.o \
	$(SRC)/hetm-threading-gpu.o \
	$(SRC)/hetm-memory.o \
	$(SRC)/hetm-log.o \
	$(SRC)/hetm-curand.o \
	$(SRC)/hetm-cmp-kernels.o \
	$(SRC)/ticket-barrier.o \
	$(SRC)/hetm-producer-consumer.o

.PHONY:	all clean

all: $(LIB)
	# Done

$(LIB): $(OBJS)
	@echo "Linking..."
	$(NVCC) $(NVFLAGS) -O2 -o $(SRC)/hetm-threading-gpu.o $(SRC)/hetm-threading-gpu.cu
	### TODO: any optimization level blocks for VERS --> TODO: do optimization for BMAP
	$(NVCC) $(NVFLAGS) -O2 -o $(SRC)/hetm-threading-cpu.o $(SRC)/hetm-threading-cpu.cu
	$(CXX) $(CXXFLAGS) -O2 -o $(SRC)/hetm-producer-consumer.o $(SRC)/hetm-producer-consumer.cpp
	$(AR) $@ $(OBJS)

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

clean:
	rm -f $(LIB) *.o $(SRC)/*.o
