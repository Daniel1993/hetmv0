CURR_DIR      ?= ~/projs/HeTM_V0/hetm

CUDA_PATH     ?= /usr/local/cuda
TINY_ROOT     ?= $(CURR_DIR)/../tinyMOD
TSX_ROOT      ?= $(CURR_DIR)/../tsxMOD
INPUT_HANDLER ?= $(CURR_DIR)/../shared

CUDA_UTIL_DIR ?= $(CURR_DIR)/../deps/cuda-utils
HTM_ARCH_DEPS ?= $(CURR_DIR)/../deps/arch_dep
HTM_SGL       ?= $(CURR_DIR)/../deps/htm_alg
PR_STM_DIR    ?= $(CURR_DIR)/../deps/pr-stm

CFG           ?= prod

INCLUDES := -I $(CUDA_UTIL_DIR)/include \
	-I $(PR_STM_DIR)/include \
	-I $(TINY_ROOT)/include \
	-I $(TINY_ROOT)/src \
	-I $(TSX_ROOT)/include \
	-I $(CUDA_PATH)/include \
	-I $(CURR_DIR)/include \
	-I $(INPUT_HANDLER)/include \
#
DEFINES  :=
#
LIBS     := \
	-L $(INPUT_HANDLER) -linput_handler \
	-L $(CUDA_PATH)/lib64 -lcudart -lcuda -lcuda-util -lnuma \
	-L $(TINY_ROOT)/lib -lstm -lpthread \
#
LIB  := libhetm.a
# NVCC := nvcc
NVCC ?= $(CUDA_PATH)/bin/nvcc
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
	-arch sm_60
LDFLAGS    := $(LIBS) \
--default-stream per-thread \
-arch sm_60

ifeq ($(USE_TSX_IMPL),1)
TSX_FLAGS += -DUSE_TSX_IMPL
endif
CFLAGS    += $(TSX_FLAGS) -march=native -mrtm
CXXFLAGS  += $(TSX_FLAGS) -march=native -mrtm
NVFLAGS   += $(TSX_FLAGS) --compiler-options='-march=native -mrtm'

ifeq ($(CFG),debug)
CFLAGS   += $(DEBUG_FLAGS)
CXXFLAGS += $(DEBUG_FLAGS)
NVFLAGS  += $(NV_DEB_FLAGS)
endif
ifneq ($(CFG),debug)
CFLAGS   += -DNDEBUG
CXXFLAGS += -DNDEBUG
NVFLAGS  += -DNDEBUG
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
	# $(NVCC) $(NVFLAGS) -O2 -o $(SRC)/hetm-threading-gpu.o $(SRC)/hetm-threading-gpu.cu
	### TODO: any optimization level blocks for VERS --> TODO: do optimization for BMAP
	$(NVCC) $(NVFLAGS) -O2 -o $(SRC)/hetm-threading-cpu.o $(SRC)/hetm-threading-cpu.cu
	$(CXX) $(CXXFLAGS) -O2 -o $(SRC)/hetm-producer-consumer.o $(SRC)/hetm-producer-consumer.cpp
	$(CXX) $(CXXFLAGS) -O2 -o $(SRC)/hetm-log.o $(SRC)/hetm-log.cpp
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
