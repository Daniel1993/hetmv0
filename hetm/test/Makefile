### CONFIG
CUDA_UTIL_DIR ?= ~/projs/cuda-utils
HETM_ROOT     ?= ..
PR_STM_DIR    ?= ~/projs/pr-stm
TINY_ROOT     ?= ../../tinyMOD
TSX_ROOT      ?= ../../tsxMOD
HTM_SGL       ?= ~/projs/htm_alg
CUDA_PATH     ?= /usr/local/cuda

PROFILE ?= 0

INCLUDES := \
	-I ../include -I $(CUDA_UTIL_DIR)/include \
	-I $(HETM_ROOT)/include \
	-I $(TINY_ROOT)/include \
	-I $(TSX_ROOT)/include \
	-I $(PR_STM_DIR)/include
DEFINES  :=
LIBS     := \
	-L $(HETM_ROOT) -lhetm \
	-L $(CUDA_UTIL_DIR) -lcuda-util \
	-L $(TINY_ROOT)/lib -lstm \
	-L $(HTM_SGL)/bin -lhtm_sgl -L $(TSX_ROOT) -ltsxMOD \
	-L $(CUDA_PATH)/lib64 -lcudart -lcuda

CFLAGS   := -g -c -O0 -std=gnu99
CXXFLAGS := -g -c -O0 -std=c++11
NVFLAGS  := -g -G -c -std=c++11
LDFLAGS  := -lcppunit $(LIBS)
NVCC     := nvcc

TEST_SRC := $(shell ls -rt -d -1 ./Test*.c ./Test*.cpp ./Test*.cu 2>/dev/null)
RUNNER   := ./runner.cpp
RUNNER_B := ./Runner
TEST_SRC += $(RUNNER)
OBJS     := $(addsuffix .o, $(basename $(TEST_SRC)))
OBJS     += pr-stm-impl.o # needed

ifeq ($(PROFILE),1)
DEFINES += -DUSE_NVTX
endif

CFLAGS   += $(INCLUDES) $(DEFINES)
CXXFLAGS += $(INCLUDES) $(DEFINES)
NVFLAGS  += $(INCLUDES) $(DEFINES)
LDFLAGS  += $(LIBS)

### RULES

all: $(RUNNER_B)

%.o: %.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

$(RUNNER_B): $(OBJS)
	$(NVCC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(RUNNER_B) $(OBJS)
