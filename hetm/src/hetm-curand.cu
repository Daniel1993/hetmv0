#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include "helper_cuda.h"
#include "helper_timer.h"
#include <time.h>

#include "bitmap.h"
#include "memman.h"
#include "hetm.cuh"

// --------------------
__constant__ unsigned PR_seed = 0xA1792F2B; // TODO: set seed
// --------------------

__global__ void HeTM_setupCurand(void *args)
{
	// curandState *state = (curandState*)args;
	// int id = threadIdx.x + blockDim.x*blockIdx.x;
	// curand_init(PR_seed, id, 0, &state[id]);
	long *state = (long*)args;
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	state[id] = id * PR_seed;
	RAND_R_FNC(state[id]);
}

__device__ int dev_memman_access_addr_gran(void *bmap, void *base, void *addr, size_t gran, size_t bits)
{
  char *bytes = (char*) bmap;
  uintptr_t addr_ptr = (uintptr_t)addr; // TODO: address translation
  uintptr_t base_ptr = (uintptr_t)base;
  uintptr_t delta    = addr_ptr - base_ptr;
  int chunk_pos      = delta >> bits;
  // printf("chunk_pos=%i bytes[chunk_pos]=%i\n", chunk_pos, bytes[chunk_pos]);
  if (bytes[chunk_pos] == 0) {
    ByteM_SET_POS(chunk_pos, bytes);
  }
  return 0;
}

__device__ int dev_memman_access_addr(void *bmap, void *addr)
{
  memman_bmap_s *key_bmap = (memman_bmap_s*)bmap;
  char *bytes = key_bmap->ptr;
  dev_memman_access_addr_gran(bytes, key_bmap->base, addr, key_bmap->gran, key_bmap->bits);
  return 0;
}
