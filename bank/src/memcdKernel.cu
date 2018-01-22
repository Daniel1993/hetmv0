#include "cuda_wrapper.cuh"

// inits the remaining stuff for memcd
void jobWithCuda_initMemcd(cuda_t *cd, int ways, int wr, int sr)
{
  queue_t *queue;

  queue = (queue_t*)malloc( sizeof(queue_t) );

  memman_alloc_gpu("HeTM_memcd_queue", cd->threadNum*cd->blockNum*sizeof(unsigned int), NULL, 0);
  memman_zero_gpu(NULL);
  queue->gpu_queue.values = (unsigned int*)memman_get_gpu(NULL);

  cd->set_percent = wr;
	cd->gpu_queue = queue->gpu_queue.values; // TODO: ???
	cd->clock = 0;

  // TODO: this was in cuda_configInit
  cd->num_ways = ways > 0 ? ways : NUMBER_WAYS;

	CUDA_CHECK_ERROR(queue_Init(queue, cd, sr, QUEUE_SIZE, (curandState*)cd->devStates), "");

  cd->q = queue;
}
