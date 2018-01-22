#include <pthread.h>

#include "tx_queue.cuh"


/****************************************************************************
 *	GLOBALS
 ****************************************************************************/

 //

/****************************************************************************
 *	KERNELS
 ****************************************************************************/

 __device__ unsigned int generate_random(curandState *state,	//random function in GPU
	int n)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int x = 0;
	/* Copy state to local memory for efficiency */
	curandState localState = state[id];

	for (int i = 0; i < n; i++) {
		x = curand(&localState);

		if (x >0) {
			break;
		}
	}
	//printf("threadID = %d, Random result: %d\n",id,x);

	state[id] = localState;
	return x;
}

// to setup random seed
__global__ void queue_generate_kernel(curandState *state, unsigned int *queue, int lower, int upper)
{
	unsigned int count = upper - lower;
	unsigned int gen = 0;
	int loop = 1;
	int id = 0; id = threadIdx.x + blockDim.x*blockIdx.x + blockIdx.y*gridDim.x*blockDim.x;
	unsigned int * local = &queue[id*BANK_NB_TRANSFERS];

	#pragma unroll
	for (int i = 0; i < BANK_NB_TRANSFERS; i++)
	{
		loop = 1;
		while(loop) {
			loop = 0;

			//Generate a new entry
			gen = generate_random(state, 100) % count;
			gen += lower;

			//Check for duplicates
			for(int j = 0; j<i ; j++) {
				if (gen == local[j]) {
					loop=1;
					break;
				}
			}

		}
		local[i] = gen;
	}
}

 /****************************************************************************
 *	FUNCTIONS
 ****************************************************************************/

extern "C"
cudaError_t queue_Init(queue_t *q, cuda_t *cd, int shared_rate, int q_size, curandState *devStates)
{
	cudaError_t cudaStatus =  cudaErrorMemoryAllocation;
	 unsigned int * temp_q;
	int err=0, lower, upper;
	int per_thread = 1 + (q_size)/(cd->blockNum * cd->threadNum);
	q_size = cd->blockNum * cd->threadNum * per_thread;

	q->shared_rate = shared_rate;

	do {
		//Host malloc for cpu queue
		q->cpu_queue.size = q_size*BANK_NB_TRANSFERS;
		q->cpu_queue.current = 0;
		pthread_mutex_init(&q->cpu_queue.lock, NULL);
		cudaStatus = cudaHostAlloc(&q->cpu_queue.values , q_size*BANK_NB_TRANSFERS*sizeof(unsigned int), cudaHostAllocPortable);
		if(cudaStatus != cudaSuccess) {
			printf("CPU Queue allocation failed.\n");
			return cudaStatus;
		}

		//Host malloc for gpu queue
		q->gpu_queue.size = q_size*BANK_NB_TRANSFERS;
		q->gpu_queue.current = 0;
		pthread_mutex_init(&q->gpu_queue.lock, NULL);
		cudaStatus = cudaHostAlloc(&q->gpu_queue.values , q_size*BANK_NB_TRANSFERS*sizeof(unsigned int), cudaHostAllocPortable);
		if(cudaStatus != cudaSuccess) {
			printf("GPU Queue allocation failed.\n");
			return cudaStatus;
		}

		//Host malloc for shared queue
		q->shared_queue.size = q_size*BANK_NB_TRANSFERS;
		q->shared_queue.current = 0;
		pthread_mutex_init(&q->shared_queue.lock, NULL);
		cudaStatus = cudaHostAlloc(&q->shared_queue.values , q_size*BANK_NB_TRANSFERS*sizeof(unsigned int), cudaHostAllocPortable);
		if(cudaStatus != cudaSuccess) {
			printf("Shared Queue allocation failed.\n");
			return cudaStatus;
		}

		cudaStatus = cudaMalloc(&temp_q,q_size*BANK_NB_TRANSFERS*sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			printf("cudaMalloc failed for queue_init!");
			continue;
		}

		dim3 blocksQueue(cd->blockNum,per_thread);
		/*
		 * GEN CPU QUEUE
		 */
		//  Kernel Gen
		upper = cd->size >> 2; lower = 0;
		queue_generate_kernel <<< blocksQueue, cd->threadNum >>>(devStates,temp_q,lower,upper);
		cudaStatus = cudaThreadSynchronize();	//synchronize threads
		if (cudaStatus != cudaSuccess) {
			printf("queue_generate_kernel failed!");
			//goto Error;
			continue;
		}
		//Copy to host
		cudaStatus = cudaMemcpy(q->cpu_queue.values , temp_q, q_size*BANK_NB_TRANSFERS*sizeof(unsigned int), cudaMemcpyDeviceToHost);	//copy new vector from CPU to GPU
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy failed for cpu_queue!");
			continue;
		}

		/*
		 * GEN SHARED QUEUE
		 */
		//  Kernel Gen
		upper = cd->size >> 1; lower = cd->size >> 2;
		queue_generate_kernel <<< blocksQueue, cd->threadNum >>>(devStates,temp_q,lower,upper);
		cudaStatus = cudaThreadSynchronize();	//synchronize threads
		if (cudaStatus != cudaSuccess) {
			printf("queue_generate_kernel failed!");
			//goto Error;
			continue;
		}
		cudaStatus = cudaMemcpy(q->shared_queue.values , temp_q, q_size*BANK_NB_TRANSFERS*sizeof(unsigned int), cudaMemcpyDeviceToHost);	//copy new vector from CPU to GPU
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy failed for shared_queue!");
			continue;
		}

		/*
		 * GEN GPU QUEUE
		 */
		//  Kernel Gen
		upper = cd->size; lower = cd->size >> 1;
		queue_generate_kernel <<< blocksQueue, cd->threadNum >>>(devStates,temp_q,lower,upper);
		cudaStatus = cudaThreadSynchronize();	//synchronize threads
		if (cudaStatus != cudaSuccess) {
			printf("queue_generate_kernel failed!");
			//goto Error;
			continue;
		}
		cudaStatus = cudaMemcpy(q->gpu_queue.values , temp_q, q_size*BANK_NB_TRANSFERS*sizeof(unsigned int), cudaMemcpyDeviceToHost);	//copy new vector from CPU to GPU
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy failed for gpu_queue!");
			continue;
		}

	} while(err);

	cudaFree(temp_q);

	return cudaStatus;
}

extern "C"
unsigned int* readQueue(queue_t *q, int size, int queue_id)
{
	int percent = rand() % 100;

	if ( percent < q->shared_rate  ) {
		if(size > q->shared_queue.size)
			return NULL;

		pthread_mutex_lock(&q->shared_queue.lock);
		int count = q->shared_queue.current;

		if ( count + size > q->shared_queue.size)
			count = q->shared_queue.current = 0;
		else {
			q->shared_queue.current += size;
		}
		pthread_mutex_unlock(&q->shared_queue.lock);

		return &q->shared_queue.values[count];
	}

	if ( queue_id == 0 ) {
		if(size > q->cpu_queue.size)
			return NULL;

		pthread_mutex_lock(&q->cpu_queue.lock);
		int count = q->cpu_queue.current;

		if ( count + size > q->cpu_queue.size) {
			count = q->cpu_queue.current = 0;
		} else {
			q->cpu_queue.current += size;
		}
		pthread_mutex_unlock(&q->cpu_queue.lock);

		return &q->cpu_queue.values[count];
	}

	if ( queue_id == 1 ) {
		if(size > q->gpu_queue.size)
			return NULL;

		pthread_mutex_lock(&q->gpu_queue.lock);
		int count = q->gpu_queue.current;

		if ( count + size > q->gpu_queue.size)
			count = q->gpu_queue.current = 0;
		else {
			q->gpu_queue.current += size;
		}
		pthread_mutex_unlock(&q->gpu_queue.lock);

		return &q->gpu_queue.values[count];
	}

	return NULL;
}

extern "C"
void queue_Delete(queue_t *q)
{
	if(q!=NULL) {
		cudaFreeHost(q->shared_queue.values);
		cudaFreeHost(q->cpu_queue.values);
		cudaFreeHost(q->gpu_queue.values);
		free(q);
	}
}
