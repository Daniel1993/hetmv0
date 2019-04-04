#include "bank.h"
#include "hetm-cmp-kernels.cuh"
#include "bank_aux.h"
#include "CheckAllFlags.h"

#include "zipf_dist.h"

thread_data_t parsedData;
int isInterBatch = 0;
size_t accountsSize;
size_t sizePool;
void* gpuMempool;

size_t currMaxCPUoutputBufferSize, currCPUoutputBufferPtr = 0;
size_t maxGPUoutputBufferSize;
size_t size_of_GPU_input_buffer, size_of_CPU_input_buffer;
int lockOutputBuffer = 0;

FILE *GPU_input_file = NULL;
FILE *CPU_input_file = NULL;

extern int *GPUoutputBuffer;
extern int *CPUoutputBuffer;
extern int *GPUInputBuffer;
extern int *CPUInputBuffer;

void GPUbufferReadFromFile_NO_CONFLS() {
  int buffer_last = size_of_GPU_input_buffer/sizeof(int);
  GPU_input_file = fopen(parsedData.GPUInputFile, "r");

  memman_select("GPU_input_buffer_good");
  int *cpu_ptr = (int*)memman_get_cpu(NULL);

  // unsigned rnd = 12345723; //RAND_R_FNC(input_seed);
  unsigned rnd; // = (*zipf_dist)(generator);
  for (int i = 0; i < buffer_last; ++i) {
    if (fscanf(GPU_input_file, "%i\n", &rnd) == EOF) {
      printf("ERROR GPU reached end-of-file at %i / %i\n", i, buffer_last);
    }
    rnd = (rnd % parsedData.CONFL_SPACE);
    int mod = rnd % 3;
    cpu_ptr[i] = (rnd - mod) + 2; // gives always 2 (mod 3) //2*i;//
  }

  memman_select("GPU_input_buffer_bad");
  cpu_ptr = (int*)memman_get_cpu(NULL);
  for (int i = 0; i < buffer_last; ++i) {
    if (fscanf(GPU_input_file, "%i\n", &rnd) == EOF) {
      printf("ERROR GPU reached end-of-file at %i / %i\n", i, buffer_last);
    }
    rnd = (rnd % parsedData.CONFL_SPACE);
    int mod = rnd % 3;
    cpu_ptr[i] = (rnd - mod) + 1; // gives always 0 (mod 3) //2*i+1;//
  }
}

void CPUbufferReadFromFile_NO_CONFLS() {
  int good_buffers_last = size_of_CPU_input_buffer/sizeof(int);
	int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int);
	CPU_input_file = fopen(parsedData.CPUInputFile, "r");

  for (int i = 0; i < good_buffers_last; ++i) {
		unsigned rnd;
		if (fscanf(CPU_input_file, "%i\n", &rnd) == EOF) {
			printf("ERROR CPU reached end-of-file at %i / %i\n", i, good_buffers_last);
		}
    rnd = (rnd % parsedData.CONFL_SPACE);
		int mod = rnd % 3;
		CPUInputBuffer[i] = (rnd - mod); // 2*i+1;//
	}
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		unsigned rnd;
		if (fscanf(CPU_input_file, "%i\n", &rnd) == EOF) {
			printf("ERROR CPU reached end-of-file at %i / %i\n", i, bad_buffers_last);
		}
    rnd = (rnd % parsedData.CONFL_SPACE);
		int mod = rnd % 3;
		CPUInputBuffer[i] = (rnd - mod) + 1; //2*i;//
	}
}

void GPUbufferReadFromFile_CONFLS() {
  int buffer_last = size_of_GPU_input_buffer/sizeof(int);
	GPU_input_file = fopen(parsedData.GPUInputFile, "r");

	// if (zipf_dist == NULL) {
	// 	generator.seed(input_seed);
	// 	zipf_dist = new zipf_distribution<int, double>(parsedData.nb_accounts * parsedData.num_ways);
	// }

	memman_select("GPU_input_buffer_good");
	int *cpu_ptr = (int*)memman_get_cpu(NULL);
  unsigned rnd; // = (*zipf_dist)(generator);
  for (int i = 0; i < buffer_last; ++i) {
    if (fscanf(GPU_input_file, "%i\n", &rnd) == EOF) {
      printf("ERROR GPU reached end-of-file at %i / %i\n", i, buffer_last);
    }
    int mod = rnd % 2;
    cpu_ptr[i] = (rnd - mod);//2*i;//
  }

  memman_select("GPU_input_buffer_bad");
  cpu_ptr = (int*)memman_get_cpu(NULL);

  for (int i = 0; i < buffer_last; ++i) {
    if (fscanf(GPU_input_file, "%i\n", &rnd) == EOF) {
      printf("ERROR GPU reached end-of-file at %i / %i\n", i, buffer_last);
    }
    int mod = rnd % 2;
    cpu_ptr[i] = (rnd - mod) + 1; // gets input from the CPU //2*i+1;//
  }
}

void GPUbufferReadFromFile_UNIF_RAND() {
  int buffer_last = size_of_GPU_input_buffer/sizeof(int);
	GPU_input_file = fopen(parsedData.GPUInputFile, "r");

	// if (zipf_dist == NULL) {
	// 	generator.seed(input_seed);
	// 	zipf_dist = new zipf_distribution<int, double>(parsedData.nb_accounts * parsedData.num_ways);
	// }

	memman_select("GPU_input_buffer_good");
	int *cpu_ptr = (int*)memman_get_cpu(NULL);
  unsigned rnd; // = (*zipf_dist)(generator);
	for (int i = 0; i < buffer_last; ++i) {
		rnd = rand();
		int mod = rnd % 2;
		cpu_ptr[i] = (rnd - mod);//2*i;//
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	for (int i = 0; i < buffer_last; ++i) {
		rnd = rand();
		int mod = rnd % 2;
		cpu_ptr[i] = (rnd - mod) + 1; // gets input from the CPU //2*i+1;//
	}
}

void CPUbufferReadFromFile_CONFLS() {
  int good_buffers_last = size_of_CPU_input_buffer/sizeof(int);
	int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int);
	CPU_input_file = fopen(parsedData.CPUInputFile, "r");

  for (int i = 0; i < good_buffers_last; ++i) {
		unsigned rnd;
		if (fscanf(CPU_input_file, "%i\n", &rnd) == EOF) {
			printf("ERROR CPU reached end-of-file at %i / %i\n", i, good_buffers_last);
		}
		int mod = rnd % 2;
		CPUInputBuffer[i] = (rnd - mod) + 1; //2*i+1;//
	}
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		unsigned rnd;
		if (fscanf(CPU_input_file, "%i\n", &rnd) == EOF) {
			printf("ERROR CPU reached end-of-file at %i / %i\n", i, bad_buffers_last);
		}
		int mod = rnd % 2;
		CPUInputBuffer[i] = (rnd - mod); // gets input from the GPU //2*i;//
	}
}

void CPUbufferReadFromFile_UNIF_RAND() {
  int good_buffers_last = size_of_CPU_input_buffer/sizeof(int);
	int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int);
	CPU_input_file = fopen(parsedData.CPUInputFile, "r");

  unsigned rnd;
	for (int i = 0; i < good_buffers_last; ++i) {
		rnd = rand();
		int mod = rnd % 2;
		CPUInputBuffer[i] = (rnd - mod) + 1; //2*i+1;//
	}
	for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
		rnd = rand();
		int mod = rnd % 2;
		CPUInputBuffer[i] = (rnd - mod); // gets input from the GPU //2*i;//
	}
}

void GPUbuffer_NO_CONFLS() {
  int buffer_last = size_of_GPU_input_buffer/sizeof(int);

	// if (zipf_dist == NULL) {
	// 	generator.seed(input_seed);
	// 	zipf_dist = new zipf_distribution<int, double>(parsedData.nb_accounts * parsedData.num_ways);
	// }

	memman_select("GPU_input_buffer_good");
	int *cpu_ptr = (int*)memman_get_cpu(NULL);
  unsigned rnd; // = (*zipf_dist)(generator);
	for (int i = 0; i < buffer_last; ++i) {
		rnd = (rand() % parsedData.CONFL_SPACE) + parsedData.CONFL_SPACE;
		int mod = rnd % 2;
		cpu_ptr[i] = (rnd - mod);//2*i;//
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	for (int i = 0; i < buffer_last; ++i) {
    if (i < parsedData.NB_CONFL_GPU_BUFFER) {
      rnd = rand() % parsedData.CONFL_SPACE;
    } else {
      rnd = (rand() % parsedData.CONFL_SPACE) + parsedData.CONFL_SPACE;
    }
		int mod = rnd % 2;
		cpu_ptr[i] = (rnd - mod) + 1; // gets input from the CPU //2*i+1;//
	}
}

void CPUbuffer_NO_CONFLS() {
  int good_buffers_last = size_of_CPU_input_buffer/sizeof(int);
  int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int);
  int sizePerThread;

  if (parsedData.NB_CONFL_CPU_BUFFER < 1) {
    sizePerThread = size_of_CPU_input_buffer;
  } else {
    sizePerThread = size_of_CPU_input_buffer / parsedData.nb_threads / parsedData.NB_CONFL_CPU_BUFFER;
  }
  unsigned rnd;
  for (int i = 0; i < good_buffers_last; ++i) {
    rnd = rand() % parsedData.CONFL_SPACE;
    int mod = rnd % 2;
    CPUInputBuffer[i] = (rnd - mod) + 1; //2*i+1;//
  }
  for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
    if (((i - good_buffers_last) % sizePerThread) == 0) {
      rnd = (rand() % parsedData.CONFL_SPACE) + parsedData.CONFL_SPACE;
    } else {
      rnd = rand() % parsedData.CONFL_SPACE;
    }
    int mod = rnd % 2;
    CPUInputBuffer[i] = (rnd - mod); // gets input from the GPU //2*i;//
  }
}

void GPUbuffer_UNIF_2()
{
  int buffer_last = size_of_GPU_input_buffer/sizeof(int);

	// if (zipf_dist == NULL) {
	// 	generator.seed(input_seed);
	// 	zipf_dist = new zipf_distribution<int, double>(parsedData.nb_accounts * parsedData.num_ways);
	// }

	memman_select("GPU_input_buffer_good");
	int *cpu_ptr = (int*)memman_get_cpu(NULL);
  unsigned rnd; // = (*zipf_dist)(generator);
	for (int i = 0; i < buffer_last; ++i) {
		rnd = (rand() % parsedData.CONFL_SPACE) + parsedData.CONFL_SPACE;
		cpu_ptr[i] = rnd;
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);

	for (int i = 0; i < buffer_last; ++i) {
    rnd = rand() % parsedData.CONFL_SPACE; // different from NO_CONFL --> fills the buffer
		cpu_ptr[i] = rnd;
	}
}

void CPUbuffer_UNIF_2()
{
  int good_buffers_last = size_of_CPU_input_buffer/sizeof(int);
  int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int);

  unsigned rnd;
  for (int i = 0; i < good_buffers_last; ++i) {
    rnd = rand() % parsedData.CONFL_SPACE;
    CPUInputBuffer[i] = rnd;
  }
  for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
      rnd = (rand() % parsedData.CONFL_SPACE) + parsedData.CONFL_SPACE;
    CPUInputBuffer[i] = rnd; // gets input from the GPU //2*i;//
  }
}

void GPUbuffer_ZIPF_2()
{
  int buffer_last = size_of_GPU_input_buffer/sizeof(int);

  // 1st item is generated 10% of the times
  unsigned maxGen = parsedData.CONFL_SPACE * parsedData.num_ways;
  zipf_setup(maxGen, 0.8);

	memman_select("GPU_input_buffer_good");
	int *cpu_ptr = (int*)memman_get_cpu(NULL);
  unsigned rnd, zipfRnd; // = (*zipf_dist)(generator);

  // int done = 0;

	for (int i = 0; i < buffer_last; ++i) {
    zipfRnd = zipf_gen();
    rnd = ((zipfRnd / parsedData.CONFL_SPACE) * 2) * parsedData.CONFL_SPACE
      + (zipfRnd % parsedData.CONFL_SPACE);
		cpu_ptr[i] = rnd;
	}

	memman_select("GPU_input_buffer_bad");
	cpu_ptr = (int*)memman_get_cpu(NULL);
  // done = 0;
	for (int i = 0; i < buffer_last; ++i) {
    zipfRnd = zipf_gen();
		cpu_ptr[i] = i % 100;// maxGen - zipfRnd;
    // rnd = ((zipfRnd / parsedData.CONFL_SPACE) * 2 + 1) * parsedData.CONFL_SPACE
    //   + (zipfRnd % parsedData.CONFL_SPACE);
		// cpu_ptr[i] = rnd;
	}
}

void CPUbuffer_ZIPF_2()
{
  int good_buffers_last = size_of_CPU_input_buffer/sizeof(int);
  int bad_buffers_last = 2*size_of_CPU_input_buffer/sizeof(int);

  unsigned maxGen = parsedData.CONFL_SPACE * parsedData.num_ways;

  // 1st item is generated 10% of the times
  zipf_setup(maxGen, 0.8);

  unsigned rnd, zipfRnd;
  for (int i = 0; i < good_buffers_last; ++i) {
    zipfRnd = zipf_gen();
    rnd = ((zipfRnd / parsedData.CONFL_SPACE) * 2 + 1) * parsedData.CONFL_SPACE
      + (zipfRnd % parsedData.CONFL_SPACE);
    CPUInputBuffer[i] = rnd;
  }
  // done = 0;
  for (int i = good_buffers_last; i < bad_buffers_last; ++i) {
    zipfRnd = zipf_gen();
    CPUInputBuffer[i] = i % 100; //maxGen - zipfRnd;
    // rnd = ((zipfRnd / parsedData.CONFL_SPACE) * 2) * parsedData.CONFL_SPACE
    //   + (zipfRnd % parsedData.CONFL_SPACE);
    // CPUInputBuffer[i] = rnd; // gets input from the GPU //2*i;//
  }
}
