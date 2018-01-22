#define _GNU_SOURCE
#include "shared.h"
#include "bank.h"
#include <sched.h>

// GLOBALS
long long int global_fix;

static char filename[512];

void bindThread(long threadId)
{
  cpu_set_t my_set;
  CPU_ZERO(&my_set);
  int offset = threadId;
  int id = threadId;

  // intel14
  offset = id % 56;
  if (id >= 14 && id < 28)
    offset += 14;
  if (id >= 28 && id < 42)
    offset -= 14;

  CPU_SET(offset, &my_set);
  sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
}

void bank_parseArgs(int argc, char **argv, thread_data_t *data)
{
  data->filename = filename; // TODO: remove this
  int i, c;
#ifndef TM_COMPILER
  data->cm = NULL;
#endif /* ! TM_COMPILER */
  data->duration          = DEFAULT_DURATION;
  data->duration2         = DEFAULT_DURATION;
  data->nb_accounts       = DEFAULT_NB_ACCOUNTS;
  data->nb_threads        = DEFAULT_NB_THREADS;
  data->nb_threadsCPU     = DEFAULT_NB_THREADS;
  data->read_all          = DEFAULT_READ_ALL;
  data->read_threads      = DEFAULT_READ_THREADS;
  data->seed              = DEFAULT_SEED;
  data->write_all         = DEFAULT_WRITE_ALL;
  data->write_threads     = DEFAULT_WRITE_THREADS;
  data->iter              = DEFAULT_ITERATIONS;
  data->trfs              = DEFAULT_NB_TRFS;
  data->trans             = DEFAULT_TransEachThread;
  data->num_ways          = NUMBER_WAYS;
  data->shared_percent    = QUEUE_SHARED_VALUE;
  data->set_percent       = WRITE_PERCENT;
  struct option long_options[] = {
    // These options don't set a flag
    {"help",                no_argument,       NULL, 'h'},
    {"accounts",            required_argument, NULL, 'a'},
    {"contention-manager",  required_argument, NULL, 'c'},
    {"duration",            required_argument, NULL, 'd'},
    {"num-threads",         required_argument, NULL, 'n'},
    {"read-all-rate",       required_argument, NULL, 'r'},
    {"read-threads",        required_argument, NULL, 'R'},
    {"seed",                required_argument, NULL, 's'},
    {"write-all-rate",      required_argument, NULL, 'w'},
    {"write-threads",       required_argument, NULL, 'W'},
    {"num-iterations",      required_argument, NULL, 'i'},
    {"trfs",                required_argument, NULL, 't'},
    {"gpu-blocks",          required_argument, NULL, 'b'},
    {"gpu-threads",         required_argument, NULL, 'x'},
    {"tperthread",          required_argument, NULL, 'T'},
    {"output-file",         required_argument, NULL, 'f'},
    {"disjoint",            no_argument,       NULL, 'j'},
    {NULL, 0, NULL, 0}
  };

  while(1) {
    i = 0;
    c = getopt_long(argc, argv, "ha:c:d:n:r:R:s:w:W:i:t:T:f:jb:x:", long_options, &i);

    if(c == -1)
      break;

    if(c == 0 && long_options[i].flag == 0)
      c = long_options[i].val;

    switch(c) {
      case 0:
        /* Flag is automatically set */
        break;
      case 'h':
        printf("bank -- STM stress test\n"
          "\n"
          "Usage:\n"
          "  bank [options...]\n"
          "\n"
          "Options:\n"
          "  -h, --help\n"
          "        Print this message\n"
          "  -a, --accounts <int>\n"
          "        Number of accounts in the bank (default=" XSTR(DEFAULT_NB_ACCOUNTS) ")\n"
  #ifndef TM_COMPILER
          "  -c, --contention-manager <string>\n"
          "        Contention manager for resolving conflicts (default=suicide)\n"
  #endif /* ! TM_COMPILER */
          "  -d, --duration <int>\n"
          "        Test duration in milliseconds (0=infinite, default=" XSTR(DEFAULT_DURATION) ")\n"
          "  -n, --num-threads <int>\n"
          "        Number of threads (default=" XSTR(DEFAULT_NB_THREADS) ")\n"
          "  -r, --read-all-rate <int>\n"
          "        Percentage of read-all transactions (default=" XSTR(DEFAULT_READ_ALL) ")\n"
          "  -R, --read-threads <int>\n"
          "        Number of threads issuing only read-all transactions (default=" XSTR(DEFAULT_READ_THREADS) ")\n"
          "  -s, --seed <int>\n"
          "        RNG seed (0=time-based, default=" XSTR(DEFAULT_SEED) ")\n"
          "  -w, --write-all-rate <int>\n"
          "        Percentage of write-all transactions (default=" XSTR(DEFAULT_WRITE_ALL) ")\n"
          "  -W, --write-threads <int>\n"
          "        Number of threads issuing only write-all transactions (default=" XSTR(DEFAULT_WRITE_THREADS) ")\n"
          "  -i, --num-iterations <int>\n"
          "        Number of iterations to execute (default=" XSTR(DEFAULT_ITERATIONS) ")\n"
          "  -t, --num-transfers <int>\n"
          "        Number of accounts to transfer between in each transaction (default=" XSTR(DEFAULT_NB_TRFS) ")\n"
          "  -T, --num-txs-pergputhread <int>\n"
          "        Number of transactions per GPU thread (default=" XSTR(DEFAULT_TransEachThread) ")\n"
          "  -f, --file-name <string>\n"
          "        Output file name (default=" XSTR(DEFAULT_OUTPUT_FILE) ")\n"
          "  -b, --gpu-blocks <int>\n"
          "        Number of blocks for the GPU (default=" XSTR(DEFAULT_NB_GPU_BLKS) ")\n"
          "  -x, --gpu-threads <int>\n"
          "        Number of threads for the GPU (default=" XSTR(DEFAULT_NB_GPU_THRS) ")\n"
        );
        exit(0);
      case 'a':
        data->nb_accounts = atoi(optarg);
        break;
  #ifndef TM_COMPILER
      case 'c':
        data->cm = optarg;
        break;
  #endif /* ! TM_COMPILER */
      case 'd':
        data->duration = atoi(optarg);
        break;
      case 'n':
        data->nb_threads = atoi(optarg);
        break;
      case 'r':
        data->read_all = atoi(optarg);
        break;
      case 'R':
        data->read_threads = atoi(optarg);
        break;
      case 's':
        data->seed = atoi(optarg);
        break;
      case 'w':
        data->write_all = atoi(optarg);
        break;
      case 'W':
        data->write_threads = atoi(optarg);
        break;
      // case 'j':
      //   data->disjoint = 1;
      //   break;
      case 'i':
        data->iter = atoi(optarg);
        break;
      case 't':
        data->trfs = atoi(optarg);
        break;
      case 'T':
        data->trans = atoi(optarg);
        break;
      case 'b':
        data->GPUblockNum = atoi(optarg);
        break;
      case 'x':
        data->GPUthreadNum = atoi(optarg);
        break;
      case '?':
        printf("Use -h or --help for help\n");
        exit(0);
      case 'f':
        data->fn = optarg;
        break;
      default:
        exit(1);
    }
  }
}

void bank_printStats(thread_data_t *data)
{
  double duration;
  double duration2;
  int ret;
  int i;
#ifndef TM_COMPILER
  unsigned long aborts                 = 0;
  unsigned long aborts_1               = 0;
  unsigned long aborts_2               = 0;
  unsigned long aborts_locked_read     = 0;
  unsigned long aborts_locked_write    = 0;
  unsigned long aborts_validate_read   = 0;
  unsigned long aborts_validate_write  = 0;
  unsigned long aborts_validate_commit = 0;
  unsigned long aborts_invalid_memory  = 0;
  unsigned long aborts_killed          = 0;
  unsigned long locked_reads_ok        = 0;
  unsigned long locked_reads_failed    = 0;
  unsigned long max_retries            = 0;
  //stm_ab_stats_t ab_stats;
#endif /* ! TM_COMPILER */

  duration = TIMER_DIFF_SECONDS(data->start, data->end) * 1000;
  duration2 = TIMER_DIFF_SECONDS(data->start, data->final) * 1000;
  data->duration  = duration;
  data->duration2 = duration2;

  ret = bank_sum(data->bank);
#if NODEBUG == 0
  for (i = 0; i < data->nb_threadsCPU; i++) {
    /* TODO:
    printf("Thread %d\n", i);
    printf("  #transfer   : %lu\n", data->dthreads[i].nb_transfer);
    printf("  #read-all   : %lu\n", data->dthreads[i].nb_read_all);
    printf("  #write-all  : %lu\n", data->dthreads[i].nb_write_all);*/

    // TM_COMPILER
    aborts                   += data->dthreads[i].nb_aborts;
    aborts_1                 += data->dthreads[i].nb_aborts_1;
    aborts_2                 += data->dthreads[i].nb_aborts_2;
    aborts_locked_read       += data->dthreads[i].nb_aborts_locked_read;
    aborts_locked_write      += data->dthreads[i].nb_aborts_locked_write;
    aborts_validate_read     += data->dthreads[i].nb_aborts_validate_read;
    aborts_validate_write    += data->dthreads[i].nb_aborts_validate_write;
    aborts_validate_commit   += data->dthreads[i].nb_aborts_validate_commit;
    aborts_invalid_memory    += data->dthreads[i].nb_aborts_invalid_memory;
    aborts_killed            += data->dthreads[i].nb_aborts_killed;
    locked_reads_ok          += data->dthreads[i].locked_reads_ok;
    locked_reads_failed      += data->dthreads[i].locked_reads_failed;
    if (max_retries < data->dthreads[i].max_retries)
      max_retries = data->dthreads[i].max_retries;

    data->updates   += data->dthreads[i].nb_transfer;
    data->reads     += data->dthreads[i].nb_read_all;
    data->writes    += data->dthreads[i].nb_write_all;
  }
  data->updates -= global_fix;
  /* Sanity check */
  ret = bank_sum(data->bank);
  printf("Bank total    : %d (expected: 0)\n", ret);
  printf("Duration      : %f (ms)\n", data->duration);
  printf("#txs          : %lu (%f / s)\n", data->reads + data->writes + data->updates, (data->reads + data->writes + data->updates) * 1000.0 / data->duration2);
  printf("#read txs     : %lu (%f / s)\n", data->reads, data->reads * 1000.0 / data->duration2);
  printf("#write txs    : %lu (%f / s)\n", data->writes, data->writes * 1000.0 / data->duration2);
  printf("#update txs   : %lu (%f / s)\n", data->updates, data->updates * 1000.0 / data->duration2);
  printf("#aborts       : %lu (%f / s)\n", data->aborts, data->aborts * 1000.0 / data->duration2);
  printf("Duration      : %f (ms)\n", data->duration);
  printf("Real duration : %f (ms)\n", data->duration2);
#endif
}

void bank_statsFile(thread_data_t *data)
{
  DEBUG_PRINT("Sorting info.\n");

  /*Sort Arrays*/
  qsort(data->tot_duration,  data->iter,sizeof(int),compare_int);
  qsort(data->tot_duration2, data->iter,sizeof(int),compare_int);
  qsort(data->tot_commits,   data->iter,sizeof(int),compare_int);
  qsort(data->tot_aborts,    data->iter,sizeof(int),compare_int);
  qsort(data->tot_commits_gpu,    data->iter,sizeof(int),compare_int);
  qsort(data->tot_throughput,     data->iter,sizeof(double),compare_double);
  qsort(data->tot_throughput_gpu, data->iter,sizeof(double),compare_double);
  qsort(data->tot_aborts_gpu,     data->iter,sizeof(int),compare_int);
  qsort(data->tot_cuda,           data->iter,sizeof(int),compare_int);
  qsort(data->tot_comp,           data->iter,sizeof(int),compare_int);
  qsort(data->tot_tx,             data->iter,sizeof(int),compare_int);
  qsort(data->tot_loop,           data->iter,sizeof(int),compare_int);
  qsort(data->tot_loops,          data->iter,sizeof(int),compare_int);
  qsort(data->tot_trf2cpu,        data->iter,sizeof(int),compare_int);
  qsort(data->tot_trf2gpu,        data->iter,sizeof(int),compare_int);
  qsort(data->tot_trfcmp,         data->iter,sizeof(int),compare_int);

    /* Save to file */
    DEBUG_PRINT("Saving to file.\n");
    FILE *f = fopen(data->filename, "a");
    if (f == NULL)
  	{
      printf("Error opening file!\n");
      exit(1);
    }
    if(ftell(f)==0) {	//New File, print headers
    	fprintf(f, "sep=;\n");
    	fprintf(f,
        "Nb Accounts(1);"
      	"Exp Duration(2);"
      	"Real Duration(3);"
      	"CPU Commits(4);"
      	"GPU Commits(5);"
      	"CPU Aborts(6);"
        "GPU Aborts(7);"
      	"CPU Throughput(8);"
      	"HeTM Throughput(9);"
      	"Total Lock Time(10);"
      	"Total Comp Time(11);"
      	"Transf CPU(12);"
      	"Transf GPU(13);"
      	"Transf Cmp(14);"
      	"Tx Time(15);"
      	"Nb GPU runs(16);"
      	"Nb success(17);"
      	"Accts per TX(18);"
      	"Batch Size(19);"
      	"Sync Balancing(20);"
      	"CPU_PART(21);"
      	"GPU_PART(22);"
      	"P_INTERSECT(23);"
      	"GPU_BLOCKS(24);"
      	"GPU_THREADS(25);"
      	"GPU-only Throughput(26);"
      	"CPU_THREADS(27)"
        "\n"
      );
    }

    fprintf(f, "%i;" , data->nb_accounts                     ); // Nb Accounts(1)
    fprintf(f, "%f;" , data->tot_duration[data->iter/2]      ); // Exp Duration(2)
    fprintf(f, "%f;" , data->tot_duration2[data->iter/2]     ); // Real Duration(3)
    fprintf(f, "%lu;", data->tot_commits[data->iter/2]       ); // CPU Commits(4)
    fprintf(f, "%lu;", data->tot_commits_gpu[data->iter/2]   ); // GPU Commits(5)
    fprintf(f, "%li;", data->tot_aborts[data->iter/2]        ); // CPU Aborts(6)
    fprintf(f, "%lu;", data->tot_aborts_gpu[data->iter/2]    ); // GPU Aborts(7)
    fprintf(f, "%f;" , data->tot_throughput[data->iter/2]    ); // CPU Throughput(8)
    fprintf(f, "%f;" , data->tot_throughput_gpu[data->iter/2]); // GPU Throughput(9)
    fprintf(f, "%li;", data->tot_cuda[data->iter/2]          ); // Total Lock Time(10)
    fprintf(f, "%li;", data->tot_comp[data->iter/2]          ); // Total Comp Time(11)
    fprintf(f, "%li;", data->tot_trf2cpu[data->iter/2]       ); // Transf CPU(12)
    fprintf(f, "%li;", data->tot_trf2gpu[data->iter/2]       ); // Transf GPU(13)
    fprintf(f, "%li;", data->tot_trfcmp[data->iter/2]        ); // Transf Cmp(14)
    fprintf(f, "%li;", data->tot_tx[data->iter/2]            ); // Tx Time(15)
    fprintf(f, "%li;", data->tot_loop[data->iter/2]          ); // Nb GPU runs(16)
    fprintf(f, "%li;", data->tot_loops[data->iter/2]         ); // Nb success(17)
    fprintf(f, "%d;" , data->trfs                            ); // Accts per TX(18)
    fprintf(f, "%d;" , data->trans                           ); // Batch Size(19)
    fprintf(f, "%d;" , SYNC_BALANCING_VALF                   ); // Sync Balancing(20)
    fprintf(f, "%f;" , CPU_PART                              ); // CPU_PART(21)
    fprintf(f, "%f;" , GPU_PART                              ); // GPU_PART(22)
    fprintf(f, "%f;" , P_INTERSECT                           ); // P_INTERSECT(23)
    fprintf(f, "%i;" , data->GPUblockNum                     ); // GPU_BLOCKS(24)
    fprintf(f, "%i;" , data->GPUthreadNum                    ); // GPU_THREADS(25)
    fprintf(f, "%f;" , data->tot_throughput_gpu_only[data->iter/2]); // GPUonly Throughput(26)
    fprintf(f, "%i\n", data->nb_threadsCPU                    ); // CPU_THREADS(27)
    fclose(f);
}

void bank_between_iter(thread_data_t *data, int j)
{
  // TODO[Ricardo]: I think there is a memory leak, the GPU runs out of memory
  // if I put too many iterations (where are the GPU mallocs/frees?)

  // TODO[Ricardo]: explain all arrays!
	/* Save info between iterations*/
  data->throughput            = (data->reads + data->writes + data->updates) * 1000.0 / data->duration2;
  // thread data->nb_threadsCPU is the GPU controller thread
  data->throughput_gpu        = (data->reads + data->writes + data->updates + data->dthreads[data->nb_threadsCPU].nb_transfer) * 1000.0 / data->duration2;
  data->throughput_gpu_only   = (data->dthreads[data->nb_threadsCPU].nb_transfer_gpu_only) * 1000.0 / data->duration2;
  data->tot_commits_gpu[j]    = data->dthreads[data->nb_threadsCPU].nb_transfer;
  data->tot_throughput_gpu[j] = data->throughput_gpu;
  printf("data->throughput_HeTM=%f\n", data->throughput_gpu);
  printf("data->throughput_gpu =%f\n", data->throughput_gpu_only);
  printf("data->throughput_cpu =%f\n", data->throughput);
  data->tot_aborts_gpu[j]     = data->dthreads[data->nb_threadsCPU].nb_aborts;
  data->tot_comp[j]           = data->dthreads[data->nb_threadsCPU].nb_aborts_2;
  data->tot_tx[j]             = data->dthreads[data->nb_threadsCPU].max_retries;
  data->tot_cuda[j]           = data->dthreads[data->nb_threadsCPU].nb_aborts_1;
  data->tot_loop[j]           = data->dthreads[data->nb_threadsCPU].nb_aborts_locked_read;
  data->tot_loops[j]          = data->dthreads[data->nb_threadsCPU].nb_aborts_locked_write;
  data->tot_trf2cpu[j]        = data->dthreads[data->nb_threadsCPU].nb_aborts_validate_read;
  data->tot_trf2gpu[j]        = data->dthreads[data->nb_threadsCPU].nb_aborts_validate_write;
  data->tot_trfcmp[j]         = data->dthreads[data->nb_threadsCPU].nb_aborts_validate_commit;
  data->tot_commits[j]        = data->reads + data->writes + data->updates;
  data->tot_duration[j]       = data->duration;
  data->tot_duration2[j]      = data->duration2;
  data->tot_throughput[j]     = data->throughput;
  data->tot_throughput_gpu_only[j] = data->throughput_gpu_only;
  for (int i = 0; i < data->nb_threadsCPU; ++i) {
    data->tot_aborts[j]      += data->dthreads[i].nb_aborts;
  }
}

void bank_check_params(thread_data_t *data)
{
  if (data->fn != NULL) {
		strncpy(data->filename, data->fn, 100);
		data->filename[100]='\0';
  } else {
		strcpy(data->filename, XSTR(DEFAULT_OUTPUT_FILE));
		data->filename[4]='\0';
  }
  strcat(data->filename, ".csv\0");

  assert(data->duration >= 0);
  assert(data->nb_accounts >= 2);
  assert(data->nb_threads > 0 && data->nb_threads < MAX_THREADS);
  assert(data->read_all >= 0 && data->write_all >= 0 && data->read_all + data->write_all <= 100);
  assert(data->read_threads + data->write_threads <= data->nb_threads);
  assert(data->iter <= MAX_ITER);
  assert(data->trfs <= MAX_ITER);
  assert(data->trans >= 0);

  printf("Nb accounts    : %d\n", data->nb_accounts);
#ifndef TM_COMPILER
  printf("CM             : %s\n", (data->cm == NULL ? "DEFAULT" : data->cm));
#endif /* ! TM_COMPILER */
  printf("Duration       : %f\n", data->duration);
  printf("Iterations     : %d\n", data->iter);
  printf("Nb threads     : %d\n", data->nb_threads);
  printf("Read-all rate  : %d\n", data->read_all);
  printf("Read threads   : %d\n", data->read_threads);
  printf("Seed           : %d\n", data->seed);
  printf("Write-all rate : %d\n", data->write_all);
  printf("Write threads  : %d\n", data->write_threads);
  printf("Transfers      : %d\n", data->trfs);
  printf("Batch Size     : %d\n", data->trans);
  printf("Type sizes     : int=%d/long=%d/ptr=%d/word=%d\n",
         (int)sizeof(int),
         (int)sizeof(long),
         (int)sizeof(void *),
         (int)sizeof(size_t));
  printf("Output file    : %s\n", data->filename);
  DEBUG_PRINT("Debug	       : Enabled\n");

#ifndef TM_COMPILER
  char *stm_flags;
  if (stm_get_parameter("compile_flags", &stm_flags))
	printf("STM flags      : %s\n", stm_flags);
#endif /* TM_COMPILER */

  data->timeout.tv_sec = data->duration / 1000;
  data->timeout.tv_nsec = ((long)data->duration % 1000) * 1000000;
	if (data->seed == 0)
		srand((int)time(NULL));
	else
		srand(data->seed);
}

/* ################################################################### *
 * Sorting
 * ################################################################### */

int compare_int (const void *a, const void *b) {
  const int *da = (const int *) a;
  const int *db = (const int *) b;

  return (*da > *db) - (*da < *db);
}

int compare_double (const void *a, const void *b) {
  const double *da = (const double *) a;
  const double *db = (const double *) b;

  return (*da > *db) - (*da < *db);
}

// ------------

int bank_sum(bank_t *bank)
{
  long i;
  long total;

  total = 0;
  for (i = 0; i < bank->size; i++) {
    total += bank->accounts[i];
  }

  return total;
}
