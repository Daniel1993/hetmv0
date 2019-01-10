#define _GNU_SOURCE // redefined
#include "shared.h"
#include "bank.h"
#include <sched.h>

#include "input_handler.h"

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
  // offset = id % 56;
  // if (id >= 14 && id < 28)
  //   offset += 14;
  // if (id >= 28 && id < 42)
  //   offset -= 14;

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
  data->stddev            = DEFAULT_SEED;
  data->hmult             = DEFAULT_HMULT;
  data->hprob             = DEFAULT_HPROB;
  data->write_all         = DEFAULT_WRITE_ALL;
  data->write_threads     = DEFAULT_WRITE_THREADS;
  data->iter              = DEFAULT_ITERATIONS;
  data->trfs              = DEFAULT_NB_TRFS;
  data->trans             = DEFAULT_TransEachThread;
  data->num_ways          = NUMBER_WAYS;
  data->read_intensive_size = 19; // TODO: add a MACRO

  data->prec_write_txs    = 2;
  data->access_controller = 1; // this is the offset of the input buffer
  data->access_offset     = 2;

  data->shared_percent    = QUEUE_SHARED_VALUE;
  data->set_percent       = WRITE_PERCENT;
  data->GPUblockNum       = DEFAULT_blockNum;
  data->GPUthreadNum      = DEFAULT_threadNum;

  data->GPUInputFile      = XSTR(DEFAULT_GPU_FILE);
  data->CPUInputFile      = XSTR(DEFAULT_CPU_FILE);

  struct option long_options[] = {
    // These options don't set a flag
    {"help",                no_argument,       NULL, 'h'},
    {"accounts",            required_argument, NULL, 'a'},
    {"contention-manager",  required_argument, NULL, 'c'},
    {"duration",            required_argument, NULL, 'd'},
    {"num-threads",         required_argument, NULL, 'n'},
    {"hmult",               required_argument, NULL, 's'},
    {"hprob",               required_argument, NULL, 'm'},
    {"num-iterations",      required_argument, NULL, 'i'},
    {"trfs",                required_argument, NULL, 't'},
    {"gpu-blocks",          required_argument, NULL, 'b'},
    {"gpu-threads",         required_argument, NULL, 'x'},
    {"tperthread",          required_argument, NULL, 'T'},
    {"output-file",         required_argument, NULL, 'f'},
    {"input-gpu",           required_argument, NULL, 'G'},
    {"input-cpu",           required_argument, NULL, 'C'},
    {"shared-rate",         required_argument, NULL, 'S'},
  	{"new-rate",            required_argument, NULL, 'N'},
  	{"number-ways",         required_argument, NULL, 'l'},
    {"precentage-read-txs", required_argument, NULL, 'R'},
    {NULL, 0, NULL, 0}
  };

  while(1) {
    i = 0;
    c = getopt_long(argc, argv, "ha:d:n:s:m:i:t:T:C:G:f:b:x:S:N:l:R:", long_options, &i);

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
          "        Number of accounts in the bank, or sets in MEMCACHED (default=" XSTR(DEFAULT_NB_ACCOUNTS) ")\n"
          "  -d, --duration <int>\n"
          "        Test duration in milliseconds (0=infinite, default=" XSTR(DEFAULT_DURATION) ")\n"
          "  -n, --num-threads <int>\n"
          "        Number of threads (default=" XSTR(DEFAULT_NB_THREADS) ")\n"
          "  -s, --hmult <float>\n"
          "        Size of the hotspot s in ]0, 1[ (rest is 3*hmult, compile with BANK_PART=2 default=" XSTR(DEFAULT_HMULT) ")\n"
          "  -m, --hprob <int>\n"
          "        Probability of hitting the hotspot (compile with BANK_PART=2 default=" XSTR(DEFAULT_HPROB) ")\n"
          "  -i, --num-iterations <int>\n"
          "        Number of iterations to execute (default=" XSTR(DEFAULT_ITERATIONS) ")\n"
          "  -t, --num-transfers <int>\n"
          "        Number of accounts to transfer between in each transaction (default=" XSTR(DEFAULT_NB_TRFS) ")\n"
          "  -S, --share-rate <int>\n"
          "        In bank is the offset of access in updateTXs\n"
          "        MEMCACHED: Percentage of transactions that are targeted at the shared zone (default=" XSTR(QUEUE_SHARED_VALUE) ")\n"
  			  "  -N, --new-rate <int>\n"
          "        In bank controls locality/contention (large N == high contention) \n"
          "        MEMCACHED: Percentage of set requests (default=" XSTR(WRITE_PERCENT) ")\n"
  			  "  -l, --num-ways <int>\n"
          "        In bank controls the precentage of write TXs\n"
          "        MEMCACHED: Number of 'ways' for each hashtable entry entry (default=" XSTR(NUMBER_WAYS) ")\n"
          "  -T, --num-txs-pergputhread <int>\n"
          "        Number of transactions per GPU thread (default=" XSTR(DEFAULT_TransEachThread) ")\n"
          "  -f, --output-file <string>\n"
          "        Output file name (default=" XSTR(DEFAULT_OUTPUT_FILE) ")\n"
          "  -G, --input-gpu <string>\n"
          "        Input file name (default=" XSTR(DEFAULT_OUTPUT_FILE) ")\n"
          "  -C, --input-cpu <string>\n"
          "        Input file name (default=" XSTR(DEFAULT_OUTPUT_FILE) ")\n"
          "  -b, --gpu-blocks <int>\n"
          "        Number of blocks for the GPU (default=" XSTR(DEFAULT_NB_GPU_BLKS) ")\n"
          "  -x, --gpu-threads <int>\n"
          "        Number of threads for the GPU (default=" XSTR(DEFAULT_NB_GPU_THRS) ")\n"
        );
        exit(0);
      case 'a':
        data->nb_accounts = atoi(optarg);
        break;
      case 'd':
        data->duration = atoi(optarg);
        break;
      case 'n':
        data->nb_threads = atoi(optarg);
        break;
      case 's':
        data->hmult = atof(optarg);
        break;
      case 'm':
        data->hprob = atoi(optarg);
        break;
      case 'i':
        data->iter = atoi(optarg);
        break;
      case 't':
        data->trfs = atoi(optarg);
        break;
      case 'S':
        data->shared_percent = atoi(optarg);
        data->read_intensive_size = atoi(optarg); // TODO: recycling same var
        data->access_offset = atoi(optarg);
        break;
      case 'N':
        data->set_percent = atof(optarg);
        data->access_controller = atof(optarg);
        break;
      case 'l':
        data->prec_write_txs = atoi(optarg);
        data->num_ways = atoi(optarg);
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
      case 'G':
        data->GPUInputFile = optarg;
        break;
      case 'C':
        data->CPUInputFile = optarg;
        break;
      case 'R':
        data->nb_read_intensive = atoi(optarg);
        break;
      default:
        exit(1);
    }
  }

  // TODO: extra parameters are fetched with input_handler
	input_parse(argc, argv); // parses input in the format <key>=<val>

  data->CPU_backoff = 100;
  if (input_exists("CPU_BACKOFF")) {
    data->CPU_backoff = input_getLong("CPU_BACKOFF");
  }

  data->CPU_steal_prob = 0;
  if (input_exists("CPU_STEAL_PROB")) {
    data->CPU_steal_prob = input_getDouble("CPU_STEAL_PROB");
  }

  data->GPU_steal_prob = 0;
  if (input_exists("GPU_STEAL_PROB")) {
    data->GPU_steal_prob = input_getDouble("GPU_STEAL_PROB");
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

  // DONE in between iters
  // duration = TIMER_DIFF_SECONDS(data->start, data->end) * 1000;
  // duration2 = TIMER_DIFF_SECONDS(data->start, data->last) * 1000;
  // data->duration  = duration;
  // data->duration2 = duration2;

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
#ifdef BENCH_BANK
  ret = bank_sum(data->bank);
  // for (int i = 0; i < data->bank->size; ++i) {
  //   if (data->bank->accounts[i] != 0)
  //   printf("[%7i=%9i]", i,  data->bank->accounts[i]);
  // }
  // printf("\n");
  printf("Bank total    : %d (expected: 0)\n", ret); // TODO
#endif
  printf("Duration      : %f (ms)\n", data->duration);
  printf("#txs          : %lu (%f / s)\n", data->reads + data->writes + data->updates, (data->reads + data->writes + data->updates) * 1000.0 / data->duration2);
  printf("#aborts       : %lu (%f / s)\n", aborts, aborts * 1000.0 / data->duration2);
  printf("Duration      : %f (ms)\n", data->duration);
  printf("Real duration : %f (ms)\n", data->duration2);
#endif

  printf(" Throughput in kernel = %14.2f TXs/s\n", (double)HeTM_stats_data.nbTxsGPU/((double)HeTM_stats_data.timeGPU/1000.0));
  printf("   -> TXs  = %.2f TXs\n", (double)HeTM_stats_data.nbTxsGPU);
  printf("   -> time = %.5f s\n", (double)HeTM_stats_data.timeGPU/1000.0);
  printf(" THROUGHPUT           = %14.2f\n", data->throughput_gpu);
  printf("   >>> THROUGHPUT GPU = %14.2f\n", data->throughput_gpu_only);
  printf("       -> time = %.5f s\n", (double)data->duration2/1000.0);
  printf("   >>> THROUGHPUT CPU = %14.2f\n", data->throughput);
  printf("   >>> AFTER BATCH    = %14.2f\n", (double)HeTM_stats_data.txsNonBlocking /
    (HeTM_stats_data.timeNonBlocking));
  printf("   >>> TXs NON_BLOCK  = %li\n", HeTM_stats_data.txsNonBlocking);
  printf("   >>> TXs            = %li\n", HeTM_stats_data.nbTxsCPU);
  printf(" TOTAL TIME IN BENCHMARK = %14.5fms\n", data->duration2);
  printf("   > ON BATCH            = %14.5f  || %14.5f\n", HeTM_stats_data.timePRSTM*1000, HeTM_stats_data.timeGPU);
  printf("   > ON COMPARING        = %14.5f\n", HeTM_stats_data.timeCMP*1000);
  printf("   > AFTER CMP           = %14.5f\n", HeTM_stats_data.timeAfterCMP*1000);
  printf("   > BLOCKING            = %14.5f\n", HeTM_stats_data.timeBlocking*1000);
  printf("   > NON-BLOCKING        = %14.5f\n", HeTM_stats_data.timeNonBlocking*1000);
  printf("   > COPING THE WSET     = %14.5f\n", HeTM_stats_data.totalTimeCpyWSet);
  printf("   > COPING THE DATASET  = %14.5f\n", HeTM_stats_data.totalTimeCpyDataset);
  printf("   > CMP (GPU time)      = %14.5f\n", HeTM_stats_data.totalTimeCmp);
  printf(" DATA TRANSFERED (B)\n");
  printf("   > LOGS (VERS, ADDR) = %zuB\n", HeTM_stats_data.sizeCpyLogs);
  printf("   > WSet (ADDR, BMAP) = %zuB\n", HeTM_stats_data.sizeCpyWSet);
  printf("   > Dataset           = %zuB\n", HeTM_stats_data.sizeCpyDataset);
}

void bank_statsFile(thread_data_t *data)
{
  DEBUG_PRINT("Sorting info.\n");

  // TODO: refactor this to be the same as the remaining statistics
  double kernelThroughput = (double)HeTM_stats_data.nbTxsGPU /
    (HeTM_stats_data.timeGPU / 1000.0);

  /*Sort Arrays*/
  qsort(data->tot_duration,       data->iter,sizeof(int),compare_int);
  qsort(data->tot_duration2,      data->iter,sizeof(int),compare_int);
  qsort(data->tot_commits,        data->iter,sizeof(int),compare_int);
  qsort(data->tot_aborts,         data->iter,sizeof(int),compare_int);
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
        "NB_ACCOUNTS(1);"
        "CPU_THREADS(2);"
        "GPU_BLOCKS(3);"
        "GPU_THREADS_PER_BLOCK(4);"
        "TXs_PER_GPU_THREAD(5);"
        "CPU_PART(6);"
        "GPU_PART(7);"
        "P_INTERSECT(8);"
      	"DURATION(9);"
      	"REAL_DURATION(10);"
      	"NB_CPU_COMMITS(11);"
      	"NB_GPU_COMMITS(12);"
      	"NB_CPU_ABORTS(13);"
        "NB_GPU_ABORTS(14);"
      	"CPU_THROUGHPUT(15);"
        "GPU_THROUGHPUT(16);"
        "Kernel_THROUGHPUT(17);"
      	"HeTM_THROUGHPUT(18);"
        "NB_BATCHES(19);"
        "NB_SUCCESS(20);"
      	"TIME_IN_KERNEL(21);"
        "TIME_BATCH(22);"
        "TIME_AFTER_BATCH(23);"
        "TIME_AFTER_CMP(24);"
      	"TIME_CPY_WSET(25);"
      	"TIME_CMP(26);"
      	"TIME_CPY_DATASET(27);"
      	"THROUGHPUT_BETWEEN_BATCHES(28);"
      	"CPU_TXs_BETWEEN_BATCHES(29);"
      	"TIME_NON_BLOCKING(30);"
      	"TIME_BLOCKING(31);"
        "SIZE_CPY_LOGS(32);"
        "SIZE_CPY_WSET(33);"
        "SIZE_CPY_DATASET(34);"
        "PROB_HOTSPOT(35);"
        "SIZE_HOTSPOT(36)"
        "\n"
      );
    }

    double throughputBetweenBatches = (double)HeTM_stats_data.txsNonBlocking /
      (HeTM_stats_data.timeBlocking+HeTM_stats_data.timeNonBlocking);

    if (HeTM_stats_data.timeBlocking+HeTM_stats_data.timeNonBlocking == 0) {
      throughputBetweenBatches = 0;
    }

    // TODO: REFACTOR NAMES OF VARIABLES!!!
    fprintf(f, "%i;" , data->nb_accounts                     ); // NB_ACCOUNTS(1)
    fprintf(f, "%i;" , data->nb_threadsCPU                   ); // CPU_THREADS(2)
    fprintf(f, "%i;" , data->GPUblockNum                     ); // GPU_BLOCKS(3)
    fprintf(f, "%i;" , data->GPUthreadNum                    ); // GPU_THREADS_PER_BLOCK(4)
    fprintf(f, "%d;" , data->trans                           ); // TXs_PER_GPU_THREAD(5)
    fprintf(f, "%f;" , CPU_PART                              ); // CPU_PART(6)
    fprintf(f, "%f;" , GPU_PART                              ); // GPU_PART(7)
    fprintf(f, "%f;" , P_INTERSECT                           ); // P_INTERSECT(8)
    fprintf(f, "%f;" , data->tot_duration[data->iter/2]      ); // DURATION(9)
    fprintf(f, "%f;" , data->tot_duration2[data->iter/2]     ); // REAL_DURATION(10)
    fprintf(f, "%lu;", data->tot_commits[data->iter/2]       ); // NB_CPU_COMMITS(11)
    fprintf(f, "%lu;", data->tot_commits_gpu[data->iter/2]   ); // NB_GPU_COMMITS(12)
    fprintf(f, "%li;", data->tot_aborts[data->iter/2]        ); // NB_CPU_ABORTS(13)
    fprintf(f, "%lu;", data->tot_aborts_gpu[data->iter/2]    ); // NB_GPU_ABORTS(14)
    fprintf(f, "%f;" , data->tot_throughput[data->iter/2]    ); // CPU_THROUGHPUT(15)
    fprintf(f, "%f;" , data->tot_throughput_gpu_only[data->iter/2]); //  GPU_THROUGHPUT(16)
    fprintf(f, "%f;" , kernelThroughput                      ); // Kernel_THROUGHPUT(17)
    fprintf(f, "%f;" , data->tot_throughput_gpu[data->iter/2]); // HeTM_THROUGHPUT(18)
    fprintf(f, "%li;", data->tot_loop[data->iter/2]          ); // NB_BATCHES(19)
    fprintf(f, "%li;", data->tot_loops[data->iter/2]         ); // NB_SUCCESS(20)
    fprintf(f, "%f;" , HeTM_stats_data.timeGPU/1000.0        ); // TIME_IN_KERNEL(21)
    fprintf(f, "%f;" , HeTM_stats_data.timePRSTM             ); // TIME_BATCH(22)
    fprintf(f, "%f;" , HeTM_stats_data.timeCMP               ); // TIME_AFTER_BATCH(23)
    fprintf(f, "%f;" , HeTM_stats_data.timeAfterCMP          ); // TIME_AFTER_CMP(24)
    fprintf(f, "%f;" , HeTM_stats_data.totalTimeCpyWSet/1000.0); // TIME_CPY_WSET(25)
    fprintf(f, "%f;" , HeTM_stats_data.totalTimeCmp/1000.0   ); // TIME_CMP(26)
    fprintf(f, "%f;" , HeTM_stats_data.totalTimeCpyDataset/1000.0); // TIME_CPY_DATASET(27)
    fprintf(f, "%f;" , throughputBetweenBatches              ); // THROUGHPUT_BETWEEN_BATCHES(28)
    fprintf(f, "%li;", HeTM_stats_data.txsNonBlocking        ); // CPU_TXs_BETWEEN_BATCHES(29)
    fprintf(f, "%f;" , HeTM_stats_data.timeNonBlocking       ); // TIME_NON_BLOCKING(30)
    fprintf(f, "%f;" , HeTM_stats_data.timeBlocking          ); // TIME_BLOCKING(31)
    fprintf(f, "%zu;", HeTM_stats_data.sizeCpyLogs           ); // SIZE_CPY_LOGS(32)
    fprintf(f, "%zu;", HeTM_stats_data.sizeCpyWSet           ); // SIZE_CPY_WSET(33)
    fprintf(f, "%zu;", HeTM_stats_data.sizeCpyDataset        ); // SIZE_CPY_DATASET(34)
    fprintf(f, "%i;" , data->hprob                           ); // PROB_HOTSPOT(35)
    fprintf(f, "%f"  , data->hmult                           ); // SIZE_HOTSPOT(36)
    fprintf(f, "\n");
    fclose(f);
}

void bank_between_iter(thread_data_t *data, int j)
{
  // TODO[Ricardo]: I think there is a memory leak, the GPU runs out of memory
  // if I put too many iterations (where are the GPU mallocs/frees?)

  // TODO[Ricardo]: explain all arrays!
	/* Save info between iterations*/
  // data->reads + data->writes + data->updates // TODO: CPU_INV changes these values

  // DONE --> do not take it again on print
  data->duration = TIMER_DIFF_SECONDS(data->start, data->end) * 1000;
  data->duration2 = TIMER_DIFF_SECONDS(data->start, data->last) * 1000.0;

  data->throughput            = (HeTM_stats_data.nbCommittedTxsCPU) / ((double)data->duration2 / 1000.0);
  // thread data->nb_threadsCPU is the GPU controller thread
  data->throughput_gpu        = (HeTM_stats_data.nbCommittedTxsCPU + HeTM_stats_data.nbCommittedTxsGPU) / ((double)data->duration2 / 1000.0);
  data->throughput_gpu_only   = (HeTM_stats_data.nbCommittedTxsGPU) / ((double)data->duration2 / 1000.0);
  data->tot_commits_gpu[j]    = data->dthreads[data->nb_threadsCPU].nb_transfer;
  data->tot_throughput_gpu[j] = data->throughput_gpu;
  data->tot_aborts_gpu[j]     = data->dthreads[data->nb_threadsCPU].nb_aborts;
  data->tot_comp[j]           = data->dthreads[data->nb_threadsCPU].nb_aborts_2;
  data->tot_tx[j]             = data->dthreads[data->nb_threadsCPU].max_retries;
  data->tot_cuda[j]           = data->dthreads[data->nb_threadsCPU].nb_aborts_1;
  data->tot_loop[j]           = data->dthreads[data->nb_threadsCPU].nb_aborts_locked_read;
  data->tot_loops[j]          = data->dthreads[data->nb_threadsCPU].nb_aborts_locked_write;
  data->tot_trf2cpu[j]        = data->dthreads[data->nb_threadsCPU].nb_aborts_validate_read;
  data->tot_trf2gpu[j]        = data->dthreads[data->nb_threadsCPU].nb_aborts_validate_write;
  data->tot_trfcmp[j]         = data->dthreads[data->nb_threadsCPU].nb_aborts_validate_commit;
  data->tot_commits[j]        = HeTM_stats_data.nbCommittedTxsCPU;
  data->tot_duration[j]       = data->duration;
  data->tot_duration2[j]      = data->duration2;
  data->tot_throughput[j]     = data->throughput;
  data->tot_throughput_gpu_only[j] = data->throughput_gpu_only;
  for (int i = 0; i < data->nb_threadsCPU; ++i) {
    data->tot_aborts[j]      += data->dthreads[i].nb_aborts;
  }

  if (data->nb_threadsCPU != 0) {
    HeTM_stats_data.timeBlocking /= data->nb_threadsCPU;
    HeTM_stats_data.timeNonBlocking /= data->nb_threadsCPU;
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
  printf("CPU_steal      : %f\n", data->CPU_steal_prob);
  printf("GPU_steal      : %f\n", data->GPU_steal_prob);
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
	if (data->seed == 0) {
    srand((int)time(NULL));
  } else {
    srand(data->seed);
  }
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
