#include "setupKernels.cuh"
#include "cmp_kernels.cuh"
#include "bankKernel.cuh"
#include "bank.h"

extern pr_tx_args_s HeTM_pr_args; // TODO: in hetm-threading-gpu.cu

static void run_bankTx(knlman_callback_params_s params);
static void run_memcdReadTx(knlman_callback_params_s params);
static void run_memcdWriteTx(knlman_callback_params_s params);
static void run_finalTxLog2(knlman_callback_params_s params);

int HeTM_setup_bankTx(int nbBlocks, int nbThreads)
{
  PR_blockNum = nbBlocks;
  PR_threadNum = nbThreads;
  memman_alloc_dual("HeTM_bankTxInput", sizeof(HeTM_bankTx_input_s), 0);
  knlman_create("HeTM_bankTx", run_bankTx, 0);
  return 0;
}

int HeTM_setup_memcdWriteTx(int nbBlocks, int nbThreads)
{
  PR_blockNum = nbBlocks;
  PR_threadNum = nbThreads;
  knlman_create("HeTM_memcdWriteTx", run_memcdWriteTx, 0);

  memman_alloc_dual("HeTM_memcdTx_input", sizeof(HeTM_memcdTx_input_s), 0);
  // memman_alloc_gpu("HeTM_memcdTx_output", sizeof(HeTM_memcdTx_input_s), 0);
  return 0;
}

int HeTM_setup_memcdReadTx(int nbBlocks, int nbThreads)
{
  PR_blockNum = nbBlocks;
  PR_threadNum = nbThreads;
  knlman_create("HeTM_memcdReadTx", run_memcdReadTx, 0);

  // already set-up HeTM_memcdTx_input

  return 0;
}

int HeTM_bankTx_cpy_IO() // TODO: not used
{
  PR_retrieveIO(&HeTM_pr_args, NULL);
  return 0;
}

int HeTM_teardown_bankTx()
{
  knlman_select("HeTM_bankTx");
  knlman_destroy();
  return 0;
}

int HeTM_teardown_memcdWriteTx()
{
  knlman_select("HeTM_memcdWriteTx");
  knlman_destroy();
  return 0;
}

int HeTM_teardown_memcdReadTx()
{
  knlman_select("HeTM_memcdReadTx");
  knlman_destroy();
  return 0;
}

int HeTM_setup_finalTxLog2()
{
  knlman_create("HeTM_finalTxLog2", run_finalTxLog2, 0);
  return 0;
}

int HeTM_teardown_finalTxLog2()
{
  knlman_select("HeTM_finalTxLog2");
  knlman_destroy();
  return 0;
}

static void run_finalTxLog2(knlman_callback_params_s params)
{
  dim3 blocks(params.blocks.x, params.blocks.y, params.blocks.z);
  dim3 threads(params.threads.x, params.threads.y, params.threads.z);
  HeTM_knl_finalTxLog2_s *data = (HeTM_knl_finalTxLog2_s*)params.entryObj;

  /* Kernel Launch */
  HeTM_knl_finalTxLog2 <<< blocks, threads >>> (*data);

  // HeTM_knl_finalTxLog2<<<blocks, threads>>>(data->knlArgs);
}

static void run_bankTx(knlman_callback_params_s params)
{
  HeTM_bankTx_s *data = (HeTM_bankTx_s*)params.entryObj;
  account_t *a = data->knlArgs.a;
  account_t *accounts = a;
  cuda_t *d = data->knlArgs.d;
  pr_buffer_s inBuf, outBuf;
  HeTM_bankTx_input_s *input, *inputDev;

  cudaSetDevice(0);

  // thread_local static unsigned long seed = 0x3F12514A3F12514A;

  memman_select("HeTM_bankTxInput");
  input    = (HeTM_bankTx_input_s*)memman_get_cpu(NULL);
  inputDev = (HeTM_bankTx_input_s*)memman_get_gpu(NULL);

  // CUDA_CHECK_ERROR(cudaThreadSynchronize(), ""); // sync the previous run

  cudaFuncSetCacheConfig(bankTx, cudaFuncCachePreferL1);

  if (a == NULL) {
    // This seems to swap the buffers if given a NULL array...
    accounts = d->dev_a;
    d->dev_a = d->dev_b;
    d->dev_b = accounts;
  }

  input->accounts = d->dev_a; // a; // d->dev_a;
  input->is_intersection = isInterBatch;
  input->nbAccounts = d->size;

  inBuf.buf = (void*)inputDev;
  inBuf.size = sizeof(HeTM_bankTx_input_s);
  outBuf.buf = NULL;
  outBuf.size = 0;
  PR_prepareIO(&HeTM_pr_args, inBuf, outBuf);

  input->input_buffer = GPUInputBuffer;
  input->output_buffer = GPUoutputBuffer;
  memman_cpy_to_gpu(NULL, NULL);

  // TODO: change PR-STM to use knlman
  // PR_blockNum = params.blocks.x;
  // PR_threadNum = params.threads.x;
  PR_run(bankTx, &HeTM_pr_args, NULL);
}

static void run_memcdReadTx(knlman_callback_params_s params)
{
  HeTM_bankTx_s *data = (HeTM_bankTx_s*)params.entryObj; // TODO
  account_t *a = data->knlArgs.a;
  account_t *accounts = a;
  cuda_t *d = data->knlArgs.d;
  pr_buffer_s inBuf, outBuf;
  HeTM_memcdTx_input_s *input, *inputDev;

  // thread_local static unsigned short seed = 1234;

  CUDA_CHECK_ERROR(cudaThreadSynchronize(), ""); // sync the previous run

  // memman_ad_hoc_free(NULL); // empties the previous parameters
  cudaFuncSetCacheConfig(memcdReadTx, cudaFuncCachePreferL1);

  if (a == NULL) {
    // This seems to swap the buffers if given a NULL array...
    accounts = d->dev_a;
    d->dev_a = d->dev_b;
    d->dev_b = accounts;
  }

  memman_select("HeTM_memcdTx_input");
  input = (HeTM_memcdTx_input_s*)memman_get_cpu(NULL);
  inputDev = (HeTM_memcdTx_input_s*)memman_get_gpu(NULL);

  input->key      = d->dev_a;
  // TODO: /sizeof(...)
  input->extraKey = input->key + (d->memcd_nbSets*d->memcd_nbWays);
  input->val      = input->extraKey + 3*(d->memcd_nbSets*d->memcd_nbWays);
  input->extraVal = input->val + (d->memcd_nbSets*d->memcd_nbWays);
  input->ts_CPU   = input->extraVal + 7*(d->memcd_nbSets*d->memcd_nbWays);
  input->ts_GPU   = input->ts_CPU + (d->memcd_nbSets*d->memcd_nbWays);
  input->state    = input->ts_GPU + (d->memcd_nbSets*d->memcd_nbWays);
  input->setUsage = input->state + (d->memcd_nbSets*d->memcd_nbWays);
  input->nbSets   = d->num_sets;
  input->nbWays   = d->num_ways;
  input->input_keys = GPUInputBuffer;
  input->input_vals = GPUInputBuffer;
  input->output     = (memcd_get_output_t*)GPUoutputBuffer;

  memman_select("memcd_global_ts");
  input->curr_clock = (int*)memman_get_gpu(NULL);

  memman_select("HeTM_memcdTx_input");
  memman_cpy_to_gpu(NULL, NULL);

  // TODO:
  // inputDev = (HeTM_memcdTx_input_s*)memman_ad_hoc_alloc(NULL, &input, sizeof(HeTM_memcdTx_input_s));
  // memman_ad_hoc_cpy(NULL);

  // TODO: change PR-STM to use knlman
  // PR_blockNum = params.blocks.x;
  // PR_threadNum = params.threads.x;
  inBuf.buf = (void*)inputDev;
  inBuf.size = sizeof(HeTM_memcdTx_input_s);
  outBuf.buf = NULL;
  outBuf.size = 0;
  PR_prepareIO(&HeTM_pr_args, inBuf, outBuf);
  PR_run(memcdReadTx, &HeTM_pr_args, NULL);
}

static void run_memcdWriteTx(knlman_callback_params_s params)
{
  HeTM_bankTx_s *data = (HeTM_bankTx_s*)params.entryObj;
  account_t *a = data->knlArgs.a;
  account_t *accounts = a;
  cuda_t *d = data->knlArgs.d;
  pr_buffer_s inBuf, outBuf;
  HeTM_memcdTx_input_s *input, *inputDev;

  cudaFuncSetCacheConfig(memcdWriteTx, cudaFuncCachePreferL1);

  if (a == NULL) {
    // This seems to swap the buffers if given a NULL array...
    accounts = d->dev_a;
    d->dev_a = d->dev_b;
    d->dev_b = accounts;
  }

  memman_select("HeTM_memcdTx_input");
  input = (HeTM_memcdTx_input_s*)memman_get_cpu(NULL);
  inputDev = (HeTM_memcdTx_input_s*)memman_get_gpu(NULL);

  input->key   = d->dev_a;
  input->extraKey = input->key + (d->memcd_nbSets*d->memcd_nbWays);
  input->val      = input->extraKey + 3*(d->memcd_nbSets*d->memcd_nbWays);
  input->extraVal = input->val + (d->memcd_nbSets*d->memcd_nbWays);
  input->ts_CPU   = input->extraVal + 7*(d->memcd_nbSets*d->memcd_nbWays);
  input->ts_GPU   = input->ts_CPU + (d->memcd_nbSets*d->memcd_nbWays);
  input->state    = input->ts_GPU + (d->memcd_nbSets*d->memcd_nbWays);
  input->setUsage = input->state + (d->memcd_nbSets*d->memcd_nbWays);
  input->nbSets   = d->num_sets;
  input->nbWays   = d->num_ways;
  input->input_keys = GPUInputBuffer;
  input->input_vals = GPUInputBuffer;
  input->output     = (memcd_get_output_t*)GPUoutputBuffer;

  memman_select("memcd_global_ts");
  input->curr_clock = (int*)memman_get_gpu(NULL);

  memman_select("HeTM_memcdTx_input");
  memman_cpy_to_gpu(NULL, NULL);

  // TODO: change PR-STM to use knlman
  // PR_blockNum = params.blocks.x;
  // PR_threadNum = params.threads.x;
  inBuf.buf = (void*)inputDev;
  inBuf.size = sizeof(HeTM_memcdTx_input_s);
  outBuf.buf = NULL;
  outBuf.size = 0;
  PR_prepareIO(&HeTM_pr_args, inBuf, outBuf);
  PR_run(memcdWriteTx, &HeTM_pr_args, NULL);
}
