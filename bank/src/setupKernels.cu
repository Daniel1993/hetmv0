#include "setupKernels.cuh"
#include "cmp_kernels.cuh"
#include "bankKernel.cuh"

static pr_tx_args_s prArgs;

static void run_bankTx(knlman_callback_params_s params);
static void run_checkTxCompressed(knlman_callback_params_s params);
static void run_checkTxExplicit(knlman_callback_params_s params);
static void run_finalTxLog2(knlman_callback_params_s params);

int HeTM_setup_bankTx(int nbBlocks, int nbThreads)
{
  PR_blockNum = nbBlocks;
  PR_threadNum = nbThreads;
  knlman_create("HeTM_bankTx", run_bankTx, 0);
  PR_createStatistics(&prArgs);
  return 0;
}

int HeTM_bankTx_cpy_IO()
{
  PR_retrieveIO(&prArgs);
  return 0;
}

int HeTM_teardown_bankTx()
{
  knlman_select("HeTM_bankTx");
  knlman_destroy();
  PR_disposeIO(&prArgs);
  return 0;
}

int HeTM_setup_checkTxCompressed()
{
  knlman_create("HeTM_checkTxCompressed", run_checkTxCompressed, 0);
  // knlman_add_mem("HeTM_flag_inter_conflict", KNLMAN_COPY); // done somewhere

  return 0;
}

int HeTM_teardown_checkTxCompressed()
{
  knlman_select("HeTM_checkTxCompressed");
  knlman_destroy();
  return 0;
}

int HeTM_setup_checkTxExplicit()
{
  knlman_create("HeTM_checkTxExplicit", run_checkTxExplicit, 0);
  return 0;
}

int HeTM_teardown_checkTxExplicit()
{
  knlman_select("HeTM_checkTxExplicit");
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

static void run_checkTxCompressed(knlman_callback_params_s params)
{
  dim3 blocks(params.blocks.x, params.blocks.y, params.blocks.z);
  dim3 threads(params.threads.x, params.threads.y, params.threads.z);
  cudaStream_t stream = (cudaStream_t)params.stream;
  HeTM_checkTxCompressed_s *data = (HeTM_checkTxCompressed_s*)params.entryObj;

  HeTM_knl_checkTxCompressed<<<blocks, threads, 0, stream>>>(data->knlArgs);

  CUDA_CHECK_ERROR(cudaStreamAddCallback(
      stream, checkCallback, data->clbkArgs, 0
    ), "");
}

static void run_checkTxExplicit(knlman_callback_params_s params)
{
  dim3 blocks(params.blocks.x, params.blocks.y, params.blocks.z);
  dim3 threads(params.threads.x, params.threads.y, params.threads.z);
  cudaStream_t stream = (cudaStream_t)params.stream;
  HeTM_checkTxExplicit_s *data = (HeTM_checkTxExplicit_s*)params.entryObj;

  cudaFuncSetCacheConfig(HeTM_knl_checkTxExplicit, cudaFuncCachePreferL1);
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

  HeTM_knl_checkTxExplicit<<<blocks, threads, 0, stream>>>(data->knlArgs);

  CUDA_CHECK_ERROR(cudaStreamAddCallback(
      stream, checkCallback, data->clbkArgs, 0
    ), "");
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
  HeTM_bankTx_input_s input, *inputDev;

  thread_local static unsigned short seed = 1234;

  CUDA_CHECK_ERROR(cudaThreadSynchronize(), ""); // sync the previous run

  memman_ad_hoc_free(NULL); // empties the previous parameters
  cudaFuncSetCacheConfig(bankTx, cudaFuncCachePreferL1);

  if (a == NULL) {
    // This seems to swap the buffers if given a NULL array...
    accounts = d->dev_a;
    d->dev_a = d->dev_b;
    d->dev_b = accounts;
  }

  memman_select("HeTM_dev_rset");
  memman_zero_gpu(NULL);

  memman_select("HeTM_flag_inter_conflict");
  memman_zero_gpu(NULL);

  DEBUG_PRINT("Launching transaction kernel.\n");

  input.accounts = d->dev_a;
  input.is_intersection = IS_INTERSECT_HIT( erand48(&seed)*100000.0 );
  input.nbAccounts = d->size;

  inputDev = (HeTM_bankTx_input_s*)memman_ad_hoc_alloc(NULL, &input, sizeof(HeTM_bankTx_input_s));
  memman_ad_hoc_cpy(NULL);

  // TODO: change PR-STM to use knlman
  // PR_blockNum = params.blocks.x;
  // PR_threadNum = params.threads.x;
  inBuf.buf = (void*)inputDev;
  inBuf.size = sizeof(HeTM_bankTx_input_s);
  outBuf.buf = NULL;
  outBuf.size = 0;
  PR_prepareIO(&prArgs, inBuf, outBuf);
  PR_run(bankTx, &prArgs, NULL);
}
