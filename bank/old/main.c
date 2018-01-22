#include "bank.h"
#include "bank_aux.h"

// TODO: some day this will become the new benchmark...

void *CPU_worker(int id)
{
  bank_do_CPU(id);
  return bank_destroy_CPU_worker(id);
}

void *GPU_worker(int id)
{
  bank_do_GPU(id);
  return bank_destroy_GPU_worker(id);
}

int main(int argc, char **argv)
{
  int i;

  bank_process_args(argc, argv);
  bank_init_main();

  bank_init_threading();
  bank_CPU_worker(CPU_worker);
  bank_GPU_worker(GPU_worker);

  for(i = 0; i < bank_get_parsed_data()->iter; i++) { // Loop for testing purposes
    bank_do_bench();
    bank_check_iter();
  }

  bank_print_stats();

  return bank_destroy_main();
}
