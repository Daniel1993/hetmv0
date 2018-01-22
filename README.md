# HeTM

HeTM: a TM library for heterogeneous systems.

---

## Sumary

HeTM is an attempt of bringing TM into the heterogenous system world, i.e., allowing a program that executes in different processing unit (GPU and CPU) to maintain a consistent shared dataset.

## Current status

Currently, HeTM is implemented in a synthetic benchmark, we are trying to make it more generic and export into other applications (e.g., MemcachedGPU).

## Dependencies

This project depends on:

cuda-utils

pr-stm

## Compiling

Go to dir bank and execute the following commands:

```bash
$ make BENCH=BANK
$ ./bank
```

Makefile flags can be found in Makefile.defines, `bank` parameters can be executing in `$ bank -h`.
