/*
 * File:
 *   log.h
 * Author(s):
 *   Ricardo Vieira
 * Description:
 *   STM transaction log.
 *
 */

#ifndef LOG_H_GUARD_
#define LOG_H_GUARD_

#include "utils.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <x86intrin.h>

/* ################################################################### *
 * Defines
 * ################################################################### */

#define LOG_AUTO        1      /* Enable or disable automatic logging */
#define LOG_SIZE        131072 // 65536  /* Initial log size */
//Use 64*1024 for explicit log, 65535 for bitmap

/* ################################################################### *
 * TYPES
 * ################################################################### */

typedef struct HeTM_CPULogEntry_t {
  long *pos;       /* Address written to */
  long val;        /* Value Written      */
  long time_stamp; /* Version counter    */
} HeTM_CPULogEntry;

typedef struct HeTM_CPULogNode {
  HeTM_CPULogEntry array[LOG_SIZE];
  int size;                   /* size of the array == LOG_SIZE */
  int curPos;                 /* index to insert               */
  int nbNodes;                /* list size                     */
  int isLast;
  int isFree;
  struct HeTM_CPULogNode *next;
} HeTM_CPULogNode_t;

typedef struct HeTM_CPULog {
  HeTM_CPULogNode_t *base;
  HeTM_CPULogNode_t *end;
  HeTM_CPULogNode_t *read;
  long *log_base_pointer;
  int shift_count;
  int nbNodes;
} HeTM_CPULog_t;

extern __thread HeTM_CPULog_t stm_log_recicled_nodes;

// TODO: log nodes garbage collector (avoid malloc/free)
// extern thread_local HeTM_CPULog_t HeTM_CPULogNode_gc;

/* ################################################################### *
 * INLINE FUNCTIONS
 * ################################################################### */

static INLINE HeTM_CPULogNode_t*
stm_log_node_alloc()
{
  HeTM_CPULogNode_t *res = NULL;
  if (stm_log_recicled_nodes.base != NULL) {
    // there are some nodes waiting for recycle
    res = stm_log_recicled_nodes.base;
    stm_log_recicled_nodes.base = res->next;
    if (res->next == NULL) {
      // no more nodes
      stm_log_recicled_nodes.end = NULL;
      stm_log_recicled_nodes.read = NULL;
    }
  } else {
    res = (HeTM_CPULogNode_t*)malloc(sizeof(HeTM_CPULogNode_t));
  }

  PRINT_DEBUG("[stm_log_node_alloc] %p\n", res);
  res->next = NULL;// memset(res, 0, sizeof(HeTM_CPULogNode_t)); // TODO: not needed
  res->isFree = 0;

  return res;
}

static INLINE void
stm_log_node_free(HeTM_CPULogNode_t *t)
{
  PRINT_DEBUG("[stm_log_node_free] %p\n", t);

  if (t->isFree) {
    PRINT_DEBUG("BUG detected!\n");
    return; // BUG
  }

  t->next = NULL; // memset(t, 0, sizeof(HeTM_CPULogNode_t)); // TODO: not needed
  t->isFree = 1;

  if (stm_log_recicled_nodes.base == NULL) {
    stm_log_recicled_nodes.base = stm_log_recicled_nodes.end = t;
    stm_log_recicled_nodes.read = stm_log_recicled_nodes.base;
    stm_log_recicled_nodes.end->next = NULL;
  } else {
    stm_log_recicled_nodes.end->next = t;
    stm_log_recicled_nodes.end = t;
    stm_log_recicled_nodes.end->next = NULL;
  }
  // postpones node destruction
}

/* Init log */
static INLINE HeTM_CPULog_t*
stm_log_init()
{
  HeTM_CPULogNode_t *t = stm_log_node_alloc();
  HeTM_CPULog_t     *p = (HeTM_CPULog_t*) malloc(sizeof(HeTM_CPULog_t));

  // memset(&t->array, 0, LOG_SIZE*sizeof(HeTM_CPULogEntry)); // TODO: not needed
  t->size   = LOG_SIZE;
  t->curPos = 0;
  t->next   = NULL;
  t->isLast = 0;

  p->end = p->base    = t;
  p->read             = NULL;
  p->log_base_pointer = NULL;
  p->nbNodes          = 1;

  return p;
};

/* BM INIT: TODO: what the hell this is doing??? reduce #memcpy from GPU? */
static INLINE int
stm_log_initBM(HeTM_CPULog_t *b, long *point, int size)
{
  int i, current;
  b->log_base_pointer = point;

  if (size < LOG_SIZE*64) {
    b->shift_count = 0;
  } else {
    for (i = 1; i <= 4; i++) {
      current = size >> i;
      if (current < LOG_SIZE*64) {
        b->shift_count = i;
        break;
      }
    }
  }

  return 1;
}

/* Add value to log */
static INLINE int
stm_log_newentry(HeTM_CPULog_t * b, long *pos, int val, long vers)
{
  HeTM_CPULogNode_t * t;

  t = b->end;

  if(t->curPos < t->size) {
    t->array[t->curPos].pos =  pos;
    t->array[t->curPos].val =  val;
    t->array[t->curPos].time_stamp =  vers;

    t->curPos = t->curPos+1;

    return 1;
  } else {
    b->end = t->next = stm_log_node_alloc();
    t = b->end;
    b->nbNodes++;

    t->array[0].pos = pos;
    t->array[0].val = val;
    t->curPos       = 1;
    t->size         = LOG_SIZE;
    t->next         = NULL;
    t->isLast       = 0;

    return 1;
  }
}

/* Read the log */
static INLINE HeTM_CPULogNode_t *
stm_log_read(HeTM_CPULog_t *t)
{
  HeTM_CPULogNode_t * newNode = stm_log_node_alloc();
  // memset(&newNode->array, 0, LOG_SIZE*sizeof(HeTM_CPULogEntry)); // TODO: not needed

  newNode->size    = LOG_SIZE;
  newNode->curPos  = 0;
  newNode->nbNodes = 0;
  newNode->isLast  = 0;
  newNode->next    = NULL;

  t->read          = t->base;
  t->read->nbNodes = t->nbNodes;
  t->end->next     = newNode;
  t->end->isLast   = 1;

  // resets the log
  t->end = t->base = newNode;
  t->nbNodes = 1;

  return t->read;
}

/* Truncate N nodes of the log, returns a pointer to the first node (user must free it) */
// static INLINE HeTM_CPULogNode_t *
// stm_log_truncate(HeTM_CPULog_t *t, int N)
// {
//   HeTM_CPULogNode_t * newNode = (HeTM_CPULogNode_t*) malloc(sizeof(HeTM_CPULogNode_t));
//   memset(&newNode->array, 0, LOG_SIZE*sizeof(HeTM_CPULogEntry));
//
//   t->read = t->base;
//   t->read->nbNodes = t->nbNodes;
//   t->end->next     = newNode;
//
//   newNode->size = LOG_SIZE;
//   newNode->curPos = 0;
//   newNode->next = NULL;
//
//   // resets the log
//   t->end = t->base = new_t;
//   t->nbNodes = 1;
//
//   return t->read;
// }

// TODO: relies on malloc/free --> if they are slow this will not be nice
/* Creates a new log and returns the previous one (user must free it) */
// static INLINE HeTM_CPULogNode_t *
// stm_log_erase(HeTM_CPULog_t *t, int N)
// {
//   HeTM_CPULogNode_t * new_t = (HeTM_CPULogNode_t*) malloc(sizeof(HeTM_CPULogNode_t));
//   memset(&new_t->array, 0, LOG_SIZE*sizeof(HeTM_CPULogEntry));
//
//   t->read = t->base;
//   t->read->nbNodes = t->nbNodes;
//
//   new_t->size = LOG_SIZE;
//   new_t->curPos = 0;
//   new_t->next = NULL;
//
//   // resets the log
//   t->end = t->base = new_t;
//   t->nbNodes = 1;
//
//   return t->read;
// }

/* Remove an element */
static INLINE void
stm_logel_free(HeTM_CPULogNode_t *p)
{
  free(p);
}

/* Clean up */
static INLINE void
stm_log_free(HeTM_CPULog_t *t)
{
  HeTM_CPULogNode_t *aux, *aux2;

  aux = aux2 = t->base;

  while (aux != NULL) {
    aux2 = aux2->next;
    stm_logel_free(aux);

    aux = aux2;
  }

  t->base = t->end = t->read = NULL;
  free(t);
  return;
}

#endif /* LOG_H_GUARD_ */
