#ifndef CHUNKED_LOG_H_GUARD_
#define CHUNKED_LOG_H_GUARD_

/**
 * Chunked log library.
 *
 * This is a thread local library that aims in providing a very efficient log
 * implementation.
 */

#include <stdlib.h>
#include <string.h>
#include "chunked-log-aux.h"

typedef struct chunked_log_node_ {
  char *chunk; // assuming sizeof(char) == 1 Byte (8 bits)
  size_t size; // total size in Bytes of the chunk
  size_t gran; // granularity of the elements in Bytes
  size_t nb_buckets;
  union {
    size_t pos;  // current position in Bytes
    size_t *posArray; // current position in Bytes (mod)
  } p;
  struct chunked_log_node_ *next, *prev;
} chunked_log_node_s;

typedef struct chunked_log_ {
  size_t size; // total number of nodes
  size_t gran; // granularity of the elements in Bytes
  size_t nb_elements; // number of elements with a given granularity
  size_t pos;  // position of the current (curr) node
  chunked_log_node_s *first, *curr, *last;
} chunked_log_s;

typedef struct mod_chunked_log_ {
  size_t nb_buckets; // total number of buckets
  size_t gran; // granularity of the elements in Bytes
  size_t nb_elements; // number of elements with a given granularity
  chunked_log_node_s *buckets, *bucketsEnd;
} mod_chunked_log_s;

/**
 * Thread local pool of log nodes.
 *
 * Each time a thread frees a node, rather than releasing the memory, the node
 * is placed here. Upon alloc, the system checks if there are freed nodes.
 */
static __thread chunked_log_s chunked_log_freeNode;

#define CHUNKED_LOG_INIT_AUX(log, granularity, nbElements) ({ \
  (log)->gran = granularity; \
  (log)->nb_elements = nbElements; \
})

#define CHUNKED_LOG_INIT(log, granularity, nbElements) ({ \
  CHUNKED_LOG_INIT_AUX(log, granularity, nbElements); \
  (log)->first = (log)->last = (log)->curr = NULL; \
  (log)->size = 0; \
  (log)->pos = 0; \
})

#define MOD_CHUNKED_LOG_INIT(log, granularity, nbElements, nbBuckets) ({ \
  CHUNKED_LOG_INIT_AUX(log, granularity, nbElements); \
  (log)->nb_buckets = nbBuckets; \
  (log)->buckets = NULL; \
  (log)->bucketsEnd = NULL; \
})

/**
 * Destroy all thread local nodes.
 */
#define CHUNKED_LOG_DESTROY(log) ({ \
  chunked_log_node_s *node = CHUNKED_LOG_POP(log); \
  while (node != NULL) { \
    CHUNKED_LOG_FREE(node); \
    node = CHUNKED_LOG_POP(log); \
  } \
  (log)->curr = (log)->last = (log)->first = NULL; \
  (log)->size = 0; \
})

#define MOD_CHUNKED_LOG_DESTROY(log) ({ \
  chunked_log_node_s *node = (log)->buckets; \
  while ((log)->buckets != NULL) { \
    node = (log)->buckets; \
    (log)->buckets = node->next; \
    CHUNKED_LOG_FREE(node); \
  } \
  (log)->buckets = (log)->bucketsEnd = NULL; \
})

#define CHUNKED_LOG_TEARDOWN() ({ \
  chunked_log_node_s *node; \
  node = CHUNKED_LOG_POP(&chunked_log_freeNode); \
  while (node != NULL) { \
    free(node->chunk); \
    if (node->nb_buckets > 0) free(node->p.posArray); \
    free(node); \
    node = CHUNKED_LOG_POP(&chunked_log_freeNode); \
  } \
  chunked_log_freeNode.curr = chunked_log_freeNode.last = chunked_log_freeNode.first = NULL; \
  chunked_log_freeNode.size = 0; \
})

/**
 * Truncates a portion of the log.
 */
#define CHUNKED_LOG_TRUNCATE(log, nb_chunks) ({ \
  CHUNKED_LOG_LOCAL_INST(res); \
  chunked_log_node_s *node = (log)->first; \
  chunked_log_node_s *next; \
  int i; \
  for (i = 0; i < nb_chunks; ++i) { \
    if (node == NULL) { /* no more available */ \
      break; \
    } \
    next = node->next; \
    CHUNKED_LOG_POP(log); /* removes before inserting */ \
    CHUNKED_LOG_EXTEND(&res, node); \
    node = next; \
  } \
  res; \
})

/**
 * Truncates a portion of the log.
 */
#define MOD_CHUNKED_LOG_TRUNCATE(log, nb_chunks) ({ \
  CHUNKED_LOG_LOCAL_INST(res); \
  int i; \
  for (i = 0; i < nb_chunks; ++i) { \
    if ((log)->buckets == NULL) { \
      break; \
    } \
    (log)->buckets->prev = NULL; /* invariant */ \
    chunked_log_node_s *node = (log)->buckets; \
    (log)->buckets = (log)->buckets->next; \
    node->next = NULL; \
    CHUNKED_LOG_EXTEND(&res, node); \
  } \
  if ((log)->buckets == NULL) { \
    (log)->bucketsEnd = NULL; \
  } \
  res; \
})

#define CHUNKED_LOG_IS_EMPTY(log) ({ \
  ((log)->size == 0 || ((log)->size == 1 && (log)->first->p.pos == 0)); \
})

#define MOD_CHUNKED_LOG_IS_EMPTY(log) ({ \
  int res = 1; \
  int i; \
  for (i = 0; i < (log)->nb_buckets; ++i) { \
    if ((log)->buckets != NULL) { \
      if ((log)->buckets->p.posArray[i] != 0) { \
        res = 0; \
        break; \
      } \
    } \
  } \
  res; \
})

#define MOD_CHUNKED_LOG_NODE_FREE_SPACE(node) ({ \
  int res = 0; \
  int i; \
  for (i = 0; i < node->nb_buckets; ++i) { \
    res += node->size - node->p.posArray[i]; \
  } \
  res; \
})

#define MOD_CHUNKED_LOG_IS_FLAT(log) ({ \
  int res = 1; \
  int i; \
  for (i = 0; i < (log)->nb_buckets; ++i) { \
    if ((log)->buckets != NULL && (log)->buckets != (log)->bucketsEnd) { \
      res = 0; \
      break; \
    } \
  } \
  res; \
})

/**
 * Allocs a new node with a given chunk size.
 */
#define CHUNKED_LOG_ALLOC(size_one_element, nb_elements) ({ \
  size_t size = size_one_element*nb_elements; \
  chunked_log_node_s *res = CHUNKED_LOG_FIND_FREE(size, 0); \
  if (res == NULL) { \
    res = (chunked_log_node_s*)malloc(sizeof(chunked_log_node_s)); \
    res->chunk = (char*)malloc(size); \
  } \
  res->nb_buckets = 0; \
  res->gran  = size_one_element; \
  res->size  = size; \
  res->p.pos = 0; \
  res->next  = res->prev = NULL; \
  res; \
})

/**
 * Allocs a new node with a given chunk size.
 */
#define MOD_CHUNKED_LOG_ALLOC(size_one_element, nb_elements, nbBuckets) ({ \
  size_t size = size_one_element*nb_elements; \
  /* TODO: must look for the same number of buckets also */ \
  chunked_log_node_s *res = CHUNKED_LOG_FIND_FREE(size, nbBuckets); \
  if (res == NULL) { \
    res = (chunked_log_node_s*)malloc(sizeof(chunked_log_node_s)); \
    res->chunk = (char*)malloc(size*nbBuckets); \
    res->p.posArray = (size_t*)malloc(sizeof(size_t)*nbBuckets); \
  } \
  memset(res->p.posArray, 0, sizeof(size_t)*nbBuckets); \
  res->nb_buckets = nbBuckets; \
  res->gran = size_one_element; \
  res->size = size; \
  res->next = res->prev = NULL; \
  res; \
})

#define CHUNKED_LOG_FREE(node) ({ \
  CHUNKED_LOG_EXTEND(&chunked_log_freeNode, node); \
}) \

#define CHUNKED_LOG_APPEND_AUX(log, node, el) ({ \
  chunked_log_node_s *newNode = node; \
  void *addr = node->chunk + node->p.pos; \
  memcpy(addr, el, node->gran); \
  node->p.pos += node->gran; \
  if (node->p.pos >= node->size) { \
    newNode = CHUNKED_LOG_ALLOC((log)->gran, (log)->nb_elements); \
    newNode->prev = node; \
    node->next = newNode; \
  } \
  newNode; \
})

#define MOD_CHUNKED_LOG_APPEND_AUX(log, node, el, bucket) ({ \
  chunked_log_node_s *newNode = node; \
  size_t bucketPos = bucket*node->gran; \
  size_t addrPos = node->p.posArray[bucket]*node->nb_buckets + bucketPos; \
  void *addr = (char*)node->chunk + addrPos; \
  memcpy(addr, el, node->gran); \
  node->p.posArray[bucket] += node->gran; \
  if (node->p.posArray[bucket] >= node->size && node->next == NULL) { \
    newNode = MOD_CHUNKED_LOG_ALLOC((log)->gran, (log)->nb_elements, node->nb_buckets); \
    newNode->prev = node; \
    newNode->next = NULL; \
    node->next = newNode; \
  } \
  newNode; \
})

/**
 * Appends an element in the log. The element must be a pointer.
 */
#define CHUNKED_LOG_APPEND(log, el) ({ \
  chunked_log_node_s *node = (log)->last; \
  if (node == NULL) { \
    node = CHUNKED_LOG_ALLOC((log)->gran, (log)->nb_elements); \
    CHUNKED_LOG_EXTEND((log), node); \
  } \
  chunked_log_node_s *newNode = CHUNKED_LOG_APPEND_AUX(log, node, el); \
  if (node != newNode) { \
    CHUNKED_LOG_EXTEND((log), newNode); \
  } \
})

#define MOD_CHUNKED_LOG_APPEND(log, el, addr) ({ \
  uintptr_t pos = (uintptr_t)(addr) % (log)->nb_buckets; /* TODO: use mod2 */ \
  chunked_log_node_s *node = (log)->bucketsEnd; \
  if (node == NULL) { \
    node = MOD_CHUNKED_LOG_ALLOC((log)->gran, (log)->nb_elements, (log)->nb_buckets); \
    (log)->buckets = (log)->bucketsEnd = node; \
  } \
  while (node->prev != NULL && node->p.posArray[pos] == 0) { \
    /* check if previous has space */ \
    if (node->prev->p.posArray[pos] < node->prev->size) { \
      node = node->prev; \
    } else { \
      break; \
    } \
  } \
  chunked_log_node_s *newNode = MOD_CHUNKED_LOG_APPEND_AUX(log, node, el, pos); \
  if (newNode != node) { \
    newNode->prev = node; \
    node->next = newNode; \
    (log)->bucketsEnd = newNode; \
  } \
})

/**
 * Removes the first node in the log. Does not free.
 */
#define CHUNKED_LOG_POP(log) ({ \
  chunked_log_node_s *node = (log)->first; \
  if (node != NULL) { \
    chunked_log_node_s *rem = node->next; \
    CHUNKED_LOG_REMOVE_FRONT(log, rem); \
    if ((log)->size > 0) (log)->size--; \
  } \
  node; \
})

#endif /* CHUNKED_LOG_H_GUARD_ */
