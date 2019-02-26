#ifndef CHUNKED_LOG_AUX_H_GUARD_
#define CHUNKED_LOG_AUX_H_GUARD_

#define CHUNKED_LOG_INIT_NODE(ptr) \
  (ptr)->first = NULL; \
  (ptr)->curr = NULL; \
  (ptr)->last = NULL; \
  (ptr)->size = 0; \
  (ptr)->pos = 0 \
//

#define CHUNKED_LOG_LOCAL_INST(var) \
  chunked_log_s var; \
  CHUNKED_LOG_INIT_NODE(&var) \
//

/**
 * Extends the log with a new chunk.
 */
#define CHUNKED_LOG_EXTEND(log, node) ({ \
  (node)->next = (node)->prev = NULL; \
  if ((log)->first == NULL || (log)->last == NULL) { \
    (log)->first = (log)->curr = (log)->last = node; \
    (log)->size = 1; \
    (log)->pos = 0; \
  } else { \
    (log)->last->next = node; \
    (node)->prev = (log)->last; \
    (log)->last = node; \
    (log)->size++; \
  } \
})

#define CHUNKED_LOG_REMOVE_FRONT(log) ({ \
  if ((log)->first == (log)->last) { \
    (log)->last = (log)->first = NULL; \
    (log)->size = 0; \
  } else { \
    (log)->first->next->prev = NULL; \
    (log)->first = (log)->first->next; \
  } \
})

// in case of warp --> i > chunked_log_free_ptr + SIZE_OF_FREE_NODES
#define CHUNKED_LOG_FIND_FREE(sizeNode, nbBuckets) ({ \
  chunked_log_node_s *res_ = NULL; \
  unsigned long i = chunked_log_alloc_ptr; \
  if (i < chunked_log_free_ptr) { \
    res_ = chunked_log_node_recycled[i % SIZE_OF_FREE_NODES]; /* barrier needed? */ \
    chunked_log_alloc_ptr++; \
  } \
  res_; \
})
/* #define CHUNKED_LOG_FIND_FREE(sizeNode, nbBuckets) ({ \
//   chunked_log_freeNode.curr = chunked_log_freeNode.first; \
//   chunked_log_node_s *res_ = NULL; \
//   while (chunked_log_freeNode.curr != NULL) { \
//     if (chunked_log_freeNode.curr->size == sizeNode && chunked_log_freeNode.curr->nb_buckets == nbBuckets) { \
//       if (chunked_log_freeNode.curr == chunked_log_freeNode.first) { \
//         chunked_log_freeNode.first = chunked_log_freeNode.curr->next; \
//       } \
//       if (chunked_log_freeNode.curr == chunked_log_freeNode.last) { \
//         chunked_log_freeNode.last = chunked_log_freeNode.curr->prev; \
//       } \
//       res_ = chunked_log_freeNode.curr; \
//       chunked_log_freeNode.curr->prev = chunked_log_freeNode.curr->next; \
//       res_->next = res_->prev = NULL; \
//       chunked_log_freeNode.size--; \
//       break; \
//     } \
//     chunked_log_freeNode.curr = chunked_log_freeNode.curr->next; \
//   } \
//   res_; \
// })*/

#endif /* CHUNKED_LOG_AUX_H_GUARD_ */
