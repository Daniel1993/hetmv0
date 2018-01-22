#ifndef SHARED_GUARD_H_
#define SHARED_GUARD_H_

#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* ################################################################### *
 * BARRIER
 * ################################################################### */

#define barrier_t             pthread_barrier_t
#define barrier_init(b, n)    pthread_barrier_init(&b, NULL, n)
#define barrier_destroy(b)    pthread_barrier_destroy(&b)
#define barrier_cross(b)      pthread_barrier_wait(&b)

// -------------

#define thread_create_or_die(thr, attr, callback, arg) \
  if (pthread_create(thr, attr, callback, (void *)(arg)) != 0) { \
    fprintf(stderr, "Error creating thread at " __FILE__ ":%i\n", __LINE__); \
    exit(1); \
  } \
//

#define thread_join_or_die(thr, res) \
  if (pthread_join(thr, res)) { \
    fprintf(stderr, "Error joining thread at " __FILE__ ":%i\n", __LINE__); \
    exit(1); \
  } \
//

#define malloc_or_die(var, nb) \
  if ((var = (__typeof__(var))malloc((nb) * sizeof(__typeof__(*var)))) == NULL) { \
    fprintf(stderr, "malloc error \"%s\" at " __FILE__":%i\n", \
      strerror(errno), __LINE__); \
    exit(1); \
  } \
//

// -------------

void bindThread(long threadId);
void sigCatcher(int sig);

#endif /* SHARED_GUARD_H_ */
