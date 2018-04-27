#include "TestLog.h"

#include "chunked-log.h"

#include <thread>

CPPUNIT_TEST_SUITE_REGISTRATION(TestLog);

using namespace std;

static void testChunking();

TestLog::TestLog()  { }
TestLog::~TestLog() { }

void TestLog::setUp() { }

void TestLog::tearDown() { }

void TestLog::TestChunk() {
  int nbElems = 10;
  chunked_log_s log;

  CHUNKED_LOG_INIT(&log, sizeof(int), nbElems);

  int i;
for (i = 0; i < nbElems; ++i) {
    CHUNKED_LOG_APPEND(&log, &i);
  }
  CPPUNIT_ASSERT(log.size == 2);
  CPPUNIT_ASSERT(log.first != log.last); // augmented by 1
  CPPUNIT_ASSERT(log.first->p.pos == log.first->size);
  CPPUNIT_ASSERT(log.last->p.pos == 0);

  CHUNKED_LOG_DESTROY(&log);
}

void TestLog::TestChunking() {
  thread t1(testChunking), t2(testChunking);

  testChunking();

  t1.join();
  t2.join();
}

void TestLog::TestTruncate() {
  int nbElems = 10;
  chunked_log_s log;
  chunked_log_s truncated;

  CHUNKED_LOG_INIT(&log, sizeof(int), nbElems);
  int i;
for (i = 0; i < 4*nbElems; ++i) {
    CHUNKED_LOG_APPEND(&log, &i);
  }
  CPPUNIT_ASSERT(log.size == 5);

  truncated = CHUNKED_LOG_TRUNCATE(&log, 2); //
  CPPUNIT_ASSERT(log.size == 3);
  CPPUNIT_ASSERT(truncated.size == 2);

  int i;
for (i = 0; i < nbElems; ++i) {
    CPPUNIT_ASSERT(((int*)truncated.first->chunk)[i] == i);
  }
  int i;
for (i = 0; i < nbElems; ++i) {
    CPPUNIT_ASSERT(((int*)truncated.first->next->chunk)[i] == i+nbElems);
  }
  int i;
for (i = 0; i < nbElems; ++i) {
    CPPUNIT_ASSERT(((int*)log.first->chunk)[i] == i+2*nbElems);
  }
  int i;
for (i = 0; i < nbElems; ++i) {
    CPPUNIT_ASSERT(((int*)log.first->next->chunk)[i] == i+3*nbElems);
  }

  CHUNKED_LOG_FREE(truncated.first);
  CHUNKED_LOG_FREE(truncated.last);

  int i;
for (i = 0; i < 2*nbElems; ++i) {
    CHUNKED_LOG_APPEND(&log, &i);
  }
  truncated = CHUNKED_LOG_TRUNCATE(&log, 2); //
  CPPUNIT_ASSERT(log.size == 3);
  CPPUNIT_ASSERT(truncated.size == 2);

  int i;
for (i = 0; i < nbElems; ++i) { // these are the old ones
    CPPUNIT_ASSERT(((int*)truncated.first->chunk)[i] == i+2*nbElems);
  }
  int i;
for (i = 0; i < nbElems; ++i) {
    CPPUNIT_ASSERT(((int*)truncated.first->next->chunk)[i] == i+3*nbElems);
  }
  int i;
for (i = 0; i < nbElems; ++i) { // these are the new ones
    CPPUNIT_ASSERT(((int*)log.first->chunk)[i] == i);
  }
  int i;
for (i = 0; i < nbElems; ++i) {
    CPPUNIT_ASSERT(((int*)log.first->next->chunk)[i] == i+nbElems);
  }
}

void TestLog::TestModLog() {
  int nbElems = 5;
  int nbBuckets = 8;
  mod_chunked_log_s log;
  chunked_log_s trnc;
  char truncated[nbBuckets*nbElems*sizeof(int)];
  char truncated3[3*nbBuckets*nbElems*sizeof(int)];

  MOD_CHUNKED_LOG_INIT(&log, sizeof(int), nbElems, nbBuckets);
  int i;
for (i = 0; i < 4*nbElems*nbBuckets; ++i) {
    MOD_CHUNKED_LOG_APPEND(&log, &i, i);
  }

  trnc = MOD_CHUNKED_LOG_TRUNCATE(&log, 1);
  memcpy(truncated, trnc.first->chunk, nbBuckets*nbElems*sizeof(int));
  CPPUNIT_ASSERT(log.buckets->next->next->next == log.bucketsEnd);
  CPPUNIT_ASSERT(log.bucketsEnd->prev->prev->prev == log.buckets);

  CPPUNIT_ASSERT(((int*)truncated)[0] == 0);
  CPPUNIT_ASSERT(((int*)truncated)[8] == 8);
  CPPUNIT_ASSERT(((int*)truncated)[16] == 16);
  CPPUNIT_ASSERT(((int*)truncated)[24] == 24);
  CPPUNIT_ASSERT(((int*)truncated)[32] == 32);
  CPPUNIT_ASSERT(((int*)truncated)[1] == 1);
  CPPUNIT_ASSERT(((int*)truncated)[2] == 2);
  MOD_CHUNKED_LOG_TRUNCATE(&log, 3); // log is flat
  CPPUNIT_ASSERT(MOD_CHUNKED_LOG_IS_FLAT(&log) == 1); // one layer of empties
  trnc = MOD_CHUNKED_LOG_TRUNCATE(&log, 1); // log is empty (need to remove an empty block)
  CPPUNIT_ASSERT(MOD_CHUNKED_LOG_IS_EMPTY(&log) == 1);

  int i;
for (i = 0; i < 4*nbElems*nbBuckets + 4*nbElems; ++i) {
    MOD_CHUNKED_LOG_APPEND(&log, &i, i);
  }
  CPPUNIT_ASSERT(log.buckets->next->next->next->next->next == log.bucketsEnd);
  CPPUNIT_ASSERT(log.bucketsEnd->prev->prev->prev->prev->prev == log.buckets);
  trnc = MOD_CHUNKED_LOG_TRUNCATE(&log, 4); // 0 has 2
  CPPUNIT_ASSERT(MOD_CHUNKED_LOG_IS_FLAT(&log) == 0);
  trnc = MOD_CHUNKED_LOG_TRUNCATE(&log, 1); // log is flat
  CPPUNIT_ASSERT(MOD_CHUNKED_LOG_IS_FLAT(&log) == 1);
  trnc = MOD_CHUNKED_LOG_TRUNCATE(&log, 1); // log is empty
  CPPUNIT_ASSERT(MOD_CHUNKED_LOG_IS_EMPTY(&log) == 1);
}

static void testChunking() {
  int nbElems = 10;
  chunked_log_s log;
  chunked_log_node_s *node;

  CHUNKED_LOG_INIT(&log, sizeof(int), nbElems);
  int i;
for (i = 0; i < nbElems; ++i) {
    CHUNKED_LOG_APPEND(&log, &i);
  }
  CPPUNIT_ASSERT(log.size == 2);
  CPPUNIT_ASSERT(log.first != log.last); // augmented by 1
  CPPUNIT_ASSERT(log.first->next == log.last);
  CPPUNIT_ASSERT(log.last->prev == log.first);
  CPPUNIT_ASSERT(log.first->p.pos == log.first->size);
  CPPUNIT_ASSERT(log.last->p.pos == 0);
  int i;
for (i = 0; i < nbElems; ++i) {
    CHUNKED_LOG_APPEND(&log, &i);
  }
  CPPUNIT_ASSERT(log.size == 3);
  CPPUNIT_ASSERT(log.last->prev->p.pos == log.last->prev->size);
  CPPUNIT_ASSERT(log.last->p.pos == 0);
  CPPUNIT_ASSERT(log.first->next->next == log.last);
  CPPUNIT_ASSERT(log.last->prev->prev == log.first);
  int i;
for (i = 0; i < nbElems; ++i) {
    CHUNKED_LOG_APPEND(&log, &i);
  }
  CPPUNIT_ASSERT(log.size == 4);
  CPPUNIT_ASSERT(log.last->prev->p.pos == log.last->prev->size);
  CPPUNIT_ASSERT(log.last->p.pos == 0);
  CPPUNIT_ASSERT(log.first->next->next->next == log.last);
  CPPUNIT_ASSERT(log.last->prev->prev->prev == log.first);
  int i;
for (i = 0; i < nbElems; ++i) {
    CHUNKED_LOG_APPEND(&log, &i);
  }
  CPPUNIT_ASSERT(log.size == 5);
  CPPUNIT_ASSERT(log.last->prev->p.pos == log.last->prev->size);
  CPPUNIT_ASSERT(log.last->p.pos == 0);
  CPPUNIT_ASSERT(log.first->next->next->next->next == log.last);
  CPPUNIT_ASSERT(log.last->prev->prev->prev->prev == log.first);

  node = CHUNKED_LOG_POP(&log);
  CPPUNIT_ASSERT(log.size == 4);
  CPPUNIT_ASSERT(log.first->prev == NULL);
  CHUNKED_LOG_FREE(node);
  CPPUNIT_ASSERT(chunked_log_freeNode.size == 1);
  CPPUNIT_ASSERT(chunked_log_freeNode.first == node);
  CPPUNIT_ASSERT(chunked_log_freeNode.last == node);
  CPPUNIT_ASSERT(chunked_log_freeNode.last->next == NULL);
  CPPUNIT_ASSERT(chunked_log_freeNode.first->prev == NULL);

  node = CHUNKED_LOG_POP(&log);
  CHUNKED_LOG_FREE(node);
  CPPUNIT_ASSERT(log.size == 3);
  CPPUNIT_ASSERT(chunked_log_freeNode.size == 2);
  CPPUNIT_ASSERT(chunked_log_freeNode.last == node);
  CPPUNIT_ASSERT(chunked_log_freeNode.last->next == NULL);
  CPPUNIT_ASSERT(chunked_log_freeNode.first->prev == NULL);

  CHUNKED_LOG_DESTROY(&log);
  CPPUNIT_ASSERT(chunked_log_freeNode.size == 0);
}
