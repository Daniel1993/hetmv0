#include "TestEvent.hpp"

using namespace std;

CPPUNIT_TEST_SUITE_REGISTRATION(TestEvent);

TestEvent::TestEvent()  { }
TestEvent::~TestEvent() { }

void TestEvent::setUp()
{
  // TODO
}

void TestEvent::tearDown()
{
}

void TestEvent::TestTODO()
{
  char msg[1024];
  CPPUNIT_ASSERT(true);
  // CPPUNIT_ASSERT_MESSAGE(msg, 5 == 6);
}
