#ifndef TEST_EVENT_H
#define TEST_EVENT_H

#include <cppunit/TestCase.h>
#include <cppunit/TestFixture.h>
#include <cppunit/ui/text/TextTestRunner.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/XmlOutputter.h>
#include <netinet/in.h>
#include <chrono>

#include "pdes.h"

class TestEvent : public CPPUNIT_NS::TestFixture
{
	CPPUNIT_TEST_SUITE(TestEvent);
	CPPUNIT_TEST(TestTODO);
	CPPUNIT_TEST_SUITE_END();

public:
	TestEvent();
	virtual ~TestEvent();
	void setUp();
	void tearDown();

private:
	void TestTODO();
} ;

#endif /* TEST_EVENT_H */
