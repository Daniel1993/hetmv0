#ifndef BARRIER_H_GUARD_
#define BARRIER_H_GUARD_

typedef struct ticket_barrier_ ticket_barrier_t;
struct ticket_barrier_
{
	unsigned count_next;
	unsigned total;
	union
	{
		struct
		{
			unsigned count_in;
			unsigned count_out;
		};
		unsigned long long reset;
	};
};

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void ticket_barrier_init(ticket_barrier_t *b, unsigned count);
void ticket_barrier_destroy(ticket_barrier_t *b);
int ticket_barrier_cross(ticket_barrier_t *b);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* BARRIER_H_GUARD_ */
