#ifndef learning_operators_h
#define learning_operators_h

#include <vector>
#include <map>

#include "asms_emoa.h"

typedef void (*crossover_operator_ptr)(sol *, sol *, sol *, sol *);
typedef void (*mutation_operator_ptr)(sol *, float);
typedef std::map<unsigned int,float> operator_selector;

struct xover_op
{
	static unsigned int num;
	static crossover_operator_ptr _operator[1];
	static operator_selector probability;
	static operator_selector uniform_prob_op;
};

struct mutation_op
{
	static unsigned int num;
	static mutation_operator_ptr _operator[4];
	static operator_selector probability;
	static operator_selector uniform_prob_op;
};

crossover_operator_ptr roulette_wheel_selection_crossover();
mutation_operator_ptr roulette_wheel_selection_mutation();

#endif
