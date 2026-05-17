#include <map>
#include <vector>
#include <algorithm>

#include "../headers/asms_emoa.h"
#include "../headers/learning_operators.h"

operator_selector xover_op::uniform_prob_op;
operator_selector xover_op::probability;
operator_selector mutation_op::uniform_prob_op;
operator_selector mutation_op::probability;


mutation_operator_ptr roulette_wheel_selection_mutation()
{
	float u = rand()/(float)RAND_MAX;

	operator_selector::iterator i = mutation_op::probability.begin();
	operator_selector::iterator end = mutation_op::probability.end();

	float acc = i->second;
	for (; i != end && acc < u; ++i, acc += i->second);

	if (i == end)
		--i;

	return mutation_op::_operator[i->first];
}

crossover_operator_ptr roulette_wheel_selection_crossover()
{
	float u = rand()/(float)RAND_MAX;

	operator_selector::iterator i = xover_op::probability.begin();
	operator_selector::iterator end = xover_op::probability.end();

	float acc = i->second;

	for (; i != end && acc < u; ++i, acc += i->second);

	if (i == end)
		--i;

	return xover_op::_operator[i->first];

}
