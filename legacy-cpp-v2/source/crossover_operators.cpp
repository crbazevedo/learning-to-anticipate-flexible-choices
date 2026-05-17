#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../headers/utils.h"
#include "../headers/mutation_operators.h"
#include "../headers/crossover_operators.h"
#include "../headers/learning_operators.h"

// Pointers array for functions implementing crossover operators
unsigned int xover_op::num = 1;
crossover_operator_ptr xover_op::_operator[1] = {&uniform_crossover};

void uniform_crossover(sol *parent1, sol *parent2, sol *offspring1, sol *offspring2)
{
	std::vector<unsigned int> index;
	for (unsigned int i = 0; i < portfolio::num_assets; ++i)
	{
		offspring1->P.investment(i) = offspring2->P.investment(i) = 0.0;
		if (parent1->P.investment(i) > 0.0 || parent2->P.investment(i) > 0.0)
			index.push_back(i);
	}

	unsigned int card1 = 0, card2 = 0;

	std::random_shuffle(index.begin(),index.end());
	for (unsigned int i = 0; i < index.size(); ++i)
	{
		if (portfolio::card(offspring1->P) < portfolio::max_cardinality)
		{
			if (uniform_zero_one() < .5)
			{
				offspring1->P.investment(index[i]) = parent1->P.investment(index[i]);
				offspring2->P.investment(index[i]) = parent2->P.investment(index[i]);
				if (offspring1->P.investment(index[i]) > 0.0)
					card1++;
				if (offspring2->P.investment(index[i]) > 0.0)
					card2++;
			}
			else
			{
				offspring1->P.investment(index[i]) = parent2->P.investment(index[i]);
				offspring2->P.investment(index[i]) = parent1->P.investment(index[i]);
				if (offspring1->P.investment(index[i]) > 0.0)
					card1++;
				if (offspring2->P.investment(index[i]) > 0.0)
					card2++;
			}
		}
		else
			break;
	}

	offspring2->P.cardinality = card2;

	bool card = false;

	while (portfolio::card(offspring1->P) < portfolio::min_cardinality)
	{
		for (unsigned c = portfolio::card(offspring1->P); c < portfolio::min_cardinality; ++c)
		{
			add_asset(offspring1,rand()%portfolio::num_assets);
			apply_threshold(offspring1->P.investment, portfolio::min_hold, portfolio::max_hold);
			card = true;
		}
	}

	while (portfolio::card(offspring2->P) < portfolio::min_cardinality)
	{
		for (unsigned c = portfolio::card(offspring2->P); c < portfolio::min_cardinality; ++c)
		{
			add_asset(offspring2,rand()%portfolio::num_assets);
			apply_threshold(offspring2->P.investment, portfolio::min_hold, portfolio::max_hold);
			card = true;
		}
	}

	if (portfolio::card(offspring1->P) < portfolio::min_cardinality)
	{
		std::cout << offspring1->P.investment;
		card = true;
	}

	if (portfolio::card(offspring2->P) < portfolio::min_cardinality)
	{
		std::cout << offspring1->P.investment;
		card = true;
	}

	if (!card)
	{
		apply_threshold(offspring1->P.investment, portfolio::min_hold, portfolio::max_hold);
		apply_threshold(offspring2->P.investment, portfolio::min_hold, portfolio::max_hold);
	}


}
