#include <map>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include "../headers/learning_operators.h"
#include "../headers/mutation_operators.h"
#include "../headers/asms_emoa.h"
#include "../headers/utils.h"

// Pointers array for functions implementing mutation operators
unsigned int mutation_op::num = 4;
mutation_operator_ptr mutation_op::_operator[4] = {&modify_investment, &modify_portfolio, &raise_entropy, &lower_entropy};

/* Decreases investment
 * Decreases allocation for a given asset.
 */
void reduce_investment(sol *offspring, unsigned int asset, float min, float max)
{
	float fator = min + uniform_zero_one()*(max - min);

	if (offspring->P.investment(asset) > .0)
		offspring->P.investment(asset) *= fator;
	offspring->anticipation = false;
	apply_threshold(offspring->P.investment, portfolio::min_hold, portfolio::max_hold);
}

/* Raises investment
 * Raises allocation for a given asset.
 */
void raise_investment(sol *offspring, unsigned int asset, float min, float max)
{
	float fator = 1.0 + (min + uniform_zero_one()*(max - min));

	if (offspring->P.investment(asset) > .0)
		offspring->P.investment(asset) *= fator;
	offspring->anticipation = false;
	apply_threshold(offspring->P.investment, portfolio::min_hold, portfolio::max_hold);
}

/* Modify investment
 *  Randomly modifies investment, according to a mutation rate p
 */
void modify_investment(sol *offspring, float p)
{
	for (int i = 0; i < offspring->P.investment.rows(); ++i)
	{
		if (offspring->P.investment(i) > 0.0)
			if (uniform_zero_one() < p)
			{
				if (uniform_zero_one() < .5)
					raise_investment(offspring,i, 0.05, 0.25);
				else
					reduce_investment(offspring,i, 0.05, 0.25);
			}
	}

	offspring->anticipation = false;
}

void remove_asset(sol *offspring, unsigned int asset)
{
	offspring->P.investment(asset) = 0.0;
	offspring->P.cardinality -= 1;
	offspring->anticipation = false;
	apply_threshold(offspring->P.investment, portfolio::min_hold, portfolio::max_hold);
}

void add_asset(sol *offspring, unsigned int asset)
{
	float investimento_uniforme = 1.0/(portfolio::card(offspring->P) + 1);
	float u = -.1 + uniform_zero_one()*.2;
	offspring->P.investment(asset) = (1 + u)*investimento_uniforme;
	offspring->P.cardinality += 1;
	offspring->anticipation = false;
	apply_threshold(offspring->P.investment, portfolio::min_hold, portfolio::max_hold);
}

/* Modify investment
 *  Randomly modifies the portfolio, according to a given mutation rate p.
 */
void modify_portfolio(sol *offspring, float p)
{
	unsigned card = offspring->P.investment.count();

	offspring->P.cardinality = card;


	for (int i = 0; i < offspring->P.investment.rows(); ++i)
	{
		if (uniform_zero_one() < p)
		{
			if (uniform_zero_one() < 0.5 && offspring->P.investment(i) > 0.0 && card > portfolio::min_cardinality)
				remove_asset(offspring,i);
			else if (card < portfolio::max_cardinality)
				add_asset(offspring,i);
		}
	}

	bool cardinality = false;
	while (portfolio::card(offspring->P) < portfolio::min_cardinality)
	{
		for (unsigned c = portfolio::card(offspring->P); c < portfolio::min_cardinality; ++c)
		{
			add_asset(offspring,rand()%portfolio::num_assets);
			if (static_cast<unsigned>(offspring->P.investment.count()) > portfolio::min_cardinality)
				prune_threshold(offspring, portfolio::min_hold);
			cardinality = true;
		}

	}

	offspring->anticipation = false;

	if (!cardinality)
		apply_threshold(offspring->P.investment, portfolio::min_hold, portfolio::max_hold);

}

void prune_threshold(sol *offspring, float l)
{
	normalize(offspring->P.investment);

	bool removed = false;
	for (int i = 0; i < offspring->P.investment.rows(); ++i)
	{
		unsigned int card = offspring->P.investment.count();
		if (offspring->P.investment(i) < l && card > portfolio::min_cardinality)
		{
			remove_asset(offspring,i);
			removed = true;
		}
	}

	if (removed)
		normalize(offspring->P.investment);
	offspring->anticipation = false;

}

void raise_entropy(sol *offspring, float dummy)
{
	unsigned int max_elem = 0;
	for (int i = 1; i < offspring->P.investment.rows(); ++i)
		if (offspring->P.investment(i) > offspring->P.investment(max_elem))
			max_elem = i;
	reduce_investment(offspring,max_elem, 0.05, 0.25);
	apply_threshold(offspring->P.investment, portfolio::min_hold, portfolio::max_hold);

	offspring->anticipation = false;
}

void lower_entropy(sol *offspring, float dummy)
{
	unsigned int max_elem = 0;
	for (int i = 1; i < offspring->P.investment.rows(); ++i)
		if (offspring->P.investment(i) > offspring->P.investment(max_elem))
			max_elem = i;
	raise_investment(offspring,max_elem, 0.05, 0.25);
	apply_threshold(offspring->P.investment, portfolio::min_hold, portfolio::max_hold);

	offspring->anticipation = false;
}


