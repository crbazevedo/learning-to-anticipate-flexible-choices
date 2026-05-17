#ifndef MUTATION_OPERATORS_H
#define MUTATION_OPERATORS_H

#include "asms_emoa.h"
#include "learning_operators.h"

void reduce_investment(sol *offspring, unsigned int asset, float min, float max);
void raise_investment(sol *offspring, unsigned int asset, float min, float max);
void add_asset(sol *offspring, unsigned int asset);
void remove_asset(sol *offspring, unsigned int asset);
void modify_investment(sol *offspring, float p);
void modify_portfolio(sol *offspring, float p);
void raise_entropy(sol *offspring, float dummy);
void lower_entropy(sol *offspring, float dummy);
void prune_threshold(sol *offspring, float l);

#endif
