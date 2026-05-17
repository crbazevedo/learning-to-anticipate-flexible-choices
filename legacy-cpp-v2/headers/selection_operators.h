#include <vector>

#ifndef OPERADORES_H
#define OPERADORES_H

#include "../headers/asms_emoa.h"

unsigned int tournament_selection_CD(std::vector<sol*>&, unsigned int);
unsigned int tournament_Delta_S(std::vector<sol*>&, const std::pair<double,double>&, unsigned int);

#endif
