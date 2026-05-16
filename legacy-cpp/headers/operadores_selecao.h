#include <vector>

#ifndef OPERADORES_H
#define OPERADORES_H

#include "../headers/nsga2.h"

unsigned int sel_torneio_CD(std::vector<sol*>&, unsigned int);
unsigned int sel_torneio_Delta_S(std::vector<sol*>&, unsigned int);

#endif
