#ifndef OPERADORES_MUTACAO_H
#define OPERADORES_MUTACAO_H

#include "nsga2.h"
#include "aprendizado_operadores.h"

void diminui_investimento(sol *filho, unsigned int asset, float min, float max);
void aumenta_investimento(sol *filho, unsigned int asset, float min, float max);
void adicionar_papel(sol *filho, unsigned int asset);
void remover_papel(sol *filho, unsigned int asset);
void modifica_investimento(sol *filho, float p);
void modifica_portfolio(sol *filho, float p);
void aumenta_entropia(sol *filho, float dummy);
void aplica_threshold(sol *filho, float h);

#endif
