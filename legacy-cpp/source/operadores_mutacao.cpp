#include <map> // Para a estrutura std::pair
#include <ctime>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm> // Para a funcao random_shuffle() de C++

#include "../headers/aprendizado_operadores.h"
#include "../headers/operadores_mutacao.h"
#include "../headers/nsga2.h"
#include "../headers/utils.h"

// Array de ponteiros para as funcoes que implementam os operadores de cruzamento
unsigned int op_mutacao::num = 2;
operador_mutacao_ptr op_mutacao::operador[2] = {&modifica_investimento, &modifica_portfolio};//, &aumenta_entropia};

/* DIMINUI INVESTIMENTO
 * Diminui a alocação relativa de investimento para um dado papel.
 */
void diminui_investimento(sol *filho, unsigned int asset, float min, float max)
{
	float fator = min + uniform_zero_one()*(max - min);

	if (filho->P.investment(asset) > .0)
		filho->P.investment(asset) *= fator;
}

/* AUMENTA INVESTIMENTO
 * Aumenta a alocação relativa de investimento para um dado papel.
 */
void aumenta_investimento(sol *filho, unsigned int asset, float min, float max)
{
	float fator = 1.0 + (min + uniform_zero_one()*(max - min));

	if (filho->P.investment(asset) > .0)
		filho->P.investment(asset) *= fator;
}

/* MODIFICA INVESTIMENTO
 *  Modifica o investimento de forma aleatória, com uma probabilidade de mutação.
 */
void modifica_investimento(sol *filho, float p)
{
	for (unsigned int i = 0; i < filho->P.available_assets_size; ++i)
	{
		if (filho->P.investment(i) > 0.0)
			if (uniform_zero_one() < p)
			{
				if (uniform_zero_one() < .5)
					aumenta_investimento(filho,i, 0.1, 0.5);
				else
					diminui_investimento(filho,i, 0.1, 0.5);
			}
	}

	aplica_threshold(filho, 0.01f);
}

void remover_papel(sol *filho, unsigned int asset)
{
	filho->P.investment(asset) = 0.0;
	filho->P.cardinality -= 1;
}

void adicionar_papel(sol *filho, unsigned int asset)
{
	float investimento_uniforme = 1.0/(filho->P.cardinality + 1);
	float u = -.1 + uniform_zero_one()*.2;
	filho->P.investment(asset) = (1 + u)*investimento_uniforme;
	filho->P.cardinality += 1;
}

/* MODIFICA INVESTIMENTO
 *  Modifica o investimento de forma aleatória, com uma probabilidade de mutação.
 */
void modifica_portfolio(sol *filho, float p)
{
	unsigned card = 0;
	for (unsigned int i = 0; i < filho->P.available_assets_size; ++i)
		if (filho->P.investment(i) > 0.0)
			card++;
	filho->P.cardinality = card;


	for (unsigned int i = 0; i < filho->P.available_assets_size; ++i)
	{
		if (uniform_zero_one() < p)
		{
			if (filho->P.investment(i) > 0.0 && card > 2)
				remover_papel(filho,i);
			else if (card < portfolio::max_cardinality)
				adicionar_papel(filho,i);
		}

	}

	aplica_threshold(filho, 0.01f);

	if (portfolio::card(filho->P) < 2)
	{
		std::cout << "[m]portfolio::card(filho->P)= " << portfolio::card(filho->P) << std::endl;
		std::cout << filho->P.investment << std::endl;
		std::cout << "[m]portfolio::card(filho->P)= " << portfolio::card(filho->P) << std::endl;
		system("pause");
	}
}

void aplica_threshold(sol *filho, float h)
{
	normalize(filho->P.investment);

	for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
		if (filho->P.investment(i) < h)
			remover_papel(filho,i);
	normalize(filho->P.investment);

}

void aumenta_entropia(sol *filho, float dummy)
{
	unsigned int max_elem = 0;
	for (unsigned int i = 1; i < portfolio::available_assets_size; ++i)
		if (filho->P.investment(i) > filho->P.investment(max_elem))
			max_elem = i;
	diminui_investimento(filho,max_elem, 0.1, 0.5);
	aplica_threshold(filho, 0.01f);
}


