#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../headers/utils.h"
#include "../headers/operadores_mutacao.h"
#include "../headers/operadores_cruzamento.h"
#include "../headers/aprendizado_operadores.h"

// Array de ponteiros para as funcoes que implementam os operadores de cruzamento
unsigned int op_cruzamento::num = 1;
operador_cruzamento_ptr op_cruzamento::operador[1] = {&cruzamento_uniforme};

void cruzamento_uniforme(sol *pai1, sol *pai2, sol *filho1, sol *filho2)
{
	std::vector<unsigned int> index;
	for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
	{
		filho1->P.investment(i) = filho2->P.investment(i) = 0.0;
		if (pai1->P.investment(i) > 0.0 || pai2->P.investment(i) > 0.0)
			index.push_back(i);
	}

	unsigned int card1 = 0, card2 = 0;

	std::random_shuffle(index.begin(),index.end());
	for (unsigned int i = 0; i < index.size(); ++i)
	{
		if (i < portfolio::max_cardinality)
		{
			if (uniform_zero_one() < .5)
			{
				filho1->P.investment(index[i]) = pai1->P.investment(index[i]);
				filho2->P.investment(index[i]) = pai2->P.investment(index[i]);
				if (filho1->P.investment(index[i]) > 0.0)
					card1++;
				if (filho2->P.investment(index[i]) > 0.0)
					card2++;
			}
			else
			{
				filho1->P.investment(index[i]) = pai2->P.investment(index[i]);
				filho2->P.investment(index[i]) = pai1->P.investment(index[i]);
				if (filho1->P.investment(index[i]) > 0.0)
					card1++;
				if (filho2->P.investment(index[i]) > 0.0)
					card2++;
			}
		}
		else
			break;
	}

	filho1->P.cardinality = card1;
	filho2->P.cardinality = card2;

	while (portfolio::card(filho1->P) < 2)
	{
		for (unsigned c = portfolio::card(filho1->P); c < 2; ++c)
		{
			adicionar_papel(filho1,rand()%portfolio::available_assets_size);
			aplica_threshold(filho1->P.investment, 0.01f);
		}
		//aplica_threshold(filho1->P.investment, 0.01f);
		//aplica_threshold(filho2->P.investment, 0.01f);
	}

	while (portfolio::card(filho2->P) < 2)
		{
			for (unsigned c = portfolio::card(filho2->P); c < 2; ++c)
			{
				adicionar_papel(filho2,rand()%portfolio::available_assets_size);
				aplica_threshold(filho2->P.investment, 0.01f);
			}
			//aplica_threshold(filho1->P.investment, 0.01f);
			//aplica_threshold(filho2->P.investment, 0.01f);
		}

	if (portfolio::card(filho1->P) < 2)
	{
		std::cout << "portfolio::card(filho1->P)= " << portfolio::card(filho1->P) << std::endl;
		std::cout << filho1->P.investment;
		std::cout << "portfolio::card(filho1->P)= " << portfolio::card(filho1->P) << std::endl;
		system("pause");
	}

	if (portfolio::card(filho2->P) < 2)
	{
		std::cout << "portfolio::card(filho1->P)= " << portfolio::card(filho2->P) << std::endl;
		std::cout << filho1->P.investment;
		std::cout << "portfolio::card(filho1->P)= " << portfolio::card(filho2->P) << std::endl;
		system("pause");
	}

}
