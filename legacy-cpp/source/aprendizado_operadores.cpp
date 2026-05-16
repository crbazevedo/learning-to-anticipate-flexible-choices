#include <map>
#include <vector>
#include <algorithm>

#include "../headers/nsga2.h"
#include "../headers/aprendizado_operadores.h"

seletor_operador op_cruzamento::prob_op_uniforme;
seletor_operador op_cruzamento::probabilidade;
seletor_operador op_mutacao::prob_op_uniforme;
seletor_operador op_mutacao::probabilidade;


operador_mutacao_ptr seletor_roulette_wheel_mutacao()
{
	float u = rand()/(float)RAND_MAX;

	seletor_operador::iterator i = op_mutacao::probabilidade.begin();
	seletor_operador::iterator end = op_mutacao::probabilidade.end();

	float acc = i->second;

	for (; i != end && acc < u; ++i, acc += i->second);

	if (i == end) --i;

	//std::cout << "OP escolhido: " << i->first << std::endl;

	return op_mutacao::operador[i->first];
}

operador_cruzamento_ptr seletor_roulette_wheel_cruzamento()
{
	float u = rand()/(float)RAND_MAX;

	seletor_operador::iterator i = op_cruzamento::probabilidade.begin();
	seletor_operador::iterator end = op_cruzamento::probabilidade.end();

	float acc = i->second;

	for (; i != end && acc < u; ++i, acc += i->second);

	if (i == end) --i;

	//std::cout << "OP escolhido: " << i->first << std::endl;

	return op_cruzamento::operador[i->first];

}
