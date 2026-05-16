#ifndef aprendizado_operadores_h
#define aprendizado_operadores_h

#include <vector>
#include <map>

#include "nsga2.h"

typedef void (*operador_cruzamento_ptr)(sol *, sol *, sol *, sol *);
typedef void (*operador_mutacao_ptr)(sol *, float);
typedef std::map<unsigned int,float> seletor_operador;

struct op_cruzamento
{
	static unsigned int num;
	static operador_cruzamento_ptr operador[1];
	static seletor_operador probabilidade;
	static seletor_operador prob_op_uniforme;
};

struct op_mutacao
{
	static unsigned int num;
	static operador_mutacao_ptr operador[2];
	static seletor_operador probabilidade;
	static seletor_operador prob_op_uniforme;
};

operador_cruzamento_ptr seletor_roulette_wheel_cruzamento();
operador_mutacao_ptr seletor_roulette_wheel_mutacao();

#endif
