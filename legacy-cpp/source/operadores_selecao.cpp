//operadores de selecao
#include "../headers/operadores_selecao.h"
#include <cstdlib>

// Selecao por torneio com reposicao. Um campeao que foi derrotado pelo desafiante 
// pode ser selecionado em rodadas futuras para um novo desafio. Na pratica isso 
// significa que o antigo "campeao" perdera com 100% de certeza do novo "campeao", 
// pois as solucoes nao sao melhoradas ao longo da disputa do torneio. Como resultado, 
// a pressao seletiva eh menor, pois um antigo campeao selecionado novamente significa 
// uma oportunidade perdida de selecionar uma solucao que teria "mais chances" de derrotar 
// o atual campeao. Na versao com reposicao, a pressao seletiva eh maior, pois cada novo 
// desafio implica um desafiante diferente, aumentando as chances de selecionar individuos 
// melhores.

// OBS.: Menor pressao seletiva geralmente retarda a convergencia e aumenta a diversidade, 
// pois as chances de selecao dos individuos piores sao mais equilibradas em relacao aquelas 
// dos individuos melhores (quanto maior o numero de competidores, maior a chance de escolher 
// individuos melhores, diminuindo a chance dos individuos piores). Consequentemente, mais 
// individuos diferentes dos melhores tem a possibilidade de sairem vencedor de um torneio. 
// Na pratica, isso significa que um subconjunto maior de pais podem ser utilizados nas operacoes 
// de cruzamento, aumentando as possibilidades de transmissao de caracteristicas para as geracoes futuras.

// Pegue o tenis como exemplo. Federer, Nadal e Djokovic sempre ganham os torneios importantes.
// Logo, a atencao da midia se volta toda para eles. Logo, a proxima geracao de tenistas, que 
// tentam melhorar as tecnicas e estilos da geracao atual, vai tender a incorporar muito mais
// uma mistura dos estilos desses tres tenistas e a ignorar possiveis boas caracteristicas de 
// tenistas que, por infortunio, nao se saiam bem nos torneios. Exemplo, Bellucci possui um jogo 
// muito bom para Saibro em alta altitude (em que a velocidade da bola eh maior). Resultado, 
// algo bom nao sera copiado.

unsigned int sel_torneio_CD(std::vector<sol*>& P, unsigned int K)
{
	// Realiza K sorteios de solucoes aleatorias, descarta as piores (por criterios de
	// nao-dominancia e crowding distance) e mantem a melhor dentre todas as K solucoes sorteadas.

	unsigned int campeao = (int)(((float)rand() / RAND_MAX)*(P.size() - 1));

	for (unsigned int k = 1; k < K; ++k)
	{
		unsigned int desafiante;

		// Sorteia um desafiante diferente do vencedor.
		do
		{
			desafiante = (int)(((float)rand() / RAND_MAX)*(P.size() - 1));
		} while(desafiante == campeao);

		// Desafiante pertence a uma classe menor. Temos um novo campeao.
		if (P[desafiante]->Pareto_rank < P[campeao]->Pareto_rank)
			campeao = desafiante;

		// Se classes sao iguais, desempate por crowding distance.
		else if (P[desafiante]->Pareto_rank == P[campeao]->Pareto_rank)
			// CD do desafiante eh maior. Temos um novo campeao.
			if (P[desafiante]->cd > P[campeao]->cd)
				campeao = desafiante;
	}

	return campeao;
}

unsigned int sel_torneio_Delta_S(std::vector<sol*>& P, unsigned int K)
{

	// Realiza K sorteios de solucoes aleatorias, descarta as piores (por criterios de 
	// nao-dominancia e crowding distance) e mantem a melhor dentre todas as K solucoes sorteadas.

	unsigned int campeao = (int)(((float)rand() / RAND_MAX)*(P.size() - 1));

	for (unsigned int k = 1; k < K; ++k)
	{
		unsigned int desafiante;

		// Sorteia um desafiante diferente do vencedor.
		do
		{
			desafiante = (int)(((float)rand() / RAND_MAX)*(P.size() - 1));
		} while(desafiante == campeao);

		// Desafiante pertence a uma classe menor. Temos um novo campeao.
		if (P[desafiante]->Pareto_rank < P[campeao]->Pareto_rank)
			campeao = desafiante;

		// Se classes sao iguais, desempate por crowding distance.
		else if (P[desafiante]->Pareto_rank == P[campeao]->Pareto_rank)
			// Delta S do desafiante eh maior. Temos um novo campeao.
			if (P[desafiante]->Delta_S > P[campeao]->Delta_S)
				campeao = desafiante;
	}

	return campeao;
}
