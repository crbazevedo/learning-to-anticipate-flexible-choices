#include "../headers/selection_operators.h"
#include <cstdlib>

unsigned int tournament_selection_CD(std::vector<sol*>& P, unsigned int K)
{

	unsigned int champion = (int)(((float)rand() / RAND_MAX)*(P.size() - 1));

	for (unsigned int k = 1; k < K; ++k)
	{
		unsigned int challenger;

		do
		{
			challenger = (int)(((float)rand() / RAND_MAX)*(P.size() - 1));
		} while(challenger == champion);

		if (P[challenger]->Pareto_front < P[champion]->Pareto_front)
			champion = challenger;

		else if (P[challenger]->Pareto_front == P[champion]->Pareto_front)
			if (P[challenger]->cd > P[champion]->cd)
				champion = challenger;
	}

	return champion;
}

unsigned int tournament_Delta_S(std::vector<sol*>& P, const std::pair<double,double>& error, unsigned int K)
{
	unsigned int champion = (int)(((float)rand() / RAND_MAX)*(P.size() - 1));

	for (unsigned int k = 1; k < K; ++k)
	{
		unsigned int challenger;

		do
		{
			challenger = (int)(((float)rand() / RAND_MAX)*(P.size() - 1));
		} while(challenger == champion);


		if (P[challenger]->Pareto_front < P[champion]->Pareto_front)
			champion = challenger;
		else if (P[challenger]->Pareto_front == P[champion]->Pareto_front)
			if (P[challenger]->Delta_S > P[champion]->Delta_S)
				champion = challenger;
	}

	return champion;
}
