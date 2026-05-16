#include "../headers/nsga2.h"
//#include "../headers/irp.h"
#include "../headers/operadores_selecao.h"
#include "../headers/operadores_mutacao.h"
#include "../headers/operadores_cruzamento.h"
//#include "../headers/operadores_busca_local.h"
#include "../headers/aprendizado_operadores.h"
#include "../headers/kalman_filter.h"
#include "../headers/statistics.h"
#include "../headers/portfolio.h"

#include <algorithm> // Para a funcao sort
#include <fstream>
#include <istream>
#include <ostream>
#include <limits>    // Para a struct numeric_limits
#include <vector>
#include <cmath>
#include <map>

#include <Eigen/Eigen/Cholesky>

using namespace std;

typedef std::pair<unsigned int, unsigned int> par_inteiros;

std::string solution::regularization_type;

void atribui_cd_objetivos(std::vector<sol*> classe, int m)
{

	// Numero de solucoes
	unsigned int l = classe.size();

	if (classe.size() <= 2)
		classe[0]->cd = 1.0f;

	else if (m == 0) // ordena por ROI
	{

		// Obtem os valores minimo e maximo encontrado no conjunto 'classe'
		// para o objetivo 'custo_trans'. Equivale a encontrar as solucoes extremas
		// da classe.

		std::sort (classe.begin(), classe.end(), cmp_ROI_ptr);
		float min_custo = classe.back()->P.ROI;
		float max_custo = classe.front()->P.ROI;

		/*for (unsigned int i = 0; i < classe.size(); ++i)
		{
			std::cout << "classe[" << i << "].id: " << classe[i]->id << std::endl;
		}*/

		//std::cout << "min_custo: " << min_custo << " ";
		//std::cout << "max_custo: " << max_custo << "\n";

		// Atribui CD infinito as solucoes extremas da classe.
		classe.front()->cd = (std::numeric_limits<float>::max()/2.0f)-1.0f;
		classe.back()->cd = (std::numeric_limits<float>::max()/2.0f)-1.0f;

		// Computa o CD (com os objetivos normalizados entre 0 e 1) para as demais solucoes.
		for (unsigned int i = 1; i < l-1; ++i)
		{
			float custo_norm1 = (min_custo == max_custo) ? 1 : (classe[i+1]->P.ROI - min_custo)/(max_custo - min_custo);
			float custo_norm2 = (min_custo == max_custo) ? 1 : (classe[i-1]->P.ROI - min_custo)/(max_custo - min_custo);

			classe[i]->cd +=  custo_norm2 - custo_norm1;
			//std::cout << "classe[i+1]->P.ROI: " << classe[i+1]->P.ROI << ", classe[i-1]->P.ROI: " << classe[i-1]->P.ROI << std::endl;
			//std::cout << "ROI(custo_norm1 - custo_norm2): " <<  (custo_norm1 - custo_norm2) << std::endl;
			//std::cout << "classe[" << i << "]->cd: " << classe[i]->cd << "\n";
		}

	} 
	else if (m == 1) // ordena por risco
	{

		// Obtem os valores minimo e maximo encontrado no conjunto 'classe'
		// para o objetivo 'custo_estoq'. Equivale a encontrar as solucoes extremas
		// da classe.

		std::sort (classe.begin(), classe.end(), cmp_risk_ptr);
		float min_custo = classe.front()->P.risk;
		float max_custo = classe.back()->P.risk;

		//std::cout << "min_custo: " << min_custo << " ";
		//std::cout << "max_custo: " << max_custo << "\n";


		// Atribui CD infinito as solucoes extremas da classe.
		classe.front()->cd = (std::numeric_limits<float>::max()/2.0f)-2.0f;
		classe.back()->cd = (std::numeric_limits<float>::max()/2.0f)-2.0f;

		// Computa o CD (com os objetivos normalizados entre 0 e 1) para as demais solucoes.
		for (unsigned int i = 1; i < l-1; ++i)
		{
			float custo_norm1 = (min_custo == max_custo) ? 1 : (classe[i+1]->P.risk - min_custo)/(max_custo - min_custo);
			float custo_norm2 = (min_custo == max_custo) ? 1 : (classe[i-1]->P.risk - min_custo)/(max_custo - min_custo);

			classe[i]->cd +=  (custo_norm1 - custo_norm2);
			//std::cout << "Risk(custo_norm1 - custo_norm2): " <<  (custo_norm1 - custo_norm2) << std::endl;
			//system("pause");
			//std::cout << "classe[" << i << "]->cd: " << classe[i]->cd << "\n";
		}
	}

}

// Requer que a populacao P esteja ja ordenada por classes.
// Ou seja, obrigatoriamente, o usuario deve ter executado 
// o procedimento fast_nondominated_sorting(P);
// Recebe o indice da classe para a qual o CD sera determinado.
void crowding_distance(std::vector<sol*> &P, unsigned int classe)
{

	unsigned int i = 0, inicio = 0, fim;

	// Vetor de ponteiros para os individuos na populacao 
	// que pertencem a classe de interesse.
	std::vector<sol*> minha_classe;

	// Encontra na populacao ja ordenada por classes o primeiro 
	// individuo da classe de interesse.
	while (i < P.size() && P[i]->Pareto_rank != classe)
		++i;

	inicio = i;

	//std::cout << "inicio: " << inicio << std::endl;

	// Inicializa CD para cada solucao da classe	
	while (i < P.size() && P[i]->Pareto_rank == classe)
	{
		// Armazena o endereco de memoria do individo no vetor de ponteiros.
		minha_classe.push_back(P[i]);
		// Inicializa o CD do individuo.
		P[i]->cd = 0;

		//std::cout << "P[" << i << "]->cd: " << P[i]->cd << std::endl;

		++i;
	}

	fim = i - 1;

	//std::cout << "fim: " << fim << std::endl;

	// Para cada objetivo
	atribui_cd_objetivos(minha_classe, 0);
	atribui_cd_objetivos(minha_classe, 1);

	//Ordena a classe de forma descendente por crowding distance
	std::sort(P.begin() + inicio, P.begin() + fim, cmp_cd_ptr);
}


unsigned int fast_nondominated_sort(std::vector<sol*> &P)
{
	unsigned int num_classes = 0;
	std::vector<std::vector<unsigned int> > F;
	std::vector<std::vector<unsigned int> > S(P.size());
	std::vector<unsigned int> n(P.size());

	std::vector<unsigned int>H ;

	for (unsigned int p = 0; p < P.size(); ++p)
	{
		for (unsigned int q = 0; q < P.size(); ++q)
			if (p != q)
			{
				if (domina_sem_restricoes(P[p],P[q]))
				//if (domina_com_restricoes(P[p],P[q]))
					S[p].push_back(q);
				else if (domina_sem_restricoes(P[q],P[p]))
				//else if (domina_com_restricoes(P[q],P[p]))
					++n[p];
			}

		//std::cout << "np[" << p << "] = " << n[p];
		//std::cout << ", sp[" << p << "].size() = " << S[p].size() << std::endl;

		if (n[p] == 0)
		{
			H.push_back(p);
			P[H.back()]->Pareto_rank = 0;
		}
	}
	// Obtém a classe F1, de soluções não-dominadas
	F.push_back(H);

	num_classes++;

	// Obtém as demais classes, se existirem
	unsigned int i = 0;

	while (!F[i].empty())
	{
		std::vector<unsigned int> H;

		for (unsigned int p = 0; p < F[i].size(); ++p)
		{
			for (unsigned int q = 0; q < S[F[i][p]].size(); ++q)
			{
				--n[S[F[i][p]][q]];
				if (n[S[F[i][p]][q]] == 0)
				{
					H.push_back(S[F[i][p]][q]);
					P[H.back()]->Pareto_rank = i + 1;

				}
			}
		}
		F.push_back(H);
		++i; ++num_classes;
	}

	// Garante que a populacao esta ordenada por classes, porem,  
	// dentro de uma mesma classe, os individuos nao estao ordenados
	// por crowding distance (o que sera feito posteriormente, dentro  
	// do procedimento crowding_distance()).
	std::sort(P.begin(),P.end(),cmp_classe_ptr);

	return num_classes-1;
}

void compute_Delta_S_class(std::vector<sol*> &minha_classe, float R_1, float R_2)
{

	//std::cout << "minha_classe.size(): " << minha_classe.size() << std::endl;
	if (minha_classe.size() == 1)
	{
		minha_classe[0]->Delta_S = (minha_classe[0]->P.ROI - R_1)*(R_2 - minha_classe[0]->P.risk);
		//std::cout << "minha_classe[0]->Delta_S: " << minha_classe[0]->Delta_S << std::endl;
		return;
	}

	sort_per_objective(minha_classe,0);

	unsigned int i = 0;
	double delta_Si = (minha_classe[i]->P.ROI - minha_classe[i+1]->P.ROI) *
						  (R_2 - minha_classe[i]->P.risk);

	delta_Si *= minha_classe[i]->stability;
	minha_classe[i]->Delta_S = delta_Si;
	//std::cout << "delta_S0: " << minha_classe[i]->Delta_S << std::endl;

	i = 1;

	while (i < minha_classe.size() - 1)
	{
		delta_Si = (minha_classe[i]->P.ROI - minha_classe[i+1]->P.ROI) *
					(minha_classe[i-1]->P.risk - minha_classe[i]->P.risk);

		delta_Si *= minha_classe[i]->stability;

		minha_classe[i]->Delta_S = delta_Si;
		//std::cout << "(" << minha_classe[i]->P.cardinality << ") delta_S" << i << ": " << minha_classe[i]->Delta_S << std::endl;
		++i;
	}

	delta_Si = (minha_classe[i]->P.ROI - R_1) *
				   (minha_classe[i-1]->P.risk - minha_classe[i]->P.risk);
	delta_Si *= minha_classe[i]->stability;
	minha_classe[i]->Delta_S = delta_Si;
	//std::cout << "delta_S" << minha_classe.size() - 1 << ": " << minha_classe[i]->Delta_S << std::endl;
	//system("pause");
}

struct stochastic_params
{
	double cov, var_ROI, var_risk, corr, var_ratio;
	double conditional_mean_ROI, conditional_var_ROI;
	double conditional_mean_risk, conditional_var_risk;

	stochastic_params(sol* w)
	{
		//std::cout << "cov=";
		cov = w->P.kalman_state.P(0,1);
		//std::cout << cov << ", var_ROI=";
		var_ROI = w->P.kalman_state.P(0,0);
		//std::cout << var_ROI << ", var_risk=";
		var_risk = w->P.kalman_state.P(1,1);
		//std::cout << var_risk << ", corr=";
		corr = cov / (sqrt(var_ROI)*sqrt(var_risk));
		//std::cout << corr << ", ratio=" << var_ratio << std::endl;
		var_ratio = sqrt(var_ROI)/sqrt(var_risk);

		conditional_mean_ROI = w->P.ROI;
		conditional_var_ROI = (1.0 - corr*corr)*var_ROI;

		conditional_mean_risk = w->P.risk;
		conditional_var_risk = (1.0 - corr*corr)*var_risk;

		//system("pause");
	}
};

double mean_delta_product(sol* w_0, sol* w_1, sol* w_2)
{

	stochastic_params sw_0(w_0);
	stochastic_params sw_1(w_1);
	stochastic_params sw_2(w_2);

	/*delta_Si = (minha_classe[i]->P.ROI - minha_classe[i+1]->P.ROI) *
				(minha_classe[i-1]->P.risk - minha_classe[i]->P.risk);*/

	double mean_delta_ROI = sw_1.conditional_mean_ROI - sw_2.conditional_mean_ROI;
	double mean_delta_risk = sw_0.conditional_mean_risk - sw_1.conditional_mean_risk;
	double var_delta_ROI = sw_1.conditional_var_ROI + sw_2.conditional_var_ROI;
	double var_delta_risk = sw_0.conditional_var_risk + sw_1.conditional_var_risk;

	return (mean_delta_ROI*var_delta_risk + mean_delta_risk*var_delta_ROI) / (var_delta_ROI + var_delta_risk);
}

void compute_stochastic_Delta_S_class(std::vector<sol*> &minha_classe, float R_1, float R_2)
{

	//std::cout << "minha_classe.size(): " << minha_classe.size() << std::endl;
	if (minha_classe.size() == 1)
	{
		stochastic_params sw_1(minha_classe[0]);

		double mean_delta_ROI = sw_1.conditional_mean_ROI - R_1;
		double mean_delta_risk = R_2 - sw_1.conditional_mean_risk;
		double var_delta_ROI = sw_1.conditional_var_ROI;
		double var_delta_risk = sw_1.conditional_var_risk;

		//double ddelta_S = (minha_classe[0]->P.ROI - R_1)*(R_2 - minha_classe[0]->P.risk);

		minha_classe[0]->Delta_S = (mean_delta_ROI*var_delta_risk + mean_delta_risk*var_delta_ROI) / (var_delta_ROI + var_delta_risk);
		//std::cout << "(" << minha_classe[0]->P.cardinality << ") delta_S0: " << minha_classe[0]->Delta_S << std::endl;
		//std::cout << "ddelta_S: " << ddelta_S << std::endl;
		return;
	}

	sort_per_objective(minha_classe,0);

	unsigned int i = 0;

	stochastic_params sw_1(minha_classe[i]);
	stochastic_params sw_2(minha_classe[i+1]);

	double mean_delta_ROI = sw_1.conditional_mean_ROI - sw_2.conditional_mean_ROI;
	double mean_delta_risk = R_2 - sw_1.conditional_mean_risk;
	double var_delta_ROI = sw_1.conditional_var_ROI + sw_2.conditional_var_ROI;
	double var_delta_risk = sw_1.conditional_var_risk;

	/*double ddelta_Si = (minha_classe[i]->P.ROI - minha_classe[i+1]->P.ROI) *
			    		  (R_2 - minha_classe[i]->P.risk);*/

	double delta_Si = minha_classe[i]->Delta_S = (mean_delta_ROI*var_delta_risk + mean_delta_risk*var_delta_ROI) / (var_delta_ROI + var_delta_risk);
	delta_Si *= minha_classe[i]->stability;
	minha_classe[i]->Delta_S = delta_Si;

	//std::cout << "(" << minha_classe[i]->P.cardinality << ") delta_S0: " << minha_classe[i]->Delta_S << std::endl;
	//std::cout << "ddelta_S0 : " << ddelta_Si << std::endl;

	i = 1;

	while (i < minha_classe.size() - 1)
	{
		/*double ddelta_Si = (minha_classe[i]->P.ROI - minha_classe[i+1]->P.ROI) *
					(minha_classe[i-1]->P.risk - minha_classe[i]->P.risk);*/
		delta_Si = mean_delta_product(minha_classe[i-1], minha_classe[i], minha_classe[i+1]);
		delta_Si *= minha_classe[i]->stability;
		minha_classe[i]->Delta_S = delta_Si;

		//std::cout << "(" << minha_classe[i]->P.cardinality << ") delta_S" << i << ": " << minha_classe[i]->Delta_S << std::endl;

		//std::cout << "delta_S" << i << ": " << minha_classe[i]->Delta_S << std::endl;
		//std::cout << "ddelta_S" << i << ": " << ddelta_Si << std::endl;
		++i;
	}

	/*ddelta_Si = (minha_classe[i]->P.ROI - R_1) *
			   (minha_classe[i-1]->P.risk - minha_classe[i]->P.risk);*/

	stochastic_params sw_l(minha_classe[i-1]);
	stochastic_params sw_r(minha_classe[i]);

	mean_delta_ROI = sw_r.conditional_mean_ROI - R_1;
	mean_delta_risk = sw_l.conditional_mean_risk - sw_r.conditional_mean_risk;
	var_delta_ROI = sw_r.conditional_var_ROI;
	var_delta_risk = sw_l.conditional_var_risk + sw_r.conditional_var_risk;

	delta_Si = minha_classe[i]->Delta_S = (mean_delta_ROI*var_delta_risk + mean_delta_risk*var_delta_ROI) / (var_delta_ROI + var_delta_risk);
	delta_Si *= minha_classe[i]->stability;
	minha_classe[i]->Delta_S = delta_Si;
	//std::cout << "(" << minha_classe[i]->P.cardinality << ") delta_S" << minha_classe.size() - 1 << ": " << minha_classe[i]->Delta_S << std::endl;
	//std::cout << "ddelta_S" << minha_classe.size() - 1 << ": " << ddelta_Si << std::endl;
	//system("pause");

}

void compute_stochastic_Delta_S(std::vector<sol*> &P, unsigned int num_classes, float R_1, float R_2)
{
	unsigned int i = 0;
	for (unsigned int c = 0; c < num_classes; ++c)
	{
		std::vector<sol*> minha_classe;

		while (P[i]->Pareto_rank != c)
			++i;

		while (i < P.size() && P[i]->Pareto_rank == c)
		{
			minha_classe.push_back(P[i]);
			++i;
		}

		//std::cout << "\ncompute_Delta_S_class(" << c << ") of (" << i << ") portfolios...\n";
		if (c == num_classes-1 && minha_classe.size() > 3)
			compute_stochastic_Delta_S_class(minha_classe,R_1,R_2);

	}
}

void compute_Delta_S(std::vector<sol*> &P, unsigned int num_classes, float R_1, float R_2)
{
	unsigned int i = 0;
	for (unsigned int c = 0; c < num_classes; ++c)
	{
		std::vector<sol*> minha_classe;

		while (P[i]->Pareto_rank != c)
			++i;

		while (i < P.size() && P[i]->Pareto_rank == c)
		{
			minha_classe.push_back(P[i]);
			++i;
		}

		//std::cout << "\ncompute_Delta_S_class(" << c << ")...\n";
		if (c == num_classes-1 && minha_classe.size() > 3)
			compute_Delta_S_class(minha_classe,R_1,R_2);
	}
}


void remove_worst_s_metric(std::vector<sol*> &P, unsigned int class_index, float R_1, float R_2)
{
	unsigned int i = 0, inicio = 0, fim, worst;
	std::vector<std::pair<unsigned int,sol*> > minha_classe;

	while (P[i]->Pareto_rank != class_index)
	{
		inicio++; i++;
	} fim = inicio;
	while (i < P.size() && P[i]->Pareto_rank == class_index)
	{
		minha_classe.push_back(std::pair<unsigned int,sol*>(i,P[i]));
		fim++; i++;
	}

	fim -= 1;

	sort_per_objective(minha_classe,0);

	if (minha_classe.size() == 1)
	{
		delete *(P.begin() + minha_classe[0].first);
		P.erase(P.begin() + minha_classe[0].first);
		return;
	}
	else if (minha_classe.size() == 2)
	{
		if (rand()%2 == 0)
		{
			delete *(P.begin() + minha_classe[0].first);
			P.erase(P.begin() + minha_classe[0].first);
			return;
		}
		else
		{
			delete *(P.begin() + minha_classe[1].first);
			P.erase(P.begin() + minha_classe[1].first);
			return;
		}
	}
	else if (minha_classe.size() == 3)
	{
		delete *(P.begin() + minha_classe[1].first);
		P.erase(P.begin() + minha_classe[1].first);
		return;
	}
	else
	{

		//for (unsigned int j = 0; j < minha_classe.size(); ++j)
			//std::cout << "<(" <<  minha_classe[j].first << "," << minha_classe[j].second->P.ROI << ");"
					  //<< "<(" <<  minha_classe[j].first << "," << minha_classe[j].second->P.risk << ")>\n";

		worst = inicio; i = 0;

		//double worst_delta_Si = minha_classe[i].second->Delta_S; i = 1;
		double worst_delta_Si = std::numeric_limits<double>::max(); i = 1;

		while (i < minha_classe.size() - 1)
		{
			//std::cout << "delta_S[" << minha_classe[i].first << "]: " << delta_Si << std::endl;
			if (minha_classe[i].second->Delta_S < worst_delta_Si)
			{
				worst_delta_Si = minha_classe[i].second->Delta_S;
				worst = minha_classe[i].first;
			}
			++i;
		}

		//std::cout << "delta_S[" << minha_classe[i].first << "]: " << delta_Si << std::endl;
		/*if (minha_classe[i].second->Delta_S < worst_delta_Si)
		{
			worst_delta_Si = minha_classe[i].second->Delta_S;
			worst = minha_classe[i].first;
		}*/

		//std::cout << "Worst delta_S: " << "(" << worst << "," << worst_delta_Si << ")\n";
		//system("pause");

		delete *(P.begin() + worst);
		P.erase(P.begin() + worst);
	}
}

void anticipatory_learning(sol* &w, unsigned int t)
{
	// Observa estado via Simulação Monte Carlo
	portfolio::observe_state(w->P, 10, t);

	Eigen::MatrixXd covar(2,2);
	covar << w->P.error_covar_prediction(0,0) + w->P.error_covar(0,0),
			 w->P.error_covar_prediction(0,1) + w->P.error_covar(0,1),
			 w->P.error_covar_prediction(1,0) + w->P.error_covar(1,0),
			 w->P.error_covar_prediction(1,1) + w->P.error_covar(1,1);

	//std::cout << "covar= " << covar << std::endl;

	Eigen::VectorXd delta1(2), delta2(2), u(2);

	// Pr{ROI(t+1) > ROI(t) & Risk(t+1) > Risk(t)}
	delta1 << w->P.ROI - w->P.ROI_prediction,
			  w->P.risk - w->P.risk_prediction;

	// Pr{ROI(t+1) < ROI(t) & Risk(t+1) < Risk(t)}
	delta2 << w->P.ROI_prediction -  w->P.ROI,
			  w->P.risk_prediction - w->P.risk;
	// Point of interest for Pr {Delta < u = [0 0]^T}
	u << 0.0, 0.0;

	// Compute the change of variables to standardize the multivariate Gaussian
	Eigen::LLT<Eigen::MatrixXd> lltOfA(covar); // compute the Cholesky decomposition of A
	Eigen::MatrixXd U = lltOfA.matrixU();

	Eigen::VectorXd z1 = U.inverse()*(u - delta1);
	Eigen::VectorXd z2 = U.inverse()*(u - delta2);
	//std::cout << "z1= " << z1 << std::endl;
	//std::cout << "z2= " << z2 << std::endl;

	// Compute probability of w(t+1) being non-dominated w.r.t. w(t)
	double nd_probability = normal_cdf(z1,Eigen::MatrixXd::Identity(2,2))
			 	 	 	  + normal_cdf(z2,Eigen::MatrixXd::Identity(2,2));
	//std::cout << "ndp= " << nd_probability << ")" << std::endl;

	// Integra conhecimento antecipatório ponderado pela incerteza futura e presente
	double alpha = 1.0 - linear_entropy(nd_probability); // Quantifica a confiança na predição
	//double alpha = 1.0 / (1.0 - covar.determinant());

	// Atualiza estimativa de retorno/risco baseado em conchecimento antecipatório
	//std::cout << "ROI(" << t << ")'= " << w->P.kalman_state.x(0) << ", risk(" << t << ")= " << w->P.kalman_state.x(1) << std::endl;
	//std::cout << "ROI(" << t+1 << ")'= " << w->P.kalman_state.x_next(0) << ", risk(" << t+1 << ")= " << w->P.kalman_state.x_next(1) << std::endl;
	w->P.kalman_state.x = w->P.kalman_state.x + alpha*(w->P.kalman_state.x_next - w->P.kalman_state.x);
	w->P.kalman_state.P = w->P.kalman_state.P + alpha*(w->P.kalman_state.P_next - w->P.kalman_state.P);
	w->P.ROI = w->P.kalman_state.x(0);
	w->P.risk = w->P.kalman_state.x(1);

	if (portfolio::robustness)
	{
		w->P.robust_ROI = w->P.ROI;
		w->P.robust_risk = w->P.risk;
	}
	else
	{
		w->P.non_robust_ROI = w->P.ROI;
		w->P.non_robust_risk = w->P.risk;
	}

	w->prediction_error = portfolio::prediction_error(w->P, t);
	//std::cout << "ROI(" << t+1 << ")= " << w->P.ROI_observed << ", risk(" << t+1 << ")= " << w->P.risk_observed << std::endl;
	//std::cout << "ROI(" << t << ")'_new= " << w->P.kalman_state.x(0) << ", risk(" << t << ")'= " << w->P.kalman_state.x(1) << "\n";

	w->anticipation = true;
	w->alpha = alpha;

	//std::cout << "(alpha= " << w->alpha << ", error= " << w->prediction_error << ")" << std::endl;
	//system("pause");
}

void anticipatory_learning(std::vector<sol*> &P, unsigned int t)
{
	for (unsigned int i = 0; i < P.size(); ++i)
		if (!P[i]->anticipation)
			anticipatory_learning(P[i], t);
}

void uma_geracao_SMS_EMOA(std::vector<sol*> &P, float mut_rate, unsigned int K, float R1, float R2, unsigned int t)
{
	op_cruzamento::probabilidade = op_cruzamento::prob_op_uniforme;
	op_mutacao::probabilidade = op_mutacao::prob_op_uniforme;

	//std::cout << "Inicio do SMS-EMOA. ";

	// Incorpora conhecimento antecipatório via filtros de Kalman
	//if (t >= 0)
	//	anticipatory_learning(P, t);

	unsigned int num_classes = fast_nondominated_sort(P);
	//std::cout << "num_classes: " << num_classes << "\n";

	//if (t >= 0)
	//	compute_stochastic_Delta_S(P,num_classes, R1, R2);
	//else
		compute_Delta_S(P,num_classes, R1, R2);


	//std::cout << "Selecao por torneio... ";
	// Para cada filho, dois pais sao sorteados por torneio de tamanho K = 3
	unsigned int pai1 = sel_torneio_Delta_S(P, K);
	unsigned int pai2;


	// Garante que o segundo pai eh diferente do primeiro.
	do
	{
		pai2 = sel_torneio_Delta_S(P, K);
	} while (pai1 == pai2);

	//std::cout << "Cruzamento e mutacao...\n";
	// Cria dois filhos a cada operacao de cruzamento.
	sol * filho1 = new sol(); sol * filho2 = new sol();

	// Aplica o operador selecionado
	operador_cruzamento_ptr operador_xover = seletor_roulette_wheel_cruzamento();
	operador_xover(P[pai1], P[pai2], filho1, filho2);
	delete filho2;

	// Aplica o operador selecionado no filho 1
	operador_mutacao_ptr operador_mut = seletor_roulette_wheel_mutacao();
	operador_mut(filho1, mut_rate);

	// Reavalia a nova solucao quanto aos custos de transporte e estoque.
	if (portfolio::robustness)
		portfolio::compute_robust_efficiency(filho1->P);
	else
		portfolio::compute_efficiency(filho1->P);

	// Incorpora conhecimento antecipatório
	//if (t >= 0)
	//	anticipatory_learning(filho1, t);

	filho1->stability = portfolio::evaluate_stability(filho1->P);

	//std::cout << "Avaliacao ok.\n";

	// Insere os dois novos filhos na populacao de offspring Q.
	P.push_back(filho1);

	// Primeiro passo: determinar as classes de nao-dominancia
	num_classes = fast_nondominated_sort(P);
	//std::cout << "num_classes: " << num_classes << "\n";

	//if (t >= 0)
	//	compute_stochastic_Delta_S(P,num_classes, R1, R2);
	//else
		compute_Delta_S(P,num_classes, R1, R2);

	//std::cout << "compute_Delta_S ok.\n";
	remove_worst_s_metric(P, num_classes-1, R1, R2);
	//std::cout << "remove_worst_s_metric ok.\n";
}

// A partir de uma populacao atual, aplica os operadores de 
// variacao e selecao apropriados para criar uma nova populacao.

// Requer que toda a populacao P atual esteja com os valores das 
// funcoes de custo determinados.

// Garante que a nova populacao estara tambem ao final da geracao 
// com os custos atualizados.

void uma_geracao_NSGAII(std::vector<sol*> &P, float mut_rate, unsigned int K, unsigned int t)
{

	op_cruzamento::probabilidade = op_cruzamento::prob_op_uniforme;
	op_mutacao::probabilidade = op_mutacao::prob_op_uniforme;

	//std::cout << "Inicio do NSGA-II. ";
	// Incorpora conhecimento antecipatório via filtros de Kalman
	//if (t >= 0)
		//anticipatory_learning(P, t);

	// Primeiro passo: determinar as classes de nao-dominancia e valores de CD.
	unsigned int num_classes = fast_nondominated_sort(P);
	//std::cout << "num_classes: " << num_classes << "\n";

	//std::cout << "fast_nondominated_sort ok\n";
	
	for (unsigned int i = 0; i < num_classes; ++i)
		crowding_distance(P, i);

	//std::cout << "crowding_distance ok\n";

	unsigned int novap_size = 2*P.size();

	// Segundo passo: cria a populacao de filhos de tamanho igual a da populacao atual, 
	// escolhendo dois pais e gerando dois filhos de cada vez.
	while (P.size() < novap_size)
	{
		// Para cada filho, dois pais sao sorteados por torneio de tamanho K = 3
		unsigned int pai1 = sel_torneio_CD(P, K);
		unsigned int pai2;

		//std::cout << "Selecao por torneio... ";
		// Garante que o segundo pai eh diferente do primeiro.
		do
		{
			pai2 = sel_torneio_CD(P, K);
		} while (pai1 == pai2);

		//std::cout << "Ok.\n";
		// Cria dois filhos a cada operacao de cruzamento.
		sol * filho1 = new sol(); sol * filho2 = new sol();

		// Aplica o operador selecionado
		operador_cruzamento_ptr operador_xover = seletor_roulette_wheel_cruzamento();
		operador_xover(P[pai1], P[pai2], filho1, filho2);

		//std::cout << "Cruzamento ok.\n";

		// Aplica o operador selecionado no filho 1
		operador_mutacao_ptr operador_mut = seletor_roulette_wheel_mutacao();
		operador_mut(filho1, mut_rate);

		operador_mut = seletor_roulette_wheel_mutacao();
		operador_mut(filho2, mut_rate);

		//std::cout << "Mutacao ok.\n";

		// Incorpora conhecimento antecipatório
		//if (t >= 0)
		//{
		//	anticipatory_learning(filho1, t);
		//	anticipatory_learning(filho2, t);
		//}

		// Reavalia a nova solucao quanto ao retorno e risco.
		if (portfolio::robustness)
		{
			portfolio::compute_robust_efficiency(filho1->P);
			portfolio::compute_robust_efficiency(filho2->P);
		}
		else
		{
			portfolio::compute_efficiency(filho1->P);
			portfolio::compute_efficiency(filho2->P);
		}

		 // TODO: Passar dados ao NSGA-II
		filho1->stability = portfolio::evaluate_stability(filho1->P);
		filho2->stability = portfolio::evaluate_stability(filho2->P);

		//std::cout << "Avaliacao ok.\n";

		// Insere os dois novos filhos na populacao de offspring Q.
		P.push_back(filho1);
		P.push_back(filho2);
	}

	// Quarto passo: determinar as classes de nao-dominancia e valores de CD sobre a 
	// populacao conjunta R.
	num_classes = fast_nondominated_sort(P);
	for (unsigned int i = 0; i < num_classes; ++i)
		crowding_distance(P, i);

	//std::cout << "Reavaliacao FNS/CD ok.\n";

	// Sexto e ultimo passo: determinar a nova geracao de individuos que irao
	// "sobreviver" em R. Os sobreviventes correspondem a primeira metade do 
	// vetor R (a mesma quantidade da populacao P original). As solucoes da 
	// segunda metade sao simplesmente descartadas.
	
	// Realiza o descarte dos piores individuos da segunda metade de R.
	std::vector<sol*>::iterator it = P.begin() + P.size()/2;
	for (; it != P.end(); ++it)
		delete *it;

	P.erase(P.begin() + P.size()/2, P.end());
	
	//std::cout << "Selecao de sobreviventes ok.\n";

}
