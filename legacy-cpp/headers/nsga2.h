#ifndef NSGA2_H
	#define NSGA2_H

#include "portfolio.h"
#include <vector>
#include <ctime>
#include <cstdlib>
#include <string>

typedef struct solution {

	portfolio P;
	double cd; // Valor de crowding distance associado
	double Delta_S; // Contribuição para o hipervolume associada
	unsigned int Pareto_rank; // A qual a classe pertence (e.g. F0, F1, ...)
	double stability; // Grau de estabilidade da solucao
	unsigned int rank_ROI; // Posicao em termos de retorno da solucao na populacao
	unsigned int rank_risk; // Posicao em termos de risco da solucao na populacao
	double alpha; // Confiança sobre a predição futura
	bool anticipation; // Já realizou a etapa de anticipatory learning?
	double prediction_error; // Erro de predição relacionado ao filtro de Kalman

	static std::string regularization_type;

	solution()
	{
		P.init();
		cd = 0.0; Pareto_rank = 0; stability = 1.0; rank_ROI = 0.0; rank_risk = 0.0; Delta_S = 0.0;
		alpha = 0.0; anticipation = false; prediction_error = 0.0;
		if (portfolio::robustness)
			portfolio::compute_robust_efficiency(P);
		else
			portfolio::compute_efficiency(P);
		portfolio::evaluate_stability(P);
	}

} sol;

sol* individuo_aleatorio(unsigned int H, unsigned int N);

inline bool cmp_cd (sol i, sol j) { return (i.cd>j.cd); }
inline bool cmp_cd_ptr (sol* i, sol* j) { return (i->cd>j->cd); }

inline bool cmp_classe (sol i, sol j) { return (i.Pareto_rank<j.Pareto_rank); }
inline bool cmp_classe_ptr (sol* i, sol* j) { return (i->Pareto_rank<j->Pareto_rank); }

inline bool cmp_ROI (sol i, sol j) { return (i.P.ROI > j.P.ROI); }
inline bool cmp_ROI_ptr (sol* i, sol* j) { return (i->P.ROI > j->P.ROI); }
inline bool cmp_ROI_ptr_pair (std::pair<unsigned int,sol*> i, std::pair<unsigned int,sol*> j) { return (i.second->P.ROI > j.second->P.ROI); }
inline bool cmp_risk (sol i, sol j) { return (i.P.risk < j.P.risk); }
inline bool cmp_risk_ptr (sol* i, sol* j) { return (i->P.risk < j->P.risk); }
inline bool cmp_risk_ptr_pair (std::pair<unsigned int,sol*> i, std::pair<unsigned int,sol*> j) { return (i.second->P.risk < j.second->P.risk); }

// Considera a classe e desempate por CD
inline bool cmp_classe_CD (sol i, sol j) 
{
	return !(i.Pareto_rank > j.Pareto_rank || (i.Pareto_rank == j.Pareto_rank && !cmp_cd(i,j)));
}

inline bool domina_sem_restricoes (sol* p, sol* q) 
{ 
	if (p->P.ROI < q->P.ROI || p->P.risk > q->P.risk)
		return false;
	else if (p->P.ROI > q->P.ROI || p->P.risk < q->P.risk)
		return true;
	else 
		return false;
}

inline bool dominancia_estocastica_icx (sol* p, sol* q)
{
	if (!domina_sem_restricoes(p,q))
			return false;

	Eigen::MatrixXd Sigma_diff = (q->P.kalman_state.P - p->P.kalman_state.P);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Sigma_diff);
	if (eigensolver.info() != Eigen::Success) abort();
	return (eigensolver.eigenvectors().sum() > 0); // TODO
}

inline bool domina_com_restricoes (sol* p, sol* q)
{
	if (p->P.cardinality > portfolio::max_cardinality
			&& q->P.cardinality <= portfolio::max_cardinality)
		return false;
	else if (p->P.cardinality <= portfolio::max_cardinality
			&& q->P.cardinality > portfolio::max_cardinality)
		return true;
	else if (p->P.cardinality > portfolio::max_cardinality
			&& p->P.cardinality > portfolio::max_cardinality)
		return (p->P.cardinality < q->P.cardinality);

	if (p->P.ROI < q->P.ROI || p->P.risk > q->P.risk)
		return false;
	else if (p->P.ROI > q->P.ROI || p->P.risk < q->P.risk)
		return true;
	else
		return false;
}

void ordena_objetivo(std::vector<sol>& P, int m);
void crowding_distance(std::vector<sol>& P, unsigned int classe);
unsigned int fast_nondominated_sort(std::vector<sol>& P);

void compute_stochastic_Delta_S(std::vector<sol*> &P, unsigned int num_classes, float R_1, float R_2);
void compute_stochastic_Delta_S_class(std::vector<sol*> &minha_classe, float R_1, float R_2);
void compute_Delta_S(std::vector<sol*> &P, unsigned int num_classes, float R_1, float R_2);
void compute_Delta_S_class(std::vector<sol*> &minha_classe, float R_1, float R_2);

void remove_worst_s_metric(std::vector<sol*> &P, unsigned int class_index, float R_1, float R_2);


void anticipatory_learning(std::vector<sol*> &P, unsigned int t);
void anticipatory_learning(sol* &w, unsigned int t);
void Kalman_filter_prediction(std::vector<sol*>& P);

void uma_geracao_NSGAII(std::vector<sol*> &P, float mut_rate, unsigned int K, unsigned int t);
void uma_geracao_SMS_EMOA(std::vector<sol*> &P, float mut_rate, unsigned int K, float R1, float R2, unsigned int t);

#endif

