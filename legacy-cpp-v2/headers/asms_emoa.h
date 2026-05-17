#ifndef ASMS_EMOA_H
	#define ASMS_EMOA_H

#include "portfolio.h"

#include <vector>
#include <ctime>
#include <cstdlib>
#include <string>

typedef struct solution {

	portfolio P;
	double cd; // Crowding distance
	double Delta_S; // Hypervolume contribution
	unsigned int Pareto_front; // Which front belongs (e.g. F0, F1, ...)
	unsigned int observed_Pareto_front; // Which front belongs in the out-of-sample scenario
	double stability; // Stability degree
	unsigned int rank_ROI; // Rank in population in terms of return
	unsigned int rank_risk; // Rank in population in terms of risk
	double alpha; // Confidence over prediction
	double anticipation_rate; // Anticipation rate
	bool anticipation; // Anticipatory learning step executed?
	double prediction_error; // Prediction residual of the Kalman filter
	double pred_hv; // Predicted hypervolume T steps ahead associated with the solution
	std::pair<double,double> epsilon_feasibility; // Feasibility probability over both objectives

	static double xover_rate, mut_rate, min_error, max_error;
	static unsigned int tourn_size, num_fronts;
	static std::vector<double> error_sequence;

	static std::vector<std::vector<solution*> > historical_populations;
	static std::vector<solution*> historical_anticipative_decisions;
	static solution* predicted_anticipative_decision;

	static double epsilon;
	static std::pair<double,double> ref_point;

	static void compute_efficiency(solution *w);

	solution()
	{
		P.init();

		cd = 0.0; Pareto_front = 0; observed_Pareto_front = 0; stability = 1.0; rank_ROI = 0.0; rank_risk = 0.0; Delta_S = 0.0;
		alpha = 0.0; anticipation = false; prediction_error = 0.0; epsilon_feasibility = std::pair<double,double>(0.0,0.0);

		compute_efficiency(this);

	}

	solution(solution *sol)
	{

		P = portfolio(sol->P);
		cd = sol->cd; Pareto_front = observed_Pareto_front = sol->Pareto_front; stability = sol->stability; rank_ROI = sol->rank_ROI; rank_risk = sol->rank_risk; Delta_S = sol->Delta_S;
		alpha = sol->alpha; anticipation = sol->anticipation; prediction_error = sol->prediction_error; anticipation_rate = sol->anticipation_rate;
		epsilon_feasibility = sol->epsilon_feasibility;

		if (P.ROI != P.ROI || P.risk != P.risk)
		{
			solution::compute_efficiency(sol);
		}

	}

	solution(const Eigen::VectorXd investment)
	{
		cd = 0.0; Pareto_front = 0; observed_Pareto_front = 0; stability = 1.0; rank_ROI = 0.0; rank_risk = 0.0; Delta_S = 0.0;
		alpha = 0.0; anticipation = false; prediction_error = 0.0;
		P.investment = investment;
		compute_efficiency(this);
	}

} sol;

static double compute_max_cost()
{
	double max_transaction_value = portfolio::current_wealth/(2.0*portfolio::num_assets);
	std::pair<double,double> fees = portfolio::BOVESPA_fees(max_transaction_value);
	double cost = max_transaction_value*fees.first + fees.second;
	return 2.0*cost*portfolio::num_assets;
}

static double get_max_cost (std::vector<sol*> &pop)
{
	double max = 0.0;
	for (unsigned int i = 0; i < pop.size(); ++i)
	{
		std::pair<double,double> cost = portfolio::transaction_cost(portfolio::current_investment, pop[i]->P.investment);
		if (cost.first > max)
			max = cost.first;
	}
	return max;
}

static double get_min_cost (std::vector<sol*> &pop)
{
	double min = 0.0;
	for (unsigned int i = 0; i < pop.size(); ++i)
	{
		std::pair<double,double> cost = portfolio::transaction_cost(portfolio::current_investment, pop[i]->P.investment);
		if (cost.first < min)
			min = cost.first;
	}
	return min;
}



sol* random_solution(unsigned int H, unsigned int N);

inline bool cmp_cd (sol i, sol j) { return (i.cd>j.cd); }
inline bool cmp_cd_ptr (sol* i, sol* j) { return (i->cd>j->cd); }
inline bool cmp_acc_confidence_ptr (sol* i, sol* j)
{
	double acc_i = (solution::min_error > 0.0) ? 1.0 - 0.5*(i->prediction_error - solution::min_error)/(solution::max_error - solution::min_error) : 1.0;
	double acc_j = (solution::min_error > 0.0) ? 1.0 - 0.5*(j->prediction_error - solution::min_error)/(solution::max_error - solution::min_error) : 1.0;

	return acc_i*i->alpha < acc_j*j->alpha;
}

inline bool cmp_confidence_ptr (sol* i, sol* j){ return i->alpha < j->alpha; }
inline bool cmp_error_ptr (sol* i, sol* j){ return i->prediction_error > j->prediction_error; }

inline bool cmp_front (sol i, sol j) { return (i.Pareto_front<j.Pareto_front); }
inline bool cmp_front_ptr (sol* i, sol* j) { return (i->Pareto_front<j->Pareto_front); }
inline bool cmp_observed_front_ptr (sol* i, sol* j) { return (i->observed_Pareto_front<j->observed_Pareto_front); }

inline bool cmp_ROI (sol i, sol j) { return (i.P.ROI > j.P.ROI); }
inline bool cmp_ROI_ptr (sol* i, sol* j) { return (i->P.ROI > j->P.ROI); }
inline bool cmp_ROI_observed_ptr (sol* i, sol* j) { return (i->P.ROI_observed > j->P.ROI_observed); }
inline bool cmp_ROI_ptr_pair (std::pair<unsigned int,sol*> i, std::pair<unsigned int,sol*> j) { return (i.second->P.ROI > j.second->P.ROI); }
inline bool cmp_risk (sol i, sol j) { return (i.P.risk < j.P.risk); }
inline bool cmp_risk_ptr (sol* i, sol* j) { return (i->P.risk < j->P.risk); }
inline bool cmp_risk_observed_ptr (sol* i, sol* j) { return (i->P.risk_observed < j->P.risk_observed); }
inline bool cmp_risk_ptr_pair (std::pair<unsigned int,sol*> i, std::pair<unsigned int,sol*> j) { return (i.second->P.risk < j.second->P.risk); }

std::pair<double,double> feasibility_p(sol * x);
bool epsilon_feasible(sol * x);

// Considera a classe e desempate por CD
inline bool compare_front_CD (sol i, sol j)
{
	return !(i.Pareto_front > j.Pareto_front || (i.Pareto_front == j.Pareto_front && !cmp_cd(i,j)));
}

inline bool unconstrained_dominance (sol* p, sol* q) 
{ 
	if (p->P.ROI < q->P.ROI || p->P.risk > q->P.risk)
		return false;
	else if (p->P.ROI > q->P.ROI || p->P.risk < q->P.risk)
		return true;
	else 
		return false;
}

inline bool Pareto_dominance (std::pair<double,double> p1, std::pair<double,double> p2)
{
	if (p1.first < p2.first || p1.second < p2.second)
		return false;
	else if (p1.first > p2.first || p1.second > p2.second)
		return true;
	else
		return false;
}

inline bool constrained_dominance (sol* p, sol* q)
{

	bool p_feasible = (p->epsilon_feasibility == std::pair<double,double>(0.0,0.0)) ? epsilon_feasible(p) :
		(p->epsilon_feasibility.first >= solution::epsilon) && (p->epsilon_feasibility.second >= solution::epsilon);

	bool q_feasible = (q->epsilon_feasibility == std::pair<double,double>(0.0,0.0)) ? epsilon_feasible(q) :
		(q->epsilon_feasibility.first >= solution::epsilon) && (q->epsilon_feasibility.second >= solution::epsilon);


	if (p_feasible && !q_feasible)
		return true;
	else if (!p_feasible && q_feasible)
		return false;
	else if (!p_feasible && !q_feasible)
		return Pareto_dominance(p->epsilon_feasibility,q->epsilon_feasibility);
	else
		return unconstrained_dominance (p,q);
}

inline bool unconstrained_observed_dominance (sol* p, sol* q)
{
	if (p->P.ROI_observed < q->P.ROI_observed || p->P.risk_observed > q->P.risk_observed)
		return false;
	else if (p->P.ROI_observed > q->P.ROI_observed || p->P.risk_observed < q->P.risk_observed)
		return true;
	else
		return false;
}

inline bool constrained_observed_dominance (sol* p, sol* q)
{
	bool p_feasible = (p->epsilon_feasibility == std::pair<double,double>(0.0,0.0)) ? epsilon_feasible(p) :
		(p->epsilon_feasibility.first >= solution::epsilon) && (p->epsilon_feasibility.second >= solution::epsilon);
	bool q_feasible = (q->epsilon_feasibility == std::pair<double,double>(0.0,0.0)) ? epsilon_feasible(q) :
		(q->epsilon_feasibility.first >= solution::epsilon) && (q->epsilon_feasibility.second >= solution::epsilon);

	if (p_feasible && !q_feasible)
		return true;
	else if (!p_feasible && q_feasible)
		return false;
	else if (!p_feasible && !q_feasible)
		return Pareto_dominance(p->epsilon_feasibility,q->epsilon_feasibility);
	else
		return unconstrained_observed_dominance (p,q);
}

inline bool stochastic_dominance (sol* p, sol* q)
{
	if (!unconstrained_dominance(p,q))
			return false;

	Eigen::MatrixXd Sigma_diff = (q->P.kalman_state.P - p->P.kalman_state.P);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Sigma_diff);
	if (eigensolver.info() != Eigen::Success) abort();
	return (eigensolver.eigenvectors().sum() > 0); // TODO
}

inline bool card_constrained_dominance (sol* p, sol* q)
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
void crowding_distance(std::vector<sol*>& P, unsigned int classe);
unsigned int fast_nondominated_sort(std::vector<sol*>& P);
unsigned int observed_fast_nondominated_sort(std::vector<sol*> &P);

void compute_stochastic_Delta_S(std::vector<sol*> &P, unsigned int num_classes, float R_1, float R_2);
void compute_stochastic_Delta_S_class(std::vector<sol*> &minha_classe, float R_1, float R_2);
void compute_Delta_S(std::vector<sol*> &P, unsigned int num_classes, float R_1, float R_2);
void compute_Delta_S_class(std::vector<sol*> &minha_classe, float R_1, float R_2);

void remove_worst_s_metric(std::vector<sol*> &P, double total_error, unsigned int class_index, float R_1, float R_2);

void apply_mutation(sol* w);

std::pair<double,double> anticipatory_learning_obj_space(std::vector<sol*> &P, unsigned int t);
void anticipatory_learning_obj_space(sol* &w, sol* anticipative_portfolio, Eigen::VectorXd& current_w, double min_error, double max_error, double min_alpha, double max_alpha, unsigned int t);
std::vector<sol*> anticipatory_learning_dec_space(const std::vector<sol*> w, unsigned int t);
sol* anticipatory_learning_dec_space(const std::vector<sol*> pop, unsigned int i, unsigned int current_t);
sol* anticipatory_learning_dec_space(std::vector<sol*> pop_hist, std::vector<sol*> pop_pred, unsigned int i, unsigned int current_t);
sol* anticipatory_learning_dec_space(sol *w, unsigned int i, unsigned int current_t);
sol* anticipatory_learning_dec_space(sol *w, unsigned int current_t);

double non_dominance_probability(sol* w);
double non_dominance_probability(sol* w1, sol* w2);

void NSGA2_iteration(std::vector<sol*> &P, unsigned int t);
void ASMS_EMOA_iteration(std::vector<sol*> &P, float R1, float R2, unsigned int t, bool anticipation);

#endif

