#include "../headers/asms_emoa.h"
#include "../headers/selection_operators.h"
#include "../headers/mutation_operators.h"
#include "../headers/crossover_operators.h"
#include "../headers/learning_operators.h"
#include "../headers/kalman_filter.h"
#include "../headers/statistics.h"
#include "../headers/portfolio.h"
#include "../headers/dirichlet.h"

#include <algorithm>
#include <fstream>
#include <istream>
#include <ostream>
#include <limits>
#include <vector>
#include <cmath>
#include <map>

#include <Eigen/Eigen/Cholesky>

using namespace std;

typedef std::pair<unsigned int, unsigned int> integer_pair;

double solution::xover_rate, solution::mut_rate, solution::min_error, solution::max_error;
std::vector<std::vector<solution*> > solution::historical_populations;
std::vector<solution*> solution::historical_anticipative_decisions;
unsigned int solution::tourn_size, solution::num_fronts;
sol* solution::predicted_anticipative_decision;
double solution::epsilon;
std::pair<double,double> solution::ref_point;

void solution::compute_efficiency(sol* w)
{

	unsigned int initial_t = (portfolio::current_period > Kalman_params::window_size)
			? (portfolio::current_period - Kalman_params::window_size) : 0;

	portfolio::observe_state(w->P, portfolio::samples_per_portfolio, initial_t, portfolio::current_period);

	w->P.cardinality = portfolio::card(w->P);
	w->prediction_error = w->P.kalman_state.error;
	w->alpha = 1.0 - non_dominance_probability(w); // Quantifies confidence over prediction

}

std::pair<double,double> feasibility_p(sol * x)
{

	Eigen::MatrixXd covar(2,2);
	covar << x->P.covar(0,0), x->P.covar(0,1), x->P.covar(1,0), x->P.covar(1,1);

	Eigen::LLT<Eigen::MatrixXd> lltOfA(covar); // Computes Cholesky decomposition
	Eigen::MatrixXd U = lltOfA.matrixU();

	Eigen::VectorXd zref_bottom(2), zref_bottom_ROI(1), zref_upper(2), zref_upper_risk(1), zpoint(2);
	zpoint << x->P.ROI, x->P.risk;

	zref_bottom << solution::ref_point.first, 0.0;
	zref_bottom = U.inverse()*(zref_bottom - zpoint);
	zref_bottom_ROI << zref_bottom(0);

	zref_upper << 0.5, solution::ref_point.second;
	zref_upper = U.inverse()*(zref_upper - zpoint);
	zref_upper_risk << zref_upper(1);


	double p1 = 1.0 - normal_cdf(zref_bottom_ROI,Eigen::MatrixXd::Identity(1,1));
	double p2 = normal_cdf(zref_upper_risk,Eigen::MatrixXd::Identity(1,1));

	return std::pair<double,double>(p1,p2);
}

bool epsilon_feasible(sol * x)
{
	x->epsilon_feasibility = feasibility_p(x);
	return (x->epsilon_feasibility.first >= solution::epsilon)
			&& (x->epsilon_feasibility.second >= solution::epsilon);
}

void assigns_crowding_distance_obj(std::vector<sol*> Pareto_front, int m)
{

	// Number of solutions
	unsigned int l = Pareto_front.size();

	if (Pareto_front.size() <= 2)
		Pareto_front[0]->cd = 1.0f;

	else if (m == 0) // Sorts by Return (ROI)
	{

		// Obtains the minimum and maximum values found in 'front_id'
		// for the objective 'transportation_cost'. This is equivalent to
		// finding extreme solutions within the front_id.

		std::sort (Pareto_front.begin(), Pareto_front.end(), cmp_ROI_ptr);
		float min_cost = Pareto_front.back()->P.ROI;
		float max_cost = Pareto_front.front()->P.ROI;

		// Assigns infinite crowding distance to extrema.
		Pareto_front.front()->cd = (std::numeric_limits<float>::max()/2.0f)-1.0f;
		Pareto_front.back()->cd = (std::numeric_limits<float>::max()/2.0f)-1.0f;

		// Computes CD (with objectives normalized between 0 and 1) for the remaining solutions
		for (unsigned int i = 1; i < l-1; ++i)
		{
			float norm_cost1 = (min_cost == max_cost) ?
					1 : (Pareto_front[i+1]->P.ROI - min_cost)/(max_cost - min_cost);
			float norm_cost2 = (min_cost == max_cost) ? 1 : (Pareto_front[i-1]->P.ROI - min_cost)/(max_cost - min_cost);

			Pareto_front[i]->cd +=  norm_cost2 - norm_cost1;
		}

	} 
	else if (m == 1) // Same thing as above, but sorting by risk
	{

		std::sort (Pareto_front.begin(), Pareto_front.end(), cmp_risk_ptr);
		float min_cost = Pareto_front.front()->P.risk;
		float max_cost = Pareto_front.back()->P.risk;

		Pareto_front.front()->cd = (std::numeric_limits<float>::max()/2.0f)-2.0f;
		Pareto_front.back()->cd = (std::numeric_limits<float>::max()/2.0f)-2.0f;

		for (unsigned int i = 1; i < l-1; ++i)
		{
			float norm_cost1 = (min_cost == max_cost) ? 1 : (Pareto_front[i+1]->P.risk - min_cost)/(max_cost - min_cost);
			float norm_cost2 = (min_cost == max_cost) ? 1 : (Pareto_front[i-1]->P.risk - min_cost)/(max_cost - min_cost);

			Pareto_front[i]->cd +=  (norm_cost1 - norm_cost2);
		}
	}

}

// Requires population P to be sorted by front_id
void crowding_distance(std::vector<sol*> &P, unsigned int front_id)
{

	unsigned int i = 0, start = 0, end;

	std::vector<sol*> Pareto_front;

	while (i < P.size() && P[i]->Pareto_front != front_id)
		++i;

	start = i;

	while (i < P.size() && P[i]->Pareto_front == front_id)
	{
		Pareto_front.push_back(P[i]);
		P[i]->cd = 0;
		++i;
	}

	end = i - 1;

	// For both objectives
	assigns_crowding_distance_obj(Pareto_front, 0);
	assigns_crowding_distance_obj(Pareto_front, 1);

	std::sort(P.begin() + start, P.begin() + end, cmp_cd_ptr);
}


unsigned int fast_nondominated_sort(std::vector<sol*> &P)
{
	unsigned int num_fronts = 0;
	std::vector<std::vector<unsigned int> > F;
	std::vector<std::vector<unsigned int> > S(P.size());
	std::vector<unsigned int> n(P.size());

	std::vector<unsigned int>H ;

	for (unsigned int p = 0; p < P.size(); ++p)
	{
		for (unsigned int q = 0; q < P.size(); ++q)
			if (p != q)
			{

				if (constrained_dominance(P[p],P[q]))
					S[p].push_back(q);
				else if (constrained_dominance(P[q],P[p]))
					++n[p];
			}

		if (n[p] == 0)
		{
			H.push_back(p);
			P[H.back()]->Pareto_front = 0;
		}
	}
	// Obtains the first front (non-dominated solutions)
	F.push_back(H);

	num_fronts++;

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
					P[H.back()]->Pareto_front = i + 1;

				}
			}
		}
		F.push_back(H);
		++i; ++num_fronts;
	}

	std::sort(P.begin(), P.end(), cmp_front_ptr);

	return num_fronts-1;
}

unsigned int observed_fast_nondominated_sort(std::vector<sol*> &P)
{
	unsigned int num_fronts = 0;
	std::vector<std::vector<unsigned int> > F;
	std::vector<std::vector<unsigned int> > S(P.size());
	std::vector<unsigned int> n(P.size());

	std::vector<unsigned int>H ;

	for (unsigned int p = 0; p < P.size(); ++p)
	{
		for (unsigned int q = 0; q < P.size(); ++q)
			if (p != q)
			{
				if (constrained_observed_dominance(P[p],P[q]))

					S[p].push_back(q);
				else if (constrained_observed_dominance(P[q],P[p]))
					++n[p];
			}

		if (n[p] == 0)
		{
			H.push_back(p);
			P[H.back()]->observed_Pareto_front = 0;
		}
	}

	F.push_back(H);

	num_fronts++;

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
					P[H.back()]->observed_Pareto_front = i + 1;

				}
			}
		}
		F.push_back(H);
		++i; ++num_fronts;
	}

	std::sort(P.begin(),P.end(),cmp_observed_front_ptr);

	return num_fronts-1;
}

void compute_Delta_S_front_id(std::vector<sol*> &Pareto_front, float R_1, float R_2)
{

	if (Pareto_front.size() == 1)
	{
		Pareto_front[0]->Delta_S = (Pareto_front[0]->P.ROI - R_1)*(R_2 - Pareto_front[0]->P.risk);
		return;
	}

	sort_per_objective(Pareto_front,0);

	unsigned int i = 0;
	double delta_Si = (Pareto_front[i]->P.ROI - Pareto_front[i+1]->P.ROI) *
						  (R_2 - Pareto_front[i]->P.risk);

	delta_Si *= Pareto_front[i]->stability;
	Pareto_front[i]->Delta_S = delta_Si;

	i = 1;

	while (i < Pareto_front.size() - 1)
	{
		delta_Si = (Pareto_front[i]->P.ROI - Pareto_front[i+1]->P.ROI) *
					(Pareto_front[i-1]->P.risk - Pareto_front[i]->P.risk);

		delta_Si *= Pareto_front[i]->stability;

		Pareto_front[i]->Delta_S = delta_Si;
		++i;
	}

	delta_Si = (Pareto_front[i]->P.ROI - R_1) *
				   (Pareto_front[i-1]->P.risk - Pareto_front[i]->P.risk);
	delta_Si *= Pareto_front[i]->stability;
	Pareto_front[i]->Delta_S = delta_Si;
}

struct stochastic_params
{
	double cov, var_ROI, var_risk, corr, var_ratio;
	double conditional_mean_ROI, conditional_var_ROI;
	double conditional_mean_risk, conditional_var_risk;

	stochastic_params(sol* w)
	{

		cov = w->P.covar(0,1);
		var_ROI = w->P.covar(0,0);
		var_risk = w->P.covar(1,1);
		corr = cov / (sqrt(var_ROI)*sqrt(var_risk));
		var_ratio = sqrt(var_ROI)/sqrt(var_risk);

		conditional_mean_ROI = w->P.ROI;
		conditional_var_ROI = (1.0 - corr*corr)*var_ROI;

		conditional_mean_risk = w->P.risk;
		conditional_var_risk = (1.0 - corr*corr)*var_risk;
	}
};

double mean_delta_product(sol* w_0, sol* w_1, sol* w_2)
{

	stochastic_params sw_0(w_0);
	stochastic_params sw_1(w_1);
	stochastic_params sw_2(w_2);

	double covar = 0.0; unsigned int N = w_0->P.S.samples.size();
	for (unsigned int j = 0; j < N; ++j)
		covar += (w_1->P.S.samples[j].roi - w_1->P.S.mean_roi)*(w_0->P.S.samples[j].risk - w_0->P.S.mean_risk);
	covar /= (N-1);

	double term1 = sw_1.conditional_mean_ROI*sw_0.conditional_mean_risk + covar;
	double term2 = sw_1.conditional_mean_ROI*sw_1.conditional_mean_risk + w_1->P.S.covar;

	covar = 0.0;
	for (unsigned int j = 0; j < N; ++j)
		covar += (w_2->P.S.samples[j].roi - w_2->P.S.mean_roi)*(w_0->P.S.samples[j].risk - w_0->P.S.mean_risk);
	covar /= (N-1);

	double term3 = sw_2.conditional_mean_ROI*sw_0.conditional_mean_risk + covar;

	covar = 0.0;
	for (unsigned int j = 0; j < N; ++j)
		covar += (w_2->P.S.samples[j].roi - w_2->P.S.mean_roi)*(w_1->P.S.samples[j].risk - w_1->P.S.mean_risk);
	covar /= (N-1);

	double term4 = sw_2.conditional_mean_ROI*sw_1.conditional_mean_risk + covar;

	return term1 - term2 - term3 + term4;
}

void compute_stochastic_Delta_S_front_id(std::vector<sol*> &Pareto_front, float R_1, float R_2)
{

	if (Pareto_front.size() == 1)
	{
		stochastic_params sw_1(Pareto_front[0]);

		//deltaS = (a - b)*(c - d) = ac - ad - bc + bd
		// ac = term1, ad = term2, bc = term3, bd = term4
		// E[xy] = E[x]E[y] + cov(x,y)

		// deltaS = (sw_1.conditional_mean_ROI - R_1)*(R_2 - sw_1.conditional_mean_risk)
		double term1 = sw_1.conditional_mean_ROI*R_2 + 0.0;
		double term2 = sw_1.conditional_mean_ROI*sw_1.conditional_mean_risk + Pareto_front[0]->P.S.covar;
		double term3 = R_1*R_2 + 0.0;
		double term4 = R_1*sw_1.conditional_mean_risk + 0.0;

		Pareto_front[0]->Delta_S = term1 - term2 - term3 + term4;

		return;
	}

	sort_per_objective(Pareto_front,0);

	unsigned int i = 0;

	stochastic_params sw_1(Pareto_front[i]);
	stochastic_params sw_2(Pareto_front[i+1]);

	//deltaS = (a - b)*(c - d) = ac - ad - bc + bd
	// ac = term1, ad = term2, bc = term3, bd = term4
	// E[xy] = E[x]E[y] + cov(x,y)

	// deltaS = (sw_1.conditional_mean_ROI - sw_2.conditional_mean_ROI)*(R_2 - sw_1.conditional_mean_risk)
	double term1 = sw_1.conditional_mean_ROI*R_2 + 0.0;
	double term2 = sw_1.conditional_mean_ROI*sw_1.conditional_mean_risk + Pareto_front[i]->P.S.covar;
	double term3 = sw_2.conditional_mean_ROI*R_2 + 0.0;

	double covar = 0.0; unsigned int N = Pareto_front[i]->P.S.samples.size();
	for (unsigned int j = 0; j < N; ++j)
		covar += (Pareto_front[i+1]->P.S.samples[j].roi - Pareto_front[i+1]->P.S.mean_roi)*(Pareto_front[i]->P.S.samples[j].risk - Pareto_front[i]->P.S.mean_risk);
	covar /= (N-1);

	double term4 = sw_2.conditional_mean_ROI*sw_1.conditional_mean_risk + covar;

	Pareto_front[i]->Delta_S = term1 - term2 - term3 + term4;


	i = 1;

	double delta_Si = 0.0;
	while (i < Pareto_front.size() - 1)
	{
		delta_Si = mean_delta_product(Pareto_front[i-1], Pareto_front[i], Pareto_front[i+1]);
		Pareto_front[i]->Delta_S = delta_Si;
		++i;
	}

	stochastic_params sw_l(Pareto_front[i-1]);
	stochastic_params sw_r(Pareto_front[i]);

	covar = 0.0;
	for (unsigned int j = 0; j < N; ++j)
		covar += (Pareto_front[i]->P.S.samples[j].roi - Pareto_front[i]->P.S.mean_roi)*(Pareto_front[i-1]->P.S.samples[j].risk - Pareto_front[i-1]->P.S.mean_risk);
	covar /= (N-1);

	term1 = sw_r.conditional_mean_ROI*sw_l.conditional_mean_risk + covar;
	term2 = sw_r.conditional_mean_ROI*sw_r.conditional_mean_risk + Pareto_front[i]->P.S.covar;
	term3 = R_1*sw_l.conditional_mean_risk + 0.0;
	term4 = R_1*sw_r.conditional_mean_risk + 0.0;

	Pareto_front[i]->Delta_S = term1 - term2 - term3 + term4;

}

void compute_stochastic_Delta_S(std::vector<sol*> &P, unsigned int num_fronts, float R_1, float R_2)
{
	unsigned int i = 0;
	for (unsigned int c = 0; c < num_fronts; ++c)
	{
		std::vector<sol*> front_id;

		while (P[i]->Pareto_front != c)
			++i;

		while (i < P.size() && P[i]->Pareto_front == c)
		{
			front_id.push_back(P[i]);
			++i;
		}

		if (c == num_fronts-1 && front_id.size() > 3)
			compute_stochastic_Delta_S_front_id(front_id,R_1,R_2);

	}
}

void compute_Delta_S(std::vector<sol*> &P, unsigned int num_fronts, float R_1, float R_2)
{
	unsigned int i = 0;
	for (unsigned int c = 0; c < num_fronts; ++c)
	{
		std::vector<sol*> front_id;

		while (P[i]->Pareto_front != c)
			++i;

		while (i < P.size() && P[i]->Pareto_front == c)
		{
			front_id.push_back(P[i]);
			++i;
		}

		if (c == num_fronts-1 && front_id.size() > 3)
			compute_Delta_S_front_id(front_id,R_1,R_2);
	}
}


void remove_worst_s_metric(std::vector<sol*> &P, double total_error, unsigned int front_id_index, float R_1, float R_2)
{
	unsigned int i = 0, start = 0, end, worst;
	std::vector<std::pair<unsigned int,sol*> > front_id;

	while (P[i]->Pareto_front != front_id_index)
	{
		start++; i++;
	} end = start;
	while (i < P.size() && P[i]->Pareto_front == front_id_index)
	{
		front_id.push_back(std::pair<unsigned int,sol*>(i,P[i]));
		end++; i++;
	}

	end -= 1;

	sort_per_objective(front_id,0);

	if (front_id.size() == 1)
	{
		delete *(P.begin() + front_id[0].first);
		P.erase(P.begin() + front_id[0].first);
		return;
	}
	else if (front_id.size() == 2)
	{

		double alpha1 = front_id[0].second->anticipation_rate;
		double alpha2 = front_id[1].second->anticipation_rate;

		if (alpha1 < alpha2)
		{
			delete *(P.begin() + front_id[0].first);
			P.erase(P.begin() + front_id[0].first);
			return;
		}
		else
		{
			delete *(P.begin() + front_id[1].first);
			P.erase(P.begin() + front_id[1].first);
			return;
		}
	}
	else if (front_id.size() == 3)
	{
		delete *(P.begin() + front_id[1].first);
		P.erase(P.begin() + front_id[1].first);
		return;
	}
	else
	{

		worst = start; i = 0;

		double worst_delta_Si = std::numeric_limits<double>::max(); i = 1;

		while (i < front_id.size() - 1)
		{
			if (front_id[i].second->Delta_S < worst_delta_Si)
			{
				worst_delta_Si = front_id[i].second->Delta_S;
				worst = front_id[i].first;
			}
			++i;
		}

		delete *(P.begin() + worst);
		P.erase(P.begin() + worst);
	}
}

double non_dominance_probability(sol* w)
{
	Eigen::MatrixXd covar(2,2);
	covar << w->P.error_covar_prediction(0,0) + w->P.error_covar(0,0),
			 w->P.error_covar_prediction(0,1) + w->P.error_covar(0,1),
			 w->P.error_covar_prediction(1,0) + w->P.error_covar(1,0),
			 w->P.error_covar_prediction(1,1) + w->P.error_covar(1,1);

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
	Eigen::LLT<Eigen::MatrixXd> lltOfA(covar); // compute Cholesky decomposition of A
	Eigen::MatrixXd U = lltOfA.matrixU();

	Eigen::VectorXd z1 = U.inverse()*(u - delta1);
	Eigen::VectorXd z2 = U.inverse()*(u - delta2);

	// Compute probability of w(t+1) being non-dominated w.r.t. w(t)
	double p = normal_cdf(z1,Eigen::MatrixXd::Identity(2,2)) + normal_cdf(z2,Eigen::MatrixXd::Identity(2,2));

	return p;
}

double non_dominance_probability(sol* w1, sol* w2)
{
	Eigen::MatrixXd covar(2,2);
	covar << w1->P.error_covar(0,0) + w2->P.error_covar(0,0),
			 w1->P.error_covar(0,1) + w2->P.error_covar(0,1),
			 w1->P.error_covar(1,0) + w2->P.error_covar(1,0),
			 w1->P.error_covar(1,1) + w2->P.error_covar(1,1);

	Eigen::VectorXd delta1(2), delta2(2), u(2);

	// Pr{ROI(t+1) > ROI(t) & Risk(t+1) > Risk(t)}
	delta1 << w1->P.ROI - w2->P.ROI,
			  w1->P.risk - w2->P.risk;

	// Pr{ROI(t+1) < ROI(t) & Risk(t+1) < Risk(t)}
	delta2 << w2->P.ROI -  w1->P.ROI,
			  w2->P.risk - w1->P.risk;
	// Point of interest for Pr {Delta < u = [0 0]^T}
	u << 0.0, 0.0;

	// Compute the change of variables to standardize the multivariate Gaussian
	Eigen::LLT<Eigen::MatrixXd> lltOfA(covar); // compute Cholesky decomposition of A
	Eigen::MatrixXd U = lltOfA.matrixU();

	Eigen::VectorXd z1 = U.inverse()*(u - delta1);
	Eigen::VectorXd z2 = U.inverse()*(u - delta2);

	// Compute probability of w(t+1) being non-dominated w.r.t. w(t)
	double p = normal_cdf(z1,Eigen::MatrixXd::Identity(2,2)) + normal_cdf(z2,Eigen::MatrixXd::Identity(2,2));
		return p;
}


void anticipatory_learning_obj_space(sol* &w, sol* anticipative_portfolio, Eigen::VectorXd& current_w, double min_error, double max_error, double min_alpha, double max_alpha, unsigned int t)
{

	if (anticipative_portfolio == NULL)
		anticipative_portfolio = w;

	if (anticipative_portfolio->prediction_error < min_error)
		min_error = anticipative_portfolio->prediction_error;
	else if (anticipative_portfolio->prediction_error > max_error)
		max_error = anticipative_portfolio->prediction_error;

	if (anticipative_portfolio->alpha < min_alpha)
		min_alpha = anticipative_portfolio->alpha;
	else if (anticipative_portfolio->alpha > max_alpha)
		max_alpha = anticipative_portfolio->alpha;

	// Updates return and risk based on the anticipatory learning rule
	double accuracy_factor = (min_error > 0.0 && (max_error - min_error) > 0.0)
			? 1.0 - (anticipative_portfolio->prediction_error - min_error)/(max_error - min_error)
					: 0.0;
	double uncertainty_factor = anticipative_portfolio->alpha;

	double rate_upb = (t == 0 || Kalman_params::window_size == 0) ? 0.0 : 0.5;
	double rate_lwb = (t == 0 || Kalman_params::window_size == 0) ? 0.0 : 0.0;

	w->anticipation_rate = rate_lwb + 0.5*uncertainty_factor*(rate_upb - rate_lwb) + 0.5*accuracy_factor*(rate_upb - rate_lwb);

	Eigen::VectorXd x_state = Eigen::VectorXd::Zero(4);
	x_state << anticipative_portfolio->P.ROI, anticipative_portfolio->P.risk, 0.0, 0.0;

	Eigen::VectorXd x = x_state; Eigen::MatrixXd C = anticipative_portfolio->P.kalman_state.P;

	if (Kalman_params::window_size > 0)
	{
		x = x_state + w->anticipation_rate*(anticipative_portfolio->P.kalman_state.x_next - x_state);

		C = anticipative_portfolio->P.kalman_state.P
				+ w->anticipation_rate*w->anticipation_rate*
				(anticipative_portfolio->P.kalman_state.P_next
						- anticipative_portfolio->P.kalman_state.P);
	}

	anticipative_portfolio->P.ROI = x(0);

	if (anticipative_portfolio != NULL)
	{
		std::pair<double,double> cost_current = portfolio::transaction_cost(portfolio::current_investment, w->P.investment);
		std::pair<double,double> cost_predicted = portfolio::transaction_cost(current_w, anticipative_portfolio->P.investment);

		double adjusted_ROI = (portfolio::current_wealth + cost_current.first) / portfolio::current_wealth
							* inverse_linear_transform(anticipative_portfolio->P.ROI,
												portfolio::min_ROI,portfolio::max_ROI);

		double nominal_ROI = (portfolio::current_wealth - cost_predicted.first)*adjusted_ROI;
		anticipative_portfolio->P.ROI = linear_transform((nominal_ROI) / portfolio::current_wealth,portfolio::min_ROI,portfolio::max_ROI);
	}

	if (x(1) > 0.0)
		anticipative_portfolio->P.risk = x(1);
	else if (x(1) < 0.0 && anticipative_portfolio->P.kalman_state.x(1) > 0.0)
		anticipative_portfolio->P.risk = anticipative_portfolio->P.kalman_state.x(1);
	else if (x(1) < 0.0 && anticipative_portfolio->P.kalman_state.x(1) <= 0.0)
		anticipative_portfolio->P.risk = 0.0; // Does not allow negative risk

	anticipative_portfolio->P.covar = C;
	w->P.covar = C;

	w->P.kalman_state = anticipative_portfolio->P.kalman_state;
	w->P.ROI = anticipative_portfolio->P.ROI;
	w->P.risk = anticipative_portfolio->P.risk;

	w->anticipation = true;
	anticipative_portfolio->anticipation = true;
}

std::pair<double,double> anticipatory_learning_obj_space(std::vector<sol*> &P, unsigned int t)
{

	double min_error = std::numeric_limits<float>::max();
	double max_error = 0.0;
	double min_alpha = 1.0, max_alpha = 0.0;

	for (unsigned int i = 0; i < P.size(); ++i)
	{
		unsigned int initial_t = (t > Kalman_params::window_size) ? (t - Kalman_params::window_size) : 0;


		if (!P[i]->anticipation)
			portfolio::observe_state(P[i]->P, portfolio::samples_per_portfolio, initial_t, t);
		else
			P[i]->prediction_error = 0.0;

		// Corrects bounds, just in case (should not happen)
		if (P[i]->alpha < min_alpha)
			min_alpha = P[i]->alpha;
		else if (P[i]->alpha > max_alpha)
			max_alpha = P[i]->alpha;

		if (P[i]->prediction_error < min_error)
			min_error = P[i]->prediction_error;
		else if (P[i]->prediction_error > max_error)
			max_error = P[i]->prediction_error;

	}

	for (unsigned int i = 0; i < P.size(); ++i)
		if (!P[i]->anticipation)
		{
			if (t > 0)
			{
				sol* anticipative_portfolio = anticipatory_learning_dec_space(P, i, t);

				anticipative_portfolio->alpha = 1.0 - non_dominance_probability(anticipative_portfolio);
				anticipatory_learning_obj_space(P[i], anticipative_portfolio, solution::predicted_anticipative_decision->P.investment, min_error, max_error, min_alpha, max_alpha, t);
			}
			else
			{
				P[i]->alpha = 1.0 - non_dominance_probability(P[i]);
				anticipatory_learning_obj_space(P[i], NULL, portfolio::current_investment, min_error, max_error, min_alpha, max_alpha, t);
			}
		}

	return std::pair<double,double>(min_error,max_error);
}

sol* anticipatory_learning_dec_space(std::vector<sol*> pop, unsigned int i, unsigned int current_t)
{
	if (current_t == 0 || Kalman_params::window_size == 0)
		return pop[i];

	unsigned int initial_t = (current_t > Kalman_params::window_size) ? (current_t - Kalman_params::window_size) : 0;

	sol* w_updated = NULL, * w_current = NULL, * w_prediction = NULL;
	sol* w_previous = new sol(solution::historical_populations[initial_t][i]);
	for (unsigned int t = initial_t+1; t < current_t; ++t)
	{
		w_current = new sol(solution::historical_populations[t][i]);
		double previous_concentration = w_previous->P.nominal_ROI;
		double current_concentration = w_current->P.nominal_ROI;
		double concentration = previous_concentration + current_concentration;

		w_prediction = dirichlet_mean_prediction(w_previous, w_current, t);
		delete w_previous; delete w_current;
		w_updated = w_prediction;

		if (t == current_t-1)
			w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
						pop[i]->P.investment, concentration);
		else
		{
			if (t+1 < current_t)
				w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
						solution::historical_populations[t+1][i]->P.investment, concentration);
			else
				w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
										pop[i]->P.investment, concentration);
		}
		w_previous = w_updated;
	}

	w_current = pop[i];
	w_prediction = dirichlet_mean_prediction(w_previous, w_current, current_t);
	return w_prediction;
}

sol* anticipatory_learning_dec_space(std::vector<sol*> pop_current, std::vector<sol*> pop_pred, unsigned int i, unsigned int current_t)
{
	if (current_t == 0 || Kalman_params::window_size == 0)
		return pop_pred[i];

	unsigned int initial_t = (current_t > Kalman_params::window_size) ? (current_t - Kalman_params::window_size) : 0;

	sol* w_updated = NULL, * w_current = NULL, * w_prediction = NULL;
	sol* w_previous = (initial_t+1 == current_t) ? new sol(pop_current[i]) : new sol(solution::historical_populations[initial_t+1][i]);
	for (unsigned int t = initial_t+1; t < current_t; ++t)
	{
		w_current = (t == current_t-1) ? new sol(pop_current[i]) : new sol(solution::historical_populations[t+1][i]);
		double previous_concentration = w_previous->P.nominal_ROI;
		double current_concentration = w_current->P.nominal_ROI;
		double concentration = previous_concentration + current_concentration;

		w_prediction = dirichlet_mean_prediction(w_previous, w_current, t);
		delete w_previous; delete w_current;
		w_updated = w_prediction;

		if (t == current_t-1)
			w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
					pop_current[i]->P.investment, concentration);
		else
		{
			if (t+1 < current_t)
				w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
						solution::historical_populations[t+1][i]->P.investment, concentration);
			else
				w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
							pop_current[i]->P.investment, concentration);
		}
		w_previous = w_updated;
	}

	w_current = pop_pred[i];
	w_prediction = dirichlet_mean_prediction(w_previous, w_current, current_t);
	return w_prediction;

}

sol* anticipatory_learning_dec_space(sol *w, unsigned int current_t)
{
	if (current_t == 0 || Kalman_params::window_size == 0)
		return w;

	sol* anticipative_portfolio;
	unsigned int initial_t = (current_t > Kalman_params::window_size) ? (current_t - Kalman_params::window_size) : 0;
	sol* w_updated = NULL, * w_current = NULL, * w_prediction = NULL;

	sol* w_previous = new sol(solution::historical_anticipative_decisions[initial_t]);
;
	for (unsigned int t = initial_t+1; t < current_t; ++t)
	{
		w_current = new sol(solution::historical_anticipative_decisions[t]);
		w_prediction = dirichlet_mean_prediction(w_previous, w_current, t);
		delete w_previous; delete w_current;
		w_updated = w_prediction;

		double previous_concentration = w_previous->P.nominal_ROI;
		double current_concentration = w_current->P.nominal_ROI;
		double concentration = previous_concentration + current_concentration;

		if (t == current_t-1)
			w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
						w->P.investment, concentration);
		else
		{
			if (t+1 < current_t)
				w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
						solution::historical_anticipative_decisions[t+1]->P.investment, concentration);
			else
				w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
										w->P.investment, concentration);
		}

		w_previous = w_updated;
	}

	w_current = w;
	w_prediction = dirichlet_mean_prediction(w_previous, w_current, current_t);

	anticipative_portfolio = w_prediction;
	return anticipative_portfolio;
}

sol* anticipatory_learning_dec_space(sol *w, unsigned int i, unsigned int current_t)
{
	if (current_t == 0 || Kalman_params::window_size == 0)
		return w;

	unsigned int initial_t = (current_t > Kalman_params::window_size) ? (current_t - Kalman_params::window_size) : 0;

	sol* w_updated = NULL, * w_current = NULL, * w_prediction = NULL;
	sol* w_previous = new sol(solution::historical_populations[initial_t][i]);

	for (unsigned int t = initial_t+1; t < current_t; ++t)
	{
		w_current = new sol(solution::historical_populations[t][i]);

		double previous_concentration = w_previous->P.nominal_ROI;
		double current_concentration = w_current->P.nominal_ROI;
		double concentration = previous_concentration + current_concentration;

		w_prediction = dirichlet_mean_prediction(w_previous, w_current, t);
		delete w_current; delete w_previous;
		w_updated = w_prediction;

		if (t == current_t-1)
			w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
						w->P.investment, concentration);
		else
		{
			if (t+1 < current_t)
				w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
						solution::historical_populations[t+1][i]->P.investment, concentration);
			else
				w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
										w->P.investment, concentration);
		}
		w_previous = w_updated;
	}

	w_current = w;
	w_prediction = dirichlet_mean_prediction(w_previous, w_current, current_t);
	return w_prediction;
}


std::vector<sol*> anticipatory_learning_dec_space(const std::vector<sol*> pop, unsigned int current_t)
{
	if (current_t == 0 || Kalman_params::window_size == 0)
		return pop;

	std::vector<sol*> anticipative_population(pop.size());

	for (unsigned int i = 0; i < pop.size(); ++i)
	{
		unsigned int initial_t = (current_t > Kalman_params::window_size) ? (current_t - Kalman_params::window_size) : 0;

		sol* w_updated = NULL, * w_current = NULL, * w_prediction = NULL;
		sol*  w_previous = new sol(solution::historical_populations[initial_t][i]);

		for (unsigned int t = initial_t+1; t < current_t; ++t)
		{

			w_current = new sol(solution::historical_populations[t][i]);
			double previous_concentration = w_previous->P.nominal_ROI;
			double current_concentration = w_current->P.nominal_ROI;
			double concentration = previous_concentration + current_concentration;

			w_prediction = dirichlet_mean_prediction(w_previous, w_current, t);
			delete w_current; delete w_previous;
			w_updated = w_prediction;

			if (t == current_t-1)
				w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
							pop[i]->P.investment, concentration);
			else
				w_updated->P.investment = dirichlet_mean_map_update(w_prediction->P.investment,
							solution::historical_populations[t+1][i]->P.investment, concentration);

			w_previous = w_updated;
		}

		w_current = pop[i];
		w_prediction = dirichlet_mean_prediction(w_previous, w_current, current_t);
		anticipative_population[i] = w_prediction;
	}

	return anticipative_population;
}

void apply_mutation(sol* w)
{
	unsigned min_weight;
	while (portfolio::card(w->P) > portfolio::max_cardinality)
	{
		min_weight = 0;
		for (unsigned int i = 0; i < portfolio::num_assets; ++i)
			if (w->P.investment(i) > 0)
			{
				min_weight = i;
				break;
			}

		for (unsigned int i = min_weight+1; i < portfolio::num_assets; ++i)
			if (w->P.investment(i) > 0 && w->P.investment(i) < w->P.investment(min_weight))
				min_weight = i;

		remove_asset(w, min_weight);
	}

	bool low_card = false; std::vector<unsigned int> indices;
	while (portfolio::card(w->P) < 2)
	{

		if (!low_card)
			for (unsigned int i = 0; i < portfolio::num_assets; ++i)
				if (w->P.investment(i) == 0)
						indices.push_back(i);

			std::random_shuffle(indices.begin(), indices.end());

			for (unsigned int i = 0; i < 2 - portfolio::card(w->P); ++i)
			{
				add_asset(w,indices.back());
				indices.pop_back();
			}
			low_card = true;
		}
}

void ASMS_EMOA_iteration(std::vector<sol*> &P, float R1, float R2, unsigned int t, bool anticipation)
{
	xover_op::probability = xover_op::uniform_prob_op;
	mutation_op::probability = mutation_op::uniform_prob_op;

	double total_error = 0.0;

	std::pair<double,double> error;


	if (anticipation == true)
		error = anticipatory_learning_obj_space(P, t);

	unsigned int num_fronts = fast_nondominated_sort(P);

	if (anticipation == true)
		compute_stochastic_Delta_S(P,num_fronts, R1, R2);
	else
		compute_Delta_S(P,num_fronts, R1, R2);

	unsigned int parent1 = tournament_Delta_S(P, error, solution::tourn_size);
	unsigned int parent2;

	sol * offspring1;
	if (uniform_zero_one() < solution::xover_rate)
	{
		do
		{
			parent2 = tournament_Delta_S(P, error, solution::tourn_size);
		} while (parent1 == parent2);

		offspring1 = new sol(); sol * offspring2 = new sol();

		crossover_operator_ptr xover_op = roulette_wheel_selection_crossover();
		xover_op(P[parent1], P[parent2], offspring1, offspring2);
		delete offspring2;
	}
	else
		offspring1 = new sol(P[parent1]);

	mutation_operator_ptr mutation_op = roulette_wheel_selection_mutation();

	double rate = solution::mut_rate;
	mutation_op(offspring1, rate);
	apply_mutation(offspring1);

	if (t >= 0 && anticipation == true)
	{
		unsigned int initial_t = (t > Kalman_params::window_size) ? (t - Kalman_params::window_size) : 0;

		portfolio::observe_state(offspring1->P, 20, initial_t, t);
		offspring1->alpha = 1.0 - non_dominance_probability(offspring1);

		double min_error = std::numeric_limits<float>::max();
		double max_error = 0.0;
		double min_alpha = 1.0, max_alpha = 0.0;

		for (unsigned int i = 0; i < P.size(); ++i)
		{
			if (P[i]->alpha < min_alpha)
				min_alpha = P[i]->alpha;
			else if (P[i]->alpha > max_alpha)
				max_alpha = P[i]->alpha;

			if (P[i]->prediction_error < min_error)
				min_error = P[i]->prediction_error;
			else if (P[i]->prediction_error > max_error)
				max_error = P[i]->prediction_error;
		}

		if (offspring1->alpha < min_alpha)
			min_alpha = offspring1->alpha;
		else if (offspring1->alpha > max_alpha)
			max_alpha = offspring1->alpha;

		if (offspring1->prediction_error < min_error)
			min_error = offspring1->prediction_error;
		else if (offspring1->prediction_error > max_error)
			max_error = offspring1->prediction_error;

		solution::num_fronts = num_fronts;
		if (t > 0)
		{
			unsigned ROI_rank = 0;
			for (unsigned int i = 0; i < P.size(); ++i)
				if (offspring1->P.ROI > P[i]->P.ROI)
				{
					ROI_rank = i;
					break;
				}
			 if (ROI_rank == P.size())
				 ROI_rank -= 1;

			sol * anticipative_offspring = anticipatory_learning_dec_space(offspring1, ROI_rank, t);
			anticipatory_learning_obj_space(offspring1, anticipative_offspring, solution::predicted_anticipative_decision->P.investment, min_error, max_error, min_alpha, max_alpha, t);
		}
		else
			anticipatory_learning_obj_space(offspring1, NULL, portfolio::current_investment, min_error, max_error, min_alpha, max_alpha, t);
	}
	else if (t > 0)
		solution::compute_efficiency(offspring1);
	else
	{
		offspring1->prediction_error = 0.0;
		solution::compute_efficiency(offspring1);
	}

	solution::num_fronts = num_fronts;
	P.push_back(offspring1);
	num_fronts = fast_nondominated_sort(P);

	if (t >= 0 && anticipation == true)
		compute_stochastic_Delta_S(P,num_fronts, R1, R2);
	else
		compute_Delta_S(P,num_fronts, R1, R2);

	remove_worst_s_metric(P, total_error, num_fronts-1, R1, R2);

	for (unsigned int p = 0; p < P.size(); ++p)
		portfolio::observed_error(P[p]->P, portfolio::current_period);
}

void NSGA2_iteration(std::vector<sol*> &P, unsigned int t)
{

	xover_op::probability = xover_op::uniform_prob_op;
	mutation_op::probability = mutation_op::uniform_prob_op;

	unsigned int num_fronts = fast_nondominated_sort(P);
	
	for (unsigned int i = 0; i < num_fronts; ++i)
		crowding_distance(P, i);


	unsigned int novap_size = 2*P.size();

	while (P.size() < novap_size)
	{
		unsigned int parent1 = tournament_selection_CD(P, solution::tourn_size);
		unsigned int parent2;

		mutation_operator_ptr mut_operator = roulette_wheel_selection_mutation();

		sol * offspring1, * offspring2;
		if (uniform_zero_one() < solution::xover_rate)
		{

			do
			{
				parent2 = tournament_selection_CD(P, solution::tourn_size);
			} while (parent1 == parent2);

			offspring1 = new sol(); offspring2 = new sol();

			crossover_operator_ptr xover_operator = roulette_wheel_selection_crossover();
			xover_operator(P[parent1], P[parent2], offspring1, offspring2);

			mut_operator = roulette_wheel_selection_mutation();
			mut_operator(offspring2, solution::mut_rate);

		}
		else
			offspring1 = P[parent1];

		mut_operator = roulette_wheel_selection_mutation();
		mut_operator(offspring1, solution::mut_rate);

		solution::compute_efficiency(offspring1);
		solution::compute_efficiency(offspring2);

		P.push_back(offspring1);
		P.push_back(offspring2);
	}

	num_fronts = fast_nondominated_sort(P);
	for (unsigned int i = 0; i < num_fronts; ++i)
		crowding_distance(P, i);

	std::vector<sol*>::iterator it = P.begin() + P.size()/2;
	for (; it != P.end(); ++it)
		delete *it;

	P.erase(P.begin() + P.size()/2, P.end());

}
