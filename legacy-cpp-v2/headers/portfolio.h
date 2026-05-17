#ifndef PORTFOLIO_H
	#define PORTFOLIO_H

#include "utils.h"
#include "kalman_filter.h"

#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <Eigen/Eigen/Dense>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>

#define SIGN(x) (x >= 0 ? 1 : -1)
#define IS_POSITIVE(x) SIGN(x) == 1
#define IS_NEGATIVE(x) SIGN(x) == -1
#define IS_ZERO(x) (x == 0) ? 1 : 0


using namespace boost;

std::vector<std::pair<Eigen::VectorXd,Eigen::MatrixXd> > load_benchmark_data(const std::string& path);

struct stats
{
	double roi, risk;
	stats (double r1, double r2) { roi = r1; risk = r2; }
	stats() { roi = risk = 0.0; }
};

struct stats_vec
{
	std::vector<stats> samples;
	double mean_roi, mean_risk, var_roi, var_risk, covar;

	stats_vec()
	{
		samples.resize(0);
		mean_roi = mean_risk = var_roi = var_risk = covar = 0.0;
	}

	stats_vec(const std::vector<stats>& s, double m_roi, double m_risk, double v_roi, double v_risk, double c)
	{
		samples = s;
		mean_roi = m_roi, mean_risk = m_risk, var_roi = v_roi, var_risk = v_risk, covar = c;
	}
};

struct portfolio
{
	stats_vec S;
	static unsigned int samples_per_portfolio;
	static unsigned int min_cardinality;
	static unsigned int max_cardinality;
	static unsigned int current_period;
	static unsigned int periodicity;
	static unsigned int num_assets;
	static unsigned int severity;
	static unsigned int periods;
	static double min_hold;
	static double max_hold;
	static double brokerage_commission;
	static double brokerage_fee;
	static double total_wealth;
	static double current_wealth;
	static double total_incurred_cost;
	static double min_cost;
	static double max_cost;
	static double max_ROI;
	static double max_risk;
	static double min_ROI;
	static double min_risk;


	static double get_max_ROI(unsigned int t)
	{
		double max_max_ROI = 0.0;
		for (unsigned int t = 0; t < portfolio::periods; ++t)
		{
			Eigen::VectorXd value = portfolio::sequence_mean_covar[t].first.colwise().maxCoeff();
			double max_ROI = value(0);

			if (max_ROI > max_max_ROI)
				max_max_ROI = max_ROI;
		}
		return max_max_ROI;
		//return max_ROI;
	}

	static double get_min_ROI(unsigned int t)
	{
		double min_min_ROI = 0.0;
		for (unsigned int t = 0; t < portfolio::periods; ++t)
		{
			Eigen::VectorXd value = portfolio::sequence_mean_covar[t].first.colwise().minCoeff();
			double min_ROI = value(0);
			if (min_ROI < min_min_ROI)
				min_min_ROI = min_ROI;
		}
		return min_min_ROI;
	}

	static double get_max_risk(unsigned int t)
	{
		double max_max_risk = 0.0;
		for (unsigned int t = 0; t < portfolio::periods; ++t)
		{
			double max_risk = 0.0;
			for (int i = 0; i < portfolio::sequence_mean_covar[t].second.rows(); ++i)
				for (int j = 0; j < portfolio::sequence_mean_covar[t].second.cols(); ++j)
					if (portfolio::sequence_mean_covar[t].second(i,j) > max_risk)
						max_risk = portfolio::sequence_mean_covar[t].second(i,j);

			if (max_risk > max_max_risk)
				max_max_risk = max_risk;
		}
		return max_max_risk;
	}

	static double get_min_risk()
	{
		return 0.0;
	}

	static std::vector<std::pair<Eigen::VectorXd,Eigen::MatrixXd> > sequence_mean_covar;

	static std::vector<std::pair<unsigned,double> > top_k_investments (const Eigen::VectorXd& investment, unsigned int k);

	static double compute_risk(portfolio &P,const Eigen::MatrixXd &covariance);
	static double compute_ROI(portfolio &P, const Eigen::VectorXd &mean_ROI);

	static Eigen::VectorXd estimate_assets_mean_ROI(const Eigen::MatrixXd &return_data);
	static void estimate_covariance(const Eigen::VectorXd &mean_ROI,
				const Eigen::MatrixXd &return_data, Eigen::MatrixXd &covariance);

	// For use in the Kalman Filter
	static void observe_state(portfolio &w, unsigned int N, unsigned int initial_t, unsigned int current_t);
	static void KF_prediction_obj_space(portfolio &w, unsigned int steps);

	static std::vector<std::pair<Eigen::VectorXd,Eigen::MatrixXd> > get_mean_covar();

	static Eigen::MatrixXd theoretical_covariance;
	static Eigen::VectorXd theoretical_mean_ROI;

	static Eigen::VectorXd current_investment;

	Eigen::MatrixXd sample_covariance;
	Eigen::VectorXd sample_mean_ROI;

	double ROI, ROI_prediction, ROI_observed;
	double risk, risk_prediction, risk_observed;
	double cardinality;
	double obs_error;
	double cid_ROI;
	double cid_risk;
	double cid;
	double current_cost;
	double nominal_ROI;
	double turnover_rate;

	Kalman_params kalman_state;
	Eigen::MatrixXd covar, error_covar, error_covar_prediction;
	Eigen::VectorXd investment;


	portfolio()
	{
		cardinality = .0;
		ROI = risk = ROI_observed = risk_observed = .0;
		ROI_prediction = risk_prediction = .0;
		obs_error = .0; current_cost = .0; turnover_rate = .0;
		nominal_ROI = .0; cid = cid_ROI = cid_risk = .0;
	}

	portfolio(portfolio& w)
	{
		cardinality = w.cardinality;
		ROI = w.ROI;
		ROI_observed = w.ROI_observed;
		risk = w.risk;
		risk_observed = w.risk_observed;
		ROI_prediction = w.ROI_prediction;
		risk_prediction = w.risk_prediction;
		obs_error = w.obs_error;
		current_cost = w.current_cost;
		turnover_rate = w.turnover_rate;
		nominal_ROI = w.nominal_ROI;
		cid = w.cid;
		cid_ROI = w.cid_ROI;
		cid_risk = w.cid_risk;

		kalman_state = w.kalman_state;
		investment = w.investment;
		covar = w.covar;
		error_covar = w.error_covar;
		error_covar_prediction = w.error_covar_prediction;
	}

	static void observed_error(portfolio &w, unsigned int t)
	{
		if (t == (periods-1))
		{
			w.obs_error = 0.0;
			return;
		}

		// What will be actually observed in terms of ROI and Risk
		Eigen::VectorXd mean_curr = portfolio::sequence_mean_covar[t].first;
		Eigen::MatrixXd covar_curr = portfolio::sequence_mean_covar[t].second;
		Eigen::VectorXd mean_next = portfolio::sequence_mean_covar[t+1].first;
		Eigen::MatrixXd covar_next = portfolio::sequence_mean_covar[t+1].second;

		double ROI_curr = compute_ROI(w, mean_curr);
		double risk_curr = compute_risk(w, covar_curr);
		double ROI_next = compute_ROI(w, mean_next);
		double risk_next = compute_risk(w, covar_next);

		std::pair<double,double> cost;

		Eigen::VectorXd Zero_vec = Eigen::VectorXd::Zero(portfolio::num_assets);
		if (t == 0) // Considers the entry cost in the share market (from nothing to something invested)
			cost = portfolio::transaction_cost(Zero_vec, w.investment);
		else
			cost = portfolio::transaction_cost(portfolio::current_investment, w.investment);


		double nominal_ROI = ROI_next*portfolio::current_wealth;
		w.ROI_observed = (nominal_ROI - cost.first) / portfolio::current_wealth;
		w.risk_observed = risk_next;

		// Compute the mean absolute error (MAE)
		double error_ROI = fabs(inverse_linear_transform(w.ROI_observed - w.ROI, min_ROI, max_ROI));
		double error_risk = fabs(inverse_linear_transform(w.risk_observed - w.risk, portfolio::min_risk,max_risk));
		w.obs_error = 0.5*(error_ROI + error_risk);

		// Compute the 1-0 losses for change in direction
		double prediction_direction_ROI = SIGN(w.kalman_state.x_next(0) - w.kalman_state.x(0));
		double prediction_direction_risk = SIGN(w.kalman_state.x_next(1) - w.kalman_state.x(1));
		double actual_direction_ROI = SIGN(ROI_next - ROI_curr);
		double actual_direction_risk = SIGN(risk_next - risk_curr);

		w.cid_ROI = IS_ZERO(actual_direction_ROI - prediction_direction_ROI);
		w.cid_risk = IS_ZERO(actual_direction_risk - prediction_direction_risk);
		w.cid = 0.5*(w.cid_ROI + w.cid_risk);

		w.ROI_observed = linear_transform(w.ROI_observed,min_ROI,max_ROI);
		w.risk_observed = linear_transform(w.risk_observed,min_ROI,max_ROI);
	}


	static double turnover(const Eigen::VectorXd& current_w, const Eigen::VectorXd& new_w)
	{
		double rate = 0.0;

		for (int i = 0; i < new_w.rows(); ++i)
			rate += fabs(current_w(i) - new_w(i));
		return rate;
	}


	static void update_cost_and_wealth(portfolio &implemented_decision, unsigned int t)
	{
		if (t == (periods-1))
		{
			return;
		}

		// What will be actually observed in terms of ROI and Risk
		Eigen::VectorXd mean = portfolio::sequence_mean_covar[t+1].first;
		std::pair<double,double> cost;
		if (t == 0) // Considers the entry cost in the share market (from nothing to something invested)
		{
			Eigen::VectorXd Zero_vec = Eigen::VectorXd::Zero(portfolio::num_assets);
			cost = portfolio::transaction_cost(Zero_vec, implemented_decision.investment);
			total_incurred_cost = cost.first;
			implemented_decision.turnover_rate = cost.second;
		}
		else
		{
			cost = portfolio::transaction_cost(portfolio::current_investment, implemented_decision.investment);
			implemented_decision.current_cost = cost.first;
			implemented_decision.turnover_rate = cost.second;
			total_incurred_cost += implemented_decision.current_cost;
		}

		portfolio::current_wealth = portfolio::current_wealth * (1.0 + compute_ROI(implemented_decision, mean)) - implemented_decision.current_cost;
	}

	static double card(portfolio& w)
	{
		w.cardinality = w.investment.count();
		return w.cardinality;
	}

	static double normalized_Entropy(portfolio &P)
	{
		double ne = 0.0;

		unsigned int card = P.investment.count();

		if (card < 2)
			return 1.0;

		for (unsigned int i = 0; i < portfolio::num_assets; ++i)
			if (P.investment(i) > 0.0)
				ne += P.investment(i) * log2(P.investment(i));
		P.cardinality = card;
		double e = -1.0/log2(P.cardinality)*ne;

		if (!(e < 0.0 || e >= 0))
			return 1.0;
		return e;

	}

	static void compute_statistics(unsigned int t)
	{
		if (t >= periods)
			return;

		portfolio::theoretical_mean_ROI = portfolio::sequence_mean_covar[t].first;
		portfolio::theoretical_covariance = portfolio::sequence_mean_covar[t].second;
	}

	static std::pair<double,double> BOVESPA_fees(double nominal_value)
	{
		std::pair<double,double> fees;

		if (nominal_value <= 135.7)
		{
			fees.first = 0.0;
			fees.second = 2.7;
		}
		else if (nominal_value > 135.7 && nominal_value <= 498.62)
		{
			fees.first = 0.02;
			fees.second = 0.0;
		}
		else if (nominal_value > 498.62 && nominal_value <= 1514.69)
		{
			fees.first = 0.015;
			fees.second = 2.49;
		}
		else if (nominal_value > 1514.69 && nominal_value <= 3029.38)
		{
			fees.first = 0.01;
			fees.second = 10.06;
		}
		else if (nominal_value > 3029.38)
		{
			fees.first = 0.005;
			fees.second = 25.21;
		}

		return fees;

	}

	static std::pair<double,double> transaction_cost(Eigen::VectorXd& w_current, Eigen::VectorXd& w_new)
	{

		if (w_current.rows() == 0 || w_new.rows() == 0 || w_new.sum() == 0.0)
			return std::pair<double,double>(0.0, 0.0);

		Eigen::VectorXd x = w_current - w_new;
		double total_cost = 0.0;

		for (int i = 0; i < x.rows(); ++i)
		{
			double value = x(i);
			x(i) = fabs(value);

			double nominal_value = value*current_wealth;

			std::pair<double,double> fees = BOVESPA_fees(nominal_value);
			double brokerage_commission = fees.first;
			double brokerage_fee = fees.second;

			total_cost += nominal_value*brokerage_commission + brokerage_fee;
		}

		return std::pair<double,double>(total_cost, turnover(w_current, w_new));
	}

	static void init_portfolio()
	{
		portfolio::compute_statistics(portfolio::current_period);
	}

	void init()
	{

		this->investment.resize(portfolio::num_assets);
		std::vector<unsigned int> selected;

		unsigned int n_assets = portfolio::min_cardinality + (rand() % (portfolio::max_cardinality - portfolio::min_cardinality));

		std::vector<unsigned int> index;
		for (unsigned int i = 0; i < portfolio::num_assets; ++i)
			index.push_back(i);

		for (unsigned int i = 0; i < n_assets; ++i)
		{
			std::random_shuffle(index.begin(), index.end());
			selected.push_back(index.back());
			index.pop_back();
		}

		// Create random weights
		Eigen::MatrixXd samples = Eigen::MatrixXd::Zero(n_assets,1);
		Eigen::VectorXd alpha = Eigen::VectorXd::Ones(n_assets);
		alpha *= 2.0;

		boost::mt19937 rng;
		Eigen::VectorXd weights = dirichlet_sample(alpha,rng);

		for (unsigned int i = 0; i < portfolio::num_assets; ++i)
			investment(i) = 0.0;
		for (unsigned int i = 0; i < n_assets; ++i)
			investment(selected[i]) = weights(i);

		apply_threshold(investment, portfolio::min_hold, portfolio::max_hold);
		cardinality = investment.count(); ROI = 0.0; risk = 0.0;

	}
};



#endif
