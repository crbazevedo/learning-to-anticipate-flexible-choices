#ifndef PORTFOLIO_H
	#define PORTFOLIO_H

#include "utils.h"
#include "../headers/kalman_filter.h"
//#include "../headers/nsga2.h"

#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <Eigen/Eigen/Dense>
#include <boost/lexical_cast.hpp>

#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/locale.hpp>

using namespace boost::gregorian;
//using namespace boost::locale;
using namespace boost;


struct asset
{
	std::string id;
	static unsigned int horizon;

	std::vector<double> historical_close_price;
	std::vector<double> validation_close_price;
	std::vector<double> complete_close_price;
};

asset load_asset_data(const std::string& path, const std::string& id);

struct portfolio
{
	static date training_start_date, validation_start_date;
	static date training_end_date, validation_end_date;
	static unsigned int window_size;
	static unsigned int tr_period;
	static unsigned int vl_period;
	static std::vector<asset> available_assets;
	static unsigned int available_assets_size;

	static Eigen::MatrixXd moving_average(const Eigen::MatrixXd &return_data);
	static Eigen::MatrixXd moving_median(const Eigen::MatrixXd &return_data);

	// Robust estimates
	static Eigen::VectorXd estimate_assets_median_ROI(const Eigen::MatrixXd &return_data);
	static double estimate_median_ROI (unsigned int, const Eigen::MatrixXd &return_data);
	static void estimate_robust_covariance(const Eigen::VectorXd &mean_ROI,
			const Eigen::MatrixXd &return_data, Eigen::MatrixXd &covariance);

	static void compute_robust_efficiency(portfolio &P);
	static double evaluate_stability(portfolio &P);

	// Non-robust estimates
	static Eigen::VectorXd estimate_assets_mean_ROI(const Eigen::MatrixXd &return_data);
	static double estimate_mean_ROI (unsigned int, const Eigen::MatrixXd &return_data);
	static void estimate_covariance(const Eigen::VectorXd &mean_ROI,
			const Eigen::MatrixXd &return_data, Eigen::MatrixXd &covariance);

	static double compute_risk(portfolio &P,const Eigen::MatrixXd &covariance);
	static double compute_ROI(portfolio &P, const Eigen::VectorXd &mean_ROI);
	static void compute_efficiency(portfolio &P);

	static void sample_autocorrelation(const Eigen::MatrixXd &return_data, unsigned int k);

	// For use in the Kalman Filter
	static void observe_state(portfolio &w, unsigned int N, unsigned int current_t);

	static Eigen::MatrixXd covariance;
	static Eigen::MatrixXd autocorrelation;
	static Eigen::MatrixXd robust_covariance;
	static Eigen::VectorXd mean_ROI;
	static Eigen::VectorXd median_ROI;
	static Eigen::MatrixXd tr_returns_data;
	static Eigen::MatrixXd vl_returns_data;
	static Eigen::MatrixXd current_returns_data;
	static Eigen::MatrixXd complete_returns_data;

	static unsigned int max_cardinality;
	static bool robustness;

	double ROI, ROI_prediction, ROI_observed;
	double risk, risk_prediction, risk_observed;
	double robust_ROI;
	double robust_risk;
	double non_robust_ROI;
	double non_robust_risk;
	double cardinality;

	Kalman_params kalman_state;
	Eigen::MatrixXd error_covar, error_covar_prediction;
	Eigen::VectorXd investment;

	portfolio()
	{
		cardinality = .0f;
		ROI = risk = robust_ROI = robust_risk = non_robust_ROI = non_robust_risk = ROI_observed = risk_observed = .0f;
		ROI_prediction = risk_prediction = .0f;
	}

	static double prediction_error(portfolio &w, unsigned int t)
	{
		// Extract next-state returns data
		int index = (t+1)*(portfolio::window_size - 1);
		Eigen::MatrixXd returns_data(portfolio::tr_returns_data.rows(),portfolio::tr_returns_data.cols());

		for (int i = 0; i < portfolio::tr_returns_data.rows(); ++i)
			returns_data.row(i) = portfolio::complete_returns_data.row(index + i);

		// What will be actually observed in terms of ROI and Risk
		Eigen::VectorXd mean = portfolio::robustness ? estimate_assets_median_ROI(returns_data) : estimate_assets_mean_ROI(returns_data);
		Eigen::MatrixXd covar;
		if (portfolio::robustness)
			estimate_robust_covariance(mean, returns_data, covar);
		else
			estimate_covariance(mean, returns_data, covar);

		w.ROI_observed = compute_ROI(w, mean); w.risk_observed = compute_risk(w, covar);

		// Return the mean square prediction error
		double error_ROI = (w.ROI - w.ROI_observed)*(w.ROI - w.ROI_observed);
		double error_risk = (w.risk - w.risk_observed)*(w.risk - w.risk_observed);
		return error_ROI + error_risk;


	}

	static double card(portfolio& w)
	{
		int card = 0;
		for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
			if (w.investment(i) > 0.0)
				card++;
		return card;
	}

	static double normalized_Entropy(portfolio &P)
	{
		double ne = 0.0;
		unsigned int card = 0;

		for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
			if (P.investment(i) > 0.0)
			{
				ne += P.investment(i) *log2(P.investment(i));
				card++;
			}
		P.cardinality = card;
		//return -1.0/log2(portfolio::available_assets_size)*ne;
		return -1.0/log2(P.cardinality)*ne;
	}

	static void compute_statistics(const Eigen::MatrixXd &returns_data)
	{
		std::cout << "Computing robust statistics...";
		portfolio::median_ROI = estimate_assets_median_ROI(returns_data);
		std::cout << " median ok. ";
		estimate_robust_covariance(portfolio::median_ROI,returns_data,portfolio::robust_covariance);
		std::cout << "Robust covariance matrix ok.\n";

		std::cout << "Computing non-robust statistics...";
		portfolio::mean_ROI = estimate_assets_mean_ROI(returns_data);
		std::cout << " mean ok. ";
		estimate_covariance(portfolio::mean_ROI,returns_data,portfolio::covariance);
		std::cout << "Non-robust covariance matrix ok.\n";
	}

	static void init_portfolio()
	{
		unsigned int length = (portfolio::tr_period + portfolio::vl_period) - (window_size - 1); length -= 2.0*portfolio::window_size;
		portfolio::tr_returns_data.resize(portfolio::tr_period - 2.0*portfolio::window_size, portfolio::available_assets_size);
		portfolio::vl_returns_data.resize(2.0*portfolio::window_size, portfolio::available_assets_size);
		portfolio::complete_returns_data.resize(length + 2.0*portfolio::window_size, portfolio::available_assets_size);

		for (unsigned int t = 0; t < length + 2.0*portfolio::window_size; ++t)
			for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
			{
				asset* Ai = &available_assets[i];

				double initial_investment_i = Ai->complete_close_price[t];
				double final_return_i = Ai->complete_close_price[t + window_size];
				double ROI_i;

				if (initial_investment_i == 0.0)
					ROI_i = final_return_i;
				else
					ROI_i = (final_return_i - initial_investment_i)/initial_investment_i;

				if (t < portfolio::tr_period - 2.0*portfolio::window_size)
					portfolio::tr_returns_data(t,i) = ROI_i;
				else if (t >= portfolio::tr_period - 2.0*portfolio::window_size && t < portfolio::tr_period)
					portfolio::vl_returns_data(t - (portfolio::tr_period - 2.0*portfolio::window_size),i) = ROI_i;
				portfolio::complete_returns_data(t,i) = ROI_i;
			}


		std::cout << "Estimating robust statistics...\n";

		// Very important: init_portfolio() must be called first!
		portfolio::current_returns_data = portfolio::tr_returns_data;

		portfolio::median_ROI = portfolio::estimate_assets_median_ROI(tr_returns_data);
		portfolio::estimate_robust_covariance(portfolio::median_ROI,tr_returns_data,portfolio::robust_covariance);

		std::cout << "Estimate non-robust statistics..\n";

		portfolio::mean_ROI = portfolio::estimate_assets_mean_ROI(tr_returns_data);
		portfolio::estimate_covariance(portfolio::mean_ROI,tr_returns_data,portfolio::covariance);

	}

	void init()
	{

		this->investment.resize(portfolio::available_assets_size);
		std::vector<unsigned int> selected;

		unsigned int num_assets = 2 + (rand() % (portfolio::max_cardinality - 1));
		//std::cout << "num_assets= " << num_assets << std::endl;
		std::vector<unsigned int> index;
		for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
			index.push_back(i);

		for (unsigned int i = 0; i < num_assets; ++i)
		{
			std::random_shuffle(index.begin(), index.end());
			selected.push_back(index.back());
			index.pop_back();
		}

		/*unsigned int num_assets = 0;

		for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
		{
			unsigned int s = rand() % 2;
			selected.push_back(s);
			if (s) num_assets++;
		}*/

		// Create random weights
		Eigen::VectorXd weights(num_assets);
		for (unsigned int i = 0; i < num_assets; ++i)
			weights(i) = uniform_zero_one();

		/*for (unsigned int i = 0, j = 0; i < portfolio::available_assets_size; ++i)
		if (selected[i])
		{
			investment(i) = weights(j);
			++j;
		}
		else portfolio::investment(i) = 0.0; */

		for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
			investment(i) = 0.0;

		for (unsigned int i = 0; i < num_assets; ++i)
			investment(selected[i]) = weights[i];

		aplica_threshold(investment,.01f);

		cardinality = num_assets;
		non_robust_ROI = robust_ROI = 0.0;
		non_robust_risk = robust_risk = 0.0;
		ROI = 0.0;
		risk = 0.0;
	}
};



#endif
