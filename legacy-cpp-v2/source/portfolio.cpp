#include "../headers/utils.h"
#include "../headers/portfolio.h"
#include "../headers/statistics.h"

#include <list>
#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <strstream>
#include <algorithm>

#include <boost/assign.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/math/constants/constants.hpp>

using namespace std;
using namespace boost;

Eigen::MatrixXd portfolio::theoretical_covariance;
Eigen::VectorXd portfolio::theoretical_mean_ROI;
std::vector<std::pair<Eigen::VectorXd,Eigen::MatrixXd> > portfolio::sequence_mean_covar;
Eigen::VectorXd portfolio::current_investment;

unsigned int portfolio::samples_per_portfolio;
unsigned int portfolio::min_cardinality;
unsigned int portfolio::max_cardinality;
unsigned int portfolio::current_period;
unsigned int portfolio::periodicity;
unsigned int portfolio::num_assets;
unsigned int portfolio::severity;
unsigned int portfolio::periods;

double portfolio::min_hold;
double portfolio::max_hold;
double portfolio::brokerage_commission;
double portfolio::brokerage_fee;
double portfolio::total_wealth;
double portfolio::current_wealth;
double portfolio::total_incurred_cost;
double portfolio::max_ROI;
double portfolio::max_risk;
double portfolio::max_cost;
double portfolio::min_ROI;
double portfolio::min_risk;
double portfolio::min_cost;


std::vector<std::pair<unsigned,double> > portfolio::top_k_investments (const Eigen::VectorXd& investment, unsigned int K)
{
	std::list<std::pair<unsigned,double> > sequence;
	std::list<std::pair<unsigned,double> >::iterator kth_element;

	for (unsigned int i = 0; i < portfolio::num_assets; ++i)
	{
		if (investment(i) > 0.0)
		{
			if (sequence.size() == 0)
				sequence.push_back(std::pair<unsigned,double>(i,investment(i)));
			else
			{
				std::list<std::pair<unsigned,double> >::iterator it = sequence.begin();
				for (; it != sequence.end(); ++it)
				{

					if (investment(i) > it->second)
					{
						sequence.insert(it,std::pair<unsigned,double>(i,investment(i)));
						break;
					}
				}
				if (it == sequence.end())
					sequence.push_back(std::pair<unsigned,double>(i,investment(i)));
			}
		}
		else
			sequence.push_back(std::pair<unsigned,double>(i,0.0));
	}

	std::list<std::pair<unsigned,double> >::iterator it = sequence.begin();
	for (unsigned int k = 0; it != sequence.end(); ++it, ++k)
		if (k == K)
		{
			kth_element = it;
			break;
		}

	return std::vector<std::pair<unsigned,double> > (sequence.begin(), kth_element);
}

std::vector<std::pair<Eigen::VectorXd,Eigen::MatrixXd> > load_benchmark_data(const std::string& path)
{
	std::vector<std::pair<Eigen::VectorXd,Eigen::MatrixXd> > sequence;

	std::string mean_path = path + "-means.dat";
	std::string covar_path = path + "-covar.dat";
	std::cout << mean_path << std::endl << covar_path << std::endl;
	std::ifstream means_file(mean_path.c_str(), std::ios::in);
	std::ifstream covar_file(covar_path.c_str(), std::ios::in);

	for (unsigned int t = 0; t < portfolio::periods; ++t)
	{
		Eigen::VectorXd mean(portfolio::num_assets);
		Eigen::MatrixXd covar(portfolio::num_assets,portfolio::num_assets);

		for (unsigned int i = 0; i < portfolio::num_assets; ++i)
		{
			means_file >> mean(i);

			for (unsigned int j = 0; j < portfolio::num_assets; ++j)
				covar_file >> covar(i,j);
		}

		bool valid_covar = check_covar(covar);
		if (!valid_covar)
		{
			std::cout << "Invalid covar!\n";
			system("pause");
		}

		std::pair<Eigen::VectorXd,Eigen::MatrixXd> mc(mean,covar);
		sequence.push_back(mc);
	}

	means_file.close();
	covar_file.close();

	return sequence;
}

Eigen::VectorXd portfolio::estimate_assets_mean_ROI(const Eigen::MatrixXd &returns_data)
{
	return returns_data.colwise().mean();
}

void portfolio::estimate_covariance(const Eigen::VectorXd &mean_ROI, const Eigen::MatrixXd &returns_data, Eigen::MatrixXd &covariance)
{
	covariance.resize(portfolio::num_assets,portfolio::num_assets);
	covariance = (returns_data.rowwise() - mean_ROI.transpose()).transpose()*(returns_data.rowwise() - mean_ROI.transpose())/(returns_data.rows()-1.0);
}

double portfolio::compute_ROI(portfolio &P, const Eigen::VectorXd &mean_ROI)
{
	return //log(P.investment.transpose()*mean_ROI);
			P.investment.transpose()*mean_ROI;
}

double portfolio::compute_risk(portfolio &P, const Eigen::MatrixXd &covariance)
{
	return P.investment.transpose()*covariance*P.investment;
}

void portfolio::observe_state(portfolio &w, unsigned int N, unsigned int initial_t, unsigned int current_t)
{

	normalize(w.investment);
	double mean_ROI = 0.0, mean_risk = 0.0;

	w.S.samples.clear();
	w.S.samples.resize(0);

	for (unsigned int t = initial_t; t <= current_t; ++t)
	{
		Eigen::VectorXd mean = portfolio::sequence_mean_covar[t].first;
		Eigen::MatrixXd covar = portfolio::sequence_mean_covar[t].second;

		std::vector<double> ROI(N), risk(N);

		mean_ROI = 0.0, mean_risk = 0.0;
		double var_ROI = 0.0, var_risk = 0.0, cov = 0.0;
		std::vector<stats> stats_samples;

		for (unsigned int i = 0; i < N; ++i)
		{
			do
			{
				Eigen::MatrixXd samples = multi_norm(mean, covar, 1000);
				samples.transposeInPlace();
				Eigen::VectorXd mean2 = estimate_assets_mean_ROI(samples);
				Eigen::MatrixXd covar2;
				estimate_covariance(mean2, samples, covar2);

				double roi = compute_ROI(w, mean2);
				double _risk = compute_risk(w, covar2);

				ROI[i] = linear_transform(roi, min_ROI, max_ROI);
				risk[i] = linear_transform(_risk, min_risk, max_risk);

			} while (ROI[i] > 1.0 || ROI[i] < 0.0 || risk[i] < 0.0 || risk[i] > 1.0);

			if (t == current_t)
				stats_samples.push_back(stats(ROI[i],risk[i]));

			mean_ROI += ROI[i];
			mean_risk += risk[i];
		}

		mean_ROI /= N;
		mean_risk /= N;

		for (unsigned int i = 0; i < N; ++i)
		{
			var_ROI += (ROI[i] - mean_ROI)*(ROI[i] - mean_ROI);
			var_risk += (risk[i] - mean_risk)*(risk[i] - mean_risk);
			cov += (ROI[i] - mean_ROI)*(risk[i] - mean_risk);
		}

		var_ROI /= (N-1);
		var_risk /= (N-1);
		cov /= (N-1);

		Eigen::VectorXd Zero_vec = Eigen::VectorXd::Zero(portfolio::num_assets);
		std::pair<double,double> cost;
		if (t == 0) // Considers the entry cost in the share market (from nothing to something invested)
			cost = portfolio::transaction_cost(Zero_vec, w.investment);
		else if (t < current_t && t > 0) // Considers the historical cost
			cost = portfolio::transaction_cost(solution::historical_anticipative_decisions[t-1]->P.investment, w.investment);
		else if (t == current_t && t > 0) // Considers the current incurring cost to change the portfolio
		{
			cost = portfolio::transaction_cost(portfolio::current_investment, w.investment);
		}

		w.current_cost = cost.first;
		w.turnover_rate = cost.second;

		// Computes the nominal expected return over investment for the current period
		double nominal_ROI = inverse_linear_transform(mean_ROI,min_ROI,max_ROI)*portfolio::current_wealth;

		// Updates the percentual expected return, considering the transaction cost
		double mean_ROI_adjusted = (nominal_ROI - cost.first) / portfolio::current_wealth;
		double nominal_ROI_adjusted = mean_ROI_adjusted*portfolio::current_wealth;
		w.nominal_ROI = nominal_ROI_adjusted;

		mean_ROI = linear_transform(mean_ROI_adjusted,portfolio::min_ROI, portfolio::max_ROI);

		if (t == current_t)
			w.S = stats_vec(stats_samples,mean_ROI,mean_risk,var_ROI,var_risk,cov);

		if (t == initial_t)
		{
			// Initial state equals the average of the noisy observations
			Eigen::VectorXd state(4);
			state << mean_ROI, mean_risk, 0.0, 0.0;


			w.kalman_state.x = state;
			w.kalman_state.u = Eigen::VectorXd::Zero(4);
			w.kalman_state.error = 0.0;

			// Initial uncertainty over the state equates the computed variances
			// Velocity uncertainty is set to an arbitrary high number
			w.kalman_state.P = Eigen::MatrixXd::Zero(4,4);
			w.kalman_state.P << var_ROI, cov, 0.0, 0.0,
								cov, var_risk, 0.0, 0.0,
								0.0, 0.0, var_ROI, 0.0,
								0.0, 0.0, 0.0, var_risk;
			w.kalman_state.P_initial = w.kalman_state.P;
			Kalman_prediction(w.kalman_state);

			w.ROI_prediction = w.kalman_state.x_next(0);
			w.risk_prediction = w.kalman_state.x_next(1);
			w.error_covar_prediction = w.kalman_state.P_next;
			w.error_covar = w.kalman_state.P;
			w.covar = w.kalman_state.P;

			// Updates the measurement uncertainty
			Kalman_params::R = Eigen::MatrixXd::Zero(2,2);
			Kalman_params::R << var_ROI/N, cov/N,
								cov/N, var_risk/N;
		}


		if (mean_ROI > 1.0 || mean_ROI < 0.0)
		{
			mean_ROI = linear_transform(compute_ROI(w, mean), min_ROI, max_ROI);
			double nominal_ROI = inverse_linear_transform(mean_ROI,min_ROI,max_ROI)*portfolio::current_wealth;

			// Updates the percentual expected return, considering the transaction cost
			double mean_ROI_adjusted = (nominal_ROI - cost.first) / portfolio::current_wealth;
			double nominal_ROI_adjusted = mean_ROI_adjusted*portfolio::current_wealth;
			w.nominal_ROI = nominal_ROI_adjusted;

			mean_ROI = mean_ROI_adjusted;
		}


		Eigen::VectorXd measurement(2);
		measurement << mean_ROI, mean_risk;

		if (current_t > initial_t)
		{
			Kalman_filter(w.kalman_state, measurement);

			w.ROI_prediction = w.kalman_state.x_next(0);
			w.risk_prediction = w.kalman_state.x_next(1);
			w.error_covar_prediction = w.kalman_state.P_next;
			w.error_covar = w.kalman_state.P;
			w.covar = w.kalman_state.P;
		}

	}

	// We finally arrive at the current time, t.
	w.ROI = mean_ROI;
	w.risk = mean_risk;

	w.kalman_state.x(0) = w.ROI;
	w.kalman_state.x(1) = w.risk;
}

void portfolio::KF_prediction_obj_space(portfolio &w, unsigned int steps_ahead)
{

	unsigned int current_step = 1;

	while (current_step < steps_ahead)
	{
		if (current_step < steps_ahead)
		{
			w.ROI = w.ROI_prediction;
			w.risk = w.risk_prediction;
			w.kalman_state.x = w.kalman_state.x_next;
			w.error_covar = w.error_covar_prediction;
			w.kalman_state.P = w.kalman_state.P_next;
			Kalman_prediction(w.kalman_state);
		}
		++current_step;
	}

}
