/*
 * main.cpp
 *
 *  Created on: 21/11/2012
 *  Last updated on 08/03/2015
 *      Author: Carlos R. B. Azevedo
 *      Laboratory of Bioinformatics and Bio-inspired Computing
 *      School of Electrical Engineering
 *      University of Campinas, Brazil
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <string>
#include <ctime>

#include <boost/lexical_cast.hpp>
#include "../headers/crossover_operators.h"
#include "../headers/mutation_operators.h"
#include "../headers/learning_operators.h"
#include "../headers/statistics.h"
#include "../headers/portfolio.h"
#include "../headers/kalman_filter.h"
#include "../headers/dirichlet.h"


using namespace boost;

// Global variables
static unsigned int pop_size;
static unsigned int max_iter;
static unsigned int trials;
static bool anticipation = false;
static std::string algo, instance, DM;
static std::vector<std::vector<Eigen::VectorXd> > investments;


unsigned int exp_id;

// Prototypes
void save_portfolio(const portfolio &P, std::ostream &out);
void save_portfolio_stochastic_obj(const portfolio &P, std::ostream &out);
void save_results(unsigned int e, unsigned int seed_initial_pop,
	unsigned int experiment_seed, std::vector<sol*> &P, unsigned int t);

void run_algorithm(unsigned int e, unsigned int initial_pop_seed,
	unsigned int experiment_seed, unsigned int pop_size, unsigned int generations);

std::vector<std::pair<std::pair<unsigned, unsigned>, std::pair<double, double> > > write_statistics(std::vector<sol*> &P, std::ostream &out, unsigned int t);
void print_pop(std::vector<sol*> &P, std::ostream &out);
void print_pop(std::vector<sol*> &P, std::pair<unsigned int, double>, std::ostream &out);


// Reads parameters from the command line and assign them to the corresponding global class variables
bool load_params (int argc, char** argv)
{
	if (argc != 27)
	{
		std::cout << "Usage: " << argv[0] << " instance num_assets exp_id trials KF_window_size"
							   << " algorithm dm_type pop_size mut_rate xover_rate tourn_size"
							   << " max_iter min_cardinality max_cardinality min_hold max_hold"
							   << " brokerage_commission brokerage_fee epsilon periods periodicity"
							   << " severity z_ref1 z_ref2 total_wealth samples_per_simulation\n";
		return false;
	}

	instance = std::string(argv[1]); // Which instance are we solving?
	portfolio::num_assets = lexical_cast<unsigned int>(argv[2]); // Number of available assets
	exp_id = lexical_cast<unsigned int>(argv[3]); // Are we resuming the experiments? From were should we start?
	trials = lexical_cast<unsigned int>(argv[4]); // Number of experiments
	Kalman_params::window_size = lexical_cast<unsigned int>(argv[5]); // How far should we look into the past?
	algo = std::string(argv[6]); // Which algorithm are we using?
	DM = std::string(argv[7]); // What is the decision-making strategy?
	pop_size = lexical_cast<unsigned int>(argv[8]); // What is the population size
	solution::mut_rate = lexical_cast<double>(argv[9]); // Mutation rate
	solution::xover_rate = lexical_cast<double>(argv[10]); // Crossover rate
	solution::tourn_size = lexical_cast<unsigned int>(argv[11]); // Selection tournament size
	max_iter = lexical_cast<unsigned int>(argv[12]); // Maximum number of iterations
	portfolio::min_cardinality = lexical_cast<unsigned int>(argv[13]);
	portfolio::max_cardinality = lexical_cast<unsigned int>(argv[14]);
	portfolio::min_hold = lexical_cast<double>(argv[15]);
	portfolio::max_hold = lexical_cast<double>(argv[16]);
	portfolio::brokerage_commission = lexical_cast<double>(argv[17]);
	portfolio::brokerage_fee = lexical_cast<double>(argv[18]);
	solution::epsilon = lexical_cast<double>(argv[19]);
	portfolio::periods = lexical_cast<unsigned int>(argv[20]);
	portfolio::periodicity = lexical_cast<unsigned int>(argv[21]);
	portfolio::severity = lexical_cast<unsigned int>(argv[22]);
	solution::ref_point.first = lexical_cast<double>(argv[23]);
	solution::ref_point.second = lexical_cast<double>(argv[24]);
	portfolio::total_wealth = lexical_cast<double>(argv[25]);
	portfolio::samples_per_portfolio = lexical_cast<unsigned int>(argv[26]);

	// We start investing the total amount and see how it goes.
	portfolio::current_wealth = portfolio::total_wealth;


	if (portfolio::max_cardinality > portfolio::num_assets)
	{
		std::cout << "Error: max_cardinality > num_assets\n";
		return false;
	}

	if (portfolio::min_cardinality < 1)
	{
		std::cout << "Error: min_cardinality < 1\n";
		return false;
	}

	if (algo.compare("asms") == 0)
		anticipation = true;

	std::string path = "../data/" + instance + "/" + instance + "-" + boost::lexical_cast<std::string>(portfolio::num_assets)
			+ "-" + boost::lexical_cast<std::string>(portfolio::periods)
			+ "-" + boost::lexical_cast<std::string>(portfolio::periodicity) + "-";
	if (portfolio::severity != 10)
		path += "0";
	path += boost::lexical_cast<std::string>(portfolio::severity);

	portfolio::sequence_mean_covar = load_benchmark_data(path);


	portfolio::init_portfolio();

	// We use uniform crossover with probability one
	xover_op::uniform_prob_op[0] = 1.0f;

	// We decide randomly which mutation operator to apply
	mutation_op::uniform_prob_op[0] = .25f;
	mutation_op::uniform_prob_op[1] = .25f;
	mutation_op::uniform_prob_op[2] = .25f;
	mutation_op::uniform_prob_op[3] = .25f;

	// Standard KF parameters initalization
	Kalman_params::F = Eigen::MatrixXd::Zero(4,4);
	Kalman_params::F << 1.0, 0.0, 1.0, 0.0,
						0.0, 1.0, 0.0, 1.0,
						0.0, 0.0, 1.0, 0.0,
						0.0, 0.0, 0.0, 1.0;
	Kalman_params::H = Eigen::MatrixXd::Zero(2,4);
	Kalman_params::H << 1.0, 0.0, 0.0, 0.0,
						0.0, 1.0, 0.0, 0.0;

	// Compute the min and max return over investment, risk, and cost among all available assets, relative to the current period.
	portfolio::max_ROI = portfolio::get_max_ROI(portfolio::current_period);
	portfolio::max_risk = portfolio::get_max_risk(portfolio::current_period);
	portfolio::min_ROI = portfolio::get_min_ROI(portfolio::current_period);
	portfolio::min_risk = 0.0;
	portfolio::max_cost = compute_max_cost();

	// We shall start from zero
	portfolio::current_investment = Eigen::VectorXd::Zero(portfolio::num_assets);

	double nominal_min_ROI = portfolio::min_ROI*portfolio::current_wealth;
	portfolio::min_ROI = (nominal_min_ROI - portfolio::max_cost)/portfolio::current_wealth;

	// Reference points properly transformed
	solution::ref_point.first = linear_transform(solution::ref_point.first, portfolio::min_ROI, portfolio::max_ROI);
	solution::ref_point.second = linear_transform(solution::ref_point.second, portfolio::min_risk, portfolio::max_risk);

	return true;
}

int main (int argc, char** argv)
{
	srand((unsigned)time(0));

	bool loaded = load_params(argc, argv);
	if (!loaded)
	{
		std::cout << "Exiting...\n";
		return 0;
	}

	investments.resize(portfolio::periods);
	portfolio::current_wealth = portfolio::total_wealth;

	// Running the experiments.
	for (unsigned int e = exp_id; e <= (exp_id + trials) - 1; ++e)
	{
		unsigned int seed_initial_pop = (unsigned)time(0);
		unsigned int seed_experiment = rand();


		portfolio::init_portfolio();
		for (unsigned int i = 0; i < solution::historical_populations.size(); ++i)
			for (unsigned int j = 0; j < solution::historical_populations[i].size(); ++j)
			{
				delete solution::historical_populations[i][j];
				solution::historical_populations[i][j] = NULL;
			}

		solution::historical_populations.clear();
		solution::historical_populations.resize(0);
		solution::historical_anticipative_decisions.clear();
		solution::historical_anticipative_decisions.resize(0);

		portfolio::current_investment = Eigen::VectorXd::Zero(portfolio::num_assets);
		run_algorithm(e, seed_initial_pop, seed_experiment, pop_size, max_iter);
	}

	return 0;
}

void print_portfolio(sol* w, std::ostream &out)
{
	out << w->P.ROI << " "
		<< w->P.risk << " "
		<< w->P.ROI_observed << " "
		<< w->P.risk_observed << " "
		<< w->rank_ROI << " "
		<< w->rank_risk << " "
		<< w->epsilon_feasibility.first << " "
		<< w->epsilon_feasibility.second << " "
		<< w->prediction_error << " "
		<< w->alpha << " "
		<< w->P.current_cost << " "
		<< w->P.nominal_ROI << " "
		<< w->pred_hv << std::endl;
}

void print_pop(std::vector<sol*> &P, std::ostream &out)
{
	sort_per_objective(P,0);
	for (unsigned int p = 0; p < P.size(); ++p)
		print_portfolio(P[p], out);
}

void print_pop(std::vector<sol*> &P, std::pair<unsigned int, double> error, std::ostream &out)
{

	out << P[error.first]->P.ROI << " "
		<< P[error.first]->P.risk << " "
		<< P[error.first]->P.ROI_observed << " "
		<< P[error.first]->P.risk_observed << " "
		<< P[error.first]->rank_ROI << " "
		<< P[error.first]->rank_risk << " "
		<< P[error.first]->epsilon_feasibility.first << " "
		<< P[error.first]->epsilon_feasibility.second << " "
		<< error.second << " "
		<< P[error.first]->alpha << " "
		<< P[error.first]->P.current_cost << " "
		<< P[error.first]->P.nominal_ROI << " "
		<< P[error.first]->pred_hv	<< std::endl;
	save_portfolio(P[error.first]->P,out);
}


std::vector<std::pair<std::pair<unsigned, unsigned>, std::pair<double, double> > > write_statistics(std::vector<sol*> &P, std::ostream &out, unsigned int t)
{

	float mean_Pareto_rank = 0.0f;
	float mean_cd = 0.0f;
	float mean_Delta_S = 0.0f;
	float mean_return = 0.0f;
	float mean_risk = 0.0f;
	float mean_return_obs = 0.0f;
	float mean_risk_obs = 0.0f;
	float mean_cardinality = 0.0f;
	float mean_entropy = 0.0f;
	float mean_alpha = 0.0f;
	float mean_anticipation_rate = 0.0f;
	float mean_prediction_error = 0.0f;
	float mean_cid_ROI = 0.0f;
	float mean_cid_risk = 0.0f;
	float mean_cid = 0.0f;
	float mean_epsilon_feasibility_ROI = 0.0f;
	float mean_epsilon_feasibility_risk = 0.0f;
	float mean_current_cost = 0.0f;
	float mean_nominal_ROI = 0.0f;
	float mean_turnover = 0.0f;
	float mean_pred_hv = 0.0f;


	double min_error = 0.0, max_error = 0.0;
	unsigned min_error_index = 0, max_error_index = 0;
	double min_uncertainty = 0.0, max_uncertainty = 0.0;
	unsigned min_uncertainty_index = 0, max_uncertainty_index = 0;

	for (unsigned int p = 0; p < P.size(); ++p)
	{
		if (P[p]->cd < 99999999.9f)
			mean_cd += P[p]->cd;

		mean_Delta_S += P[p]->Delta_S;
		mean_return += P[p]->P.ROI;
		mean_risk += P[p]->P.risk;
		mean_return_obs += P[p]->P.ROI_observed;
		mean_risk_obs += P[p]->P.risk_observed;
		mean_cardinality += P[p]->P.investment.count();
		mean_entropy += portfolio::normalized_Entropy(P[p]->P);
		mean_alpha += P[p]->alpha;
		mean_anticipation_rate += P[p]->anticipation_rate;
		mean_epsilon_feasibility_ROI += P[p]->epsilon_feasibility.first;
		mean_epsilon_feasibility_risk += P[p]->epsilon_feasibility.second;
		mean_current_cost += P[p]->P.current_cost;
		mean_nominal_ROI += P[p]->P.nominal_ROI;
		mean_turnover += P[p]->P.turnover_rate;
		mean_pred_hv += P[p]->pred_hv;

		double error = P[p]->P.obs_error;
		double uncertainty = 1.0 - P[p]->alpha;

		if (P[p]->Pareto_front == 0)
		{
			if (error < min_error)
			{
				min_error_index = p;
				min_error = error;
			}
			else if (error > max_error)
			{
				max_error_index = p;
				max_error = error;
			}

			if (uncertainty < min_uncertainty)
			{
				min_uncertainty_index = p;
				min_uncertainty = uncertainty;
			}
			else if (uncertainty > max_uncertainty)
			{
				max_uncertainty_index = p;
				max_uncertainty = uncertainty;
			}
		}

		mean_prediction_error += error;
		mean_cid_ROI += P[p]->P.cid_ROI;
		mean_cid_risk += P[p]->P.cid_risk;
		mean_cid += P[p]->P.cid;
	}

	mean_Pareto_rank = solution::num_fronts;
	mean_Delta_S /= P.size();
	mean_Pareto_rank /= P.size();
	mean_cd /= P.size();
	mean_return /= P.size();
	mean_risk /= P.size();
	mean_return_obs /= P.size();
	mean_risk_obs /= P.size();
	mean_cardinality /= P.size();
	mean_entropy /= P.size();
	mean_alpha /= P.size();
	mean_anticipation_rate /= P.size();
	mean_prediction_error /= P.size();
	mean_cid_ROI += P.size();
	mean_cid_risk += P.size();
	mean_cid += P.size();
	mean_epsilon_feasibility_ROI /= P.size();
	mean_epsilon_feasibility_risk /= P.size();
	mean_current_cost /= P.size();
	mean_nominal_ROI /= P.size();
	mean_turnover /= P.size();
	mean_pred_hv /= P.size();


	out << mean_Pareto_rank << " " << mean_cd	<< " " << mean_Delta_S << " " << mean_return << " " << mean_risk << " "
		<< mean_return_obs << " " << mean_risk_obs << " " << spread(P) << " " << coverage(P) << " " << mean_cardinality << " " << mean_entropy << " "
		<< coherence(P) << " " << mean_epsilon_feasibility_ROI << " " << mean_epsilon_feasibility_risk <<  " "
		<< mean_alpha << " " << mean_anticipation_rate << " " << mean_prediction_error << " " << mean_cid_ROI << " " << mean_cid_risk << " " << mean_cid
		<< " " << mean_current_cost << " " << portfolio::total_incurred_cost << " " << mean_turnover << " " << mean_nominal_ROI  << " "
		<< portfolio::current_wealth << " " << hypervolume(P, solution::ref_point.first, solution::ref_point.second) << " " << mean_pred_hv;

	if (!(P[0]->anticipation == false && P[0]->epsilon_feasibility.first == 0.0 && P[0]->epsilon_feasibility.second == 0.0))
		out << " " << observed_hypervolume(P, 1000, solution::ref_point.first, solution::ref_point.second);
	else out << " " << 0.0;
	out << "\n";

	std::pair<std::pair<unsigned, unsigned>, std::pair<double, double> > error_pair (std::pair<unsigned, unsigned>(min_error_index,max_error_index),std::pair<double, double>(min_error,max_error));
	std::pair<std::pair<unsigned, unsigned>, std::pair<double, double> > uncertainty_pair (std::pair<unsigned, unsigned>(min_uncertainty_index,max_uncertainty_index),std::pair<double, double>(min_uncertainty,max_uncertainty));
	std::vector<std::pair<std::pair<unsigned, unsigned>, std::pair<double, double> > > res;
	res.push_back(error_pair);
	res.push_back(uncertainty_pair);

	return res;
}

void save_portfolio(const portfolio &P, std::ostream &out)
{
	for (unsigned int i = 0; i < portfolio::num_assets-1; ++i)
			out << std::setiosflags(std::ios::fixed) << std::setprecision(2) << P.investment(i) << " ";
	out << std::setiosflags(std::ios::fixed) << std::setprecision(2) << P.investment(portfolio::num_assets-1);
	out << std::endl;
}

void save_portfolio_stochastic_obj(const portfolio &P, std::ostream &out)
{
	out << P.ROI << " " << P.risk
		<< " " << P.covar(0,0) << " " << P.covar(1,1)
		<< " " << P.covar(0,1);
	out << std::endl;
}

void save_results(unsigned int e, unsigned int seed_initial_pop,
	unsigned int experiment_seed, std::vector<sol*> &P, unsigned int t)
{


	// Filename for storing final population data
	std::stringstream experiment_id; experiment_id << e;
	std::stringstream card; card << portfolio::max_cardinality;
	std::stringstream period; period << t;

	std::string file_path = "exp_" + instance + "_card" + card.str() + "_" + algo  + "_id" + experiment_id.str() + "_t" + period.str() + ".dat";

	std::ofstream arquivo(file_path.c_str(), std::ios::out);
	arquivo << seed_initial_pop << " " << experiment_seed << std::endl;
	std::vector<std::pair<std::pair<unsigned, unsigned>, std::pair<double, double> > > res = write_statistics(P, arquivo, t);
	std::pair<std::pair<unsigned, unsigned>, std::pair<double, double> > error = res[0];
	std::pair<std::pair<unsigned, unsigned>, std::pair<double, double> > uncertainty = res[1];
	arquivo.close();

	file_path = "pop_" + instance + "_card" + card.str() + "_" + algo + "_id" + experiment_id.str() + "_t" + period.str() + ".dat";
	arquivo.open(file_path.c_str(), std::ios::out);
	print_pop(P, arquivo);
	arquivo.close();

	file_path = "lowest_error_port_" + instance + "_card" + card.str() + "_" + algo  + "_id" + experiment_id.str() + "_t" + period.str() + ".dat";
	arquivo.open(file_path.c_str(), std::ios::out);
	print_pop(P, std::pair<unsigned int, double>(error.first.first,error.second.first), arquivo);
	arquivo.close();

	file_path = "lowest_uncertainty_port_" + instance + "_card" + card.str() + "_" + algo  + "_id" + experiment_id.str() + "_t" + period.str() + ".dat";
	arquivo.open(file_path.c_str(), std::ios::out);
	print_pop(P, std::pair<unsigned int, double>(uncertainty.first.first,uncertainty.second.first), arquivo);
	arquivo.close();

	file_path = "highest_error_port_" + instance + "_card" + card.str() + "_" + algo  + "_id" + experiment_id.str() + "_t" + period.str() + ".dat";
	arquivo.open(file_path.c_str(), std::ios::out);
	print_pop(P, std::pair<unsigned int, double>(error.first.second,error.second.second), arquivo);
	arquivo.close();

	file_path = "highest_uncertainty_port_" + instance + "_card" + card.str() + "_" + algo  + "_id" + experiment_id.str() + "_t" + period.str() + ".dat";
	arquivo.open(file_path.c_str(), std::ios::out);
	print_pop(P, std::pair<unsigned int, double>(uncertainty.first.second,uncertainty.second.second), arquivo);
	arquivo.close();

}

void run_algorithm(unsigned int e, unsigned int initial_pop_seed,
	unsigned int experiment_seed, unsigned int pop_size, unsigned int generations)
{
	srand(initial_pop_seed);

	portfolio::current_period = 0;
	portfolio::total_incurred_cost = 0.0;
	portfolio::current_wealth = portfolio::total_wealth;

	xover_op::probability = xover_op::uniform_prob_op;
	mutation_op::probability = mutation_op::uniform_prob_op;

	std::vector<sol*> active_population;
	for (unsigned int p = 0; p < pop_size; ++p)
		active_population.push_back(new sol());

	srand(experiment_seed);

	std::stringstream experiment_id; experiment_id << e;
	std::stringstream card; card << portfolio::max_cardinality;
	std::string filename;

	do
	{

		portfolio::max_ROI = portfolio::get_max_ROI(portfolio::current_period);
		portfolio::max_risk = portfolio::get_max_risk(portfolio::current_period);
		portfolio::min_ROI = portfolio::get_min_ROI(portfolio::current_period);
		portfolio::min_risk = 0.0;

		portfolio::max_cost = compute_max_cost();

		solution::ref_point.first = linear_transform(0.0, portfolio::min_ROI, portfolio::max_ROI);
		solution::ref_point.second = linear_transform(1.0, portfolio::min_risk, portfolio::max_risk);

		double nominal_min_ROI = portfolio::min_ROI*portfolio::current_wealth;
		double adjusted_min_ROI = nominal_min_ROI - portfolio::max_cost;
		portfolio::min_ROI = adjusted_min_ROI/portfolio::current_wealth;

		save_results(e, initial_pop_seed, experiment_seed, active_population, portfolio::current_period);

		std::stringstream period; period << portfolio::current_period;

		if (portfolio::current_period  > 0)
		for (unsigned int p = 0; p < active_population.size(); ++p)
			solution::compute_efficiency(active_population[p]);

		unsigned int num_classes = fast_nondominated_sort(active_population);
		for (unsigned int k = 0; k < num_classes; ++k)
			crowding_distance(active_population, k);

		filename = "saida-pop_" + instance + "_card" + card.str() + "_" + algo + "_id" + experiment_id.str() + "_t" + period.str() + ".dat";
		std::ofstream arquivo_geracao(filename.c_str(), std::ios::app);
		print_pop(active_population, arquivo_geracao);
		arquivo_geracao.close();

		Eigen::VectorXd acc_investments(portfolio::num_assets);
		for (unsigned int i = 0; i < active_population.size(); ++i)
			acc_investments += active_population[i]->P.investment;
		acc_investments /= active_population.size();

		investments[portfolio::current_period].push_back(acc_investments); acc_investments = sum(investments[portfolio::current_period])/investments[portfolio::current_period].size();

		unsigned int K = (portfolio::num_assets < 10) ? portfolio::num_assets : 10;
		std::vector<std::pair<unsigned,double> > top_k = portfolio::top_k_investments(acc_investments, K);

		filename = "top_k_" + instance + "_card" + card.str() + "_" + algo  + "e" + experiment_id.str() + ".dat";

		std::ofstream file_top(filename.c_str(), std::ios::app);

		for (unsigned k = 0; k < K-1; ++k)
			file_top << top_k[k].first << " " << top_k[k].second << " ";
		file_top << top_k[K-1].first << " " << top_k[K-1].second << std::endl;
		file_top.close();

		for (unsigned int p = 0; p < active_population.size(); ++p)
			portfolio::observed_error(active_population[p]->P, portfolio::current_period);

		filename = "stats_" + instance + "_card" + card.str() + "_" + algo + "_id" + experiment_id.str() + ".dat";
		std::ofstream experiment_file(filename.c_str(), std::ios::app);
		write_statistics(active_population, experiment_file, portfolio::current_period);
		experiment_file.close();

		for (unsigned int g = 1; g <= generations; ++g)
		{

			if (algo == "sms" || algo == "asms")
			{
				// ASMS-EMOA is a steady state algorithm
				for (unsigned int s = 0; s < active_population.size(); ++s)
					ASMS_EMOA_iteration(active_population, solution::ref_point.first, solution::ref_point.second, portfolio::current_period, anticipation);
			}
			else if (algo == "nsga2")
				// NSGA-II is a generational algorithm
				NSGA2_iteration(active_population, portfolio::current_period);

			//if (g == generations)
			for (unsigned int i = 0; i < active_population.size(); ++i)
			{
				active_population[i]->anticipation = false;
				if (Kalman_params::window_size == 0)
				{
					active_population[i]->P.ROI_prediction = active_population[i]->P.ROI;
					active_population[i]->P.risk_prediction = active_population[i]->P.risk;
				}

			}

			filename = "stats_" + instance + "_card" + card.str() + "_" + algo + "_id" + experiment_id.str() + ".dat";
			std::ofstream experiment_file(filename.c_str(), std::ios::app);
			write_statistics(active_population, experiment_file, portfolio::current_period);
			experiment_file.close();

			float current_hypervolume = hypervolume(active_population, solution::ref_point.first, solution::ref_point.second);
			float observed_hypv = observed_hypervolume(active_population, 1000, solution::ref_point.first, solution::ref_point.second);

			std::cout << "current/observed hypervolume [" << e << "][" << portfolio::current_period << "][" << g << "]: ("
					<< current_hypervolume << ", " << observed_hypv << ")" << std::endl;

			compute_ranks(active_population);
			sort_per_objective(active_population,0);
			std::string file_path = "pop_" + instance + "_card" + card.str() + "_" + algo  + "_id" + experiment_id.str() + "t_" + period.str() + "_end.dat";
			std::ofstream best_population_file(file_path.c_str(), std::ios::out);

			for (unsigned int i = 0; i < active_population.size(); ++i)
				save_portfolio_stochastic_obj(active_population[i]->P,best_population_file);
			best_population_file.close();

			std::stringstream generation_id; generation_id << g;
			std::stringstream period; period << portfolio::current_period;
			filename = "saida-pop_" + instance + "_card" + card.str() + "_" + algo + "_id" + experiment_id.str() + "_t" + period.str() + ".dat";
			std::ofstream arquivo_geracao(filename.c_str(), std::ios::app);
			print_pop(active_population, arquivo_geracao);
			arquivo_geracao.close();
		}


		// Pushing the population found at time t into the history
		std::vector<sol*> previous_population;
		for (unsigned int i = 0; i < active_population.size(); ++i)
			previous_population.push_back(new sol(active_population[i]));
		solution::historical_populations.push_back(previous_population);

		// Sorts the population according to ROI
		std::sort(solution::historical_populations.back().begin(),solution::historical_populations.back().end(),cmp_ROI_ptr);
		std::sort(active_population.begin(),active_population.end(),cmp_ROI_ptr);


		std::vector<double> pred_hypervolume(active_population.size());
		double best_hyp_value = std::numeric_limits<float>::min();

		unsigned int best_decision;
		if (Kalman_params::window_size > 0 && DM.compare("flexibility") == 0)
		{

			// Identifies the best candidate decision maximizing the future hypervolume
			best_decision = 0;

			// Obtains the anticipative population
			std::vector<sol*> anticipative_population;
			if (portfolio::current_period > 0)
				anticipative_population = anticipatory_learning_dec_space(active_population, portfolio::current_period);
			else
				anticipative_population = active_population;


			double min_error = std::numeric_limits<float>::max();
			double max_error = 0.0;
			double min_alpha = 1.0, max_alpha = 0.0;

			for (unsigned int j = 0; j < anticipative_population.size(); ++j)
			{

				portfolio::KF_prediction_obj_space(anticipative_population[j]->P, 1);

				anticipative_population[j]->prediction_error = anticipative_population[j]->P.kalman_state.error;

				if (anticipative_population[j]->alpha < min_alpha)
					min_alpha = anticipative_population[j]->alpha;
				else if (anticipative_population[j]->alpha > max_alpha)
					max_alpha = anticipative_population[j]->alpha;

				if (anticipative_population[j]->prediction_error < min_error)
					min_error = anticipative_population[j]->prediction_error;
				else if (anticipative_population[j]->prediction_error > max_error)
					max_error = anticipative_population[j]->prediction_error;
			}

			for (unsigned int i = 0; i < active_population.size(); ++i)
			{
				Eigen::VectorXd candidate_decision = active_population[i]->P.investment;

				for (unsigned int j = 0; j < anticipative_population.size(); ++j)
					anticipatory_learning_obj_space(active_population[j], anticipative_population[j], candidate_decision, min_error, max_error, min_alpha, max_alpha, portfolio::current_period);


				unsigned int num_classes = fast_nondominated_sort(anticipative_population);
				for (unsigned int k = 0; k < num_classes; ++k)
						crowding_distance(anticipative_population, k);

				pred_hypervolume[i] = hypervolume(anticipative_population, solution::ref_point.first, solution::ref_point.second);
				active_population[i]->pred_hv = pred_hypervolume[i];

				std::cout << "pred_hv[" << i << "](" << pred_hypervolume[i] << ")x";
				epsilon_feasible(active_population[i]);
				double weighted_pred_hypv = pred_hypervolume[i];//*active_population[i]->epsilon_feasibility.first*active_population[i]->epsilon_feasibility.second;

				std::cout << "(" << active_population[i]->epsilon_feasibility.first << ")x("
						  << active_population[i]->epsilon_feasibility.second << ") = " << weighted_pred_hypv << std::endl;

				if (weighted_pred_hypv > best_hyp_value)
				{
					best_hyp_value = weighted_pred_hypv;
					best_decision = i;
					filename = "saida-anticipative_pop_" + instance + "_card" + card.str() + "_" + algo + "_id" + experiment_id.str() + "_t" + period.str() + ".dat";
					arquivo_geracao.open(filename.c_str(), std::ios::out);
					print_pop(anticipative_population, arquivo_geracao);
					arquivo_geracao.close();
				}

				std::sort(anticipative_population.begin(),anticipative_population.end(),cmp_ROI_ptr);
			}
		}
		else if (DM.compare("median") == 0)
		{
			std::cout << "Median point decision maker\n";
			// Median decision maker. This assumes the population is already
			// sorted by the first objective function. We know that this is the case
			// because we sorted the population before entering the DM scope code.
			best_decision = active_population.size() % 2 == 0 ? active_population.size() / 2 : (active_population.size() + 1) / 2;
			epsilon_feasible(active_population[best_decision]);
		}
		else if (DM.compare("random") == 0)
		{
			std::cout << "Random decision maker\n";
			// Random decision maker
			best_decision = floor(active_population.size()*uniform_zero_one());
			epsilon_feasible(active_population[best_decision]);
		}

		std::cout << std::endl;

		std::cout << "\nApplying best decision -> ";
		std::cout << "P[" << best_decision << "]: pred_hypv=" << best_hyp_value;
		// Updating incurred transaction cost and wealth
		portfolio::update_cost_and_wealth(active_population[best_decision]->P, portfolio::current_period);
		std::cout << ", ROI_observed=" << active_population[best_decision]->P.ROI_observed << ", nominal_ROI="
				<< active_population[best_decision]->P.nominal_ROI << ", incurred_cost=" << active_population[best_decision]->P.current_cost
				 << ", total_incurred_cost=" << portfolio::total_incurred_cost << ", current_wealth=" << portfolio::current_wealth << std::endl;

		// "Applies" the best decision at the current investment environment
		portfolio::current_investment = active_population[best_decision]->P.investment;
		solution::historical_anticipative_decisions.push_back(new sol(active_population[best_decision]));

		if (portfolio::current_period == 0)
			solution::predicted_anticipative_decision = new sol(active_population[best_decision]);
		else
		{

			delete solution::predicted_anticipative_decision;
			solution::predicted_anticipative_decision = new sol(anticipatory_learning_dec_space(active_population[best_decision], portfolio::current_period));
			unsigned int initial_t = (portfolio::current_period > Kalman_params::window_size) ? (portfolio::current_period - Kalman_params::window_size) : 0;
			portfolio::observe_state(solution::predicted_anticipative_decision->P, 20, initial_t, portfolio::current_period + 1);
		}

		portfolio::current_period++;

		// Saves the decision taken
		std::string file_path = "decision_" + instance + "_card" + card.str() + "_" + algo  + "_id" + experiment_id.str() + "t_" + period.str() + "_obj.dat";
		std::ofstream decision_file(file_path.c_str(), std::ios::out);
		save_portfolio_stochastic_obj(active_population[best_decision]->P, decision_file); decision_file.close();

		file_path = "decision_" + instance + "_card" + card.str() + "_" + algo  + "_id" + experiment_id.str() + "t_" + period.str() + "_stats.dat";
		decision_file.open(file_path.c_str(), std::ios::out);
		print_portfolio(active_population[best_decision], decision_file); decision_file.close();

		file_path = "decision_" + instance + "_card" + card.str() + "_" + algo  + "_id" + experiment_id.str() + "t_" + period.str() + "_dec.dat";
		decision_file.open(file_path.c_str(), std::ios::out);
		save_portfolio(active_population[best_decision]->P, decision_file); decision_file.close();


		if (portfolio::current_period == portfolio::periods-1)
		{
			std::cout << "Ending experiment...\n";
			save_results(e, initial_pop_seed, experiment_seed, active_population, portfolio::current_period);

			for (unsigned int p = 0; p < active_population.size(); ++p)
			{
				delete active_population[p];
				active_population[p] = NULL;
			}
			active_population.clear();
			return;
		}

		std::cout << "Re-starting...\n";
		portfolio::compute_statistics(portfolio::current_period);

	} while (true);

}
