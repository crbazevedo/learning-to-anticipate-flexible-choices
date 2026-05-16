#include "../headers/utils.h"
#include "../headers/portfolio.h"
#include "../headers/statistics.h"

#include <cmath>
#include <string>
#include <fstream>
#include <iostream>
#include <strstream>
#include <algorithm>

#include <boost/assign.hpp>
#include <boost/locale.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>

using namespace std;
using namespace boost;
//using namespace boost::locale;
using namespace boost::gregorian;


date portfolio::training_start_date;
date portfolio::training_end_date;
date portfolio::validation_start_date;
date portfolio::validation_end_date;
unsigned int asset::horizon;
unsigned int portfolio::available_assets_size;
unsigned int portfolio::window_size;
unsigned int portfolio::tr_period;
unsigned int portfolio::vl_period;
Eigen::MatrixXd portfolio::covariance;
Eigen::MatrixXd portfolio::robust_covariance;
Eigen::VectorXd portfolio::mean_ROI;
Eigen::VectorXd portfolio::median_ROI;
Eigen::MatrixXd portfolio::tr_returns_data;
Eigen::MatrixXd portfolio::vl_returns_data;
Eigen::MatrixXd portfolio::current_returns_data;
Eigen::MatrixXd portfolio::complete_returns_data;
std::vector<asset> portfolio::available_assets;
Eigen::MatrixXd portfolio::autocorrelation;

unsigned int portfolio::max_cardinality;
bool portfolio::robustness;


void portfolio::sample_autocorrelation(const Eigen::MatrixXd &returns_data, unsigned int lags)
{
	portfolio::autocorrelation.resize(lags,portfolio::available_assets_size);
	double var;

	for (unsigned int a = 0; a < portfolio::available_assets_size; ++a)
	{
		Eigen::VectorXd autocorr(lags);
		Eigen::VectorXd v = static_cast<Eigen::VectorXd>(returns_data.col(a));
		double average = mean(v);
		var = 0.0;

		for (int i = 0; i < v.rows(); ++i)
		{
			var += (v(i) - average, 2.0)*(v(i) - average, 2.0);
		}

		for (unsigned int k = 0; k < lags; ++k)
		{
			double sum = 0.0;
			for (unsigned int i = 0; i < returns_data.rows()-k; ++i)
				sum += (v(i) - average)*(v(i+k) - average);
			autocorr(k) = sum / var;
		}

		portfolio::autocorrelation.col(a) = autocorr;
	}

}

asset load_asset_data(const std::string& path, const std::string& id)
{

	using namespace boost::adaptors;
	using namespace boost::assign;

	asset A;
	A.id = id;

	fstream fin(path.c_str(), ios_base::in);

	string line = "";
	getline(fin, line); // Reads and discards the first line.
						// Date, Open, High, Low, Close, Volume, Adj Close

	std::vector<double> training, validation, complete;

	getline(fin, line); // Reads the first data row.

	char_separator<char> sep(",");
	tokenizer<char_separator<char> > tokens(line, sep);
	date current_date, next_date;

	unsigned int j = 0;
	BOOST_FOREACH (const string& t, tokens)	{
		if (j == 0)
		{
			current_date = date(from_simple_string(t));
			if (current_date != portfolio::validation_end_date)
			{
				//std::cout << "CRITICAL ERROR!\n";
				fin.close();
				return A;
			}
		}
		else if (j == 6)
			if (current_date <= portfolio::validation_end_date)
			{
				stringstream s;
				s << t;
				double value;
				s >> value;
				//std::cout << "close value: " << value << "\n";
				validation.push_back(value);
				complete.push_back(value);
				break;
			}
		++j;
	}
	//std::cout << "Current date: " << current_date << std::endl;
	//std::cout << "Validation Start Date: " << portfolio::validation_start_date << std::endl;

	//std::cout << "Loading validation data...\n";
	//system("pause");

	// Load validation data
	while (!fin.eof() && current_date > portfolio::validation_start_date)
	{
		getline(fin, line);

		char_separator<char> sep(",");
		tokenizer<char_separator<char> > tokens(line, sep);

		unsigned int j = 0;
		BOOST_FOREACH (const string& t, tokens)	{
			if (j == 0)
			{
				next_date = date(from_simple_string(t));
				//std::cout << "Next date: " << next_date << std::endl;
				if (sequential_date(next_date,current_date))
					current_date = next_date;

				//std::cout << "Current date: " << current_date << " ";
				if (current_date > portfolio::validation_end_date)
					continue;
				else if (current_date < portfolio::validation_start_date)
					break;
			}
			else if (j == 6) // Pula para a sétima coluna
			{
				stringstream s;
				s << t;
				double value;
				s >> value;
				//std::cout << "close value: " << value;

				if (current_date != next_date)
				{
					day_iterator day_itr (current_date);
					for (; day_itr != next_date; --day_itr)
						if (day_itr->day_of_week().as_enum() != Saturday
								&& day_itr->day_of_week().as_enum() != Sunday)
						{
							validation.push_back(value);
							complete.push_back(value);
						}
					current_date = next_date;

				}
				else
				{
					validation.push_back(value);
					complete.push_back(value);
				}

				break;
			}
			++j;
		}
		//system("pause");
	}

	getline(fin, line); // Reads the first data row.
	tokenizer<char_separator<char> > tokens_tr(line, sep);

	j = 0;
	BOOST_FOREACH (const string& t, tokens_tr)	{
		if (j == 0)
			current_date = date(from_simple_string(t));
		else if (j == 6) // Pula para a sétima coluna
			if (current_date <= portfolio::training_end_date)
			{
				stringstream s;
				s << t;
				double value;
				s >> value;
				//std::cout << "close value: " << value;
				training.push_back(value);
				complete.push_back(value);
				break;
			}
		++j;
	}

	//std::cout << "Loading training data...\n";
	//system("pause");
	// Load training data
	while (!fin.eof() && current_date > portfolio::training_start_date)
	{

		getline(fin, line);

		char_separator<char> sep(",");
		tokenizer<char_separator<char> > tokens(line, sep);

		unsigned int j = 0;
		BOOST_FOREACH (const string& t, tokens)	{
			if (j == 0)
			{
				next_date = date(from_simple_string(t));
				if (sequential_date(next_date,current_date))
					current_date = next_date;

				//std::cout << "Current date: " << current_date << " ";
				if (current_date > portfolio::training_end_date)
					continue;
				else if (current_date < portfolio::training_start_date)
					break;
			}
			else if (j == 6)
			{
				stringstream s;
				s << t;
				double value;
				s >> value;

				//std::cout << "close value: " << value;
				if (current_date != next_date)
				{
					day_iterator day_itr (current_date);
					for (; day_itr != next_date; --day_itr)
						if (day_itr->day_of_week().as_enum() != Saturday
								&& day_itr->day_of_week().as_enum() != Sunday)
						{
							training.push_back(value);
							complete.push_back(value);
						}
					current_date = next_date;

				}
				else
				{
					training.push_back(value);
					complete.push_back(value);
				}
				break;
			}
			++j;
		}
		//system("pause");
	}

	fin.close();

	if (current_date > portfolio::training_start_date)
		return A;

	boost::copy(training | reversed, std::back_inserter(A.historical_close_price));
	boost::copy(validation | reversed, std::back_inserter(A.validation_close_price));
	boost::copy(complete | reversed, std::back_inserter(A.complete_close_price));


	return A;
}

Eigen::VectorXd portfolio::estimate_assets_median_ROI(const Eigen::MatrixXd &returns_data)
{
	Eigen::VectorXd median_ROI(portfolio::available_assets_size);
	for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
	{
		Eigen::VectorXd col = returns_data.col(i);
		median_ROI(i) = median(col);
	}

	return median_ROI;
}

Eigen::VectorXd portfolio::estimate_assets_mean_ROI(const Eigen::MatrixXd &returns_data)
{
	return returns_data.colwise().mean();
}

void portfolio::estimate_covariance(const Eigen::VectorXd &mean_ROI, const Eigen::MatrixXd &returns_data, Eigen::MatrixXd &covariance)
{
	covariance.resize(portfolio::available_assets_size,portfolio::available_assets_size);
	covariance = (returns_data.rowwise() - mean_ROI.transpose()).transpose()*(returns_data.rowwise() - mean_ROI.transpose())/(returns_data.rows()-1.0);
	//portfolio::covariance = (portfolio::raw_data.transpose()*portfolio::raw_data)/portfolio::raw_data.rows();
	//std::cout << "covariance.rows(): " << covariance.rows() << ", covariance.cols(): " << covariance.cols() << std::endl;
}

void portfolio::estimate_robust_covariance(const Eigen::VectorXd &mean_ROI, const Eigen::MatrixXd &returns_data, Eigen::MatrixXd &covariance)
{

	covariance.resize(portfolio::available_assets_size,portfolio::available_assets_size);
	Eigen::VectorXd medians(portfolio::available_assets_size), IQDs(portfolio::available_assets_size);

	// Compute robust statistics for each column of the data matrix
	for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
	{
		Eigen::VectorXd v = returns_data.col(i);

		medians(i) = median(v);
		IQDs(i) = unbiased_IQD(v);
		//std::cout << "median(" << i << "): " << medians(i) <<  " IQD(" << i << ")" << IQDs(i) << std::endl;
	}

	// Compute the transformed data
	Eigen::MatrixXd transformed_data;
	transformed_data.resize(returns_data.rows(),portfolio::available_assets_size);
	std::vector<unsigned int> non_zero(portfolio::available_assets_size);

	//std::cout << "Compute the transformed data\n";
	for (unsigned int j = 0; j < portfolio::available_assets_size; ++j)
	{
		// A sliding window to compute the sample mean
		// of return rate of the requested period of investment.
		for (int t = 0; t < returns_data.rows(); ++t)
			transformed_data(t,j) = returns_data(t,j) - medians(j);
	}

	// Compute the bias-adjusted QC estimates
	//std::cout << "Compute the bias-adjusted QC estimates\n";
	Eigen::MatrixXd correlation_matrix;
	correlation_matrix.resize(portfolio::available_assets_size,portfolio::available_assets_size);
	for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
		for (unsigned int j = 0; j < portfolio::available_assets_size; ++j)
		{
			unsigned int n_0 = 0;
			for (int t = 0; t < returns_data.rows(); ++t)
			{
				double r = transformed_data(t,i)* transformed_data(t,j);
				correlation_matrix(i,j) += r;
				if (r != 0.0)
					n_0++;
			}

			correlation_matrix(i,j) = 1.0;
			const double PI = boost::math::constants::pi<double>();
			if (i != j)
				correlation_matrix(i,j) = sin(PI/2.0*correlation_matrix(i,j)/n_0);

			// Robust Covariance Matrix (still not positive-definite)
			covariance(i,j) = IQDs(i)*IQDs(j)*correlation_matrix(i,j);
			//std::cout << "r(" << i << "," << j << ") = " << portfolio::robust_covariance(i,j) << std::endl;
		}


		//std::cout << "Computing the positive-definite covariance matrix.\n";
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covariance);
		if (eigensolver.info() != Eigen::Success) abort();
		//std::cout << "Spectral decomposition done.\n";
		//std::cout << "Projecting the data... done.\n";

		// Project the data
		Eigen::MatrixXd data_projection;
		data_projection.resize(returns_data.rows(),portfolio::available_assets_size);
		std::vector<double> unbiased_IQDs(portfolio::available_assets_size);
		//std::cout << "eigensolver.eigenvectors().rows() = " << eigensolver.eigenvectors().rows() << std::endl;
		//std::cout << "eigensolver.eigenvectors().cols() = " << eigensolver.eigenvectors().cols() << std::endl;


		// A sliding window to compute the sample mean
		// of return rate of the requested period of investment.
		for (int t = 0; t < returns_data.rows(); ++t)
		{
			Eigen::VectorXd v = eigensolver.eigenvectors()*returns_data.row(t).transpose();
			//std::cout << "v.rows() = " << v.rows() << std::endl;
			//std::cout << "v.cols() = " << v.cols() << std::endl;
			//std::cout << "data_projection.rows() = " << data_projection.rows() << std::endl;
			//std::cout << "data_projection.cols() = " << data_projection.cols() << std::endl;
			data_projection.row(t) = v.transpose();
		}

		//std::cout << "Computing robust statistics...\n";
		// Compute robust statistics for each column of the projected data matrix
		for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
		{
			Eigen::VectorXd v = data_projection.col(i);
			unbiased_IQDs[i] = unbiased_IQD(v);
			//std::cout << "unbiased_IQDs[" << i << "] = " << unbiased_IQDs[i] << std::endl;
		}


		//std::cout << "Sorting things out...\n";
		std::sort(unbiased_IQDs.begin(),unbiased_IQDs.end(),std::greater<double>());
		for (unsigned int i = 0; i < portfolio::available_assets_size; ++i)
			IQDs(i) = unbiased_IQDs.at(i)*unbiased_IQDs.at(i);


		//std::cout << "Computing the final robust covariance matrix\n";
		covariance = eigensolver.eigenvectors()*IQDs.asDiagonal()*eigensolver.eigenvectors().transpose();
		//std::cout << "robust_covariance.rows(): " << covariance.rows() << ", robust_covariance.cols(): " << covariance.cols() << std::endl;

}

double portfolio::estimate_median_ROI (unsigned int asset, const Eigen::MatrixXd &returns_data)
{
	Eigen::VectorXd v = returns_data.col(asset);
	return median(v);
}

double portfolio::estimate_mean_ROI (unsigned int asset, const Eigen::MatrixXd &returns_data)
{
	Eigen::VectorXd v = returns_data.col(asset);
	return mean(v);
}

void portfolio::compute_robust_efficiency(portfolio &P)
{
	P.ROI  = P.robust_ROI = compute_ROI(P,portfolio::median_ROI);
	P.risk = P.robust_risk = compute_risk(P,portfolio::robust_covariance);

	P.non_robust_ROI = compute_ROI(P,portfolio::mean_ROI);
	P.non_robust_risk = compute_risk(P,portfolio::covariance);
	P.cardinality = portfolio::card(P);
}

void portfolio::compute_efficiency(portfolio &P)
{
	P.ROI  = P.non_robust_ROI = compute_ROI(P,portfolio::mean_ROI);
	P.risk = P.non_robust_risk = compute_risk(P,portfolio::covariance);

	P.robust_ROI  = compute_ROI(P,portfolio::median_ROI);
	P.robust_risk = compute_risk(P,portfolio::robust_covariance);
	P.cardinality = portfolio::card(P);
}

double portfolio::compute_ROI(portfolio &P, const Eigen::VectorXd &mean_ROI)
{
	return P.investment.transpose()*mean_ROI;
}

double portfolio::compute_risk(portfolio &P, const Eigen::MatrixXd &covariance)
{
	return P.investment.transpose()*covariance*P.investment;
}

Eigen::MatrixXd portfolio::moving_average(const Eigen::MatrixXd &returns_data)
{
	Eigen::MatrixXd ma(portfolio::window_size,portfolio::available_assets_size);
	unsigned int lags = 2*portfolio::window_size;
	unsigned int t_ini = returns_data.rows() - 2*portfolio::window_size;
	unsigned int t_fin = returns_data.rows();

	for (unsigned int a = 0; a < portfolio::available_assets_size; ++a)
	{

		/*std::cout << "returns_data(" << a << "):\n";
		for (unsigned int t = t_ini; t < t_fin; ++t)
			std::cout << returns_data(t,a) << " ";
		std::cout << std::endl;
		system("pause");*/

		double sum_w = 0.0;

		unsigned int cont = 0;
		for (unsigned int i = 0; i < 2.0*portfolio::window_size; ++i)
		{
			double w = portfolio::autocorrelation(i,a);
			if (w > 0.0)
			{
				++cont;
				//std::cout << "w(" << i << ")(" << a << "): " << w << std::endl;
				sum_w += w;
			}
		}

		if (cont == 1)
		{
			//std::cout << "Now I wanna see that!\n";
			sum_w = 0.0;
			for (unsigned int i = 0; i < 2.0*portfolio::window_size; ++i)
			{
				portfolio::autocorrelation(i,a) = 1.0/(2.0*portfolio::window_size);
				sum_w += portfolio::autocorrelation(i,a);
			}
		}


		for (unsigned int i = 0; i < portfolio::window_size; ++i)
		{
			ma(i,a) = 0.0;
			//std::cout << "sum_w(" << i << "): " << sum_w << std::endl;

			for (unsigned int t = t_ini + i; t < t_fin + i; ++t)
			{
				unsigned index = t - (t_ini + i);
				//std::cout << "index(" << index << ")lags-(index+1)(" << lags-(index+1) << ")\n";
				double w = portfolio::autocorrelation(lags-(index+1),a);
				if (w < 0.0)
					continue;
				//std::cout << "(" << w << ")*";
				if (t < t_fin)
				{
					//std::cout << "i(" << i << ")a(" << a << ")t(" << t << ")rd...\n";
					ma(i,a) += w*returns_data(t,a);
					//std::cout << returns_data(t,a) << " ";
				}
				else
				{
					//std::cout << "i(" << i << ")a(" << a << ")t(" << t << ")ma\n";
					ma(i,a) += w*ma((t-t_fin),a);
					//std::cout << "(t-t_fin)(" << t-t_fin << "): " << ma((t-t_fin),a) << "\n";
				}
			}
			//std::cout << std::endl;
			ma(i,a) /= sum_w;//2*portfolio::window_size;
			//std::cout << ma(i,a) << "\n";

			//system("pause");

		}
		//system("pause");
	}

	return ma;
}

Eigen::MatrixXd portfolio::moving_median(const Eigen::MatrixXd &returns_data)
{
	Eigen::MatrixXd mm(portfolio::window_size,portfolio::available_assets_size);

	for (unsigned int a = 0; a < portfolio::available_assets_size; ++a)
	{
		for (unsigned int i = 0; i < portfolio::window_size; ++i)
		{
			mm(i,a) = 0.0;
			unsigned int t_ini = returns_data.rows() - 2*portfolio::window_size;
			unsigned int t_fin = returns_data.rows();
			Eigen::VectorXd v(2*portfolio::window_size);
			for (unsigned int t = t_ini + i; t < t_fin + i; ++t)
				if (t < t_fin)
					v(t) =  returns_data(t,a);
				else
					v(t) = mm((t-t_fin),a);
			mm(i,a) = median(v);
		}
	}

	return mm;
}

double portfolio::evaluate_stability(portfolio &P)
{
	Eigen::MatrixXd covariance;
	//Eigen::MatrixXd ma = moving_average(portfolio::current_returns_data);
	//std::cout << "moving_average ok.\n";
	double ROI_unseen, risk_unseen;

	if (portfolio::robustness)
	{
		Eigen::VectorXd median_ROI = estimate_assets_median_ROI(portfolio::vl_returns_data);
		//std::cout << "estimate_assets_median_ROI ok.\n";
		estimate_robust_covariance(median_ROI,portfolio::vl_returns_data, covariance);
		//std::cout << "estimate_robust_covariance ok.\n";
		ROI_unseen = compute_ROI(P,median_ROI);
		//std::cout << "ROI_unseen ok.\n";
		risk_unseen = compute_risk(P,covariance);
	}
	else
	{
		Eigen::VectorXd mean_ROI = estimate_assets_mean_ROI(portfolio::vl_returns_data);
		//std::cout << "estimate_assets_median_ROI ok.\n";
		estimate_covariance(mean_ROI,portfolio::vl_returns_data, covariance);
		//std::cout << "estimate_robust_covariance ok.\n";
		ROI_unseen = compute_ROI(P,mean_ROI);
		//std::cout << "ROI_unseen ok.\n";
		risk_unseen = compute_risk(P,covariance);
	}
	//std::cout << "risk_unseen ok.\n";

	double stability = 1.0;

	if (solution::regularization_type == "stability")
		stability = 1.0/(1.0 + pow(ROI_unseen - P.ROI, 2.0) + pow(risk_unseen - P.risk, 2.0));
	else if (solution::regularization_type == "entropy")
		stability = normalized_Entropy(P);

	return stability;
}

void portfolio::observe_state(portfolio &w, unsigned int N, unsigned int current_t)
{

	for (unsigned int t = 0; t <= current_t; ++t)
	{
		//std::cout << ":::[" << t << "]:::\n";

		int index = t*(portfolio::window_size - 1);
		Eigen::MatrixXd returns_data(portfolio::tr_returns_data.rows(),portfolio::tr_returns_data.cols());

		for (int i = 0; i < portfolio::tr_returns_data.rows(); ++i)
			returns_data.row(i) = portfolio::complete_returns_data.row(index + i);

		Eigen::VectorXd mean = portfolio::robustness ? estimate_assets_median_ROI(returns_data) : estimate_assets_mean_ROI(returns_data);
		Eigen::MatrixXd covar;

		if (portfolio::robustness)
			estimate_robust_covariance(mean, returns_data, covar);
		else
			estimate_covariance(mean, returns_data, covar);


		std::vector<double> ROI(N+1), risk(N+1);
		ROI[0] = compute_ROI(w, mean); risk[0] = compute_risk(w, covar);

		//std::cout << "ROI[" << 0 << "]= " << ROI[0] << " Risk[" << 0 << "]= " << risk[0] << std::endl;

		double mean_ROI = ROI[0], mean_risk = risk[0];
		double var_ROI = 0.0, var_risk = 0.0, cov = 0.0;

		for (unsigned int i = 1; i <= N; ++i)
		{


			Eigen::MatrixXd samples = multi_norm(mean, covar, returns_data.rows());
			samples.transposeInPlace();
			Eigen::VectorXd mean = portfolio::robustness ? estimate_assets_median_ROI(samples) : estimate_assets_mean_ROI(samples);
			Eigen::MatrixXd covar;
			if (portfolio::robustness)
					estimate_robust_covariance(mean, samples, covar);
				else
					estimate_covariance(mean, samples, covar);

			ROI[i] = compute_ROI(w, mean);
			risk[i] = compute_risk(w, covar);
			//std::cout << "ROI[" << i << "]= " << ROI[i] << " Risk[" << i << "]= " << risk[i] << std::endl;

			mean_ROI += ROI[i];
			mean_risk += risk[i];
		}

		mean_ROI /= N+1;
		mean_risk /= N+1;

		for (unsigned int i = 0; i <= N; ++i)
		{
			var_ROI += (ROI[i] - mean_ROI)*(ROI[i] - mean_ROI);
			var_risk += (risk[i] - mean_risk)*(risk[i] - mean_risk);
			cov += (ROI[i] - mean_ROI)*(risk[i] - mean_risk);
		}

		var_ROI /= N;
		var_risk /= N;
		cov /= N;

		if (t == 0)
		{
			// Initial state equals the average of the noisy observations
			Eigen::VectorXd state(4);
			state << ROI[0], risk[0], 0.0, 0.0;
			w.kalman_state.x = state;
			w.kalman_state.u = Eigen::VectorXd::Zero(4);

			// Initial uncertainty over the state equates the computed variances
			// Velocity uncertainty is set to an arbitrary high number
			w.kalman_state.P = Eigen::MatrixXd::Zero(4,4);
			/*w.kalman_state.P << var_ROI, 0.0, 0.0, 0.0,
								0.0, var_risk, 0.0, 0.0,
								0.0, 0.0, 1000.0, 0.0,
								0.0, 0.0, 0.0, 1000.0;*/
			w.kalman_state.P << var_ROI, cov, 0.0, 0.0,
								cov, var_risk, 0.0, 0.0,
								0.0, 0.0, 1000.0, 0.0,
								0.0, 0.0, 0.0, 1000.0;
		}

		// Updates the measurement uncertainty
		Kalman_params::R = Eigen::MatrixXd::Zero(2,2);
		Kalman_params::R << var_ROI/N, cov/N,
							cov/N, var_risk/N;
		/*Kalman_params::R << var_ROI, 0.0,
							0.0, var_risk;*/

		Eigen::VectorXd measurement(2);
		measurement << mean_ROI, mean_risk;
		//std::cout << "mean_ROI= " << mean_ROI << ", mean_risk= " << mean_risk << std::endl;

		Kalman_filter(w.kalman_state, measurement);
		//std::cout << "x= " << w.kalman_state.x << std::endl;
		//std::cout << "P= " << w.kalman_state.P << std::endl;
	}

	// We finally arrive at the current time, t.
	Kalman_prediction(w.kalman_state);
	w.ROI_prediction = w.kalman_state.x_next(0);
	w.risk_prediction = w.kalman_state.x_next(1);
	w.error_covar_prediction = w.kalman_state.P_next;

	w.ROI = w.kalman_state.x(0);
	w.risk = w.kalman_state.x(1);
	//w.non_robust_ROI = w.ROI = r1;
	//w.non_robust_risk = w.risk = r2;
	w.error_covar = w.kalman_state.P;
	//std::cout << "ROI= " << w.ROI << " ";
	//std::cout << "ROI_prediction= " << w.ROI_prediction << std::endl;
	//std::cout << "risk= " << w.risk << " ";
	//std::cout << "risk_prediction= " << w.risk_prediction << std::endl;
	//std::cout << "P= " << w.kalman_state.P << std::endl;

}

