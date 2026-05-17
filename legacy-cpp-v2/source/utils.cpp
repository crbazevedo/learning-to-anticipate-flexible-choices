/*
 * utils.cpp
 *
 *  Created on: 21/11/2012
 *      Author: LBiC
 */

#include "../headers/utils.h"
#include <algorithm>

double linear_transform(double value, double min, double max)
{
	double scaled_value = (value - min)/(max - min);
	return scaled_value;
}

double inverse_linear_transform(double value, double min, double max)
{
	return value*(max-min)+min;
}

Eigen::VectorXd sum(const std::vector<Eigen::VectorXd> &M)
{
	Eigen::VectorXd v(M[0].rows());

	for (unsigned int i = 0; i < M.size(); ++i)
		v += M[i];

	return v;
}

void normalize(Eigen::VectorXd& v)
{
	v = (1.0/ v.sum())*v;
}

float uniform_zero_one()
{
	return (float)rand() / RAND_MAX;
}

bool sequential_date(date current_date, date next_date)
{
	if (current_date.day_of_week().as_enum() == Monday)
		return next_date.day_of_week().as_enum() == Tuesday;
	if (current_date.day_of_week().as_enum() == Tuesday)
		return next_date.day_of_week().as_enum() == Wednesday;
	if (current_date.day_of_week().as_enum() == Wednesday)
		return next_date.day_of_week().as_enum() == Thursday;
	if (current_date.day_of_week().as_enum() == Thursday)
		return next_date.day_of_week().as_enum() == Friday;
	if (current_date.day_of_week().as_enum() == Friday)
		return next_date.day_of_week().as_enum() == Monday;
	return true;
}

double mean(std::vector<double> &v)
{
	double m = 0.0;
	for (unsigned int i = 0; i < v.size(); ++i)
		m += v[i];
	return m / v.size();
}

double mean(Eigen::VectorXd &v)
{
	return v.mean();
}

double median(std::vector<double> &v)
{

	if (v.size() % 2 != 0)
	{
		size_t n = v.size() / 2;
		std::nth_element(v.begin(), v.begin()+n, v.end());
		return v[n];
	}
	else
	{
		size_t n1 = (v.size()-1) / 2;
		size_t n2 = (v.size()-1) / 2 + 1;
		std::nth_element(v.begin(), v.begin()+n1, v.end());
		std::nth_element(v.begin(), v.begin()+n2, v.end());

		return (v[n1] + v[n2]) / 2.0;
	}
}

double median(Eigen::VectorXd &v)
{

	if (v.count() == 0)
		return 0.0;

	std::vector<double> v2;
	v2.resize(v.rows());

	for (int i = 0; i < v.rows(); ++i)
		v2[i] = v(i);

	return median(v2);
}

double break_down(std::vector<double>&v, double cutoff)
{
	size_t n = v.size()*cutoff;
	std::nth_element(v.begin(), v.begin()+n, v.end());
	return v[n];
}

double break_down(Eigen::VectorXd &v, double cutoff)
{
	std::vector<double> v2;
		v2.resize(v.rows());
		for (int i = 0; i < v.rows(); ++i)
			v2[i] = v(i);
	return break_down(v2,cutoff);
}

double unbiased_IQD(Eigen::VectorXd& v)
{

	double Q1 = break_down(v, 0.25);
	double Q3 = break_down(v, 0.75);
	return .7413*(Q3 - Q1); // The constant is such that
							// the IQD is unbiased when the
							// columns are normally distributed.
}

double variance(Eigen::VectorXd& v)
{
	return 0.0;
}

void apply_threshold(Eigen::VectorXd &w, double l, double u)
{
	normalize(w);
	for (int i = 0; i < w.rows(); ++i)
	{
		if (w(i) > 0.0 && w(i) < l)
			w(i) = l;
		else if (w(i) > u)
			w(i) = u;
		normalize(w);
	}

}

Eigen::VectorXd dirichlet_sample(const Eigen::VectorXd& alpha, boost::mt19937& rng)
{
	Eigen::VectorXd sample(alpha.rows());

	for (int i = 0; i < alpha.rows(); ++i)
	{
		boost::gamma_distribution<> pdf(alpha(i));
		boost::variate_generator<boost::mt19937&, boost::gamma_distribution<> > generator(rng, pdf);
		sample(i) = generator();
	}

	return sample / sample.sum();
}

Eigen::VectorXd dirichlet_sample(const Eigen::VectorXd& proportions, double concentratrion, boost::mt19937& rng)
{
	Eigen::VectorXd alpha(proportions.rows());

	for (int i = 0; i < proportions.rows(); ++i)
		alpha(i) = proportions(i)*concentratrion;

	return dirichlet_sample(alpha, rng);
}
