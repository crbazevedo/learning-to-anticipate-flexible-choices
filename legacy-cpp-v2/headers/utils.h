/*
 * utils.h
 *
 *  Created on: 21/11/2012
 *      Author: LBiC
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <Eigen/Eigen/Dense>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/random.hpp>
using namespace boost::gregorian;

Eigen::VectorXd sum(const std::vector<Eigen::VectorXd> &M);
double mean(std::vector<double> &v);
double mean(Eigen::VectorXd &v);
double median(std::vector<double> &v);
double median(Eigen::VectorXd &v);
double break_down(std::vector<double>&v, double cutoff);
double break_down(Eigen::VectorXd &v, double cutoff);
double unbiased_IQD(Eigen::VectorXd& v);
double variance(Eigen::VectorXd& v);
bool sequential_date(date current_date, date next_date);
float uniform_zero_one();
void normalize(Eigen::VectorXd& v);
void apply_threshold(Eigen::VectorXd &w, double l, double u);
double linear_transform(double value, double min, double max);
double inverse_linear_transform(double value, double min, double max);
Eigen::VectorXd dirichlet_sample(const Eigen::VectorXd& proportions, double concentratrion, boost::mt19937& rng);
Eigen::VectorXd dirichlet_sample(const Eigen::VectorXd& alpha, boost::mt19937& rng);


#endif /* UTILS_H_ */
