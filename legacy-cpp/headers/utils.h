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
using namespace boost::gregorian;

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
void aplica_threshold(Eigen::VectorXd &w, float h);


#endif /* UTILS_H_ */
