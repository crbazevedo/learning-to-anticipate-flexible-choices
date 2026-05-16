#ifndef STATISTICS_H
	#define STATISTICS_H

#include "nsga2.h"

// Sorting functions and comparison operators
void sort_per_objective(std::vector<sol*>&,unsigned);
void sort_per_objective(std::vector<std::pair<unsigned int, sol*> >&,unsigned);
// Diversity measures
double spread(std::vector<sol*>&);
// Approximation metrics
double hypervolume(std::vector<sol*>&, double, double);
// Robustness metrics
void compute_ranks(std::vector<sol*>&);

// Draw a sample from a multivariate Gaussian distributions
Eigen::MatrixXd multi_norm(Eigen::VectorXd mu, Eigen::MatrixXd Sigma, int num_samples);
double normal_cdf(Eigen::VectorXd x, Eigen::MatrixXd Sigma);
double entropy(double p);
double linear_entropy(double p);

// Operations with Gaussian distributions


#endif
