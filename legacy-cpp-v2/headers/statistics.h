#ifndef STATISTICS_H
	#define STATISTICS_H

#include "asms_emoa.h"

// Sorting functions and comparison operators
void sort_per_objective(std::vector<sol*>&,unsigned);
void sort_per_observed_objective(std::vector<sol*>&,unsigned);
void sort_per_objective(std::vector<std::pair<unsigned int, sol*> >&,unsigned);
// Diversity measures
double spread(std::vector<sol*>&);
double coverage(std::vector<sol*>& P);
double turnover(const Eigen::VectorXd& current_w, const Eigen::VectorXd& new_w);
Eigen::VectorXd concept_vector(std::vector<sol*>& P);
double coherence(std::vector<sol*>& P);
// Approximation metrics
double hypervolume(std::vector<sol*>&, double, double);
double observed_hypervolume(std::vector<sol*>&, unsigned int, double, double);
// Robustness metrics
void compute_ranks(std::vector<sol*>&);
bool check_covar(Eigen::MatrixXd Sigma);

// Draw a sample from a multivariate Gaussian distributions
Eigen::MatrixXd multi_norm(Eigen::VectorXd mu, Eigen::MatrixXd Sigma, int num_samples);
double normal_cdf(Eigen::VectorXd x, Eigen::MatrixXd Sigma);
double entropy(double p);
double linear_entropy(double p);

// Operations with Gaussian distributions


#endif
