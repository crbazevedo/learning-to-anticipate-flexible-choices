/*
 * dirichlet.h
 *
 *  Created on: 10/11/2013
 *      Author: LBiC
 */

#ifndef DIRICHLET_H_
#define DIRICHLET_H_

#include <Eigen/Eigen/Dense>
#include <boost/random.hpp>

#include "asms_emoa.h"

Eigen::VectorXd dirichlet_mean(const Eigen::VectorXd& alpha);
Eigen::VectorXd dirichlet_variance(const Eigen::VectorXd& alpha);
Eigen::VectorXd dirichlet_variance(const Eigen::VectorXd& proportions, double concentration);
Eigen::VectorXd dirichlet_mean_map_estimate(const Eigen::VectorXd& alpha);

// Inference
sol* dirichlet_mean_prediction(sol* prev_w, sol* current_w, unsigned int current_t);
Eigen::VectorXd dirichlet_mean_prediction_vec(Eigen::VectorXd& prev_proportions, Eigen::VectorXd& current_proportions, double anticipative_rate);
Eigen::VectorXd dirichlet_mean_map_update(Eigen::VectorXd& p_old, Eigen::VectorXd& p_obs, double concentration);
sol* dirichlet_mean_map_update(sol* w_predicted, sol* w_obs, unsigned int current_t);


#endif /* DIRICHLET_H_ */
