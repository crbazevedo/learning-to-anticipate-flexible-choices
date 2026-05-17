/*
 * dirichlet.cpp
 *
 *  Created on: 10/11/2013
 *      Author: LBiC
 */
#include "../headers/dirichlet.h"
#include "../headers/portfolio.h"
#include "../headers/statistics.h"
#include "../headers/asms_emoa.h"

#include <boost/random/gamma_distribution.hpp>

#include <vector>

Eigen::VectorXd dirichlet_mean(const Eigen::VectorXd& alpha)
{
	return alpha / alpha.sum();
}

Eigen::VectorXd dirichlet_variance(const Eigen::VectorXd& alpha)
{
	double factor = alpha.sum()*alpha.sum()*alpha.sum() + alpha.sum()*alpha.sum();
	Eigen::VectorXd alpha_square = alpha.array().square();
	return (alpha.sum()*alpha - alpha_square) / factor;
}

Eigen::VectorXd dirichlet_variance(const Eigen::VectorXd& proportions, double concentration)
{
	Eigen::VectorXd alpha = concentration*proportions;
	return dirichlet_variance(alpha);
}

Eigen::VectorXd dirichlet_mean_map_estimate(const Eigen::VectorXd& alpha)
{
	return (alpha - Eigen::VectorXd::Ones(alpha.rows())) / (alpha.sum() - alpha.rows());
}

Eigen::VectorXd dirichlet_mean_prediction_vec(Eigen::VectorXd& prev_proportions, Eigen::VectorXd& current_proportions, double anticipative_rate)
{
	anticipative_rate = 0.5*anticipative_rate;
	Eigen::VectorXd prediction = prev_proportions + anticipative_rate*(current_proportions - prev_proportions);
	normalize(prediction);

	for (int i = 0; i < prediction.rows(); ++i)
		if (prediction(i) < 0.0)
			prediction(i) = 0.0;
		else if (prediction(i) > 1.0)
			prediction(i) = 1.0;

	return prediction / prediction.sum();
}

sol* dirichlet_mean_prediction(sol* prev_w, sol* current_w, unsigned int current_t)
{
	sol* predicted_w = new sol(current_w);
	double anticipative_rate = 2.0 - non_dominance_probability(prev_w, current_w);
	predicted_w->P.investment = dirichlet_mean_prediction_vec(prev_w->P.investment, current_w->P.investment, anticipative_rate);

	unsigned int initial_t = (current_t > Kalman_params::window_size) ? (current_t - Kalman_params::window_size) : 0;
	portfolio::observe_state(predicted_w->P, portfolio::samples_per_portfolio, initial_t, current_t);

	if (current_t == portfolio::current_period)
		portfolio::KF_prediction_obj_space(predicted_w->P, 1); // 2 steps ahead

	return predicted_w;
}

Eigen::VectorXd dirichlet_mean_map_update(Eigen::VectorXd& p_predicted, Eigen::VectorXd& p_obs, double concentration)
{
	Eigen::VectorXd p_updated = Eigen::VectorXd::Zero(p_predicted.rows());
	Eigen::VectorXd var = dirichlet_variance(p_predicted, concentration);

	for (int i = 0; i < p_predicted.rows(); ++i)
		if (p_predicted(i) == 0.0 || var(i) == 0.0)
			p_updated(i) = 0.0;
		else
			p_updated(i) = p_predicted(i) + 1.0*var(i)*(p_obs(i) - p_predicted(i)) / (p_predicted(i)*(1 - p_predicted(i)));

	return p_updated / p_updated.sum();
}

sol* dirichlet_mean_map_update(sol* w_predicted, sol* w_obs, unsigned int current_t)
{
	Eigen::VectorXd p_updated = dirichlet_mean_map_update(w_predicted->P.investment, w_obs->P.investment, 20);
	sol* w_updated = new sol(w_obs); w_updated->P.investment = p_updated;
	return w_updated;
}
