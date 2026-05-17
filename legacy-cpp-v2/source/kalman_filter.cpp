/*
 * kalman_filter.cpp
 *
 *  Created on: 03/03/2013
 *      Author: LBiC
 */


#include <iostream>
#include "../headers/kalman_filter.h"

Eigen::MatrixXd Kalman_params::F; // Next State Function
Eigen::MatrixXd Kalman_params::H; // Measurement Function
Eigen::MatrixXd Kalman_params::R; // Measurement Covariance Matrix

unsigned int Kalman_params::window_size;

void Kalman_prediction(Kalman_params& params)
{
	params.x_next = (params.F * params.x) + params.u;
	params.P_next = params.F * params.P * params.F.transpose();
}
void Kalman_update(Kalman_params& params, Eigen::VectorXd measurement)
{

	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(params.F.rows(),params.F.cols());
	Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(1,2);
			        Z << measurement(0), measurement(1);
	Eigen::VectorXd y = Z.transpose() - (params.H * params.x_next);
	Eigen::MatrixXd S = params.H * params.P_next * params.H.transpose() + params.R;
	Eigen::MatrixXd K = params.P_next * params.H.transpose() * S.inverse();
	params.x = params.x_next + (K * y);
	params.P = (I - (K * params.H)) * params.P_next;

	params.error += y.norm();

}
void Kalman_filter(Kalman_params& params, Eigen::VectorXd measurement)
{
	Kalman_update(params, measurement);
	Kalman_prediction(params);
}



