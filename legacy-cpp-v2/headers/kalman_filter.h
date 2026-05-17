/*
 * kalman_filter.h
 *
 *  Created on: 03/03/2013
 *      Author: LBiC
 */

#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include <Eigen/Eigen/Dense>

struct Kalman_params
{
	static Eigen::MatrixXd F; // Next State Function
	static Eigen::MatrixXd H; // Measurement Function
	static Eigen::MatrixXd R; // Measurement Covariance Matrix

	Eigen::VectorXd x; // State Vector
	Eigen::VectorXd x_next; // Next state vector
	Eigen::VectorXd u; // External control action
	Eigen::MatrixXd P; // Error Covariance Matrix
	Eigen::MatrixXd P_initial; // Error Covariance Matrix
	Eigen::MatrixXd P_next; // Next covariance

	double error;

	static unsigned int window_size; // Window size for the KF estimation
};


void Kalman_prediction(Kalman_params& params);
void Kalman_update(Kalman_params& params, Eigen::VectorXd measurement);
void Kalman_filter(Kalman_params& params, Eigen::VectorXd measurement);


#endif /* KALMAN_FILTER_H_ */
