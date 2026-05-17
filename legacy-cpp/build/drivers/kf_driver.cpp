// W19-2 cross-check D: Kalman filter predict + update cross-validation.
//
// Reads from stdin a fixed KF fixture:
//   ###KF_META
//   state_dim,obs_dim,n_steps
//   ###F           (state_dim x state_dim)
//   ###H           (obs_dim x state_dim)
//   ###R           (obs_dim x obs_dim)
//   ###x0          (state_dim)
//   ###P0          (state_dim x state_dim)
//   ###MEAS_<i>    (obs_dim)  for i = 0..n_steps-1
//
// Runs N predict-update cycles; emits state + covariance at each step:
//   step,kind,row_idx,c0,c1,...
//   0,x,0,...
//   0,P,0,...
//   0,P,1,...
//   ...

#include <iostream>
#include <iomanip>
#include <sstream>
#include <Eigen/Dense>

#include "kalman_filter.h"

static Eigen::MatrixXd read_matrix(std::istream& in, int rows, int cols) {
    Eigen::MatrixXd M(rows, cols);
    std::string line;
    for (int r = 0; r < rows; ++r) {
        std::getline(in, line);
        std::stringstream ss(line);
        for (int c = 0; c < cols; ++c) {
            char comma; ss >> M(r, c);
            if (c + 1 < cols) ss >> comma;
        }
    }
    return M;
}

static void emit_state(int step, const Kalman_params& p, int state_dim) {
    std::cout << std::setprecision(17);
    // x
    std::cout << step << ",x,0";
    for (int i = 0; i < state_dim; ++i) std::cout << "," << p.x(i);
    std::cout << "\n";
    // P (row by row)
    for (int i = 0; i < state_dim; ++i) {
        std::cout << step << ",P," << i;
        for (int j = 0; j < state_dim; ++j) std::cout << "," << p.P(i, j);
        std::cout << "\n";
    }
}

int main()
{
    std::string line;
    std::getline(std::cin, line);  // ###KF_META
    std::getline(std::cin, line);
    int state_dim=0, obs_dim=0, n_steps=0;
    { std::stringstream ss(line); char c; ss >> state_dim >> c >> obs_dim >> c >> n_steps; }

    std::getline(std::cin, line);  // ###F
    Kalman_params::F = read_matrix(std::cin, state_dim, state_dim);
    std::getline(std::cin, line);  // ###H
    Kalman_params::H = read_matrix(std::cin, obs_dim, state_dim);
    std::getline(std::cin, line);  // ###R
    Kalman_params::R = read_matrix(std::cin, obs_dim, obs_dim);

    Kalman_params params;
    std::getline(std::cin, line);  // ###x0
    params.x = read_matrix(std::cin, state_dim, 1).col(0);
    params.u = Eigen::VectorXd::Zero(state_dim);
    std::getline(std::cin, line);  // ###P0
    params.P = read_matrix(std::cin, state_dim, state_dim);
    params.x_next = params.x;
    params.P_next = params.P;

    // Emit step 0 state (initial)
    std::cout << "step,kind,row_idx";
    for (int j = 0; j < state_dim; ++j) std::cout << ",c" << j;
    std::cout << "\n";
    emit_state(0, params, state_dim);

    for (int s = 1; s <= n_steps; ++s) {
        std::getline(std::cin, line);  // ###MEAS_<i>
        Eigen::VectorXd z = read_matrix(std::cin, obs_dim, 1).col(0);
        Kalman_filter(params, z);
        emit_state(s, params, state_dim);
    }
    return 0;
}
