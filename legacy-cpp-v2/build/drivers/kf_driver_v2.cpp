// W19-2 re-assessment vs v2 KF.
// SAME fixture as v1 kf_driver; tests v2's REVERSED lifecycle:
//   v1: Kalman_filter() = prediction → update
//   v2: Kalman_filter() = update → prediction
//
// On identical inputs, post-step (x, P) will DIFFER from v1 starting step 1.

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
    std::cout << step << ",x,0";
    for (int i = 0; i < state_dim; ++i) std::cout << "," << p.x(i);
    std::cout << "\n";
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
    params.error = 0.0;  // v2 addition

    std::getline(std::cin, line);  // ###x0
    params.x = read_matrix(std::cin, state_dim, 1).col(0);
    params.u = Eigen::VectorXd::Zero(state_dim);
    std::getline(std::cin, line);  // ###P0
    params.P = read_matrix(std::cin, state_dim, state_dim);
    params.x_next = params.x;
    params.P_next = params.P;

    std::cout << "step,kind,row_idx";
    for (int j = 0; j < state_dim; ++j) std::cout << ",c" << j;
    std::cout << "\n";
    emit_state(0, params, state_dim);

    for (int s = 1; s <= n_steps; ++s) {
        std::getline(std::cin, line);  // ###MEAS_<i>
        Eigen::VectorXd z = read_matrix(std::cin, obs_dim, 1).col(0);
        Kalman_filter(params, z);  // v2 internal order: update → prediction
        emit_state(s, params, state_dim);
    }
    return 0;
}
