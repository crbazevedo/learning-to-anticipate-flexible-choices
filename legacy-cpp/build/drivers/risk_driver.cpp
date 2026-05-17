// W18-2 cross-check A: C++ driver for portfolio::compute_risk.
//
// Reads a fixture in the W18-1 format from stdin:
//   ###META
//   n_portfolios,n_assets,n_days
//   ###PORTFOLIOS
//   <n_portfolios rows of n_assets floats>
//   ###RETURNS
//   <n_days rows of n_assets floats>
//
// Computes the COVARIANCE matrix of returns (sample, ddof=1) and
// for each portfolio emits compute_risk(P, cov) to stdout as CSV:
//   portfolio_idx,risk
//
// Per thesis §7.2 Eq (7.4):
//   risk = u^T Σ̂ u  (the QUADRATIC FORM, i.e. variance)

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

#include "portfolio.h"

int main()
{
    std::string line;

    // Parse META
    if (!std::getline(std::cin, line) || line != "###META") {
        std::cerr << "W18-2 risk_driver: expected ###META header, got '"
                  << line << "'\n";
        return 1;
    }
    std::getline(std::cin, line);
    unsigned int n_portfolios = 0, n_assets = 0, n_days = 0;
    {
        std::stringstream ss(line);
        char comma;
        ss >> n_portfolios >> comma >> n_assets >> comma >> n_days;
    }

    // Parse ###PORTFOLIOS
    std::getline(std::cin, line);  // ###PORTFOLIOS
    Eigen::MatrixXd portfolios(n_portfolios, n_assets);
    for (unsigned int p = 0; p < n_portfolios; ++p) {
        std::getline(std::cin, line);
        std::stringstream ss(line);
        for (unsigned int a = 0; a < n_assets; ++a) {
            char comma;
            ss >> portfolios(p, a);
            if (a + 1 < n_assets) ss >> comma;
        }
    }

    // Parse ###RETURNS
    std::getline(std::cin, line);  // ###RETURNS
    Eigen::MatrixXd returns(n_days, n_assets);
    for (unsigned int d = 0; d < n_days; ++d) {
        std::getline(std::cin, line);
        std::stringstream ss(line);
        for (unsigned int a = 0; a < n_assets; ++a) {
            char comma;
            ss >> returns(d, a);
            if (a + 1 < n_assets) ss >> comma;
        }
    }

    // Compute mean_ROI + covariance (sample, ddof=1 to match numpy default)
    Eigen::VectorXd mean_ROI(n_assets);
    for (unsigned int a = 0; a < n_assets; ++a) {
        mean_ROI(a) = returns.col(a).mean();
    }

    Eigen::MatrixXd centered(n_days, n_assets);
    for (unsigned int d = 0; d < n_days; ++d) {
        centered.row(d) = returns.row(d) - mean_ROI.transpose();
    }
    // ddof=1
    Eigen::MatrixXd cov = (centered.transpose() * centered) / static_cast<double>(n_days - 1);

    // Set Portfolio static state so we can use the real compute_risk
    portfolio::mean_ROI = mean_ROI;
    portfolio::covariance = cov;
    portfolio::available_assets_size = n_assets;

    // Emit CSV header
    std::cout << "portfolio_idx,risk\n";
    std::cout << std::setprecision(17);

    for (unsigned int p = 0; p < n_portfolios; ++p) {
        portfolio P;
        P.investment = portfolios.row(p).transpose();
        // compute_risk = u^T Σ u (quadratic form)
        double risk = portfolio::compute_risk(P, portfolio::covariance);
        std::cout << p << "," << risk << "\n";
    }

    return 0;
}
