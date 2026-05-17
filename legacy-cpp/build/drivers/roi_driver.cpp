// W18-3 cross-check B: C++ driver for portfolio::compute_ROI.
//
// Same fixture format as W18-2 (W18-1 fixture spec). Emits per-portfolio
// ROI to stdout:
//   portfolio_idx,roi
//
// Per thesis §7.2 Eq (7.4):
//   ROI = μ̂^T u

#include <iostream>
#include <iomanip>
#include <sstream>
#include <Eigen/Dense>

#include "portfolio.h"

int main()
{
    std::string line;
    if (!std::getline(std::cin, line) || line != "###META") {
        std::cerr << "roi_driver: expected ###META\n"; return 1;
    }
    std::getline(std::cin, line);
    unsigned int n_portfolios=0, n_assets=0, n_days=0;
    { std::stringstream ss(line); char c;
      ss >> n_portfolios >> c >> n_assets >> c >> n_days; }

    std::getline(std::cin, line);  // ###PORTFOLIOS
    Eigen::MatrixXd portfolios(n_portfolios, n_assets);
    for (unsigned int p = 0; p < n_portfolios; ++p) {
        std::getline(std::cin, line);
        std::stringstream ss(line);
        for (unsigned int a = 0; a < n_assets; ++a) {
            char c; ss >> portfolios(p, a);
            if (a + 1 < n_assets) ss >> c;
        }
    }
    std::getline(std::cin, line);  // ###RETURNS
    Eigen::MatrixXd returns(n_days, n_assets);
    for (unsigned int d = 0; d < n_days; ++d) {
        std::getline(std::cin, line);
        std::stringstream ss(line);
        for (unsigned int a = 0; a < n_assets; ++a) {
            char c; ss >> returns(d, a);
            if (a + 1 < n_assets) ss >> c;
        }
    }

    Eigen::VectorXd mean_ROI(n_assets);
    for (unsigned int a = 0; a < n_assets; ++a)
        mean_ROI(a) = returns.col(a).mean();

    portfolio::mean_ROI = mean_ROI;
    portfolio::available_assets_size = n_assets;

    std::cout << "portfolio_idx,roi\n" << std::setprecision(17);
    for (unsigned int p = 0; p < n_portfolios; ++p) {
        portfolio P;
        P.investment = portfolios.row(p).transpose();
        double roi = portfolio::compute_ROI(P, portfolio::mean_ROI);
        std::cout << p << "," << roi << "\n";
    }
    return 0;
}
