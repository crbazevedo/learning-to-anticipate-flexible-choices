// W19-1 cross-check C: bivariate Gaussian inputs (mean + covariance).
// Reads W18-1 fixture; emits per-asset mean + full covariance matrix.

#include <iostream>
#include <iomanip>
#include <sstream>
#include <Eigen/Dense>

#include "portfolio.h"

int main()
{
    std::string line;
    if (!std::getline(std::cin, line) || line != "###META") {
        std::cerr << "expected ###META\n"; return 1;
    }
    std::getline(std::cin, line);
    unsigned int n_p=0, n_a=0, n_d=0;
    { std::stringstream ss(line); char c; ss >> n_p >> c >> n_a >> c >> n_d; }

    // Skip portfolios block (not used here)
    std::getline(std::cin, line);
    for (unsigned int p = 0; p < n_p; ++p) std::getline(std::cin, line);

    std::getline(std::cin, line);  // ###RETURNS
    Eigen::MatrixXd returns(n_d, n_a);
    for (unsigned int d = 0; d < n_d; ++d) {
        std::getline(std::cin, line);
        std::stringstream ss(line);
        for (unsigned int a = 0; a < n_a; ++a) {
            char c; ss >> returns(d, a);
            if (a + 1 < n_a) ss >> c;
        }
    }

    portfolio::available_assets_size = n_a;
    Eigen::VectorXd mean = portfolio::estimate_assets_mean_ROI(returns);
    Eigen::MatrixXd cov;
    portfolio::estimate_covariance(mean, returns, cov);

    // Emit CSV. Wide format: row 0 = mean (one row, n_a cols);
    // rows 1..n_a = covariance.
    std::cout << "row_type,row_idx";
    for (unsigned int a = 0; a < n_a; ++a) std::cout << ",c" << a;
    std::cout << "\n" << std::setprecision(17);
    std::cout << "mean,0";
    for (unsigned int a = 0; a < n_a; ++a) std::cout << "," << mean(a);
    std::cout << "\n";
    for (unsigned int i = 0; i < n_a; ++i) {
        std::cout << "cov," << i;
        for (unsigned int j = 0; j < n_a; ++j)
            std::cout << "," << cov(i, j);
        std::cout << "\n";
    }
    return 0;
}
