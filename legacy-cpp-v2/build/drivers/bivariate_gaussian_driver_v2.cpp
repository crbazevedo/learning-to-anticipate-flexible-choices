// W19-1 re-assessment vs v2.

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

    std::getline(std::cin, line);  // ###PORTFOLIOS
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

    portfolio::num_assets = n_a;
    Eigen::VectorXd mean = portfolio::estimate_assets_mean_ROI(returns);
    Eigen::MatrixXd cov;
    portfolio::estimate_covariance(mean, returns, cov);

    std::cout << "row_type,row_idx";
    for (unsigned int a = 0; a < n_a; ++a) std::cout << ",c" << a;
    std::cout << "\n" << std::setprecision(17);
    std::cout << "mean,0";
    for (unsigned int a = 0; a < n_a; ++a) std::cout << "," << mean(a);
    std::cout << "\n";
    for (unsigned int i = 0; i < n_a; ++i) {
        std::cout << "cov," << i;
        for (unsigned int j = 0; j < n_a; ++j) std::cout << "," << cov(i, j);
        std::cout << "\n";
    }
    return 0;
}
