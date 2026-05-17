// W19-3 cross-check E: Dirichlet predictor (v2-only — v1 has no Dirichlet).
//
// Reads a fixture from stdin:
//   ###DIRICHLET_META
//   n_assets,n_test_cases
//   ###CASES
//   <test_case_id>,<rate>,<concentration>
//   ###prev_<id>: n_assets floats
//   ###curr_<id>: n_assets floats
//   ###obs_<id>: n_assets floats
//
// Emits per-case:
//   test_case_id,kind,c0,c1,...
//   where kind ∈ {mean_pred, map_update}

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <Eigen/Dense>

#include "dirichlet.h"

static Eigen::VectorXd read_vec(std::istream& in, int n) {
    Eigen::VectorXd v(n);
    std::string line;
    std::getline(in, line);
    std::stringstream ss(line);
    for (int i = 0; i < n; ++i) {
        char c; ss >> v(i);
        if (i + 1 < n) ss >> c;
    }
    return v;
}

int main()
{
    std::string line;
    std::getline(std::cin, line);  // ###DIRICHLET_META
    std::getline(std::cin, line);
    int n_assets=0, n_cases=0;
    { std::stringstream ss(line); char c; ss >> n_assets >> c >> n_cases; }

    std::getline(std::cin, line);  // ###CASES
    struct Case { int id; double rate; double concentration; };
    std::vector<Case> cases;
    for (int i = 0; i < n_cases; ++i) {
        std::getline(std::cin, line);
        std::stringstream ss(line);
        Case k; char c;
        ss >> k.id >> c >> k.rate >> c >> k.concentration;
        cases.push_back(k);
    }

    std::cout << "test_case_id,kind";
    for (int a = 0; a < n_assets; ++a) std::cout << ",c" << a;
    std::cout << "\n" << std::setprecision(17);

    for (auto& k : cases) {
        std::getline(std::cin, line);  // ###prev_<id>
        Eigen::VectorXd prev = read_vec(std::cin, n_assets);
        std::getline(std::cin, line);  // ###curr_<id>
        Eigen::VectorXd curr = read_vec(std::cin, n_assets);
        std::getline(std::cin, line);  // ###obs_<id>
        Eigen::VectorXd obs = read_vec(std::cin, n_assets);

        Eigen::VectorXd pred = dirichlet_mean_prediction_vec(prev, curr, k.rate);
        Eigen::VectorXd updated = dirichlet_mean_map_update(pred, obs, k.concentration);

        std::cout << k.id << ",mean_pred";
        for (int a = 0; a < n_assets; ++a) std::cout << "," << pred(a);
        std::cout << "\n";
        std::cout << k.id << ",map_update";
        for (int a = 0; a < n_assets; ++a) std::cout << "," << updated(a);
        std::cout << "\n";
    }
    return 0;
}
