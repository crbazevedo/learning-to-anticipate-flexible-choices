#include "../headers/statistics.h"
#include "../headers/nsga2.h"
#include "../headers/mvtnorm.h"

#include <cmath>
#include <vector>
#include <limits>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <ctime>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

// Sort the candidates set using the values of a
// specific objective function
void sort_per_objective(std::vector<sol*>& candidates_set, unsigned obj_id)
{
	if (obj_id == 0)
		std::sort(candidates_set.begin(),candidates_set.end(),cmp_ROI_ptr);
	else if (obj_id == 1)
		std::sort(candidates_set.begin(),candidates_set.end(),cmp_risk_ptr);
}

void sort_per_objective(std::vector<std::pair<unsigned int, sol*> >& candidates_set, unsigned obj_id)
{
	if (obj_id == 0)
		std::sort(candidates_set.begin(),candidates_set.end(),cmp_ROI_ptr_pair);
	else if (obj_id == 1)
		std::sort(candidates_set.begin(),candidates_set.end(),cmp_risk_ptr_pair);
}

void compute_ranks(std::vector<sol*> &P)
{
	unsigned int i = 0;
	std::vector<sol*> classe;
	while (i < P.size() && P[i]->Pareto_rank == 0)
	{
		classe.push_back(P[i]);
		++i;
	}

	sort_per_objective(classe,0);

	for (unsigned int i = 0; i < classe.size(); ++i)
		classe[i]->rank_ROI = i;

	sort_per_objective(classe,1);

	for (unsigned int i = 0; i < classe.size(); ++i)
		classe[i]->rank_risk = i;
}

double spread(std::vector<sol*>& P)
{
	double average = 0.0;
	unsigned i = 1;
	std::vector<double> distances;

	while(i < P.size() && P[i]->Pareto_rank == 0)
	{
		double sum = 0.0;
		for (unsigned j = 0; j < 2; ++j)
		{
			double diff;
			if (j == 0)
				diff = P[i-1]->P.ROI - P[i]->P.ROI;
			else
				diff = P[i-1]->P.risk - P[i]->P.risk;

			diff = diff*diff;
			sum += diff;
		}
		double d = sqrt(sum);
		distances.push_back(d);
		average += d;
		i++;
	}
	unsigned pareto_size = i;
	average /= pareto_size;
	
	double sum = 0.0;
	for (i = 0; i < distances.size(); ++i)
		sum += fabs(distances[i] - average);

	return sum /= pareto_size;
}

double hypervolume(std::vector<sol*>& P, double rx, double ry)
{
	std::vector<sol*> pareto;
	for (unsigned i = 0; i < P.size(); ++i)
		if (P[i]->Pareto_rank == 0)
		{
			pareto.push_back(P[i]);
			/*if (pareto.back()->P.ROI != pareto.back()->P.ROI || pareto.back()->P.risk != pareto.back()->P.risk)
			{
				std::cout << pareto.back()->P.investment << std::endl;
				std::cout << "pareto.back()->Delta_S=" << pareto.back()->Delta_S << std::endl;
				system("pause");
			}*/
		}


	sort_per_objective(pareto,0);

	double volume = 0.0;

	for (int i = pareto.size() - 1; i > 0; --i) 
	{
		double v = (pareto[i-1]->P.risk - pareto[i]->P.risk);
		v = v * (pareto[i]->P.ROI - rx);
		volume += v;
	}

	double v = (ry - pareto[0]->P.risk);
	v = v * (pareto[0]->P.ROI - rx);
	volume += v;

	return volume;	
}

/*
  We need a functor that can pretend it's const,
  but to be a good random number generator
  it needs mutable state.
*/
namespace Eigen {
namespace internal {
template<typename Scalar>
struct scalar_normal_dist_op
{
  static boost::mt19937 rng;    // The uniform pseudo-random algorithm
  mutable boost::normal_distribution<Scalar> norm;  // The Gaussian combinator

  EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

  template<typename Index>
  inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
};

template<typename Scalar> boost::mt19937 scalar_normal_dist_op<Scalar>::rng;

template<typename Scalar>
struct functor_traits<scalar_normal_dist_op<Scalar> >
{ enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };
} // end namespace internal
} // end namespace Eigen

/*
  Draw nn samples from a size-dimensional normal distribution
  with a specified mean and covariance
*/
Eigen::MatrixXd multi_norm(Eigen::VectorXd mu, Eigen::MatrixXd Sigma, int num_samples)
{
  int size = mu.rows(); // Dimensionality (rows)
  int nn=num_samples;     // How many samples (columns) to draw
  Eigen::internal::scalar_normal_dist_op<double> randN; // Gaussian functor
  Eigen::internal::scalar_normal_dist_op<double>::rng.seed(rand()); // Seed the rng

  // Define mean and covariance of the distribution
  /*Eigen::VectorXd mean(size);
  Eigen::MatrixXd covar(size,size);

  mean  <<  0,  0;
  covar <<  1, .5,
           .5,  1;*/

  Eigen::MatrixXd normTransform(size,size);

  Eigen::LLT<Eigen::MatrixXd> cholSolver(Sigma);

  // We can only use the cholesky decomposition if
  // the covariance matrix is symmetric, pos-definite.
  // But a covariance matrix might be pos-semi-definite.
  // In that case, we'll go to an EigenSolver
  if (cholSolver.info()==Eigen::Success) {
    // Use cholesky solver
    normTransform = cholSolver.matrixL();
  } else {
    // Use eigen solver
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(Sigma);
    normTransform = eigenSolver.eigenvectors()
                   * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }

  Eigen::MatrixXd samples = (normTransform
                           * Eigen::MatrixXd::NullaryExpr(size,nn,randN)).colwise()
                           + mu;

  //std::cout << "Mean\n" << mu << std::endl;
  //std::cout << "Covar\n" << Sigma << std::endl;
  //std::cout << "Samples\n" << samples << std::endl;

  return samples;
}

double normal_cdf(Eigen::VectorXd z, Eigen::MatrixXd Sigma)
{
	unsigned int d = z.rows();
	unsigned int off_diagonal_elems = Sigma.rows()*(Sigma.rows()-1)/2;

	double * covar = new double[off_diagonal_elems];
	for (unsigned int i = 0; i < off_diagonal_elems; ++i)
		covar[i] = 0.0;

	double * upper = new double[d];
	for (unsigned int i = 0; i < d; ++i)
		upper[i] = z(i);

	double error;
	double p = pmvnorm_P(d, upper, covar, &error);

	delete []covar;
	delete []upper;

	return p;
}

double entropy(double p)
{
	if (p == 0.0 || p == 1.0)
		return 0.0;
	return - (p*log2(p) + (1.0 - p)*log2(1.0 - p));
}

double linear_entropy(double p)
{
	if (p <= .5)
		return 2.0*p;
	return 2.0*(1.0 - p);
}

