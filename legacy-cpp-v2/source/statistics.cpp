#include "../headers/statistics.h"
#include "../headers/asms_emoa.h"
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

#define MIN(a,b) ((a)>(b)?(b):(a))
#define MAX(a,b) ((a)>(b)?(a):(b))

#if 0
#include "../headers/rmg-master/wishart_distribution.hpp"
#include "../headers/rmg-master/inverted_wishart_distribution.hpp"
#include "../headers/rmg-master/matrix_variate_t_distribution.hpp"
#include "../headers/rmg-master/inverted_matrix_variate_t_distribution.hpp"

std::vector<Eigen::MatrixXd> wishart_matrix(const Eigen::MatrixXd& covar, unsigned int sample_size)
{
    using namespace boost::numeric::ublas;
    boost::random::mt19937 rng;

    symmetric_matrix<double> A(covar.rows()), B;
    for (unsigned int i = 0; i < covar.rows(); ++i)
    	for (unsigned int j = i; j < covar.rows(); ++i)
    		A(i,j) = covar(i,j);

    wishart_distribution<> IW(covar.rows(),A);
    std::vector<Eigen::MatrixXd> sample;

    for (unsigned int n = 0; n < sample_size; ++n)
    {

		B = IW(rng);
		Eigen::MatrixXd covar_sample = Eigen::MatrixXd::Zero(covar.rows(),covar.rows());
		for (unsigned int i = 0; i < covar.rows(); ++i)
			for (unsigned int j = 0; j < covar.rows(); ++i)
				covar_sample(i,j) = B(i,j);
		sample.push_back(covar_sample);
    }

    return sample;
}
#endif


bool check_covar(Eigen::MatrixXd Sigma)
{
	unsigned size = Sigma.rows();
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

	for (unsigned int i = 0; i < size; ++i)
		for (unsigned j = 0; j < size; ++j)
			if (normTransform(i,j) != normTransform(i,j))
			  return false;
	return true;
}

// Sort the candidates set using the values of a
// specific objective function
void sort_per_objective(std::vector<sol*>& candidates_set, unsigned obj_id)
{
	if (obj_id == 0)
		std::sort(candidates_set.begin(),candidates_set.end(),cmp_ROI_ptr);
	else if (obj_id == 1)
		std::sort(candidates_set.begin(),candidates_set.end(),cmp_risk_ptr);
}

void sort_per_observed_objective(std::vector<sol*>& candidates_set, unsigned obj_id)
{
	if (obj_id == 0)
		std::sort(candidates_set.begin(),candidates_set.end(),cmp_ROI_observed_ptr);
	else if (obj_id == 1)
		std::sort(candidates_set.begin(),candidates_set.end(),cmp_risk_observed_ptr);
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
	std::vector<sol*> front;
	while (i < P.size())
	{
		front.push_back(P[i]);
		++i;
	}

	sort_per_objective(front,0);

	for (unsigned int i = 0; i < front.size(); ++i)
		front[i]->rank_ROI = i;

	sort_per_objective(front,1);

	for (unsigned int i = 0; i < front.size(); ++i)
		front[i]->rank_risk = i;
}

double spread(std::vector<sol*>& P)
{
	double average = 0.0;
	unsigned i = 1;
	std::vector<double> distances;

	while(i < P.size() && P[i]->Pareto_front == 0)
	{
		double sum = 0.0;
		for (unsigned j = 0; j < 2; ++j)
		{
			double diff;
			if (j == 0)
				diff = P[i-1]->P.ROI_observed - P[i]->P.ROI_observed;
			else
				diff = P[i-1]->P.risk_observed - P[i]->P.risk_observed;

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

double coverage(std::vector<sol*>& P)
{

	double min_risk, min_ROI, max_risk, max_ROI;

	min_risk = std::numeric_limits<double>::max();
	max_risk = std::numeric_limits<double>::min();
	min_ROI = min_risk; max_ROI = max_risk;

	for (unsigned i = 0; i < P.size(); ++i)
		if(P[i]->Pareto_front == 0)
		{
			if (P[i]->P.risk_observed < min_risk)
				min_risk = P[i]->P.risk_observed;
			else if (P[i]->P.risk_observed > max_risk)
				max_risk = P[i]->P.risk_observed;

			if (P[i]->P.ROI_observed < min_ROI)
				min_ROI = P[i]->P.ROI_observed;
			else if (P[i]->P.ROI_observed > max_ROI)
				max_ROI = P[i]->P.ROI_observed;
		}


	double d_ROI = MIN(max_ROI, 1.0) - MAX(min_ROI, solution::ref_point.first);
	double d_risk = MIN(max_risk, solution::ref_point.second) - MAX(min_risk, 0.0);
	double sum = d_risk*d_risk+d_ROI*d_ROI;

	double c = sqrt(sum / 2.0);

	if (c != c)
		return 0.0;

	return c;
}

double observed_hypervolume(std::vector<sol*>& P, unsigned int samples, double rx, double ry)
{
	double out_of_sample_future_hypv = 0.0;
	
	std::vector<std::pair<double,double> > original_values;

	for (unsigned i = 0; i < P.size(); ++i)
		original_values.push_back(std::pair<double,double>(P[i]->P.ROI_observed,P[i]->P.risk_observed));

	for (unsigned int e = 0; e < samples; ++e)
	{

		for (unsigned i = 0; i < P.size(); ++i)
		{
			original_values.push_back(std::pair<double,double>(P[i]->P.ROI_observed,P[i]->P.risk_observed));
			Eigen::MatrixXd objective_values = multi_norm(P[i]->P.kalman_state.x, P[i]->P.covar, 1);
			P[i]->P.ROI_observed = objective_values(0,0);
			P[i]->P.risk_observed = objective_values(0,1);
		}
	
		observed_fast_nondominated_sort(P);

		std::vector<sol*> pareto;
		for (unsigned i = 0; i < P.size(); ++i)
			if (P[i]->observed_Pareto_front == 0)
				pareto.push_back(P[i]);

		sort_per_observed_objective(pareto,0);

		double volume = 0.0;

		for (int i = pareto.size() - 1; i > 0; --i)
		{
			double v = (pareto[i-1]->P.risk_observed - pareto[i]->P.risk_observed);
			v = v * (pareto[i]->P.ROI_observed - rx);
			volume += v;
		}

		double v = (ry - pareto[0]->P.risk_observed);
		v = v * (pareto[0]->P.ROI_observed - rx);
		volume += v;
		
		out_of_sample_future_hypv += volume;		
		++e;
		
	}

	// Restore solutions original mean objective values
	for (unsigned i = 0; i < P.size(); ++i)
	{
		P[i]->P.ROI_observed = original_values[i].first;
		P[i]->P.risk_observed = original_values[i].second;
	}

	// Restore original sorting and class divisions
	observed_fast_nondominated_sort(P);

	return (1.0/samples)*out_of_sample_future_hypv;
}


double hypervolume(std::vector<sol*>& P, double rx, double ry)
{
	std::vector<sol*> pareto;
	for (unsigned i = 0; i < P.size(); ++i)
		if (P[i]->Pareto_front == 0)
			pareto.push_back(P[i]);

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

  Eigen::MatrixXd normTransform(size,size);
  Eigen::LLT<Eigen::MatrixXd> cholSolver(Sigma);

  // We can only use the cholesky decomposition if
  // the covariance matrix is symmetric, pos-definite.
  // But a covariance matrix might be pos-semi-definite.
  // In that case, we'll go to an EigenSolver
  if (cholSolver.info()==Eigen::Success) {
    // Use cholesky solver
	  //std::cout << "Use cholesky solver...\n";
    normTransform = cholSolver.matrixL();
  } else {
    // Use eigen solver
	  //std::cout << " Use eigen solver...\n";
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(Sigma);
    normTransform = eigenSolver.eigenvectors()
                   * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }

  Eigen::MatrixXd samples = (normTransform
                           * Eigen::MatrixXd::NullaryExpr(size,nn,randN)).colwise()
                           + mu;
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

Eigen::VectorXd concept_vector(std::vector<sol*>& P)
{
	Eigen::VectorXd c = Eigen::VectorXd::Zero(P[0]->P.investment.rows());
	for (unsigned int i = 0; i < P.size(); ++i)
		c += P[i]->P.investment;
	c /= P.size();
	return c/c.norm();
}

double coherence(std::vector<sol*>& P)
{
	Eigen::VectorXd c = concept_vector(P);
	double coherence = 0.0;
	for (unsigned int i = 0; i < P.size(); ++i)
		coherence += P[i]->P.investment.transpose()*c;
	return coherence/P.size();
}
