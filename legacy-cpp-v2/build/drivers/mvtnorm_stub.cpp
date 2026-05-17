// W18 substrate-v2 build adapter: stub for pmvnorm_P (the multivariate
// normal CDF) which v2 statistics.cpp calls but the v2 source tree
// doesn't ship a corresponding mvtnorm.cpp (only mvtnorm.h).
//
// Drivers that don't actually invoke pmvnorm_P at runtime (like
// risk_driver_v2 — pure quadratic form) will link clean. Drivers that
// DO call it at runtime (TIP, anticipative rate) need either a real
// pmvnorm implementation OR a documented stub-result fallback.

#include <cstring>

// v2 mvtnorm.h wraps everything in extern "C" — match that.
extern "C" {

double pmvnorm_P(int n, double* upper, double* covar, double* error)
{
    (void)n; (void)upper; (void)covar;
    if (error) *error = -1.0;
    return 0.5;
}

double pmvnorm_Q(int n, double* lower, double* covar, double* error)
{
    (void)n; (void)lower; (void)covar;
    if (error) *error = -1.0;
    return 0.5;
}

}  // extern "C"
