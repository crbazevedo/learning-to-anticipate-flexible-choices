// W18-1 build adapter: stub for the Fortran `mvtdst_` symbol that
// mvtnorm.o references. Apple silicon has no easy Fortran toolchain
// to compile the original mvtdst.f, but cross-validation drivers that
// don't actually call multivariate normal CDF (like risk_driver,
// roi_driver) just need the LINK to succeed.
//
// If a driver DOES call pmvnorm / pmvnorm_P / pmvnorm_Q at runtime,
// this stub will fill in zeros and that driver should be flagged as
// not-runnable on Apple silicon (or get a real Fortran build).

#include <cstring>

extern "C" {
    // Original signature from mvtdst.f:
    //   SUBROUTINE MVTDST(N, NU, LOWER, UPPER, INFIN, CORREL, DELTA,
    //                     MAXPTS, ABSEPS, RELEPS, ERROR, VALUE, INFORM)
    // We implement a stub that fills VALUE=0, INFORM=99 (signal-not-computed).
    void mvtdst_(int* N, int* NU,
                  double* LOWER, double* UPPER,
                  int* INFIN, double* CORREL, double* DELTA,
                  int* MAXPTS, double* ABSEPS, double* RELEPS,
                  double* ERROR, double* VALUE, int* INFORM)
    {
        (void)N; (void)NU; (void)LOWER; (void)UPPER; (void)INFIN;
        (void)CORREL; (void)DELTA; (void)MAXPTS; (void)ABSEPS; (void)RELEPS;
        if (ERROR) *ERROR = 0.0;
        if (VALUE) *VALUE = 0.0;
        if (INFORM) *INFORM = 99;  // signal: STUB; multivariate CDF not available
    }
}
