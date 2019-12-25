#ifndef TIME_SERIES_H
#define TIME_SERIES_H

#include <NumCpp.hpp>
#include <assert.h>

using namespace std;
using namespace nc;

tuple<double, double> AR(NdArray<double> ts, int p);
tuple<double, double> ARMA(NdArray<double> ts, int p, int q);
tuple<double, double> ARIMA(NdArray<double> ts, int p, int d, int q);
tuple<double, double> SARMA(NdArray<double> ts, int p, int q, int P, int Q, int m);
tuple<double, double> SARIMA(NdArray<double> ts, int p, int d, int q, int P, int D, int Q, int m);

#endif