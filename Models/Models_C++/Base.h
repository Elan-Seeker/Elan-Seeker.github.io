#ifndef BASE_H
#define BASE_H

#include <NumCpp.hpp>
#include <assert.h>

using namespace nc;

NdArray<double> Matrix_T(NdArray<double> inp_matrix);

class Linear_Regression {

public:
	Linear_Regression(NdArray<double> inp_X, NdArray<double> inp_y, double inp_bias = 1);
	~Linear_Regression();
	NdArray<double> Predict(NdArray<double> X_test);
	double SDS();

private:
	NdArray<double> X, y, omega, X_aug, matrix_F, matrix_G;
	double bias;
	int n_x, n_y;

};

#endif