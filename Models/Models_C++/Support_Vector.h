#ifndef SUPPORT_VECTOR_H
#define SUPPORT_VECTOR_H

#include <NumCpp.hpp>
#include <assert.h>
#include <string>

using namespace std;
using namespace nc;

class Support_Vector_Classifier {

public:

	Support_Vector_Classifier(int input_length, int output_length, NdArray<double> inp_X_train, NdArray<double> inp_Y_train, string inp_kernel = "linear");
	~Support_Vector_Classifier();
	void Learn();
	NdArray<int> Predict(NdArray<double> X_test);

private:

	string kernel;
	int inp_len, out_len, train_num, adjust_epoch;
	NdArray<double> X_train, Y_train, lambda, bias, kernel_matrix;

	NdArray<double> Linear(NdArray<double> x1, NdArray<double> x2);
	NdArray<double> Poly(NdArray<double> x1, NdArray<double> x2, int degree = 2);
	NdArray<double> Gaussian(NdArray<double> x1, NdArray<double> x2, double delta = 1);

};

#endif