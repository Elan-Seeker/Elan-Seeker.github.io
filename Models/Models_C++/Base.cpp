#include "Base.h"

// generate a transposed matrix
NdArray<double> Matrix_T(NdArray<double> inp_matrix) {

	int row = inp_matrix.numRows(), col = inp_matrix.numCols();
	NdArray<double> ret_matrix = zeros<double>(Shape(col, row));

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; ++j) {
			ret_matrix(j, i) = inp_matrix(i, j);
		}
	}

	return ret_matrix;
}

// initialize a linear regression
Linear_Regression::Linear_Regression(NdArray<double> inp_X, NdArray<double> inp_y, double inp_bias) {

	X = inp_X;
	y = inp_y;
	bias = inp_bias;

	n_x = X.numCols(), n_y = X.numRows();
	assert((y.numRows() == n_y) && (y.numCols() == 1));

	omega = zeros<double>(Shape(1, n_x + 1));
	X_aug = hstack({X, ones<double>(Shape(n_y, 1))});

	matrix_F = dot(Matrix_T(X_aug), X_aug);
	matrix_G = dot(Matrix_T(X_aug), y);

	omega = dot(linalg::inv(matrix_F), matrix_G);

}

// destructor of a linear regression
Linear_Regression::~Linear_Regression() {

}

// predict the output of X_test
NdArray<double> Linear_Regression::Predict(NdArray<double> X_test) {

	assert(X_test.numCols() == n_x);
	int n_test = X_test.numRows();
	NdArray<double> X_test_aug = hstack({ X_test, ones<double>(Shape(n_test, 1)) });

	NdArray<double> y_pred = dot(X_test_aug, omega);

	return y_pred;

}

// get the standard deviation of sample
double Linear_Regression::SDS() {
	
	NdArray<double> y_pred = Predict(X);

	return (sqrt(sum(power(y_pred - y, 2), Axis::NONE)) / n_y)(0, 0);

}