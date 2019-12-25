#include "Support_Vector.h"
#include "Base.h"

// generate a support vector classifier
Support_Vector_Classifier::Support_Vector_Classifier(int input_length, int output_length, NdArray<double> inp_X_train, NdArray<double> inp_Y_train, string inp_kernel) {

	kernel = inp_kernel;
	inp_len = input_length;
	out_len = output_length;

	X_train = inp_X_train;
	Y_train = inp_Y_train;

	assert(X_train.numRows() == Y_train.numRows() && inp_len == X_train.numCols() && out_len == Y_train.numCols());

	train_num = X_train.numRows();
	lambda = zeros<double>(Shape(train_num, out_len));
	bias = zeros<double>(Shape(out_len, 1));
	adjust_epoch = 0;

	// construct the kernel matrix to be used to lambda adjustments
	kernel_matrix = zeros<double>(Shape(train_num, train_num));
	if (kernel == "linear") {
		kernel_matrix = dot(X_train, Matrix_T(X_train));
	}
	else if (kernel == "poly") {
		int degree = 2;
		NdArray<double> kernel_matrix_temp = dot(X_train, Matrix_T(X_train));
		for (int i = 0; i < degree + 1; i++) {
			kernel_matrix += power(kernel_matrix_temp, i);
		}
	}
	else if (kernel == "gauss") {
		double delta = 1, i_j_temp = 0;
		for (int i = 0; i < train_num; i++) {
			for (int j = i; j < train_num; j++) {
				i_j_temp = sum(power(X_train(i, X_train.cSlice()) - X_train(j, X_train.cSlice()), 2), Axis::NONE)(0, 0);
				kernel_matrix(i, j) = i_j_temp;
				kernel_matrix(j, i) = i_j_temp;
			}
		}
		kernel_matrix = exp(-sqrt(kernel_matrix) / (2 * power(delta, 2)));
	}

}

//destructor of a support vector classifier
Support_Vector_Classifier::~Support_Vector_Classifier() {

}

// use SMO algorithm to adjust the omega and bias
void Support_Vector_Classifier::Learn() {

	adjust_epoch++;

	for (int i = 0; i < out_len; i++) {
		
		int j1 = 0, j2 = 1;
		while (true) {

			double adjust_temp_one = kernel_matrix(j1, j1) - 2 * kernel_matrix(j1, j2) + kernel_matrix(j2, j2);
			double adjust_temp_two = Y_train(j1, i) - Y_train(j2, i) - sum(
				lambda(lambda.rSlice(), i) * Y_train(Y_train.rSlice(), i) * (
					kernel_matrix(kernel_matrix.rSlice(), j1) - kernel_matrix(kernel_matrix.rSlice(), j2)), Axis::NONE)(0, 0);

			// adjust two lambdas each time
			lambda(j1, i) += adjust_temp_two / (adjust_temp_one * Y_train(j1, i));
			lambda(j2, i) -= adjust_temp_two / (adjust_temp_one * Y_train(j2, i));

			if (j2 == 0 || j2 == train_num - 1) {
				break;
			}

			j1 = (j1 + 2) % train_num;
			j2 = (j2 + 2) % train_num;

		}

		NdArray<double> lambda_c_i = lambda(lambda.rSlice(), i);
		lambda_c_i.putMask(lambda_c_i != 0, 1);
		double SV_num = sum(lambda_c_i, Axis::NONE)(0, 0);
		double b_SV_sum = sum(lambda_c_i * Y_train(Y_train.rSlice(), i), Axis::NONE)(0, 0);

		NdArray<double> lambda_c_i_train = lambda(lambda.rSlice(), i) * Y_train(Y_train.rSlice(), i);
		for (int j = 0; j < train_num; j++) {
			b_SV_sum -= sum(lambda_c_i_train * kernel_matrix(kernel_matrix.rSlice(), j), Axis::NONE)(0, 0);
		}

		bias(i, 0) = b_SV_sum / SV_num;

	}

}

// Predict test data
NdArray<int> Support_Vector_Classifier::Predict(NdArray<double> X_test) {

	assert(X_test.numCols() == inp_len);
	int num_test = X_test.numRows();

	NdArray<int> y_out = zeros<int>(Shape(num_test, 1));
	NdArray<double> kernel_dot, y_out_temp, lambda_Y_train = Matrix_T(lambda * Y_train);

	if (kernel == "linear") {
		for (int i = 0; i < num_test; i++) {
			kernel_dot = Linear(X_train, X_test(i, X_test.cSlice()));
			y_out_temp = dot(lambda_Y_train, kernel_dot) + bias;
			y_out(i, 0) = argmax(y_out_temp)(0, 0);
		}
	}
	else if (kernel == "poly") {
		for (int i = 0; i < num_test; i++) {
			kernel_dot = Poly(X_train, X_test(i, X_test.cSlice()));
			y_out_temp = dot(lambda_Y_train, kernel_dot) + bias;
			y_out(i, 0) = argmax(y_out_temp)(0, 0);
		}
	}
	else if (kernel == "gauss") {
		for (int i = 0; i < num_test; i++) {
			kernel_dot = Gaussian(X_train, X_test(i, X_test.cSlice()));
			y_out_temp = dot(lambda_Y_train, kernel_dot) + bias;
			y_out(i, 0) = argmax(y_out_temp)(0, 0);
		}
	}

	return y_out;

}

// Linear kernel
NdArray<double> Support_Vector_Classifier::Linear(NdArray<double> x1, NdArray<double> x2) {
	return dot(x1, Matrix_T(x2));
}

// Poly kernel
NdArray<double> Support_Vector_Classifier::Poly(NdArray<double> x1, NdArray<double> x2, int degree) {

	NdArray<double> poly_ori = dot(x1, Matrix_T(x2)), return_array = zeros<double>(poly_ori.numRows(), 1);
	for (int i = 0; i < degree + 1; i++) {
		return_array += power(poly_ori, i);
	}
	return return_array;
}

// Gaussian kernel
NdArray<double> Support_Vector_Classifier::Gaussian(NdArray<double> x1, NdArray<double> x2, double delta) {
	NdArray<double> return_array, x2_multi = x2;
	for (int i = 0; i < x1.numRows() - 1; i++) {
		x2_multi = vstack({ x2_multi, x2 });
	}
	return_array = x1 - x2_multi;
	return_array = sqrt(Matrix_T(sum(power(return_array, 2), Axis::COL)));
	return_array = exp(-return_array / (2 * power(delta, 2)));
	
	return return_array;
}