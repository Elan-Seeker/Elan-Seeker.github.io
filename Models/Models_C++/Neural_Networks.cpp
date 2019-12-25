#include "Neural_Networks.h"
#include "Base.h"

// Initialize a neural network
Neural_Network::Neural_Network(int input_length, int hidden_length, int output_length, string inp_act_func, double inp_learning_rate, string init_method) {

	inp_weights = Init_Trans(random::randN<double>(Shape(hidden_length, input_length + 1)), init_method);
	for (int i = 0; i < hidden_length; i++) {
		inp_weights(i, input_length) = 0;
	}

	hid_weights = Init_Trans(random::randN<double>(Shape(output_length, hidden_length + 1)), init_method);
	for (int i = 0; i < output_length; i++) {
		hid_weights(i, hidden_length) = 0;
	}

	learning_rate = inp_learning_rate;
	act_func = inp_act_func;
	inp_len = input_length;
	hid_len = hidden_length;
	out_len = output_length;
	// bias is fixed to 1, for which the weights for it can be adjusted
	bias = 1.0;
	epsilon = 1e-10;
	adjust_time = 0;
	// vec_one, vec_two, PD: used in the different optimization methods
	inp_vec_one = zeros<double>(Shape(hid_len, inp_len + 1));
	inp_vec_two = zeros<double>(Shape(hid_len, inp_len + 1));
	hid_vec_one = zeros<double>(Shape(out_len, hid_len + 1));
	hid_vec_two = zeros<double>(Shape(out_len, hid_len + 1));
	inp_PD = zeros<double>(Shape(hid_len, inp_len + 1));
	hid_PD = zeros<double>(Shape(out_len, hid_len + 1));

}

// Predict tested data
NdArray<double> Neural_Network::Predict(NdArray<double> input_points) {

	int num_predict = input_points.numRows();
	assert(input_points.numCols() == inp_len);

	NdArray<double> z_hid = dot(hstack({ input_points, ones<double>(Shape(num_predict, 1)) }), Matrix_T(inp_weights));
	if (act_func == "Sigmoid" || act_func == "Tanh") {
		z_hid.putMask(z_hid < -100, -100);
	}
	NdArray<double> y_hid = Act_Func_Dict(z_hid, act_func);

	NdArray<double> z_out = dot(hstack({ y_hid, ones<double>(Shape(num_predict, 1)) }), Matrix_T(hid_weights));
	if (act_func == "Sigmoid" || act_func == "Tanh") {
		z_out.putMask(z_out < -100, 100);
	}
	NdArray<double> y_out = z_out;

	// use softmax to convert y_out into the probabilistic forms
	// avoid invalid calculation in exp
	y_out.putMask(y_out > 100, 100);
	NdArray<double> y_out_exp = exp(y_out);
	double Ohm = sum(y_out_exp, Axis::NONE)(0, 0);
	NdArray<double> Y_out = y_out_exp / Ohm;

	return Y_out;

}

// Destructor of a neural network
Neural_Network::~Neural_Network() {

}

// Adjust each weight from the training data
void Neural_Network::Adjust(NdArray<double> target_result, NdArray<double> input_points, string optimizer, double dropout) {

	adjust_time++;
	assert(target_result.numRows() == input_points.numRows() && target_result.numCols() == out_len && input_points.numCols() == inp_len);
	int batch_size = target_result.numRows();

	hid_PD = 0;
	inp_PD = 0;

	NdArray<double> hid_mask = random::rand<double>(Shape(1, hid_len));
	hid_mask.putMask(hid_mask > dropout, 1.0);
	hid_mask.putMask(hid_mask != 1.0, 0);

	for (int row_ind = 0; row_ind < batch_size; row_ind++) {

		NdArray<double> z_hid = dot(hstack({ input_points(row_ind, input_points.cSlice()), ones<double>(Shape(1, 1)) }), Matrix_T(inp_weights));
		if (act_func == "Sigmoid" || act_func == "Tanh") {
			z_hid.putMask(z_hid < -100, -100);
		}
		NdArray<double> y_hid = Act_Func_Dict(z_hid, act_func);
		// dropout technique to avoid over-fitting
		y_hid *= hid_mask;

		NdArray<double> z_out = dot(hstack({ y_hid, ones<double>(Shape(1, 1)) }), Matrix_T(hid_weights));
		if (act_func == "Sigmoid" || act_func == "Tanh") {
			z_out.putMask(z_out < -100, 100);
		}
		NdArray<double> y_out = z_out;

		// use softmax to convert y_out into the probabilistic forms
		// avoid invalid calculation in exp
		y_out.putMask(y_out > 100, 100);
		NdArray<double> y_out_exp = exp(y_out);
		double Ohm = sum(y_out_exp, Axis::NONE)(0, 0);
		NdArray<double> Y_out = y_out_exp / Ohm;

		// calculate the partial derivative matrix of loss function in input layer and hidden layer
		// loss function: C = -Sigma(Y_i * ln(Y_out_i)), i = 1 - 10)
		NdArray<double> M1 = Y_out - target_result(row_ind, target_result.cSlice());
		NdArray<double> M2 = dot(M1, hid_weights(hid_weights.rSlice(), Slice(0, hid_len)) * Array_Broadcast(hid_mask, 0, out_len)) * Act_Func_PD_Dict(z_hid, act_func);

		hid_PD += Array_Broadcast(Matrix_T(M1), 1, hid_len + 1) * Array_Broadcast(hstack({ y_hid * hid_mask, ones<double>(Shape(1,1)) * bias }), 0, out_len);
		inp_PD += Array_Broadcast(Matrix_T(M2 * hid_mask), 1, inp_len + 1) * Array_Broadcast(hstack({ input_points(row_ind, input_points.cSlice()), ones<double>(Shape(1,1)) }), 0, hid_len);

	}

	hid_PD /= batch_size;
	inp_PD /= batch_size;

	// classic gradient descent :
		// X(t) = X(t - 1) - ita * gradient(X, t - 1)
	if (optimizer == "GD") {
		hid_weights -= hid_PD * learning_rate;
		inp_weights -= inp_PD * learning_rate;
	}

	// Momentum optimizer:
		// v(t) = beta * v(t-1) - ita * gradient(X, t-1)
		// X(t) = X(t-1) + v(t)
	else if (optimizer == "Momentum") {
		double beta = 0.9;
		hid_vec_one = hid_vec_one * beta - hid_PD * learning_rate;
		inp_vec_one = inp_vec_one * beta - inp_PD * learning_rate;
		hid_weights += hid_vec_one;
		inp_weights += inp_vec_one;
	}

	// AdaGrad optimizer:
		// v(t) = v(t-1) + gradient(X, t-1) ^ 2
		// X(t) = X(t-1) - ita * gradient(X, t-1) / (sqrt(v(t)) + epsilon)
	else if (optimizer == "AdaGrad") {
		hid_vec_one += power(hid_PD, 2);
		inp_vec_one += power(inp_PD, 2);
		hid_weights -= (hid_PD / sqrt(hid_vec_one + epsilon)) * learning_rate;
		inp_weights -= (inp_PD / sqrt(inp_vec_one + epsilon)) * learning_rate;
	}

	// Adam optimizer:
		// v1(t) = beta1 * v1(t-1) + (1 - beta1) * gradient(X, t-1)
		// v2(t) = beta2 * v2(t-1) + (1 - beta2) * gradient(X, t-1) ^ 2
		// v1_regu(t) = v1(t) / (1 - beta1 ^ t)
		// v2_regu(t) = v2(t) / (1 - beta2 ^ t)
		// X(t) = X(t-1) - ita * v1_regu(t) / (sqrt(v2_regu(t)) + epsilon)
	else if (optimizer == "Adam") {
		double beta_one = 0.9, beta_two = 0.99;
		hid_vec_one = hid_vec_one * beta_one + hid_PD * (1 - beta_one);
		hid_vec_two = hid_vec_two * beta_two + power(hid_PD, 2) * (1 - beta_two);
		inp_vec_one = inp_vec_one * beta_one + inp_PD * (1 - beta_one);
		inp_vec_two = inp_vec_two * beta_two + power(inp_PD, 2) * (1 - beta_two);
		hid_weights -= (hid_vec_one / (1 - power(beta_one, adjust_time)) * learning_rate) / (
			sqrt(hid_vec_two / (1 - power(beta_two, adjust_time))) + epsilon);
		inp_weights -= (inp_vec_one / (1 - power(beta_one, adjust_time)) * learning_rate) / (
			sqrt(inp_vec_two / (1 - power(beta_two, adjust_time))) + epsilon);
	}

}

// Initialize a deep neural network
Deep_Neural_Network::Deep_Neural_Network(int input_length, int hid_a_length, int hid_b_length, int output_length, string inp_act_func, double inp_learning_rate, string init_method) {

	inp_weights = Init_Trans(random::randN<double>(Shape(hid_a_length, input_length + 1)), init_method);
	for (int i = 0; i < hid_a_length; i++) {
		inp_weights(i, input_length) = 0;
	}

	hid_a_weights = Init_Trans(random::randN<double>(Shape(hid_b_length, hid_a_length + 1)), init_method);
	for (int i = 0; i < hid_b_length; i++) {
		hid_a_weights(i, hid_a_length) = 0;
	}

	hid_b_weights = Init_Trans(random::randN<double>(Shape(output_length, hid_b_length + 1)), init_method);
	for (int i = 0; i < output_length; i++) {
		hid_b_weights(i, hid_b_length) = 0;
	}

	learning_rate = inp_learning_rate;
	act_func = inp_act_func;
	inp_len = input_length;
	hid_a_len = hid_a_length;
	hid_b_len = hid_b_length;
	out_len = output_length;
	// bias is fixed to 1, for which the weights for it can be adjusted
	bias = 1.0;
	epsilon = 1e-10;
	adjust_time = 0;
	// vec_one, vec_two, PD: used in the different optimization methods
	inp_vec_one = zeros<double>(Shape(hid_a_len, inp_len + 1));
	inp_vec_two = zeros<double>(Shape(hid_a_len, inp_len + 1));
	hid_a_vec_one = zeros<double>(Shape(hid_b_len, hid_a_len + 1));
	hid_a_vec_two = zeros<double>(Shape(hid_b_len, hid_a_len + 1));
	hid_b_vec_one = zeros<double>(Shape(out_len, hid_b_len + 1));
	hid_b_vec_two = zeros<double>(Shape(out_len, hid_b_len + 1));
	inp_PD = zeros<double>(Shape(hid_a_len, inp_len + 1));
	hid_a_PD = zeros<double>(Shape(hid_b_len, hid_a_len + 1));
	hid_b_PD = zeros<double>(Shape(out_len, hid_b_len + 1));

}

// Predict tested data
NdArray<double> Deep_Neural_Network::Predict(NdArray<double> input_points) {

	int num_predict = input_points.numRows();
	assert(input_points.numCols() == inp_len);

	NdArray<double> z_hid_a = dot(hstack({ input_points, ones<double>(Shape(num_predict, 1)) }), Matrix_T(inp_weights));
	if (act_func == "Sigmoid" || act_func == "Tanh") {
		z_hid_a.putMask(z_hid_a < -100, -100);
	}
	NdArray<double> y_hid_a = Act_Func_Dict(z_hid_a, act_func);

	NdArray<double> z_hid_b = dot(hstack({ y_hid_a, ones<double>(Shape(num_predict, 1)) }), Matrix_T(hid_a_weights));
	if (act_func == "Sigmoid" || act_func == "Tanh") {
		z_hid_b.putMask(z_hid_b < -100, -100);
	}
	NdArray<double> y_hid_b = Act_Func_Dict(z_hid_b, act_func);

	NdArray<double> z_out = dot(hstack({ y_hid_b, ones<double>(Shape(num_predict, 1)) }), Matrix_T(hid_b_weights));
	if (act_func == "Sigmoid" || act_func == "Tanh") {
		z_out.putMask(z_out < -100, 100);
	}
	NdArray<double> y_out = z_out;

	// use softmax to convert y_out into the probabilistic forms
	// avoid invalid calculation in exp
	y_out.putMask(y_out > 100, 100);
	NdArray<double> y_out_exp = exp(y_out);
	double Ohm = sum(y_out_exp, Axis::NONE)(0, 0);
	NdArray<double> Y_out = y_out_exp / Ohm;

	return Y_out;

}

// Destructor of a deep neural network
Deep_Neural_Network::~Deep_Neural_Network() {

}

// Adjust each weight from the training data
void Deep_Neural_Network::Adjust(NdArray<double> target_result, NdArray<double> input_points, string optimizer, double dropout) {

	adjust_time++;
	assert(target_result.numRows() == input_points.numRows() && target_result.numCols() == out_len && input_points.numCols() == inp_len);
	int batch_size = target_result.numRows();

	inp_PD = 0;
	hid_a_PD = 0;
	hid_b_PD = 0;

	NdArray<double> hid_a_mask = random::rand<double>(Shape(1, hid_a_len));
	NdArray<double> hid_b_mask = random::rand<double>(Shape(1, hid_b_len));
	hid_a_mask.putMask(hid_a_mask > dropout, 1.0);
	hid_a_mask.putMask(hid_a_mask != 1.0, 0);
	hid_b_mask.putMask(hid_b_mask > dropout, 1.0);
	hid_b_mask.putMask(hid_b_mask != 1.0, 0);

	for (int row_ind = 0; row_ind < batch_size; row_ind++) {

		NdArray<double> z_hid_a = dot(hstack({ input_points(row_ind, input_points.cSlice()), ones<double>(Shape(1, 1)) }), Matrix_T(inp_weights));
		if (act_func == "Sigmoid" || act_func == "Tanh") {
			z_hid_a.putMask(z_hid_a < -100, -100);
		}
		NdArray<double> y_hid_a = Act_Func_Dict(z_hid_a, act_func);
		// dropout technique to avoid over-fitting
		y_hid_a *= hid_a_mask;

		NdArray<double> z_hid_b = dot(hstack({ y_hid_a, ones<double>(Shape(1, 1)) }), Matrix_T(hid_a_weights));
		if (act_func == "Sigmoid" || act_func == "Tanh") {
			z_hid_b.putMask(z_hid_b < -100, -100);
		}
		NdArray<double> y_hid_b = Act_Func_Dict(z_hid_b, act_func);
		// dropout technique to avoid over-fitting
		y_hid_b *= hid_b_mask;

		NdArray<double> z_out = dot(hstack({ y_hid_b, ones<double>(Shape(1, 1)) }), Matrix_T(hid_b_weights));
		if (act_func == "Sigmoid" || act_func == "Tanh") {
			z_out.putMask(z_out < -100, 100);
		}
		NdArray<double> y_out = z_out;

		// use softmax to convert y_out into the probabilistic forms
		// avoid invalid calculation in exp
		y_out.putMask(y_out > 100, 100);
		NdArray<double> y_out_exp = exp(y_out);
		double Ohm = sum(y_out_exp, Axis::NONE)(0, 0);
		NdArray<double> Y_out = y_out_exp / Ohm;

		// calculate the partial derivative matrix of loss function in input layer and hidden layer
		// loss function: C = -Sigma(Y_i * ln(Y_out_i)), i = 1 - 10)
		NdArray<double> M1 = Y_out - target_result(row_ind, target_result.cSlice());
		NdArray<double> M2 = dot(M1, hid_b_weights(hid_b_weights.rSlice(), Slice(0, hid_b_len)) * Array_Broadcast(hid_b_mask, 0, out_len)) * Act_Func_PD_Dict(z_hid_b, act_func);
		NdArray<double> M3 = dot(M2, hid_a_weights(hid_a_weights.rSlice(), Slice(0, hid_a_len)) * Array_Broadcast(hid_a_mask, 0, hid_b_len) * Array_Broadcast(Matrix_T(hid_b_mask), 1, hid_a_len)) * Act_Func_PD_Dict(z_hid_a, act_func);
		
		hid_b_PD += Array_Broadcast(Matrix_T(M1), 1, hid_b_len + 1) * Array_Broadcast(hstack({ y_hid_b * hid_b_mask, ones<double>(Shape(1,1)) * bias }), 0, out_len);
		hid_a_PD += Array_Broadcast(Matrix_T(M2 * hid_b_mask), 1, hid_a_len + 1) * Array_Broadcast(hstack({ y_hid_a * hid_a_mask, ones<double>(Shape(1,1)) * bias }), 0, hid_b_len);
		inp_PD += Array_Broadcast(Matrix_T(M3 * hid_a_mask), 1, inp_len + 1) * Array_Broadcast(hstack({ input_points(row_ind, input_points.cSlice()), ones<double>(Shape(1,1)) }), 0, hid_a_len);

	}

	hid_b_PD /= batch_size;
	hid_a_PD /= batch_size;
	inp_PD /= batch_size;

	// classic gradient descent :
		// X(t) = X(t - 1) - ita * gradient(X, t - 1)
	if (optimizer == "GD") {
		hid_b_weights -= hid_b_PD * learning_rate;
		hid_a_weights -= hid_a_PD * learning_rate;
		inp_weights -= inp_PD * learning_rate;
	}

	// Momentum optimizer:
		// v(t) = beta * v(t-1) - ita * gradient(X, t-1)
		// X(t) = X(t-1) + v(t)
	else if (optimizer == "Momentum") {
		double beta = 0.9;
		hid_b_vec_one = hid_b_vec_one * beta - hid_b_PD * learning_rate;
		hid_a_vec_one = hid_a_vec_one * beta - hid_a_PD * learning_rate;
		inp_vec_one = inp_vec_one * beta - inp_PD * learning_rate;
		hid_b_weights += hid_b_vec_one;
		hid_a_weights += hid_a_vec_one;
		inp_weights += inp_vec_one;
	}

	// AdaGrad optimizer:
		// v(t) = v(t-1) + gradient(X, t-1) ^ 2
		// X(t) = X(t-1) - ita * gradient(X, t-1) / (sqrt(v(t)) + epsilon)
	else if (optimizer == "AdaGrad") {
		hid_b_vec_one += power(hid_b_PD, 2);
		hid_a_vec_one += power(hid_a_PD, 2);
		inp_vec_one += power(inp_PD, 2);
		hid_b_weights -= (hid_b_PD / sqrt(hid_b_vec_one + epsilon)) * learning_rate;
		hid_a_weights -= (hid_a_PD / sqrt(hid_a_vec_one + epsilon)) * learning_rate;
		inp_weights -= (inp_PD / sqrt(inp_vec_one + epsilon)) * learning_rate;
	}

	// Adam optimizer:
		// v1(t) = beta1 * v1(t-1) + (1 - beta1) * gradient(X, t-1)
		// v2(t) = beta2 * v2(t-1) + (1 - beta2) * gradient(X, t-1) ^ 2
		// v1_regu(t) = v1(t) / (1 - beta1 ^ t)
		// v2_regu(t) = v2(t) / (1 - beta2 ^ t)
		// X(t) = X(t-1) - ita * v1_regu(t) / (sqrt(v2_regu(t)) + epsilon)
	else if (optimizer == "Adam") {
		double beta_one = 0.9, beta_two = 0.99;
		hid_b_vec_one = hid_b_vec_one * beta_one + hid_b_PD * (1 - beta_one);
		hid_b_vec_two = hid_b_vec_two * beta_two + power(hid_b_PD, 2) * (1 - beta_two);
		hid_a_vec_one = hid_a_vec_one * beta_one + hid_a_PD * (1 - beta_one);
		hid_a_vec_two = hid_a_vec_two * beta_two + power(hid_a_PD, 2) * (1 - beta_two);
		inp_vec_one = inp_vec_one * beta_one + inp_PD * (1 - beta_one);
		inp_vec_two = inp_vec_two * beta_two + power(inp_PD, 2) * (1 - beta_two);
		hid_b_weights -= (hid_b_vec_one / (1 - power(beta_one, adjust_time)) * learning_rate) / (
			sqrt(hid_b_vec_two / (1 - power(beta_two, adjust_time))) + epsilon);
		hid_a_weights -= (hid_a_vec_one / (1 - power(beta_one, adjust_time)) * learning_rate) / (
			sqrt(hid_a_vec_two / (1 - power(beta_two, adjust_time))) + epsilon);
		inp_weights -= (inp_vec_one / (1 - power(beta_one, adjust_time)) * learning_rate) / (
			sqrt(inp_vec_two / (1 - power(beta_two, adjust_time))) + epsilon);
	}

}

// Initialize a convolutional neural network
Convolutional_Neural_Network::Convolutional_Neural_Network(int input_number, int input_width, int input_height, int filter_num, int filter_len, int filter_stride,
	int hid_a_length, int hid_b_length, int output_length, string inp_act_func, double inp_learning_rate, string init_method) {

	inp_num = input_number;
	inp_wid = input_width;
	inp_hei = input_height;
	f_num = filter_num;
	f_len = filter_len;
	f_s = filter_stride;
	hid_a_len = hid_a_length;
	hid_b_len = hid_b_length;
	out_len = output_length;

	f_weights = Init_Trans(random::randN<double>(Shape(f_num, inp_num * f_len * f_len)), init_method);
	f_wid = int((inp_wid - f_len) / f_s) + 1;
	f_hei = int((inp_hei - f_len) / f_s) + 1;

	flat_num = f_num * f_wid * f_hei;

	flat_weights = Init_Trans(random::randN<double>(Shape(hid_a_len, flat_num + 1)), init_method);
	for (int i = 0; i < hid_a_len; i++) {
		flat_weights(i, flat_num) = 0;
	}

	hid_a_weights = Init_Trans(random::randN<double>(Shape(hid_b_len, hid_a_len + 1)), init_method);
	for (int i = 0; i < hid_b_len; i++) {
		hid_a_weights(i, hid_a_len) = 0;
	}

	hid_b_weights = Init_Trans(random::randN<double>(Shape(out_len, hid_b_len + 1)), init_method);
	for (int i = 0; i < out_len; i++) {
		hid_b_weights(i, hid_b_len) = 0;
	}

	learning_rate = inp_learning_rate;
	act_func = inp_act_func;
	// bias is fixed to 1, for which the weights for it can be adjusted
	bias = 1.0;
	epsilon = 1e-10;
	adjust_time = 0;
	// vec_one, vec_two, PD: used in the different optimization methods
	f_PD = zeros<double>(Shape(f_num, inp_num * f_len * f_len));
	flat_PD = zeros<double>(Shape(hid_a_len, flat_num + 1));
	hid_a_PD = zeros<double>(Shape(hid_b_len, hid_a_len + 1));
	hid_b_PD = zeros<double>(Shape(out_len, hid_b_len + 1));

	f_vec_one = zeros<double>(Shape(f_num, inp_num * f_len * f_len));
	f_vec_two = zeros<double>(Shape(f_num, inp_num * f_len * f_len));
	flat_vec_one = zeros<double>(Shape(hid_a_len, flat_num + 1));
	flat_vec_two = zeros<double>(Shape(hid_a_len, flat_num + 1));
	hid_a_vec_one = zeros<double>(Shape(hid_b_len, hid_a_len + 1));
	hid_a_vec_two = zeros<double>(Shape(hid_b_len, hid_a_len + 1));
	hid_b_vec_one = zeros<double>(Shape(out_len, hid_b_len + 1));
	hid_b_vec_two = zeros<double>(Shape(out_len, hid_b_len + 1));

}

// Predict tested data
NdArray<double> Convolutional_Neural_Network::Predict(NdArray<double> input_points) {

	int num_predict = input_points.numRows();
	assert(input_points.numCols() == inp_num * inp_wid * inp_hei);

	NdArray<double> y_flat = Convolution(input_points);

	NdArray<double> z_hid_a = dot(hstack({ y_flat, ones<double>(Shape(num_predict, 1)) }), Matrix_T(flat_weights));
	if (act_func == "Sigmoid" || act_func == "Tanh") {
		z_hid_a.putMask(z_hid_a < -100, -100);
	}
	NdArray<double> y_hid_a = Act_Func_Dict(z_hid_a, act_func);

	NdArray<double> z_hid_b = dot(hstack({ y_hid_a, ones<double>(Shape(num_predict, 1)) }), Matrix_T(hid_a_weights));
	if (act_func == "Sigmoid" || act_func == "Tanh") {
		z_hid_b.putMask(z_hid_b < -100, -100);
	}
	NdArray<double> y_hid_b = Act_Func_Dict(z_hid_b, act_func);

	NdArray<double> z_out = dot(hstack({ y_hid_b, ones<double>(Shape(num_predict, 1)) }), Matrix_T(hid_b_weights));
	if (act_func == "Sigmoid" || act_func == "Tanh") {
		z_out.putMask(z_out < -100, 100);
	}
	NdArray<double> y_out = z_out;

	// use softmax to convert y_out into the probabilistic forms
	// avoid invalid calculation in exp
	y_out.putMask(y_out > 100, 100);
	NdArray<double> y_out_exp = exp(y_out);
	double Ohm = sum(y_out_exp, Axis::NONE)(0, 0);
	NdArray<double> Y_out = y_out_exp / Ohm;

	return Y_out;

}

// Destructor of a convolutional neural network
Convolutional_Neural_Network::~Convolutional_Neural_Network() {

}

// Adjust each weight from the training data
void Convolutional_Neural_Network::Adjust(NdArray<double> target_result, NdArray<double> input_points, string optimizer, double dropout) {

	adjust_time++;
	assert(target_result.numRows() == input_points.numRows() && target_result.numCols() == out_len && input_points.numCols() == inp_num * inp_wid * inp_hei);
	int batch_size = target_result.numRows();

	f_PD = 0;
	flat_PD = 0;
	hid_a_PD = 0;
	hid_b_PD = 0;

	NdArray<double> hid_a_mask = random::rand<double>(Shape(1, hid_a_len));
	NdArray<double> hid_b_mask = random::rand<double>(Shape(1, hid_b_len));
	hid_a_mask.putMask(hid_a_mask > dropout, 1.0);
	hid_a_mask.putMask(hid_a_mask != 1.0, 0);
	hid_b_mask.putMask(hid_b_mask > dropout, 1.0);
	hid_b_mask.putMask(hid_b_mask != 1.0, 0);

	for (int row_ind = 0; row_ind < batch_size; row_ind++) {

		NdArray<double> y_flat = Convolution(input_points(row_ind, input_points.cSlice()));

		NdArray<double> z_hid_a = dot(hstack({ y_flat, ones<double>(Shape(1, 1)) }), Matrix_T(flat_weights));
		if (act_func == "Sigmoid" || act_func == "Tanh") {
			z_hid_a.putMask(z_hid_a < -100, -100);
		}
		NdArray<double> y_hid_a = Act_Func_Dict(z_hid_a, act_func);
		// dropout technique to avoid over-fitting
		y_hid_a *= hid_a_mask;

		NdArray<double> z_hid_b = dot(hstack({ y_hid_a, ones<double>(Shape(1, 1)) }), Matrix_T(hid_a_weights));
		if (act_func == "Sigmoid" || act_func == "Tanh") {
			z_hid_b.putMask(z_hid_b < -100, -100);
		}
		NdArray<double> y_hid_b = Act_Func_Dict(z_hid_b, act_func);
		// dropout technique to avoid over-fitting
		y_hid_b *= hid_b_mask;

		NdArray<double> z_out = dot(hstack({ y_hid_b, ones<double>(Shape(1, 1)) }), Matrix_T(hid_b_weights));
		if (act_func == "Sigmoid" || act_func == "Tanh") {
			z_out.putMask(z_out < -100, 100);
		}
		NdArray<double> y_out = z_out;

		// use softmax to convert y_out into the probabilistic forms
		// avoid invalid calculation in exp
		y_out.putMask(y_out > 100, 100);
		NdArray<double> y_out_exp = exp(y_out);
		double Ohm = sum(y_out_exp, Axis::NONE)(0, 0);
		NdArray<double> Y_out = y_out_exp / Ohm;

		// calculate the partial derivative matrix of loss function in input layer and hidden layer
		// loss function: C = -Sigma(Y_i * ln(Y_out_i)), i = 1 - 10)
		NdArray<double> M1 = Y_out - target_result(row_ind, target_result.cSlice());
		NdArray<double> M2 = dot(M1, hid_b_weights(hid_b_weights.rSlice(), Slice(0, hid_b_len)) * Array_Broadcast(hid_b_mask, 0, out_len)) * Act_Func_PD_Dict(z_hid_b, act_func);
		NdArray<double> M3 = dot(M2, hid_a_weights(hid_a_weights.rSlice(), Slice(0, hid_a_len)) * Array_Broadcast(hid_a_mask, 0, hid_b_len) * Array_Broadcast(Matrix_T(hid_b_mask), 1, hid_a_len)) * Act_Func_PD_Dict(z_hid_a, act_func);
		NdArray<double> M4 = dot(M3, flat_weights(flat_weights.rSlice(), Slice(0, flat_num)) * Array_Broadcast(Matrix_T(hid_a_mask), 1, flat_num)).reshape(f_num, f_hei * f_wid);
		NdArray<double> inp_twist = Twist(input_points(row_ind, input_points.cSlice()));

		hid_b_PD += Array_Broadcast(Matrix_T(M1), 1, hid_b_len + 1) * Array_Broadcast(hstack({ y_hid_b * hid_b_mask, ones<double>(Shape(1,1)) * bias }), 0, out_len);
		hid_a_PD += Array_Broadcast(Matrix_T(M2 * hid_b_mask), 1, hid_a_len + 1) * Array_Broadcast(hstack({ y_hid_a * hid_a_mask, ones<double>(Shape(1,1)) * bias }), 0, hid_b_len);
		flat_PD += Array_Broadcast(Matrix_T(M3 * hid_a_mask), 1, flat_num + 1) * Array_Broadcast(hstack({ y_flat, ones<double>(Shape(1,1)) }), 0, hid_a_len);
		f_PD += dot(M4, Matrix_T(inp_twist));

	}

	hid_b_PD /= batch_size;
	hid_a_PD /= batch_size;
	flat_PD /= batch_size;
	f_PD /= batch_size;

	// classic gradient descent :
		// X(t) = X(t - 1) - ita * gradient(X, t - 1)
	if (optimizer == "GD") {
		hid_b_weights -= hid_b_PD * learning_rate;
		hid_a_weights -= hid_a_PD * learning_rate;
		flat_weights -= flat_PD * learning_rate;
		f_weights -= f_PD * learning_rate;
	}

	// Momentum optimizer:
		// v(t) = beta * v(t-1) - ita * gradient(X, t-1)
		// X(t) = X(t-1) + v(t)
	else if (optimizer == "Momentum") {
		double beta = 0.9;
		hid_b_vec_one = hid_b_vec_one * beta - hid_b_PD * learning_rate;
		hid_a_vec_one = hid_a_vec_one * beta - hid_a_PD * learning_rate;
		flat_vec_one = flat_vec_one * beta - flat_PD * learning_rate;
		f_vec_one = f_vec_one * beta - f_PD * learning_rate;
		hid_b_weights += hid_b_vec_one;
		hid_a_weights += hid_a_vec_one;
		flat_weights += flat_vec_one;
		f_weights += f_vec_one;
	}

	// AdaGrad optimizer:
		// v(t) = v(t-1) + gradient(X, t-1) ^ 2
		// X(t) = X(t-1) - ita * gradient(X, t-1) / (sqrt(v(t)) + epsilon)
	else if (optimizer == "AdaGrad") {
		hid_b_vec_one += power(hid_b_PD, 2);
		hid_a_vec_one += power(hid_a_PD, 2);
		flat_vec_one += power(flat_PD, 2);
		f_vec_one += power(f_PD, 2);
		hid_b_weights -= (hid_b_PD / sqrt(hid_b_vec_one + epsilon)) * learning_rate;
		hid_a_weights -= (hid_a_PD / sqrt(hid_a_vec_one + epsilon)) * learning_rate;
		flat_weights -= (flat_PD / sqrt(flat_vec_one + epsilon)) * learning_rate;
		f_weights -= (f_PD / sqrt(f_vec_one + epsilon)) * learning_rate;
	}

	// Adam optimizer:
		// v1(t) = beta1 * v1(t-1) + (1 - beta1) * gradient(X, t-1)
		// v2(t) = beta2 * v2(t-1) + (1 - beta2) * gradient(X, t-1) ^ 2
		// v1_regu(t) = v1(t) / (1 - beta1 ^ t)
		// v2_regu(t) = v2(t) / (1 - beta2 ^ t)
		// X(t) = X(t-1) - ita * v1_regu(t) / (sqrt(v2_regu(t)) + epsilon)
	else if (optimizer == "Adam") {
		double beta_one = 0.9, beta_two = 0.99;
		hid_b_vec_one = hid_b_vec_one * beta_one + hid_b_PD * (1 - beta_one);
		hid_b_vec_two = hid_b_vec_two * beta_two + power(hid_b_PD, 2) * (1 - beta_two);
		hid_a_vec_one = hid_a_vec_one * beta_one + hid_a_PD * (1 - beta_one);
		hid_a_vec_two = hid_a_vec_two * beta_two + power(hid_a_PD, 2) * (1 - beta_two);
		flat_vec_one = flat_vec_one * beta_one + flat_PD * (1 - beta_one);
		flat_vec_two = flat_vec_two * beta_two + power(flat_PD, 2) * (1 - beta_two);
		f_vec_one = f_vec_one * beta_one + f_PD * (1 - beta_one);
		f_vec_two = f_vec_two * beta_two + power(f_PD, 2) * (1 - beta_two);
		hid_b_weights -= (hid_b_vec_one / (1 - power(beta_one, adjust_time)) * learning_rate) / (
			sqrt(hid_b_vec_two / (1 - power(beta_two, adjust_time))) + epsilon);
		hid_a_weights -= (hid_a_vec_one / (1 - power(beta_one, adjust_time)) * learning_rate) / (
			sqrt(hid_a_vec_two / (1 - power(beta_two, adjust_time))) + epsilon);
		flat_weights -= (flat_vec_one / (1 - power(beta_one, adjust_time)) * learning_rate) / (
			sqrt(flat_vec_two / (1 - power(beta_two, adjust_time))) + epsilon);
		f_weights -= (f_vec_one / (1 - power(beta_one, adjust_time)) * learning_rate) / (
			sqrt(f_vec_two / (1 - power(beta_two, adjust_time))) + epsilon);
	}

}

// construct the convolution function to extract the image features
NdArray<double> Convolutional_Neural_Network::Convolution(NdArray<double> input_image) {

	int num_predict = input_image.numRows();
	int ind_f_num, ind_f_wid, ind_f_hei, ind_inp_wid_beg, ind_inp_hei_beg, ind_image_beg;
	assert(input_image.numCols() == inp_num * inp_wid * inp_hei);

	NdArray<double> out_matrix = zeros<double>(Shape(num_predict, flat_num)), out_matrix_slice;

	// Only two-dimension arrays are permitted in NumCpp, so a new method is created
	for (int i = 0; i < flat_num; i++) {

		// flat_num = f_num * f_wid * f_hei
		// inp_image: [num_predict, inp_num, inp_wid, inp_hei]
		// filter: [f_num, inp_num, f_len, f_len]

		ind_f_num = int(i / (f_wid * f_hei));
		ind_f_wid = int((i - ind_f_num * f_wid * f_hei) / f_hei);
		ind_f_hei = i % f_hei;

		ind_inp_wid_beg = ind_f_wid * f_s;
		ind_inp_hei_beg = ind_f_hei * f_s;

		// construct a flat matrix reflecting inp_image matrix: [:, :, ind_inp_wid_beg: ind_inp_wid_beg + f_len, ind_inp_hei_beg: ind_inp_hei_beg + f_len]
		// then multiply it with filter[ind_f_num, :, :, :]
		NdArray<double> input_image_selected = zeros<double>(Shape(num_predict, 0));
		for (int j = 0; j < inp_num; j++) {
			for (int k = 0; k < f_len; k++) {

				ind_image_beg = j * inp_wid * inp_hei + (ind_inp_wid_beg + k) * inp_hei + ind_inp_hei_beg;
				input_image_selected = hstack({input_image_selected, input_image(input_image.rSlice(), Slice(ind_image_beg, ind_image_beg + f_len))});

			}
		}

		out_matrix_slice = sum(input_image_selected * Array_Broadcast(f_weights(ind_f_num, f_weights.cSlice()), 0, num_predict), Axis::COL);

		for (int j = 0; j < num_predict; j++) {
			out_matrix(j, i) = out_matrix_slice(0, j);
		}

	}

	return out_matrix;

}

// the twist function to be used in weights adjustments
NdArray<double> Convolutional_Neural_Network::Twist(NdArray<double> input_image) {

	assert(input_image.numRows() == 1 && input_image.numCols() == inp_num * inp_hei * inp_wid);

	NdArray<double> twist_image = zeros<double>(inp_num * f_len * f_len, f_hei * f_wid);
	int ind_inp_num, ind_f_len_hei, ind_f_len_wid, ind_f_hei, ind_f_wid;

	for (int i = 0; i < inp_num * f_len * f_len; i++) {
		for (int j = 0; j < f_hei * f_wid; j++) {

			ind_inp_num = int(i / (f_len * f_len));
			ind_f_len_hei = int((i - ind_inp_num * f_len * f_len) / f_len);
			ind_f_len_wid = i % f_len;
			ind_f_hei = int(j / f_wid);
			ind_f_wid = j % f_wid;

			twist_image(i, j) = input_image(0, ind_inp_num * inp_hei * inp_wid + 
				(ind_f_hei * f_s + ind_f_len_hei) * inp_wid + ind_f_wid * f_s + ind_f_len_wid);

		}
	}

	return twist_image;

}

// Broadcast one-dimension array into two-dimension space. Python's Numpy already includes it.
NdArray<double> Array_Broadcast(NdArray<double> inp_array, int broad_axis, int broad_length) {
	
	NdArray<double> out_array;
	if (broad_axis == 0) {
		assert(inp_array.numRows() == 1);
		out_array = inp_array;
		for (int i = 0; i < broad_length - 1; i++) {
			out_array = vstack({ out_array, inp_array });
		}
	}
	else {
		assert(inp_array.numCols() == 1);
		out_array = inp_array;
		for (int i = 0; i < broad_length - 1; i++) {
			out_array = hstack({ out_array, inp_array });
		}
	}

	return out_array;

}

// Convert the binary Y_out to decimal output
NdArray<int> Bin_to_Dec(NdArray<double> Y_out_bin) {

	int num_out = Y_out_bin.numRows();
	NdArray<int> Y_out = zeros<int>(Shape(num_out, 1));
	for (int i = 0; i < num_out; i++) {
		Y_out(i, 0) = argmax(Y_out_bin(i, Y_out_bin.cSlice()))(0, 0);
	}
	return Y_out;

}

// different initialization methods
	// Tiny: normal distribution multiplied by a small number
	// Xavier: normal distribution divided by sqrt(last dimension). Best for Sigmoid and Tanh
	// He: normal distribution divided by sqrt(last dimension / 2).Best for ReLu and ReLu_Leak
NdArray<double> Init_Trans(NdArray<double> inp_matrix, string init_method) {

	int inp_col = inp_matrix.numCols() - 1;
	double weight_multiplier = 0.01;
	if (init_method == "Xavier") {
		weight_multiplier = sqrt(1.0 / inp_col);
	}
	else if (init_method == "He") {
		weight_multiplier = sqrt(2.0 / inp_col);
	}
	return inp_matrix * weight_multiplier;

}

// Activation functions and according partial difference functions

// Sigmoid:
	// Pros: range from 0 - 1, always positive derivative, good classifier
	// Cons: if x significantly different from zero, then the optimization progress is slow even to halt.
NdArray<double> Sigmoid(NdArray<double> x) {
	return power((exp(-x) + 1.0), -1);
}
NdArray<double> Sigmoid_PD(NdArray<double> x) {
	return exp(-x) / power((exp(-x) + 1), 2);
}

// Tanh:
	// Pros: range from 0 - 1, always positive derivative, good classifier, and derivative larger than Sigmoid
	// Cons: same as Sigmoid
NdArray<double> Tanh(NdArray<double> x) {
	return power((exp(-x * 2.0) + 1.0), -1) * 2.0;
}
NdArray<double> Tanh_PD(NdArray<double> x) {
	return (exp(-x * 2.0) * 4.0) / power((exp(-x * 2.0) + 1.0), 2);
}

// ReLu:
	// Pros: not limited to 0 - 1, always positive derivative, good regressor and classifier. Easy to calculate
	// Cons: easier to diverge than Sigmoid and Tanh. Neurons smaller than zeros will be dead forever
NdArray<double> ReLu(NdArray<double> x) {
	return (x > 0).astype<double>() * x;
}
NdArray<double> ReLu_PD(NdArray<double> x) {
	return (x > 0).astype<double>();
}

// ReLu_Leak:
	// Pros: same as Relu. And no neurons will be dead
	// Cons: easier to diverge than Sigmoid and Tanh.
	// Special property: larger the difference between the slopes of >0 and <=0, more quickly the network learns.
NdArray<double> ReLu_Leak(NdArray<double> x) {
	return x - (x < 0).astype<double>() * x * 0.5;
}
NdArray<double> ReLu_Leak_PD(NdArray<double> x) {
	return -(x < 0).astype<double>() * 0.5 + 1;
}

// dictionaries for activation functions and partial difference functions
NdArray<double> Act_Func_Dict(NdArray<double> x, string act_func) {

	if (act_func == "Sigmoid") {
		return Sigmoid(x);
	}
	else if (act_func == "Tanh") {
		return Tanh(x);
	}
	else if (act_func == "ReLu") {
		return ReLu(x);
	}
	return ReLu_Leak(x);

}

NdArray<double> Act_Func_PD_Dict(NdArray<double> x, string act_func) {

	if (act_func == "Sigmoid") {
		return Sigmoid_PD(x);
	}
	else if (act_func == "Tanh") {
		return Tanh_PD(x);
	}
	else if (act_func == "ReLu") {
		return ReLu_PD(x);
	}
	return ReLu_Leak_PD(x);

}