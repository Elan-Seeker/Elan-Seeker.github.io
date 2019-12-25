#ifndef NEURAL_NETWORKS_H
#define NEURAL_NETWORKS_H

#include <NumCpp.hpp>
#include <assert.h>
#include "base.h"

using namespace std;

class Neural_Network {

public:

	Neural_Network(int input_length, int hidden_length, int output_length, string inp_act_func = "Sigmoid", 
		double inp_learning_rate = 0.001, string init_method = "Xavier");
	~Neural_Network();
	NdArray<double> Predict(NdArray<double> input_points);
	void Adjust(NdArray<double> target_result, NdArray<double> input_points, string optimizer = "GD", double dropout = 0.3);

private:

	NdArray<double> inp_weights, hid_weights, inp_vec_one, inp_vec_two, hid_vec_one, hid_vec_two, hid_PD, inp_PD;
	double learning_rate, bias, epsilon;
	int adjust_time, inp_len, hid_len, out_len;
	string act_func;

};

class Deep_Neural_Network {

public:

	Deep_Neural_Network(int input_length, int hid_a_length, int hid_b_length, int output_length, 
		string inp_act_func = "Sigmoid", double inp_learning_rate = 0.001, string init_method = "Xavier");
	~Deep_Neural_Network();
	NdArray<double> Predict(NdArray<double> input_points);
	void Adjust(NdArray<double> target_result, NdArray<double> input_points, string optimizer = "GD", double dropout = 0.3);

private:

	NdArray<double> inp_weights, hid_a_weights, hid_b_weights, inp_PD, hid_a_PD, hid_b_PD;
	NdArray<double> inp_vec_one, inp_vec_two, hid_a_vec_one, hid_a_vec_two, hid_b_vec_one, hid_b_vec_two;
	double learning_rate, bias, epsilon;
	int adjust_time, inp_len, hid_a_len, hid_b_len, out_len;
	string act_func;

};

class Convolutional_Neural_Network {

public:

	Convolutional_Neural_Network(int input_number, int input_width, int input_height, int filter_num, int filter_len, int filter_stride, 
		int hid_a_length, int hid_b_length, int output_length, string inp_act_func = "Sigmoid", double inp_learning_rate = 0.001, string init_method = "Xavier");
	~Convolutional_Neural_Network();
	NdArray<double> Predict(NdArray<double> input_points);
	void Adjust(NdArray<double> target_result, NdArray<double> input_points, string optimizer = "GD", double dropout = 0.3);
	NdArray<double> Convolution(NdArray<double> input_image);
	NdArray<double> Twist(NdArray<double> input_image);

private:

	NdArray<double> f_weights, flat_weights, hid_a_weights, hid_b_weights, f_PD, flat_PD, hid_a_PD, hid_b_PD;
	NdArray<double> f_vec_one, f_vec_two, flat_vec_one, flat_vec_two, hid_a_vec_one, hid_a_vec_two, hid_b_vec_one, hid_b_vec_two;
	double learning_rate, bias, epsilon;
	int adjust_time, inp_num, inp_wid, inp_hei,f_num, f_len, f_s, hid_a_len, hid_b_len, out_len, f_wid, f_hei, flat_num;
	string act_func;

};

// Broadcast one-dimension array into two-dimension space
NdArray<double> Array_Broadcast(NdArray<double> inp_array, int broad_axis, int broad_length);

// Convert the binary Y_out to decimal output
NdArray<int> Bin_to_Dec(NdArray<double> Y_out_bin);

// different initialization methods
	// Tiny: normal distribution multiplied by a small number
	// Xavier: normal distribution divided by sqrt(last dimension). Best for Sigmoid and Tanh
	// He: normal distribution divided by sqrt(last dimension / 2).Best for ReLu and ReLu_Leak
NdArray<double> Init_Trans(NdArray<double> inp_matrix, string init_method);

// Activation functions and according partial difference functions
NdArray<double> Sigmoid(NdArray<double> x);
NdArray<double> Sigmoid_PD(NdArray<double> x);
NdArray<double> Tanh(NdArray<double> x);
NdArray<double> Tanh_PD(NdArray<double> x);
NdArray<double> ReLu(NdArray<double> x);
NdArray<double> ReLu_PD(NdArray<double> x);
NdArray<double> ReLu_Leak(NdArray<double> x);
NdArray<double> ReLu_Leak_PD(NdArray<double> x);

// dictionaries for activation functions and partial difference functions
NdArray<double> Act_Func_Dict(NdArray<double> x, string act_func);
NdArray<double> Act_Func_PD_Dict(NdArray<double> x, string act_func);

#endif