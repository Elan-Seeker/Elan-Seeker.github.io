#include <iostream>
#include <fstream>
#include <sstream>
#include <NumCpp.hpp>
#include "Neural_Networks.h"

using namespace std;
using namespace nc;

int main() {

	cout << "Program begins." << endl;

	int train_num = 2500, watershed = 2000, inp_len = 196, out_len = 10;
	NdArray<double> X_train = zeros<double>(Shape(train_num, inp_len)), Y_train = zeros<double>(Shape(train_num, out_len));

	// load train_sample.csv
	ifstream csv_reader("train_sample_small.csv");
	string line, pixel;
	// ignore the first line as it's the index line
	getline(csv_reader, line);
	for (int i = 0; i < train_num; i++) {
		// load the ith line into the line string
		getline(csv_reader, line);
		istringstream str_in(line);
		// load data of the first column into Y_train
		getline(str_in, pixel, ',');
		// ont-hot encoder
		Y_train(i, stoi(pixel)) = 1;
		for (int j = 0; j < inp_len; j++) {
			getline(str_in, pixel, ',');
			X_train(i, j) = stod(pixel);
		}
	}

	cout << "Loading process finished." << endl;

	// separate train data into cross data and validation data
	NdArray<double> X_cross = X_train(Slice(0, watershed), X_train.cSlice());
	NdArray<double> X_valid = X_train(Slice(watershed, train_num), X_train.cSlice());
	NdArray<double> Y_cross = Y_train(Slice(0, watershed), Y_train.cSlice());
	NdArray<double> Y_valid = Y_train(Slice(watershed, train_num), Y_train.cSlice());

	// construct the neural networks
	Neural_Network NN = Neural_Network(inp_len, 200, out_len, "ReLu", 0.002, "He");
	Deep_Neural_Network DNN = Deep_Neural_Network(inp_len, 300, 100, out_len, "ReLu", 0.002, "He");
	Convolutional_Neural_Network CNN = Convolutional_Neural_Network(1, 14, 14, 16, 3, 2, 200, 100, out_len, "ReLu", 0.002, "He");
	cout << "Neural Networks are constructed, then the training begins." << endl;

	int num_cross = X_cross.numRows(), num_valid = X_valid.numRows();
	int num_accurate_NN, num_accurate_DNN, num_accurate_CNN;
	
	// batch training:
		// stochastic training: batch_size == 1;
		// mini-batch training: batch_size ~ (1, num_cross);
		// full-batch training: batch_size == num_cross;
	int epoch_max = 100, batch_size = 10, adjust_time_max = int(num_cross * epoch_max / batch_size);
	int batch_pos_beg, batch_pos_end;

	NdArray<int> Y_valid_pred_NN, Y_valid_pred_DNN, Y_valid_pred_CNN;

	for (int i = 0; i < adjust_time_max; i++) {
		
		if (i % 5 == 0) {

			num_accurate_NN = 0;
			num_accurate_DNN = 0;
			num_accurate_CNN = 0;
			// use the neural networks to predict valid data
			Y_valid_pred_NN = Bin_to_Dec(NN.Predict(X_valid));
			Y_valid_pred_DNN = Bin_to_Dec(DNN.Predict(X_valid));
			Y_valid_pred_CNN = Bin_to_Dec(CNN.Predict(X_valid));

			// calculate the accuracy rate of Y_valid_pred
			for (int j = 0; j < num_valid; j++) {
				if (Y_valid(j, Y_valid_pred_NN(j, 0)) == 1.0) {
					num_accurate_NN++;
				}
				if (Y_valid(j, Y_valid_pred_DNN(j, 0)) == 1.0) {
					num_accurate_DNN++;
				}
				if (Y_valid(j, Y_valid_pred_CNN(j, 0)) == 1.0) {
					num_accurate_CNN++;
				}
			}

			cout << "After " << i << " times of adjustments:" << endl;
			cout << "The accuracy of NN prediction is " << double(num_accurate_NN) / num_valid << endl;
			cout << "The accuracy of DNN prediction is " << double(num_accurate_DNN) / num_valid << endl;
			cout << "The accuracy of CNN prediction is " << double(num_accurate_CNN) / num_valid << endl;

		}

		// use train data to adjust the weights in neural networks
		batch_pos_beg = (i * batch_size) % num_cross;
		batch_pos_end = batch_pos_beg + batch_size;
		NN.Adjust(Y_cross(Slice(batch_pos_beg, batch_pos_end), Y_cross.cSlice()), X_cross(Slice(batch_pos_beg, batch_pos_end), X_cross.cSlice()), "Adam", 0.3);
		DNN.Adjust(Y_cross(Slice(batch_pos_beg, batch_pos_end), Y_cross.cSlice()), X_cross(Slice(batch_pos_beg, batch_pos_end), X_cross.cSlice()), "Adam", 0.3);
		CNN.Adjust(Y_cross(Slice(batch_pos_beg, batch_pos_end), Y_cross.cSlice()), X_cross(Slice(batch_pos_beg, batch_pos_end), X_cross.cSlice()), "Adam", 0.3);

		cout << i + 1 << " times of adjustments." << endl;

	}

	system("pause");

}