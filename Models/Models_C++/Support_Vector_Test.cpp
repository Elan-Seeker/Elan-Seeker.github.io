#include <iostream>
#include <fstream>
#include <sstream>
#include <NumCpp.hpp>
#include "Support_Vector.h"

using namespace std;
using namespace nc;

int main() {

	cout << "Program begins." << endl;

	int train_num = 2500, watershed = 2000, inp_len = 196, out_len = 10;
	NdArray<double> X_train = zeros<double>(Shape(train_num, inp_len)), Y_train = ones<double>(Shape(train_num, out_len)) * -1;

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

	// construct the support vector classifier and load the data to be trained
	Support_Vector_Classifier SVC_Linear = Support_Vector_Classifier(inp_len, out_len, X_cross, Y_cross, "linear");
	Support_Vector_Classifier SVC_Poly = Support_Vector_Classifier(inp_len, out_len, X_cross, Y_cross, "poly");
	Support_Vector_Classifier SVC_Gauss = Support_Vector_Classifier(inp_len, out_len, X_cross, Y_cross, "gauss");

	cout << "SVCs are constructed, then the training begins." << endl;

	int epoch_max = 5, epoch_trained = 0, num_cross = X_cross.numRows(), num_valid = X_valid.numRows();
	int num_accurate_linear, num_accurate_polyno, num_accurate_gaussi;

	NdArray<int> Y_valid_pred_linear, Y_valid_pred_polyno, Y_valid_pred_gaussi;

	for (int i = 0; i < epoch_max; i++) {

		num_accurate_linear = 0;
		num_accurate_polyno = 0;
		num_accurate_gaussi = 0;
		// use the SVC to predict valid data
		Y_valid_pred_linear = SVC_Linear.Predict(X_valid);
		Y_valid_pred_polyno = SVC_Poly.Predict(X_valid);
		Y_valid_pred_gaussi = SVC_Gauss.Predict(X_valid);
		// calculate the accuracy rate of Y_valid_pred
		for (int j = 0; j < num_valid; j++) {
			if (Y_valid(j, Y_valid_pred_linear(j, 0)) == 1.0) {
				num_accurate_linear++;
			}
			if (Y_valid(j, Y_valid_pred_polyno(j, 0)) == 1.0) {
				num_accurate_polyno++;
			}
			if (Y_valid(j, Y_valid_pred_gaussi(j, 0)) == 1.0) {
				num_accurate_gaussi++;
			}
		}

		cout << "After " << i << " epochs of training:" << endl;
		cout << "The accuracy of SVC_Linear prediction is " << double(num_accurate_linear) / num_valid << endl;
		cout << "The accuracy of SVC_Poly prediction is " << double(num_accurate_polyno) / num_valid << endl;
		cout << "The accuracy of SVC_Gauss prediction is " << double(num_accurate_gaussi) / num_valid << endl;

		// use train data to adjust the lambda weights in SVC
		SVC_Linear.Learn();
		SVC_Poly.Learn();
		SVC_Gauss.Learn();

	}

	system("pause");

}