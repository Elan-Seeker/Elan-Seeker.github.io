#include <iostream>
#include "Base.h"
#include "Time_Series.h"

using namespace std;

const double PI = 3.14159265358979323846;

void Test_Linear_Regression() {

	int n_y = 50;
	NdArray<double> X = random::randN<double>(Shape(n_y, 4));
	NdArray<double> y = X(X.rSlice(), 0) * 3.5 + X(X.rSlice(), 1) * 0.29 - X(X.rSlice(), 2) + X(X.rSlice(), 3) * 10;
	// add noises into y
	y += random::randN<double>(Shape(n_y, 1)) * 0.05;

	Linear_Regression LR = Linear_Regression(X, y);
	NdArray<double> y_pred = LR.Predict(X);
	double sds = LR.SDS();

	cout << "y_real:" << endl;
	cout << y << endl;
	cout << "y_pred:" << endl;
	cout << y_pred << endl;
	cout << "standard deviation of the sample: " << sds << endl;

}

void Test_AR() {

	int len_ts = 50;
	NdArray<double> ts = arange<double>(0, len_ts) * 10;
	// add noises into ts
	ts += random::randN<double>(Shape(1, len_ts)) * 0.05;

	double sds = 0, y_pred = 0;
	tie(sds, y_pred) = AR(ts(0, Slice(0, len_ts - 1)), 5);

	cout << "y_real: " << ts(0, len_ts - 1) << endl;
	cout << "y_pred: " << y_pred << endl;
	cout << "standard error of sample: " << sds << endl;

}

void Test_ARMA() {

	int len_ts = 50;
	NdArray<double> ts = arange<double>(0, len_ts) * 10;
	// add noises into ts
	ts += random::randN<double>(Shape(1, len_ts)) * 0.05;

	double sds, y_pred;
	tie(sds, y_pred) = ARMA(ts(0, Slice(0, len_ts - 1)), 5, 5);

	cout << "y_real: " << ts(0, len_ts - 1) << endl;
	cout << "y_pred: " << y_pred << endl;
	cout << "standard error of sample: " << sds << endl;

}

void Test_ARIMA() {

	int len_ts = 50;
	NdArray<double> ts = arange<double>(0, len_ts) * 10;
	// add noises into ts
	ts += random::randN<double>(Shape(1, len_ts)) * 0.05;

	double sds, y_pred;
	tie(sds, y_pred) = ARIMA(ts(0, Slice(0, len_ts - 1)), 5, 2, 5);

	cout << "y_real: " << ts(0, len_ts - 1) << endl;
	cout << "y_pred: " << y_pred << endl;
	cout << "standard error of sample: " << sds << endl;

}

void Test_SARMA() {

	int len_ts = 200, m = 12;
	NdArray<double> ts = arange<double>(0, len_ts) * 10;
	// add noises into ts
	ts += random::randN<double>(Shape(1, len_ts)) * 0.05;
	// add seasonal factors into ts
	for (int i = 0; i >= len_ts; i++) {
		ts(0, i) += sin<double>(PI * i / m) * 10;
	}

	double sds, y_pred;
	tie(sds, y_pred) = SARMA(ts(0, Slice(0, len_ts - 1)), 5, 5, 1, 1, m);

	cout << "y_real: " << ts(0, len_ts - 1) << endl;
	cout << "y_pred: " << y_pred << endl;
	cout << "standard error of sample: " << sds << endl;

}

void Test_SARIMA() {

	int len_ts = 200, m = 12;
	NdArray<double> ts = arange<double>(0, len_ts) * 10;
	// add noises into ts
	ts += random::randN<double>(Shape(1, len_ts)) * 0.05;
	// add seasonal factors into ts
	for (int i = 0; i >= len_ts; i++) {
		ts(0, i) += sin<double>(PI * i / m) * 10;
	}

	double sds, y_pred;
	tie(sds, y_pred) = SARIMA(ts(0, Slice(0, len_ts - 1)), 5, 2, 5, 1, 1, 1, m);

	cout << "y_real: " << ts(0, len_ts - 1) << endl;
	cout << "y_pred: " << y_pred << endl;
	cout << "standard error of sample: " << sds << endl;

}

int main() {
	
	Test_Linear_Regression();
	Test_AR();
	Test_ARMA();
	Test_ARIMA();
	Test_SARMA();
	Test_SARIMA();

	system("pause");
}