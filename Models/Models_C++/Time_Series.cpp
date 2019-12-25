#include "Time_Series.h"
#include "Base.h"

// auto regression prediction: y(t, p) = C + omega_1 * y(t-1) + omega_2 * y(t-2) + ... + omega_p * y(t-p)
tuple<double, double> AR(NdArray<double> ts, int p) {

	int len_ts = ts.numCols();
	assert((len_ts - p >= 2) && (ts.numRows() == 1));

	NdArray<double> X_ts = zeros<double>(Shape(len_ts - p, p));
	NdArray<double> y_ts = zeros<double>(Shape(len_ts - p, 1));

	for (int i = 0; i < len_ts - p; i++) {
		for (int j = 0; j < p; j++) {
			X_ts(i, j) = ts(0, i + j);
		}
		y_ts(i, 0) = ts(0, i + p);
	}

	Linear_Regression AR_model = Linear_Regression(X_ts, y_ts);

	double y_pred = AR_model.Predict(ts(0, Slice(len_ts - p, len_ts)))(0, 0);

	// return the standard error and predicted value
	return make_tuple(AR_model.SDS(), y_pred);

}

// moving average prediction: y(t, p, q) = AR(t, p) + alpha_1 * err(t-1) + alpha_2 * err(t-2) + ... + alpha_q * err(t-q)
tuple<double, double> ARMA(NdArray<double> ts, int p, int q) {
	
	int len_ts = ts.numCols();
	assert((len_ts - p - q >= 2) && (ts.numRows() == 1));

	NdArray<double> X_ts = zeros<double>(Shape(len_ts - p, p));
	NdArray<double> y_ts = zeros<double>(Shape(len_ts - p, 1));

	for (int i = 0; i < len_ts - p; i++) {
		for (int j = 0; j < p; j++) {
			X_ts(i, j) = ts(0, i + j);
		}
		y_ts(i, 0) = ts(0, i + p);
	}

	Linear_Regression AR_model = Linear_Regression(X_ts, y_ts);

	double y_pred = AR_model.Predict(ts(0, Slice(len_ts - p, len_ts)))(0, 0);

	if (q > 0) {

		NdArray<double> y_ts_pred = AR_model.Predict(X_ts);
		NdArray<double> err_ts = Matrix_T(y_ts - y_ts_pred);

		NdArray<double> err_x_ts = zeros<double>(Shape(len_ts - p - q, q));
		NdArray<double> err_y_ts = zeros<double>(Shape(len_ts - p - q, 1));

		for (int i = 0; i < len_ts - p - q; i++) {
			for (int j = 0; j < q; j++) {
				err_x_ts(i, j) = err_ts(0, i + j);
			}
			err_y_ts(i, 0) = err_ts(0, i + q);
		}

		Linear_Regression MA_model = Linear_Regression(err_x_ts, err_y_ts, 0);

		y_pred += MA_model.Predict(err_ts(0, Slice(len_ts - p - q, len_ts - p)))(0, 0);

	}

	// return the standard error and predicted value
	return make_tuple(AR_model.SDS(), y_pred);

}

// ARIMA prediction: y(t, p, 1, q) = y(t-1) + ARMA(t, p, q)
tuple<double, double> ARIMA(NdArray<double> ts, int p, int d, int q) {
	
	int len_ts = ts.numCols();
	assert((len_ts - p - d - q >= 2) && (ts.numRows() == 1));

	double y_pred_add = 0;

	NdArray<double> ts_diff_temp = ts.copy();

	for (int i = 0; i < d; i++) {
		y_pred_add += ts_diff_temp(0, len_ts - i - 1);
		ts_diff_temp = diff(ts_diff_temp);
	}

	double sds, y_pred;
	tie(sds, y_pred) = ARMA(ts_diff_temp, p, q);

	y_pred += y_pred_add;

	// return the standard error and predicted value
	return make_tuple(sds, y_pred);

}

// SARMA prediction
tuple<double, double> SARMA(NdArray<double> ts, int p, int q, int P, int Q, int m) {
	
	int len_ts = ts.numCols();
	assert((len_ts - p - q - m * (P + Q) >= 2) && (ts.numRows() == 1));
	assert(p + q < m);

	NdArray<double> X_ts = zeros<double>(Shape(len_ts - p - m * P, p + P * (p + 1)));
	NdArray<double> y_ts = zeros<double>(Shape(len_ts - p - m * P, 1));

	for (int i = 0; i < len_ts - p - m * P; i++) {
		for (int j = 0; j < p; j++) {
			X_ts(i, P * (p + 1) + j) = ts(0, i + m * P + j);
		}
		y_ts(i, 0) = ts(0, i + m * P + p);
		for (int j = 0; j < P; j++) {
			for (int k = 0; k < p + 1; k++) {
				X_ts(i, j * (p + 1) + k) = ts(0, i + j * m + k);
			}
		}
	}

	Linear_Regression AR_model = Linear_Regression(X_ts, y_ts);

	NdArray<double> ts_test = zeros<double>(Shape(1, p + P * (p + 1)));
	for (int i = 0; i < p; i++) {
		ts_test(0, P * (p + 1) + i) = ts(0, len_ts - p + i);
	}
	for (int i = 0; i < P; i++) {
		for (int j = 0; j < p + 1; j++) {
			ts_test(0, i * (p + 1) + j) = ts(0, len_ts - m * (P - i) - p + j);
		}
	}

	double y_pred = AR_model.Predict(ts_test)(0, 0);

	if (q > 0) {

		NdArray<double> y_ts_pred = AR_model.Predict(X_ts);
		NdArray<double> err_ts = Matrix_T(y_ts - y_ts_pred);

		int len_err_ts = err_ts.numCols();

		NdArray<double> err_x_ts = zeros<double>(Shape(len_err_ts - q - m * Q, q + Q * (q + 1)));
		NdArray<double> err_y_ts = zeros<double>(Shape(len_err_ts - q - m * Q, 1));

		for (int i = 0; i < len_err_ts - q - m * Q; i++) {
			for (int j = 0; j < q; j++) {
				err_x_ts(i, Q * (q + 1) + j) = err_ts(0, i + m * Q + j);
			}
			err_y_ts(i, 0) = err_ts(0, i + m * Q + q);
			for (int j = 0; j < Q; j++) {
				for (int k = 0; k < q + 1; k++) {
					err_x_ts(i, j * (q + 1) + k) = err_ts(0, i + j * m + k);
				}
			}
		}

		Linear_Regression MA_model = Linear_Regression(err_x_ts, err_y_ts, 0);

		NdArray<double> err_ts_test = zeros<double>(Shape(1, q + Q * (q + 1)));
		for (int i = 0; i < q; i++) {
			err_ts_test(0, Q * (q + 1) + i) = err_ts(0, len_err_ts - q + i);
		}
		for (int i = 0; i < Q; i++) {
			for (int j = 0; j < q + 1; j++) {
				err_ts_test(0, i * (q + 1) + j) = err_ts(0, len_err_ts - m * (Q - i) - q + j);
			}
		}

		y_pred += MA_model.Predict(err_ts_test)(0, 0);
	}

	// return the standard error and predicted value
	return make_tuple(AR_model.SDS(), y_pred);

}

// SARIMA prediction
tuple<double, double> SARIMA(NdArray<double> ts, int p, int d, int q, int P, int D, int Q, int m) {
	
	int len_ts = ts.numCols(), len_diff_temp = 0;
	assert((len_ts - p - d - q - m * (P + D + Q) >= 2) && (ts.numRows() == 1));

	double y_pred_add = 0;

	NdArray<double> ts_diff_temp = ts.copy();

	for (int i = 0; i < d; i++) {
		y_pred_add += ts_diff_temp(0, len_ts - i - 1);
		ts_diff_temp = diff(ts_diff_temp);
	}
	for (int i = 0; i < D; i++) {
		len_diff_temp = ts_diff_temp.numCols();
		y_pred_add += ts_diff_temp(0, len_diff_temp - 1);
		ts_diff_temp = ts_diff_temp(0, Slice(m, len_diff_temp)) - ts_diff_temp(0, Slice(0, len_diff_temp - m));
	}

	double sds, y_pred;
	tie(sds, y_pred) = SARMA(ts_diff_temp, p, q, P, Q, m);

	y_pred += y_pred_add;

	// return the standard error and predicted value
	return make_tuple(sds, y_pred);

}