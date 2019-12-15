import numpy as np
from Optimization import Linear_Regression

# basic method: Moving average of a time series with a parameter
def Moving_Avg(Ts_Matrix, horizon = 5):

	Ts_shape = Ts_Matrix.shape
	Ts_dim = np.size(Ts_shape)

	assert Ts_shape[-1] >= horizon, 'invalid horizon.'
	assert Ts_dim >= 2 and Ts_dim <= 5, 'invalid Ts_Matrix.'

	if Ts_dim == 2:
		Ts_chosen = Ts_Matrix[:, -horizon:]
	elif Ts_dim == 3:
		Ts_chosen = Ts_Matrix[:, :, -horizon:]
	elif Ts_dim == 4:
		Ts_chosen = Ts_Matrix[:, :, :, -horizon:]
	else:
		Ts_chosen = Ts_Matrix[:, :, :, :, -horizon:]

	Ts_avg = np.mean(Ts_chosen, axis=-1)

	return Ts_avg

# basic method: Exponential Smooth average of a time series with a parameter
def Exp_Smooth_Avg(Ts_Matrix, alpha = 0.6):

	Ts_shape = Ts_Matrix.shape
	Ts_dim = np.size(Ts_shape)

	assert Ts_dim >= 2 and Ts_dim <= 5, 'invalid Ts_Matrix.'

	Return_shape = Ts_shape[:-1]

	Time_horizon = Ts_shape[-1]
	Ts_avg = np.zeros(shape=Return_shape, dtype=np.float)

	for i in range(Time_horizon):
		Ts_avg = (1 - alpha) * Ts_avg
		if Ts_dim == 2:
			Ts_avg += alpha * Ts_Matrix[:, i]
		elif Ts_dim == 3:
			Ts_avg += alpha * Ts_Matrix[:, :, i]
		elif Ts_dim == 4:
			Ts_avg += alpha * Ts_Matrix[:, :, :, i]
		else:
			Ts_avg += alpha * Ts_Matrix[:, :, :, :, i]

	return Ts_avg

# auto regression prediction: y(t, p) = C + omega_1 * y(t-1) + omega_2 * y(t-2) + ... + omega_p * y(t-p)
def AR(ts, p):

	len_ts = np.size(ts)
	# ensure the number of slices exceeds the least required number
	assert len_ts - p >= 2, 'invalid time series.'

	X_ts = np.zeros((len_ts - p, p))
	y_ts = np.zeros(len_ts - p)

	for i in range(len_ts - p):
		X_ts[i] = ts[i: i + p]
		y_ts[i] = ts[i + p]

	AR_model = Linear_Regression(X=X_ts, y=y_ts)

	# return the standard error and predicted value
	y_pred = AR_model.predict(X_test=ts[-p:])
	return AR_model.SDS(), y_pred

# moving average prediction: y(t, p, q) = AR(t, p) + alpha_1 * err(t-1) + alpha_2 * err(t-2) + ... + alpha_q * err(t-q)
def ARMA(ts, p, q):

	len_ts = np.size(ts)
	# ensure the number of slices exceeds the least required number
	assert len_ts - p - q >= 2, 'invalid time series.'

	X_ts = np.zeros((len_ts - p, p))
	y_ts = np.zeros(len_ts - p)

	for i in range(len_ts - p):
		X_ts[i] = ts[i: i + p]
		y_ts[i] = ts[i + p]

	AR_model = Linear_Regression(X=X_ts, y=y_ts)

	y_pred = AR_model.predict(X_test=ts[-p:])

	if q > 0:

		y_ts_pred = AR_model.predict(X_test=X_ts)
		err_ts = y_ts - y_ts_pred

		err_x_ts = np.zeros((len_ts - p - q, q))
		err_y_ts = np.zeros(len_ts - p - q)

		for i in range(len_ts - p - q):
			err_x_ts[i] = err_ts[i: i + q]
			err_y_ts[i] = err_ts[i + q]

		MA_model = Linear_Regression(X=err_x_ts, y=err_y_ts, bias=0)

		y_pred += MA_model.predict(X_test=err_ts[-q:])

	return AR_model.SDS(), y_pred

# moving average prediction: y(t, p, 1, q) = y(t-1) + ARMA(t, p, q)
def ARIMA(ts, p, d, q):

	len_ts = np.size(ts)
	# ensure the number of slices exceeds the least required number
	assert len_ts - p - d - q >= 2, 'invalid time series.'

	ts_diff = []
	ts_diff.append(ts.copy())
	ts_diff_temp = ts.copy()

	for i in range(d):
		ts_diff_temp = np.diff(ts_diff_temp)
		ts_diff.append(ts_diff_temp)

	sds, y_pred = ARMA(ts_diff_temp, p, q)

	for i in range(d):
		y_pred += ts_diff[i][-1]

	return sds, y_pred

# moving average prediction: SARMA
def SARMA(ts, p, q, P, Q, m):

	len_ts = np.size(ts)
	# ensure the number of slices exceeds the least required number
	assert len_ts - p - q - m * (P + Q) >= 2, 'invalid time series.'
	# ensure the number of data selected each season less than the season length
	assert p + q < m, 'invalid parameters.'

	X_ts = np.zeros((len_ts - p - m * P, p + P * (p + 1)))
	y_ts = np.zeros(len_ts - p - m * P)

	for i in range(len_ts - p - m * P):
		X_ts[i, -p:] = ts[i + m * P: i + m * P + p]
		y_ts[i] = ts[i + m * P + p]
		for j in range(P):
			X_ts[i, j * (p + 1): (j + 1) * (p + 1)] = ts[i + j * m: i + j * m + p + 1]

	AR_model = Linear_Regression(X=X_ts, y=y_ts)

	ts_test = np.zeros(p + P * (p + 1))
	ts_test[-p:] = ts[-p:]
	for i in range(P):
		ts_test[i * (p + 1): (i + 1) * (p + 1)] = ts[len_ts - m * (P - i) - p: len_ts - m * (P - i) + 1]

	y_pred = AR_model.predict(X_test=ts_test)

	if q > 0:

		y_ts_pred = AR_model.predict(X_test=X_ts)
		err_ts = y_ts - y_ts_pred

		len_err_ts = np.size(err_ts)

		err_x_ts = np.zeros((len_err_ts - q - m * Q, q + Q * (q + 1)))
		err_y_ts = np.zeros(len_err_ts - q - m * Q)

		for i in range(len_err_ts - q - m * Q):
			err_x_ts[i, -q:] = err_ts[i + m * Q: i + m * Q + q]
			err_y_ts[i] = err_ts[i + m * Q + q]
			for j in range(Q):
				err_x_ts[i, j * (q + 1): (j + 1) * (q + 1)] = err_ts[i + j * m: i + j * m + q + 1]

		MA_model = Linear_Regression(X=err_x_ts, y=err_y_ts, bias=0)

		err_ts_test = np.zeros(q + Q * (q + 1))
		err_ts_test[-q:] = err_ts[-q:]
		for i in range(Q):
			err_ts_test[i * (q + 1): (i + 1) * (q + 1)] = err_ts[len_err_ts - m * (Q - i) - q: len_err_ts - m * (Q - i) + 1]

		y_pred += MA_model.predict(X_test=err_ts_test)

	return AR_model.SDS(), y_pred

# moving average prediction: SARIMA
def SARIMA(ts, p, d, q, P, D, Q, m):

	len_ts = np.size(ts)
	# ensure the number of slices exceeds the least required number
	assert len_ts - p - d - q - m * (P + D + Q) >= 2, 'invalid time series.'

	ts_diff = []
	ts_diff.append(ts.copy())
	ts_diff_temp = ts.copy()

	for i in range(d):
		ts_diff_temp = np.diff(ts_diff_temp)
		ts_diff.append(ts_diff_temp)

	for i in range(D):
		ts_diff_temp = ts_diff_temp[m:] - ts_diff_temp[:-m]
		ts_diff.append(ts_diff_temp)

	sds, y_pred = SARMA(ts_diff_temp, p, q, P, Q, m)

	for i in range(d + D):
		y_pred += ts_diff[i][-1]

	return sds, y_pred
