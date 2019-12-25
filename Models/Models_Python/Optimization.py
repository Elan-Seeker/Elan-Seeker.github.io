import numpy as np

# construct the support vector classification class
class SVC:

	# use this function to initialize a SVC
	def __init__(self, input_length, output_length, X_train, Y_train, kernel = 'linear'):
		
		self.kernel = kernel
		self.inp_len = input_length
		self.out_len = output_length

		X_num, X_len = X_train.shape
		Y_num, Y_len = Y_train.shape

		assert X_num == Y_num and X_len == self.inp_len and Y_len == self.out_len, 'Invalid x and y dimensions.'

		max_num = 30000
		self.train_num = np.min([X_num, max_num])
		if X_num > max_num:
			print('Scale of training data has been truncated.')
		self.X_train = X_train[:self.train_num, :]
		self.Y_train = Y_train[:self.train_num, :]

		self.lamda = np.zeros((self.train_num, self.out_len))
		self.bias = np.zeros(self.out_len)
		self.adjust_epoch = 0

		# construct the kernel matrix to be used to lamda adjustments
		self.kernel_matrix = np.zeros((self.train_num, self.train_num), dtype = np.float)
		if self.kernel == 'linear':
			for i in range(self.inp_len):
				temp = self.X_train[:, i][:, np.newaxis]
				self.kernel_matrix += temp.T * temp
		elif self.kernel == 'poly':
			degree = 4
			temp_kernel_matrix = np.zeros((self.train_num, self.train_num), dtype = np.float)
			for i in range(self.inp_len):
				temp = self.X_train[:, i][:, np.newaxis]
				temp_kernel_matrix += temp.T * temp
			for i in range(degree + 1):
				self.kernel_matrix += np.power(temp_kernel_matrix, i)
		elif self.kernel == 'gauss':
			delta = 1
			for i in range(self.train_num):
				gauss_ori = np.sqrt(np.sum(np.power(self.X_train - self.X_train[i, :], 2), axis=1))
				self.kernel_matrix[i, :] = np.exp(-gauss_ori / (2 * np.power(delta, 2)))

	# use this function to predict tested data
	def __call__(self, inp_point):

		assert np.size(inp_point) == self.inp_len, 'Invalid input dimension.'
		kernel_dict = {'linear': SVC.Linear, 'poly': SVC.Poly, 'gauss': SVC.Gaussian}

		kernel_dot = kernel_dict[self.kernel](self.X_train, inp_point) #dimension: self.train_num
		y_out = np.sum((self.lamda * self.Y_train).T * kernel_dot, axis=1) + self.bias

		return np.argmax(y_out)

	# use SMO algorithm to adjust the omega and bias
	def Learn(self):
		
		self.adjust_epoch += 1

		for i in range(self.out_len):
			j1 = 0
			j2 = 1
			while True:

				adjust_temp_one = self.kernel_matrix[j1, j1] - 2 * self.kernel_matrix[j1, j2] + self.kernel_matrix[j2, j2]
				adjust_temp_two = self.Y_train[j1, i] - self.Y_train[j2, i] - np.sum(
					self.lamda[:, i] * self.Y_train[:, i] * (self.kernel_matrix[:, j1] - self.kernel_matrix[:, j2]))

				# adjust two lamda each time
				self.lamda[j1, i] += adjust_temp_two / (adjust_temp_one * self.Y_train[j1, i])
				self.lamda[j2, i] -= adjust_temp_two / (adjust_temp_one * self.Y_train[j2, i])

				if j2 == 0 or j2 == self.train_num - 1:
					break

				j1 = (j1 + 2) % self.train_num
				j2 = (j2 + 2) % self.train_num

			SV_num = np.sum((self.lamda[:, i] != 0).astype(np.int))
			b_SV_sum = np.sum((self.lamda[:, i] != 0).astype(np.int) * self.Y_train[:, i]) - np.sum(self.lamda[:, i] * self.Y_train[:, i] * self.kernel_matrix)
			# when each lamda has been adjusted in this epoch, adjust the bias
			self.bias[i] = b_SV_sum / SV_num

	@staticmethod
	def Linear(x1, x2):
		return np.sum(x1 * x2, axis=1)

	@staticmethod
	def Poly(x1, x2, degree = 3):
		poly_ori = np.sum(x1 * x2, axis=1)
		return_array = 0
		for i in range(degree + 1):
			return_array += np.power(poly_ori, i)
		return return_array

	@staticmethod
	def Gaussian(x1, x2, delta = 1):
		gauss_ori = np.sqrt(np.sum(np.power(x1 - x2, 2), axis=1))
		return np.exp(-gauss_ori / (2 * np.power(delta, 2)))

# construct the multivariate linear regression class
class Linear_Regression:

	# use this function to initialize a linear regression
	def __init__(self, X, y, bias = 1):

		self.X = X
		self.y = y
		self.bias = bias

		self.n_y, self.n_x = X.shape
		assert np.size(self.y) == self.n_y, 'invalid input data.'

		self.omega = np.zeros(self.n_x + 1, dtype=np.float)
		self.X_aug = np.column_stack([self.X, np.ones(self.n_y)[:, np.newaxis] * self.bias])

		X_reshape_one = np.reshape(self.X_aug, (self.n_y, self.n_x + 1, 1))
		X_reshape_two = np.reshape(self.X_aug, (self.n_y, 1, self.n_x + 1))

		self.matrix_F = np.sum(X_reshape_one * X_reshape_two, axis=0)
		self.matrix_G = np.sum(self.X_aug * self.y[:, np.newaxis], axis=0)

		if np.linalg.matrix_rank(self.matrix_F) == self.n_x + 1:
			self.omega = np.linalg.inv(self.matrix_F).dot(self.matrix_G)
		else:
			self.omega = np.linalg.pinv(self.matrix_F).dot(self.matrix_G)

	# use this function to predict test data
	def predict(self, X_test):

		if np.size(X_test.shape) == 2:
			n_y, n_x = X_test.shape
			assert n_x == self.n_x, 'invalid X_test.'

			X_test_aug = np.column_stack([X_test, np.ones(n_y)[:, np.newaxis] * self.bias])

			y_pred = X_test_aug.dot(self.omega)

		else:
			n_x = np.size(X_test)
			assert n_x == self.n_x, 'invalid X_test.'

			y_pred = np.append(X_test, [self.bias]).dot(self.omega)

		return y_pred

	# use this function to get the standard deviation of the sample
	def SDS(self):

		y_pred = self.predict(X_test=self.X)

		return np.sqrt(np.sum(np.power(y_pred - self.y, 2)) / self.n_y)