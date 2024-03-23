import numpy as np


# construct the neural network class
class NN:

    # use this function to initialize a neural network
    def __init__(self, input_length, hidden_length, output_length, act_func = 'Sigmoid', inp_weights = None, hid_weights = None, learning_rate = 0.1, init_method = 'Xavier'):
        # different weight initialization methods
            # Tiny: normal distribution multiplied by a small number
            # Xavier: normal distribution divided by sqrt(last dimension). Best for Sigmoid and Tanh
            # He: normal distribution divided by sqrt(last dimension / 2).Best for ReLu and ReLu_Leak
        weight_multiplier = 0.01
        if inp_weights == None:
            if init_method == 'Xavier':
                weight_multiplier = np.sqrt(1 / input_length)
            elif init_method == 'He':
                weight_multiplier = np.sqrt(2 / input_length)
            self.inp_weights = np.random.normal(size = (hidden_length, input_length + 1)) * weight_multiplier
            self.inp_weights[:, -1] = 0
        if hid_weights == None:
            if init_method == 'Xavier':
                weight_multiplier = np.sqrt(1 / hidden_length)
            elif init_method == 'He':
                weight_multiplier = np.sqrt(2 / hidden_length)
            self.hid_weights = np.random.normal(size = (output_length, hidden_length + 1)) * weight_multiplier
            self.hid_weights[:, -1] = 0
        self.learning_rate = learning_rate
        self.act_func = act_func
        # bias is fixed to 1, for which the weights for it can be adjusted
        self.bias = 1
        self.epsilon = 1e-10
        self.adjust_time = 0
        # vec_one, vec_two, PD: used in the different optimization methods
        self.inp_vec_one, self.inp_vec_two = np.zeros((2, hidden_length, input_length + 1), dtype=np.float)
        self.hid_vec_one, self.hid_vec_two = np.zeros((2, output_length, hidden_length + 1), dtype=np.float)
        self.hid_PD = np.zeros((self.hid_weights.shape[0], self.hid_weights.shape[1]), dtype=np.float)
        self.inp_PD = np.zeros((self.inp_weights.shape[0], self.inp_weights.shape[1]), dtype=np.float)

    # use this function to predict tested data
    def __call__(self, inp_point):

        act_func_dict = {'Sigmoid': NN.Sigmoid, 'Tanh': NN.Tanh, 'ReLu': NN.ReLu, 'ReLu_Leak': NN.ReLu_Leak}

        z_hid = (self.inp_weights * np.append(inp_point, [self.bias])).sum(axis=1)
        if self.act_func in ['Sigmoid', 'Tanh']:
            z_hid[z_hid < -100] = -100
        y_hid = act_func_dict[self.act_func](z_hid)

        z_out = (self.hid_weights * np.append(y_hid, [self.bias])).sum(axis=1)
        if self.act_func in ['Sigmoid', 'Tanh']:
            z_out[z_out < -100] = -100
        # y_out = act_func_dict[self.act_func](z_out)
        y_out = z_out
        # use softmax to convert y_out into the probabilistic forms
        # avoid invalid calculation in np.exp
        y_out[y_out > 100] = 100
        y_out_exp = np.exp(y_out)
        Ohm = np.sum(y_out_exp)
        Y_out = y_out_exp / Ohm

        return Y_out

    # use this function to adjust each weight from the training data
    def Adjust(self, target_result, inp_point, optimizer = 'GD', dropout = 0.3):

        self.adjust_time += 1

        act_func_dict = {'Sigmoid': NN.Sigmoid, 'Tanh': NN.Tanh, 'ReLu': NN.ReLu, 'ReLu_Leak': NN.ReLu_Leak}
        act_func_PD_dict = {'Sigmoid': NN.Sigmoid_PD, 'Tanh': NN.Tanh_PD, 'ReLu': NN.ReLu_PD, 'ReLu_Leak': NN.ReLu_Leak_PD}

        assert np.shape(target_result)[0] == np.shape(inp_point)[0], 'Demensions of X and Y do not match.'
        row_num = np.shape(target_result)[0]

        self.hid_PD[:, :] = 0
        self.inp_PD[:, :] = 0

        for row_ind in range(row_num):

            hid_mask = (np.random.random(self.inp_weights.shape[0]) > dropout).astype(np.int)
            z_hid = (self.inp_weights * np.append(inp_point[row_ind], [self.bias])).sum(axis=1)
            if self.act_func in ['Sigmoid', 'Tanh']:
                z_hid[z_hid < -100] = -100

            y_hid = act_func_dict[self.act_func](z_hid)
            y_hid = y_hid * hid_mask

            z_out = (self.hid_weights * np.append(y_hid, [self.bias])).sum(axis=1) / (1 - dropout)
            if self.act_func in ['Sigmoid', 'Tanh']:
                z_out[z_out < -100] = -100
            # y_out = act_func_dict[self.act_func](z_out)
            y_out = z_out

            # use softmax to convert y_out into the probabilistic forms
            # avoid invalid calculation in np.exp
            y_out[y_out > 100] = 100
            y_out_exp = np.exp(y_out)
            Ohm = np.sum(y_out_exp)
            Y_out = y_out_exp / Ohm

            # calculate the partial derivative matrix of loss function in input layer and hidden layer
            # loss function: C = -Sigma(Y_i * ln(Y_out_i)), i = 1 - 10)
            # M1 = (Y_out - target_result[row_ind, :]) * act_func_PD_dict[self.act_func](z_out)
            M1 = (Y_out - target_result[row_ind, :])
            M1 = np.reshape(M1, (1, np.size(M1)))
            M2 = np.dot(M1, self.hid_weights[:, :-1] * hid_mask) * act_func_PD_dict[self.act_func](z_hid)
            self.hid_PD += (M1.T * np.append(y_hid, [self.bias])) * np.append(hid_mask, [1])
            self.inp_PD += M2.T * np.append(inp_point[row_ind], [self.bias]) * hid_mask[:, np.newaxis]

        self.hid_PD = self.hid_PD / row_num
        self.inp_PD = self.inp_PD / row_num

        # classic gradient descent:
            # X(t) = X(t-1) - ita * gradient(X, t-1)
        if optimizer == 'GD':
            self.hid_weights -= self.learning_rate * self.hid_PD
            self.inp_weights -= self.learning_rate * self.inp_PD

        # Momentum optimizer:
            # v(t) = beta * v(t-1) - ita * gradient(X, t-1)
            # X(t) = X(t-1) + v(t)
        elif optimizer == 'Momentum':
            beta = 0.9
            self.hid_vec_one = beta * self.hid_vec_one - self.learning_rate * self.hid_PD
            self.inp_vec_one = beta * self.inp_vec_one - self.learning_rate * self.inp_PD
            self.hid_weights += self.hid_vec_one
            self.inp_weights += self.inp_vec_one

        # AdaGrad optimizer:
            # v(t) = v(t-1) + gradient(X, t-1) ^ 2
            # X(t) = X(t-1) - ita * gradient(X, t-1) / (sqrt(v(t)) + epsilon)
        elif optimizer == 'AdaGrad':
            self.hid_vec_one += np.power(self.hid_PD, 2)
            self.inp_vec_one += np.power(self.inp_PD, 2)
            self.hid_weights -= self.learning_rate * self.hid_PD / np.sqrt(self.hid_vec_one + self.epsilon)
            self.inp_weights -= self.learning_rate * self.inp_PD / np.sqrt(self.inp_vec_one + self.epsilon)

        # Adam optimizer:
            # v1(t) = beta1 * v1(t-1) + (1 - beta1) * gradient(X, t-1)
            # v2(t) = beta2 * v2(t-1) + (1 - beta2) * gradient(X, t-1) ^ 2
            # v1_regu(t) = v1(t) / (1 - beta1 ^ t)
            # v2_regu(t) = v2(t) / (1 - beta2 ^ t)
            # X(t) = X(t-1) - ita * v1_regu(t) / (sqrt(v2_regu(t)) + epsilon)
        elif optimizer == 'Adam':
            beta_one = 0.9
            beta_two = 0.99
            self.hid_vec_one = beta_one * self.hid_vec_one + (1 - beta_one) * self.hid_PD
            self.hid_vec_two = beta_two * self.hid_vec_two + (1 - beta_two) * np.power(self.hid_PD, 2)
            self.inp_vec_one = beta_one * self.inp_vec_one + (1 - beta_one) * self.inp_PD
            self.inp_vec_two = beta_two * self.inp_vec_two + (1 - beta_two) * np.power(self.inp_PD, 2)
            self.hid_weights -= self.learning_rate * (self.hid_vec_one / (1 - np.power(beta_one, self.adjust_time))) / (
                np.sqrt(self.hid_vec_two / (1 - np.power(beta_two, self.adjust_time))) + self.epsilon
            )
            self.inp_weights -= self.learning_rate * (self.inp_vec_one / (1 - np.power(beta_one, self.adjust_time))) / (
                np.sqrt(self.inp_vec_two / (1 - np.power(beta_two, self.adjust_time))) + self.epsilon
            )

    # different activation functions
    # Sigmoid:
        # Pros: range from 0 - 1, always positive derivative, good classifier
        # Cons: if x significantly different from zero, then the optimization progress is slow even to halt.

    @staticmethod
    def Sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def Sigmoid_PD(x):
        return np.exp(-x) / np.power((1 + np.exp(-x)), 2)

    # Tanh:
        # Pros: range from 0 - 1, always positive derivative, good classifier, and derivative larger than Sigmoid
        # Cons: same as Sigmoid

    @staticmethod
    def Tanh(x):
        return 2 / (1 + np.exp(-2 * x))

    @staticmethod
    def Tanh_PD(x):
        return 4 * np.exp(-2 * x) / np.power((1 + np.exp(-2 * x)), 2)

    # ReLu:
        # Pros: not limited to 0 - 1, always positive derivative, good regressor and classifier. Easy to calculate
        # Cons: easier to diverge than Sigmoid and Tanh. Neurons smaller than zeros will be dead forever

    @staticmethod
    def ReLu(x):
        return (x > 0).astype(np.int) * x

    @staticmethod
    def ReLu_PD(x):
        return (x > 0).astype(np.int)

    # ReLu_Leak:
        # Pros: same as Relu. And no neurons will be dead
        # Cons: easier to diverge than Sigmoid and Tanh.
        # Special property: larger the difference between the slopes of >0 and <=0, more quickly the network learns.

    @staticmethod
    def ReLu_Leak(x):
        return x - 0.5 * (x < 0).astype(np.int) * x

    @staticmethod
    def ReLu_Leak_PD(x):
        return 1 - 0.5 * (x < 0).astype(np.int)

# construct the deep neural network class
class DNN:

    # use this function to initialize a deep neural network
    def __init__(self, input_length, hid_a_length, hid_b_length, output_length, act_func = 'ReLu_Leak', inp_weights = None, hid_a_weights = None, hid_b_weights = None, learning_rate = 0.001, init_method = 'He'):
        if inp_weights == None:
            self.inp_weights = self.Init_Trans(np.random.normal(size = (hid_a_length, input_length + 1)), init_method)
            self.inp_weights[:, -1] = 0
        if hid_a_weights == None:
            self.hid_a_weights = self.Init_Trans(np.random.normal(size = (hid_b_length, hid_a_length + 1)), init_method)
            self.hid_a_weights[:, -1] = 0
        if hid_b_weights == None:
            self.hid_b_weights = self.Init_Trans(np.random.normal(size = (output_length, hid_b_length + 1)), init_method)
            self.hid_b_weights[:, -1] = 0
        self.learning_rate = learning_rate
        self.act_func = act_func
        # bias is fixed to 1, for which the weights for it can be adjusted
        self.bias = 1
        self.epsilon = 1e-10
        self.adjust_time = 0
        # PD, vec_one, vec_two, act_PD: used in the different optimization methods
        self.inp_PD = np.zeros((self.inp_weights.shape[0], self.inp_weights.shape[1]), dtype = np.float)
        self.hid_a_PD = np.zeros((self.hid_a_weights.shape[0], self.hid_a_weights.shape[1]), dtype = np.float)
        self.hid_b_PD = np.zeros((self.hid_b_weights.shape[0], self.hid_b_weights.shape[1]), dtype = np.float)
        self.inp_vec_one, self.inp_vec_two = np.zeros((2, hid_a_length, input_length + 1), dtype = np.float)
        self.hid_a_vec_one, self.hid_a_vec_two = np.zeros((2, hid_b_length, hid_a_length + 1), dtype = np.float)
        self.hid_b_vec_one, self.hid_b_vec_two = np.zeros((2, output_length, hid_b_length + 1), dtype = np.float)

    # use this function to predict tested data
    def __call__(self, inp_point):

        act_func_dict = {'Sigmoid': DNN.Sigmoid, 'Tanh': DNN.Tanh, 'ReLu': DNN.ReLu, 'ReLu_Leak': DNN.ReLu_Leak}

        z_hid_a = (self.inp_weights * np.append(inp_point, [self.bias])).sum(axis = 1)
        if self.act_func in ['Sigmoid', 'Tanh']:
            z_hid_a[z_hid_a < -100] = -100
        y_hid_a = act_func_dict[self.act_func](z_hid_a)

        z_hid_b = (self.hid_a_weights * np.append(y_hid_a, [self.bias])).sum(axis = 1)
        if self.act_func in ['Sigmoid', 'Tanh']:
            z_hid_b[z_hid_b < -100] = -100
        y_hid_b = act_func_dict[self.act_func](z_hid_b)

        z_out = (self.hid_b_weights * np.append(y_hid_b, [self.bias])).sum(axis = 1)
        if self.act_func in ['Sigmoid', 'Tanh']:
            z_out[z_out < -100] = -100
        # y_out = act_func_dict[self.act_func](z_out)
        y_out = z_out

        # use softmax to convert y_out into the probabilistic forms
        # avoid invalid calculation in np.exp
        y_out[y_out > 100] = 100
        y_out_exp = np.exp(y_out)
        Ohm = np.sum(y_out_exp)
        Y_out = y_out_exp / Ohm

        return Y_out

    # use this function to adjust each weight from the training data
    def Adjust(self, target_result, inp_point, optimizer = 'GD', dropout = 0.3):

        self.adjust_time += 1

        act_func_dict = {'Sigmoid': DNN.Sigmoid, 'Tanh': DNN.Tanh, 'ReLu': DNN.ReLu, 'ReLu_Leak': DNN.ReLu_Leak}
        act_func_PD_dict = {'Sigmoid': DNN.Sigmoid_PD, 'Tanh': DNN.Tanh_PD, 'ReLu': DNN.ReLu_PD, 'ReLu_Leak': DNN.ReLu_Leak_PD}

        # if row_num == 1, then it's stochastic gradient descent
        # if row_num > 1, then it's mini-batch gradient descent
        assert np.shape(target_result)[0] == np.shape(inp_point)[0], 'Demensions of X and Y do not match.'
        row_num = np.shape(target_result)[0]

        self.inp_PD[:, :] = 0
        self.hid_a_PD[:, :] = 0
        self.hid_b_PD[:, :] = 0

        for row_ind in range(row_num):

            hid_a_mask = (np.random.random(self.inp_weights.shape[0]) > dropout).astype(np.int)
            hid_b_mask = (np.random.random(self.hid_a_weights.shape[0]) > dropout).astype(np.int)

            z_hid_a = (self.inp_weights * np.append(inp_point[row_ind], [self.bias])).sum(axis = 1)
            if self.act_func in ['Sigmoid', 'Tanh']:
                z_hid_a[z_hid_a < -100] = -100

            y_hid_a = act_func_dict[self.act_func](z_hid_a)
            y_hid_a = y_hid_a * hid_a_mask

            z_hid_b = (self.hid_a_weights * np.append(y_hid_a, [self.bias])).sum(axis = 1) / (1 - dropout)
            if self.act_func in ['Sigmoid', 'Tanh']:
                z_hid_b[z_hid_b < -100] = -100
            y_hid_b = act_func_dict[self.act_func](z_hid_b)
            y_hid_b = y_hid_b * hid_b_mask

            z_out = (self.hid_b_weights * np.append(y_hid_b, [self.bias])).sum(axis = 1) / (1 - dropout)
            if self.act_func in ['Sigmoid', 'Tanh']:
                z_out[z_out < -100] = -100
            # y_out = act_func_dict[self.act_func](z_out)
            y_out = z_out

            # use softmax to convert y_out into the probabilistic forms
            # avoid invalid calculation in np.exp
            y_out[y_out > 100] = 100
            y_out_exp = np.exp(y_out)
            Ohm = np.sum(y_out_exp)
            Y_out = y_out_exp / Ohm

            # calculate the partial derivative of loss function to each weight in input layer and hidden layer
            # loss function: C = -Sigma(Y_i * ln(Y_out_i)), i = 1 - 10)

            # M1 = (Y_out - target_result[row_ind, :]) * act_func_PD_dict[self.act_func](z_out)
            M1 = (Y_out - target_result[row_ind, :])
            M1 = np.reshape(M1, (1, np.size(M1)))
            M2 = np.dot(M1, self.hid_b_weights[:, :-1] * hid_b_mask) * act_func_PD_dict[self.act_func](z_hid_b)
            M3 = np.dot(M2, self.hid_a_weights[:, :-1] * hid_a_mask * hid_b_mask[:, np.newaxis]) * act_func_PD_dict[self.act_func](z_hid_a)

            self.hid_b_PD += M1.T * np.append(y_hid_b, [self.bias]) * np.append(hid_b_mask, [1])
            self.hid_a_PD += M2.T * np.append(y_hid_a, [self.bias]) * np.append(hid_a_mask, [1]) * hid_b_mask[:, np.newaxis]
            self.inp_PD += M3.T * np.append(inp_point[row_ind], [self.bias]) * hid_a_mask[:, np.newaxis]

            # for i in range(self.hid_b_weights.shape[0]):
            #     for j in range(self.hid_b_weights.shape[1] - 1):
            #         self.hid_b_PD[i, j] += (Y_out[i] - target_result[row_ind, i]) * self.out_act_PD[i] * y_hid_b[j]
            #     self.hid_b_PD[i, -1] += (Y_out[i] - target_result[row_ind, i]) * self.out_act_PD[i]
            #
            # for i in range(self.hid_a_weights.shape[0]):
            #     delta_C_y_hid_b = 0
            #     for j in range(self.hid_b_weights.shape[0]):
            #         delta_C_y_hid_b += (Y_out[j] - target_result[row_ind, j]) * self.out_act_PD[j] * self.hid_b_weights[j, i]
            #     for j in range(self.hid_a_weights.shape[1] - 1):
            #         self.hid_a_PD[i, j] += delta_C_y_hid_b * self.hid_b_act_PD[i] * y_hid_a[j]
            #     self.hid_a_PD[i, -1] += delta_C_y_hid_b * self.hid_b_act_PD[i]
            #
            # for i in range(self.inp_weights.shape[0]):
            #     delta_C_y_hid_a = 0
            #     for j in range(self.hid_b_weights.shape[0]):
            #         delta_z_out_y_hid_a = 0
            #         for k in range(self.hid_a_weights.shape[0]):
            #             delta_z_out_y_hid_a += self.hid_b_weights[j, k] * self.hid_b_act_PD[k] * self.hid_a_weights[k, i]
            #         delta_C_y_hid_a += (Y_out[j] - target_result[row_ind, j]) * self.out_act_PD[j] * delta_z_out_y_hid_a
            #     for j in range(self.inp_weights.shape[1] - 1):
            #         self.inp_PD[i, j] += delta_C_y_hid_a * self.hid_a_act_PD[i] * inp_point[row_ind, j]
            #     self.inp_PD[i, -1] += delta_C_y_hid_a * self.hid_a_act_PD[i]

        self.inp_PD[:, :] /= row_num
        self.hid_a_PD[:, :] /= row_num
        self.hid_b_PD[:, :] /= row_num

        # classic gradient descent:
            # X(t) = X(t-1) - ita * gradient(X, t-1)
        if optimizer == 'GD':
            self.hid_b_weights -= self.learning_rate * self.hid_b_PD
            self.hid_a_weights -= self.learning_rate * self.hid_a_PD
            self.inp_weights -= self.learning_rate * self.inp_PD

        # Momentum optimizer:
            # v(t) = beta * v(t-1) - ita * gradient(X, t-1)
            # X(t) = X(t-1) + v(t)
        elif optimizer == 'Momentum':
            beta = 0.9
            self.hid_b_vec_one = beta * self.hid_b_vec_one - self.learning_rate * self.hid_b_PD
            self.hid_a_vec_one = beta * self.hid_a_vec_one - self.learning_rate * self.hid_a_PD
            self.inp_vec_one = beta * self.inp_vec_one - self.learning_rate * self.inp_PD
            self.hid_b_weights += self.hid_b_vec_one
            self.hid_a_weights += self.hid_a_vec_one
            self.inp_weights += self.inp_vec_one

        # AdaGrad optimizer:
            # v(t) = v(t-1) + gradient(X, t-1) ^ 2
            # X(t) = X(t-1) - ita * gradient(X, t-1) / (sqrt(v(t)) + epsilon)
        elif optimizer == 'AdaGrad':
            self.hid_b_vec_one += np.power(self.hid_b_PD, 2)
            self.hid_a_vec_one += np.power(self.hid_a_PD, 2)
            self.inp_vec_one += np.power(self.inp_PD, 2)
            self.hid_b_weights -= self.learning_rate * self.hid_b_PD / np.sqrt(self.hid_b_vec_one + self.epsilon)
            self.hid_a_weights -= self.learning_rate * self.hid_a_PD / np.sqrt(self.hid_a_vec_one + self.epsilon)
            self.inp_weights -= self.learning_rate * self.inp_PD / np.sqrt(self.inp_vec_one + self.epsilon)

        # Adam optimizer:
            # v1(t) = beta1 * v1(t-1) + (1 - beta1) * gradient(X, t-1)
            # v2(t) = beta2 * v2(t-1) + (1 - beta2) * gradient(X, t-1) ^ 2
            # v1_regu(t) = v1(t) / (1 - beta1 ^ t)
            # v2_regu(t) = v2(t) / (1 - beta2 ^ t)
            # X(t) = X(t-1) - ita * v1_regu(t) / (sqrt(v2_regu(t)) + epsilon)
        elif optimizer == 'Adam':
            beta_one = 0.9
            beta_two = 0.99
            self.hid_b_vec_one = beta_one * self.hid_b_vec_one + (1 - beta_one) * self.hid_b_PD
            self.hid_b_vec_two = beta_two * self.hid_b_vec_two + (1 - beta_two) * np.power(self.hid_b_PD, 2)
            self.hid_a_vec_one = beta_one * self.hid_a_vec_one + (1 - beta_one) * self.hid_a_PD
            self.hid_a_vec_two = beta_two * self.hid_a_vec_two + (1 - beta_two) * np.power(self.hid_a_PD, 2)
            self.inp_vec_one = beta_one * self.inp_vec_one + (1 - beta_one) * self.inp_PD
            self.inp_vec_two = beta_two * self.inp_vec_two + (1 - beta_two) * np.power(self.inp_PD, 2)

            self.hid_b_weights -= self.learning_rate * (self.hid_b_vec_one / (1 - np.power(beta_one, self.adjust_time))) / (
                np.sqrt(self.hid_b_vec_two / (1 - np.power(beta_two, self.adjust_time))) + self.epsilon
            )
            self.hid_a_weights -= self.learning_rate * (self.hid_a_vec_one / (1 - np.power(beta_one, self.adjust_time))) / (
                np.sqrt(self.hid_a_vec_two / (1 - np.power(beta_two, self.adjust_time))) + self.epsilon
            )
            self.inp_weights -= self.learning_rate * (self.inp_vec_one / (1 - np.power(beta_one, self.adjust_time))) / (
                np.sqrt(self.inp_vec_two / (1 - np.power(beta_two, self.adjust_time))) + self.epsilon
            )

    # different initialization methods
        # Tiny: normal distribution multiplied by a small number
        # Xavier: normal distribution divided by sqrt(last dimension). Best for Sigmoid and Tanh
        # He: normal distribution divided by sqrt(last dimension / 2).Best for ReLu and ReLu_Leak

    @staticmethod
    def Init_Trans(inp_matrix, init_method):
        inp_col = inp_matrix.shape[1] - 1
        weight_multiplier = 0.01
        if init_method == 'Xavier':
            weight_multiplier = 1 / np.sqrt(inp_col)
        elif init_method == 'He':
            weight_multiplier = 2 / np.sqrt(inp_col)
        return inp_matrix * weight_multiplier

    # different activation functions
    # Sigmoid:
        # Pros: range from 0 - 1, always positive derivative, good classifier
        # Cons: if x significantly different from zero, then the optimization progress is slow even to halt.

    @staticmethod
    def Sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def Sigmoid_PD(x):
        return np.exp(-x) / np.power((1 + np.exp(-x)), 2)

    # Tanh:
        # Pros: range from 0 - 1, always positive derivative, good classifier, and derivative larger than Sigmoid
        # Cons: same as Sigmoid
    @staticmethod
    def Tanh(x):
        return 2 / (1 + np.exp(-2 * x))

    @staticmethod
    def Tanh_PD(x):
        return 4 * np.exp(-2 * x) / np.power((1 + np.exp(-2 * x)), 2)

    # ReLu:
        # Pros: not limited to 0 - 1, always positive derivative, good regressor and classifier. Easy to calculate
        # Cons: easier to diverge than Sigmoid and Tanh. Neurons smaller than zeros will be dead forever
    @staticmethod
    def ReLu(x):
        return (x > 0).astype(np.int) * x

    @staticmethod
    def ReLu_PD(x):
        return (x > 0).astype(np.int)

    # ReLu_Leak:
        # Pros: same as Relu. And no neurons will be dead
        # Cons: easier to diverge than Sigmoid and Tanh.
    @staticmethod
    def ReLu_Leak(x):
        return x - 0.5 * (x < 0).astype(np.int) * x

    @staticmethod
    def ReLu_Leak_PD(x):
        return 1 - 0.5 * (x < 0).astype(np.int)

# construct the convolutional neural network class
class CNN:

    # use this function to initialize a deep neural network
    def __init__(self, input_num, input_width, input_height, f_num, f_len, f_s, hid_a_length, hid_b_length, output_length,
                 act_func = 'ReLu_Leak', f_weights = None, flat_weights = None, hid_a_weights = None, hid_b_weights = None, learning_rate = 0.001, init_method = 'He'):
        self.input_num = input_num
        self.input_wid = input_width
        self.input_hei = input_height
        self.f_num = f_num
        self.f_len = f_len
        self.f_s = f_s
        self.hid_a_length = hid_a_length
        self.hid_b_length = hid_b_length
        self.output_length = output_length
        if f_weights == None:
            self.f_weights = self.Init_Trans(np.random.normal(size=(self.f_num, self.input_num, self.f_len, self.f_len)), init_method)
        self.f_wid = np.int((self.input_wid - self.f_len) / self.f_s) + 1
        self.f_hei = np.int((self.input_hei - self.f_len) / self.f_s) + 1
        self.flat_num = self.f_num * self.f_wid * self.f_hei
        if flat_weights == None:
            self.flat_weights = self.Init_Trans(np.random.normal(size=(self.hid_a_length, self.flat_num + 1)), init_method)
            self.flat_weights[:, -1] = 0
        if hid_a_weights == None:
            self.hid_a_weights = self.Init_Trans(np.random.normal(size=(self.hid_b_length, self.hid_a_length + 1)), init_method)
            self.hid_a_weights[:, -1] = 0
        if hid_b_weights == None:
            self.hid_b_weights = self.Init_Trans(np.random.normal(size=(self.output_length, self.hid_b_length + 1)), init_method)
            self.hid_b_weights[:, -1] = 0
        self.learning_rate = learning_rate
        self.act_func = act_func
        # bias is fixed to 1, for which the weights for it can be adjusted
        self.bias = 1
        self.epsilon = 1e-10
        self.adjust_time = 0
        # PD, vec_one, vec_two, act_PD: used in the different optimization methods
        self.f_PD = np.zeros(self.f_weights.shape, dtype=np.float)
        self.flat_PD = np.zeros(self.flat_weights.shape, dtype=np.float)
        self.hid_a_PD = np.zeros(self.hid_a_weights.shape, dtype=np.float)
        self.hid_b_PD = np.zeros(self.hid_b_weights.shape, dtype=np.float)

        self.f_vec_one, self.f_vec_two = np.zeros(np.append([2], self.f_weights.shape), dtype=np.float)
        self.flat_vec_one, self.flat_vec_two = np.zeros(np.append([2], self.flat_weights.shape), dtype=np.float)
        self.hid_a_vec_one, self.hid_a_vec_two = np.zeros(np.append([2], self.hid_a_weights.shape), dtype=np.float)
        self.hid_b_vec_one, self.hid_b_vec_two = np.zeros(np.append([2], self.hid_b_weights.shape), dtype=np.float)

    # use this function to predict tested data
    def __call__(self, inp_image):

        act_func_dict = {'Sigmoid': CNN.Sigmoid, 'Tanh': CNN.Tanh, 'ReLu': CNN.ReLu, 'ReLu_Leak': CNN.ReLu_Leak}

        num_image, len_image_y, len_image_x = inp_image.shape
        assert num_image == self.input_num and len_image_y == self.input_hei and len_image_x == self.input_wid, 'Dimensions of the image do not match this CNN.'

        # x_twist = np.reshape(CNN.Twist(inp_image, self.f_len, self.f_s),
        #                      (1, self.input_num, self.f_len, self.f_len, self.f_hei, self.f_wid))
        # w_c_reshape = np.reshape(self.f_weights, (self.f_num, self.input_num, self.f_len, self.f_len, 1, 1))
        #
        # y_conv = np.sum(x_twist * w_c_reshape, axis=(1, 2, 3))

        y_conv = CNN.Convolution(inp_image, self.f_weights, stride=self.f_s)

        y_flat = np.reshape(y_conv, np.size(y_conv))

        z_hid_a = (self.flat_weights * np.append(y_flat, [self.bias])).sum(axis=1)
        if self.act_func in ['Sigmoid', 'Tanh']:
            z_hid_a[z_hid_a < -100] = -100
        y_hid_a = act_func_dict[self.act_func](z_hid_a)

        z_hid_b = (self.hid_a_weights * np.append(y_hid_a, [self.bias])).sum(axis=1)
        if self.act_func in ['Sigmoid', 'Tanh']:
            z_hid_b[z_hid_b < -100] = -100
        y_hid_b = act_func_dict[self.act_func](z_hid_b)

        z_out = (self.hid_b_weights * np.append(y_hid_b, [self.bias])).sum(axis=1)
        if self.act_func in ['Sigmoid', 'Tanh']:
            z_out[z_out < -100] = -100
        # y_out = act_func_dict[self.act_func](z_out)
        y_out = z_out

        # use softmax to convert y_out into the probabilistic forms
        # avoid invalid calculation in np.exp
        y_out[y_out > 100] = 100
        y_out_exp = np.exp(y_out)
        Ohm = np.sum(y_out_exp)
        Y_out = y_out_exp / Ohm

        return Y_out

    # use this function to adjust each weight from the training data
    def Adjust(self, target_result, inp_image, optimizer = 'GD', dropout = 0.3):

        self.adjust_time += 1

        act_func_dict = {'Sigmoid': CNN.Sigmoid, 'Tanh': CNN.Tanh, 'ReLu': CNN.ReLu, 'ReLu_Leak': CNN.ReLu_Leak}
        act_func_PD_dict = {'Sigmoid': CNN.Sigmoid_PD, 'Tanh': CNN.Tanh_PD, 'ReLu': CNN.ReLu_PD, 'ReLu_Leak': CNN.ReLu_Leak_PD}

        # if row_num == 1, then it's stochastic gradient descent
        # if row_num > 1, then it's mini-batch gradient descent
        assert np.shape(target_result)[0] == np.shape(inp_image)[0], 'Demensions of X and Y do not match.'
        row_num = np.shape(target_result)[0]

        self.f_PD[:, :, :, :] = 0
        self.flat_PD[:, :] = 0
        self.hid_a_PD[:, :] = 0
        self.hid_b_PD[:, :] = 0

        for row_ind in range(row_num):

            num_image, len_image_y, len_image_x = inp_image[row_ind].shape
            assert num_image == self.input_num and len_image_y == self.input_hei and len_image_x == self.input_wid, 'Dimensions of the image do not match this CNN.'

            hid_a_mask = (np.random.random(self.flat_weights.shape[0]) > dropout).astype(np.int)
            hid_b_mask = (np.random.random(self.hid_a_weights.shape[0]) > dropout).astype(np.int)

            # x_twist = np.reshape(CNN.Twist(inp_image[row_ind], self.f_len, self.f_s),
            #                      (1, self.input_num, self.f_len, self.f_len, self.f_hei, self.f_wid))
            # w_c_reshape = np.reshape(self.f_weights, (self.f_num, self.input_num, self.f_len, self.f_len, 1, 1))
            #
            # y_conv = np.sum(x_twist * w_c_reshape, axis=(1, 2, 3))

            y_conv = CNN.Convolution(inp_image[row_ind], self.f_weights, stride=self.f_s)

            y_flat = np.reshape(y_conv, np.size(y_conv))

            z_hid_a = (self.flat_weights * np.append(y_flat, [self.bias])).sum(axis=1)
            if self.act_func in ['Sigmoid', 'Tanh']:
                z_hid_a[z_hid_a < -100] = -100
            y_hid_a = act_func_dict[self.act_func](z_hid_a)
            y_hid_a = y_hid_a * hid_a_mask

            z_hid_b = (self.hid_a_weights * np.append(y_hid_a, [self.bias])).sum(axis=1) / (1 - dropout)
            if self.act_func in ['Sigmoid', 'Tanh']:
                z_hid_b[z_hid_b < -100] = -100
            y_hid_b = act_func_dict[self.act_func](z_hid_b)
            y_hid_b = y_hid_b * hid_b_mask

            z_out = (self.hid_b_weights * np.append(y_hid_b, [self.bias])).sum(axis=1) / (1 - dropout)
            if self.act_func in ['Sigmoid', 'Tanh']:
                z_out[z_out < -100] = -100
            # y_out = act_func_dict[self.act_func](z_out)
            y_out = z_out

            # use softmax to convert y_out into the probabilistic forms
            # avoid invalid calculation in np.exp
            y_out[y_out > 100] = 100
            y_out_exp = np.exp(y_out)
            Ohm = np.sum(y_out_exp)
            Y_out = y_out_exp / Ohm

            # calculate the partial derivative of loss function to each weight in input layer and hidden layer
            # loss function: C = -Sigma(Y_i * ln(Y_out_i)), i = 1 - 10)

            # M1 = (Y_out - target_result[row_ind, :]) * act_func_PD_dict[self.act_func](z_out)
            M1 = Y_out - target_result[row_ind]
            M1 = np.reshape(M1, (1, np.size(M1)))
            M2 = np.dot(M1, self.hid_b_weights[:, :-1] * hid_b_mask) * act_func_PD_dict[self.act_func](z_hid_b)
            M3 = np.dot(M2, self.hid_a_weights[:, :-1] * hid_a_mask * hid_b_mask[:, np.newaxis]) * act_func_PD_dict[self.act_func](z_hid_a)
            M4 = np.reshape(np.dot(M3, self.flat_weights[:, :-1] * hid_a_mask[:, np.newaxis]), (self.f_num, 1, 1, 1, self.f_hei, self.f_wid))
            inp_twist = CNN.Twist(inp_image[row_ind], self.f_len, self.f_s)

            self.hid_b_PD += M1.T * np.append(y_hid_b, [self.bias]) * np.append(hid_b_mask, [1])
            self.hid_a_PD += M2.T * np.append(y_hid_a, [self.bias]) * np.append(hid_a_mask, [1]) * hid_b_mask[:, np.newaxis]
            self.flat_PD += M3.T * np.append(y_flat, [self.bias]) * hid_a_mask[:, np.newaxis]
            self.f_PD += (M4 * inp_twist).sum(axis=(4,5))

        self.f_PD[:, :, :, :] /= row_num
        self.flat_PD[:, :] /= row_num
        self.hid_a_PD[:, :] /= row_num
        self.hid_b_PD[:, :] /= row_num

        # classic gradient descent:
            # X(t) = X(t-1) - ita * gradient(X, t-1)
        if optimizer == 'GD':
            self.hid_b_weights -= self.learning_rate * self.hid_b_PD
            self.hid_a_weights -= self.learning_rate * self.hid_a_PD
            self.flat_weights -= self.learning_rate * self.flat_PD
            self.f_weights -= self.learning_rate * self.f_PD

        # Momentum optimizer:
            # v(t) = beta * v(t-1) - ita * gradient(X, t-1)
            # X(t) = X(t-1) + v(t)
        elif optimizer == 'Momentum':
            beta = 0.9
            self.hid_b_vec_one = beta * self.hid_b_vec_one - self.learning_rate * self.hid_b_PD
            self.hid_a_vec_one = beta * self.hid_a_vec_one - self.learning_rate * self.hid_a_PD
            self.flat_vec_one = beta * self.flat_vec_one - self.learning_rate * self.flat_PD
            self.f_vec_one = beta * self.f_vec_one - self.learning_rate * self.f_PD
            self.hid_b_weights += self.hid_b_vec_one
            self.hid_a_weights += self.hid_a_vec_one
            self.flat_weights += self.flat_vec_one
            self.f_weights += self.f_vec_one

        # AdaGrad optimizer:
            # v(t) = v(t-1) + gradient(X, t-1) ^ 2
            # X(t) = X(t-1) - ita * gradient(X, t-1) / (sqrt(v(t)) + epsilon)
        elif optimizer == 'AdaGrad':
            self.hid_b_vec_one += np.power(self.hid_b_PD, 2)
            self.hid_a_vec_one += np.power(self.hid_a_PD, 2)
            self.flat_vec_one += np.power(self.flat_PD, 2)
            self.f_vec_one += np.power(self.f_PD, 2)
            self.hid_b_weights -= self.learning_rate * self.hid_b_PD / np.sqrt(self.hid_b_vec_one + self.epsilon)
            self.hid_a_weights -= self.learning_rate * self.hid_a_PD / np.sqrt(self.hid_a_vec_one + self.epsilon)
            self.flat_weights -= self.learning_rate * self.flat_PD / np.sqrt(self.flat_vec_one + self.epsilon)
            self.f_weights -= self.learning_rate * self.f_PD / np.sqrt(self.f_vec_one + self.epsilon)

        # Adam optimizer:
            # v1(t) = beta1 * v1(t-1) + (1 - beta1) * gradient(X, t-1)
            # v2(t) = beta2 * v2(t-1) + (1 - beta2) * gradient(X, t-1) ^ 2
            # v1_regu(t) = v1(t) / (1 - beta1 ^ t)
            # v2_regu(t) = v2(t) / (1 - beta2 ^ t)
            # X(t) = X(t-1) - ita * v1_regu(t) / (sqrt(v2_regu(t)) + epsilon)
        elif optimizer == 'Adam':
            beta_one = 0.9
            beta_two = 0.99
            self.hid_b_vec_one = beta_one * self.hid_b_vec_one + (1 - beta_one) * self.hid_b_PD
            self.hid_b_vec_two = beta_two * self.hid_b_vec_two + (1 - beta_two) * np.power(self.hid_b_PD, 2)
            self.hid_a_vec_one = beta_one * self.hid_a_vec_one + (1 - beta_one) * self.hid_a_PD
            self.hid_a_vec_two = beta_two * self.hid_a_vec_two + (1 - beta_two) * np.power(self.hid_a_PD, 2)
            self.flat_vec_one = beta_one * self.flat_vec_one + (1 - beta_one) * self.flat_PD
            self.flat_vec_two = beta_two * self.flat_vec_two + (1 - beta_two) * np.power(self.flat_PD, 2)
            self.f_vec_one = beta_one * self.f_vec_one + (1 - beta_one) * self.f_PD
            self.f_vec_two = beta_two * self.f_vec_two + (1 - beta_two) * np.power(self.f_PD, 2)

            self.hid_b_weights -= self.learning_rate * (self.hid_b_vec_one / (1 - np.power(beta_one, self.adjust_time))) / (
                np.sqrt(self.hid_b_vec_two / (1 - np.power(beta_two, self.adjust_time))) + self.epsilon
            )
            self.hid_a_weights -= self.learning_rate * (self.hid_a_vec_one / (1 - np.power(beta_one, self.adjust_time))) / (
                np.sqrt(self.hid_a_vec_two / (1 - np.power(beta_two, self.adjust_time))) + self.epsilon
            )
            self.flat_weights -= self.learning_rate * (self.flat_vec_one / (1 - np.power(beta_one, self.adjust_time))) / (
                np.sqrt(self.flat_vec_two / (1 - np.power(beta_two, self.adjust_time))) + self.epsilon
            )
            self.f_weights -= self.learning_rate * (self.f_vec_one / (1 - np.power(beta_one, self.adjust_time))) / (
                np.sqrt(self.f_vec_two / (1 - np.power(beta_two, self.adjust_time))) + self.epsilon
            )

    # construct the convolution function to extract the image features
    @staticmethod
    def Convolution(inp_image, cov_filter, stride=1):

        num_filter, num_image_f, len_filter_y, len_filter_x = cov_filter.shape
        num_image, len_image_y, len_image_x = inp_image.shape
        # ensure that inp_image matches filter data
        assert num_image_f == num_image, 'Dimensions of the filter must match dimensions of inp_image.'
        len_out_y = int((len_image_y - len_filter_y) / stride) + 1
        len_out_x = int((len_image_x - len_filter_x) / stride) + 1
        out_matrix = np.zeros((num_filter, len_out_y, len_out_x))

        for i in range(num_filter):
            pos_image_x = 0
            pos_out_x = 0
            while pos_image_x + len_filter_x <= len_image_x:
                pos_image_y = 0
                pos_out_y = 0
                while pos_image_y + len_filter_y <= len_image_y:
                    out_matrix[i, pos_out_y, pos_out_x] = np.sum(cov_filter[i] * inp_image[:, pos_image_y: pos_image_y + len_filter_y, pos_image_x: pos_image_x + len_filter_x])
                    pos_image_y += stride
                    pos_out_y += 1
                pos_image_x += stride
                pos_out_x += 1

        return out_matrix

    # construct the twist function to be used in weights adjustments
    @staticmethod
    def Twist(inp_image, f_len, f_s):

        num_image, len_image_y, len_image_x = inp_image.shape
        twist_hei = np.int((len_image_y - f_len) / f_s) + 1
        twist_wid = np.int((len_image_x - f_len) / f_s) + 1

        twist_image = np.zeros((num_image, f_len, f_len, twist_hei, twist_wid))
        for i in range(num_image):
            for j in range(f_len):
                for k in range(f_len):
                    twist_image[i, j, k] = inp_image[i, j: j + (twist_hei - 1) * f_s + 1: f_s, k: k + (twist_wid - 1) * f_s + 1: f_s]

        return twist_image

    # construct the down sampling function
    @staticmethod
    def MaxPool(inp_image, len_max_window=2, stride=1):

        num_image, len_image_y, len_image_x = inp_image.shape
        len_out_y = int((len_image_y - len_max_window) / stride) + 1
        len_out_x = int((len_image_x - len_max_window) / stride) + 1
        out_matrix = np.zeros((num_image, len_out_y, len_out_x))

        for i in range(num_image):
            pos_image_x = 0
            pos_out_x = 0
            while pos_image_x + len_max_window <= len_image_x:
                pos_image_y = 0
                pos_out_y = 0
                while pos_image_y + len_max_window <= len_image_y:
                    out_matrix[i, pos_out_y, pos_out_x] = np.max(
                        inp_image[i, pos_image_y: pos_image_y + len_max_window, pos_image_x: pos_image_x + len_max_window])
                    pos_image_y += len_max_window
                    pos_out_y += 1
                pos_image_x += len_max_window
                pos_out_x += 1

        return out_matrix

    # different initialization methods
        # Tiny: normal distribution multiplied by a small number
        # Xavier: normal distribution divided by sqrt(last dimension). Best for Sigmoid and Tanh
        # He: normal distribution divided by sqrt(last dimension / 2).Best for ReLu and ReLu_Leak

    @staticmethod
    def Init_Trans(inp_matrix, init_method):
        inp_shape_size = np.size(inp_matrix.shape)
        weight_multiplier = 0.01
        if inp_shape_size == 2:
            inp_col = inp_matrix.shape[1]
            if init_method == 'Xavier':
                weight_multiplier = 1 / np.sqrt(inp_col)
            elif init_method == 'He':
                weight_multiplier = 2 / np.sqrt(inp_col)
        else:
            inp_col = inp_matrix.shape[3]
            if init_method == 'Xavier':
                weight_multiplier = 1 / inp_col
            elif init_method == 'He':
                weight_multiplier = 2 / inp_col
        return inp_matrix * weight_multiplier

    # different activation functions
    # Sigmoid:
        # Pros: range from 0 - 1, always positive derivative, good classifier
        # Cons: if x significantly different from zero, then the optimization progress is slow even to halt.

    @staticmethod
    def Sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def Sigmoid_PD(x):
        return np.exp(-x) / np.power((1 + np.exp(-x)), 2)

    # Tanh:
        # Pros: range from 0 - 1, always positive derivative, good classifier, and derivative larger than Sigmoid
        # Cons: same as Sigmoid

    @staticmethod
    def Tanh(x):
        return 2 / (1 + np.exp(-2 * x))

    @staticmethod
    def Tanh_PD(x):
        return 4 * np.exp(-2 * x) / np.power((1 + np.exp(-2 * x)), 2)

    # ReLu:
        # Pros: not limited to 0 - 1, always positive derivative, good regressor and classifier. Easy to calculate
        # Cons: easier to diverge than Sigmoid and Tanh. Neurons smaller than zeros will be dead forever

    @staticmethod
    def ReLu(x):
        return (x > 0).astype(np.int) * x

    @staticmethod
    def ReLu_PD(x):
        return (x > 0).astype(np.int)

    # ReLu_Leak:
        # Pros: same as Relu. And no neurons will be dead
        # Cons: easier to diverge than Sigmoid and Tanh.

    @staticmethod
    def ReLu_Leak(x):
        return x - 0.5 * (x < 0).astype(np.int) * x

    @staticmethod
    def ReLu_Leak_PD(x):
        return 1 - 0.5 * (x < 0).astype(np.int)

# construct the linear regression network class
class Linear_Regression:

    # use this function to initialize a linear regression
    def __init__(self, x_dim, alpha = None, learning_rate = 0.1):
        if alpha == None:
            self.alpha = np.random.normal(size = x_dim + 1)
        # self.alpha[-1] = 0
        self.bias = 1
        self.x_dim = x_dim
        self.learning_rate = learning_rate
        self.adjust_time = 0
        self.epsilon = 1e-10
        self.alpha_vec_one, self.alpha_vec_two = np.zeros((2, x_dim + 1), dtype = np.float)
        self.alpha_PD = np.zeros(x_dim + 1, dtype = np.float)

    # use this function to predict tested data
    def __call__(self, inp_x):

        y_out = np.sum(self.alpha * np.append(inp_x, [self.bias]))

    # use this function to adjust each alpha from the training data
    def Adjust(self, y, inp_x, optimizer = 'GD'):
        
        self.adjust_time += 1

        num_y = np.size(y)
        num_x, x_dim = np.shape(inp_x)

        assert num_y == num_x and x_dim == self.x_dim, 'Dimensions of X and Y do not match.'

        self.alpha_PD[:] = 0

        for row_ind in range(num_x):

            y_out = np.sum(self.alpha * np.append(inp_x[row_ind], [self.bias]))

            self.alpha_PD += (y_out - y[row_ind]) * np.append(inp_x[row_ind], [self.bias])

        self.alpha_PD /= num_x

        if optimizer == 'GD':
            self.alpha -= self.learning_rate * self.alpha_PD

        elif optimizer == 'Momentum':
            beta = 0.9
            self.alpha_vec_one = beta * self.alpha_vec_one - self.learning_rate * self.alpha_PD
            self.alpha += self.alpha_vec_one

        elif optimizer == 'AdaGrad':
            self.alpha_vec_one += np.power(self.alpha_PD, 2)
            self.alpha -= self.learning_rate * self.alpha_PD / np.sqrt(self.alpha_vec_one + self.epsilon)

        elif optimizer == 'Adam':
            beta_one = 0.9
            beta_two = 0.99
            self.alpha_vec_one = beta_one * self.alpha_vec_one + (1 - beta_one) * self.alpha_PD
            self.alpha_vec_two = beta_two * self.alpha_vec_two + (1 - beta_two) * np.power(self.alpha_PD, 2)
            self.alpha -= self.learning_rate * (self.alpha_vec_one / (1 - np.power(beta_one, self.adjust_time))) / (
                np.sqrt(self.alpha_vec_two / (1 - np.power(beta_two, self.adjust_time))) + self.epsilon)

    # use this function to assess the performance of this regressor
    def R_Square(self, y, inp_x):

        num_y = np.size(y)
        num_x, x_dim = np.shape(inp_x)

        assert num_y == num_x and x_dim == self.x_dim, 'Dimensions of X and Y do not match.'

        y_out = np.sum(self.alpha[:-1] * inp_x, axis = 1) + self.alpha[-1] * self.bias
        y_mean = np.mean(y)
        R_Square = 1 - np.sum(np.power(y_out - y, 2)) / np.sum(np.power(y - y_mean, 2))

        return R_Square


