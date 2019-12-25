This is my online C++ repository for constructed models.

Neural_Networks.cpp & Neural_Networks.h:

Models contained: NN: Neural network with single hidden layer; DNN: Deep neural network with multiple hidden layers; CNN: Convolutional neural network with single convolutional layer and multiple hidden layers; Linear regression: A specific condition for NN;

Crucial algorithm: Back-propagation algorithm to adjust weights, convolution function to extract image features; Activation functions realized: Sigmoid, Tanh, ReLu, ReLu_Leak, Softmax; Weight initialization: Tiny normal, Xavier, He; Optimizers realized: Classic GD, Momentum, AdamGrad, Adam; Training batches: Stochastic training, mini-batch training;

Neural_Networks_Test.cpp: Testing code for NN, DNN, CNN;

Support_Vector.cpp & Support_Vector.h:

Models contained: Support vector classifier; Crucial algorithm: SMO algorithm to adjust parameters for trained data; Kernels constructed: Linear, Poly, Gaussian;

Support_Vector_Test.cpp: Testing code for SVC in three kernels(Linear, Poly, Gaussian);

Time_Series.cpp & Time_Series.h:

Basic averaging method: Moving agerage, Exponential smooth average; Regression method: AR, Auto regressive predictor; ARMA, Auto regressive moving average predictor; ARIMA, Auto regressive integrated moving average predictor; SARMA, Seasonal auto regressive moving average predictor; SARIMA, Seasonal auto regressive integrated moving average predictor;

Time_Series_Test.cpp: Testing code for AR, ARMA, ARIMA, SARMA, SARIMA;

Base.cpp & Base.h: Linear_Regression, which is used in time-series prediction models.

train_sample_small.csv: a small sample of MNIST digits data, which is standardized. Used in testing of neural_networks and support_vector.
