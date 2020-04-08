# Basic_DNN
An adapted implementation of the MATLAB code provided in [[1]](https://epubs.siam.org/doi/10.1137/18M1165748). The original code provided with the paper is available at: https://www.maths.ed.ac.uk/~dhigham/algfiles.html

[[1] Higham, C.F. and Higham, D.J., 2019. Deep Learning: An Introduction for Applied Mathematicians. SIAM Review, 61(4), pp.860-891](https://epubs.siam.org/doi/10.1137/18M1165748).


## Notable Additional Functionality:
* arbitrary number of hidden layers with arbitrary number of nodes in each
* choice of activation function including linear, sigmoid, tanh, ReLU, leaky ReLU and ELU
* batch stochastic Gradient descent rather than (single sample) stochastic gradient descent
* parameters such as number of iterations and learning rate not fixed

## To-Do Items:
* (potential:) vectorised version - less 'readable' but quicker to perform experiments
* adaptive learning rate
* arbitrary number of data points in training set

## Python Version Available:
A Python implementation of this code has been written by [Alex Hiles](https://github.com/alexhiles) and is available at https://github.com/alexhiles/NN
