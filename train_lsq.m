function [ Wb, t_train ] = train_lsq( x1, x2, y, act_fn, nlv )

%{
FUNCTION DESCRIPTION:
---------------------------------------------------------------------------
Train neural net using non-linear least squares.
---------------------------------------------------------------------------

INPUTS:
---------------------------------------------------------------------------
   x1,x2 -- co-ordinates of training data {vectors},
       y -- labels for trianing data {vector},
  act_fn -- type of activation function being used {string},
     nlv -- number of nodes in each layer {vector}.
---------------------------------------------------------------------------

OUTPUTS:
---------------------------------------------------------------------------
      Wb -- weight matrices and bias vectors {cell array},
 t_train -- time taken to train network {scalar}.
---------------------------------------------------------------------------


Written by: C F Higham and D J Higham, August 2017
Available at: https://arxiv.org/abs/1801.05894

Adapted by: James Rynn
Last edited: 19/03/2020
%}


% Number of weights at each layer.
numW = [0,nlv(1:end-1).*nlv(2:end)];

% Number of bias terms at each layer.
numB = [0,nlv(2:end)];

% Total number of parameters (weights + biases).
nP = sum(numW + numB);


% Initialise weights and biases.
Pzero = 0.5*randn(nP,1);


% Perform optimisation.
tic;
lsqP = lsqnonlin(@(P) cost_vecP(P, x1,x2,y, act_fn, nlv, numW, numB), Pzero);
t_train = toc;


% Convert weights/biases parameter vector into cell array.
Wb = params_to_weights(lsqP, nlv, numW, numB);
    
end