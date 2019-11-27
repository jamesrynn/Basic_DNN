function [y, dyx] = activate(x,W,b)
%SIGMOID  Evaluates sigmoid function.
%
%   x is the input vector
%   y is the output vector
%
%   W contains the weights, b contains the shifts
%
%   If x has s components and y has r components, then
%   W is an r by s matrix and b is a vector of length s
% 
%   The ith component of y is activate((Wx+b)_i)
%   where activate(z) = 1/(1+exp(-z)).
%
%   C F Higham and D J Higham, August 2017


% Sigmoid.
if(isequal(fn_type,'sigmoid'))
    y = 1./(1+exp(-(W*x+b)));   % Sigmoid
    dyx = y*(1-y);              % Derivative
    

% Linear.
elseif(isequal(fn_type,'linear'))
    y = W*x + b;    % Linear
    dyx = W;        % Derivative
    

% Hyperbolic Tanh (Tanh).
elseif(isequal(fn_type,'hyperbolic'))
    y = tahn(W*x + b);
    dyx = sech(W*x + b).^2;


% Rectified Linear Unit (ReLU)
elseif(isequal(fn_type,'ReLU'))
    % ReLU
    y = max(0,x);   

    % Derivative
    dyx = 0.5*ones(size(x)); 
    dyx(x>=0) = 1;
    dyx(x<0) = 0;
    
elseif(isequal(fn_type,'leaky_ReLU'))
    y = ;
    dyx = ;
else
    error(['Activation function type' fn_type 'is not supported.'])
end