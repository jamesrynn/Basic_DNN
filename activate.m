function y = activate(x,W,b)
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

y = 1./(1+exp(-(W*x+b))); 

