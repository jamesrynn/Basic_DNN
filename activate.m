function [y, dy] = activate( x, W, b, act_fn )

%{
FUNCTION DESCRIPTION:
---------------------------------------------------------------------------
Activation function.
Choose between: linear, sigmoid, Tanh, ReLU, leaky ReLU and ELU.
Returns:
 -  y = activate(z)  = activate(Wx+b)
 - dy = activate'(z) = activate'(Wx+b)
The ith component of y is activate((Wx+b)_i) and similarly for dy.
---------------------------------------------------------------------------

INPUTS:
---------------------------------------------------------------------------
  x -- input vector {vector in R^{s}},
  W -- weights {matrix in R^{r x s}},
  b -- bias values {vector in R^{s}}.
---------------------------------------------------------------------------

OUTPUTS:
---------------------------------------------------------------------------
  y -- output vector {vector in R^{r}},
 dy -- derivative of activation function at these points {vector in R^{r}}.
---------------------------------------------------------------------------

Written by: James Rynn
Last edited: 18/12/2019
%}



%% TREAT INPUT:

% Input after weightings and bias applied.
z = W*x+b;



%% ACTIVATION FUNCTION:

% LINEAR:
% -------------------------------------------------------------------------
if(isequal(act_fn,'linear'))
     y = z;                % Activation
    dy = ones(size(z));    % Derivative
% -------------------------------------------------------------------------



% SIGMOID:
elseif(isequal(act_fn,'sigmoid'))
      y = 1./(1+exp(-z));   % Activation
     dy = y.*(1-y);         % Derivative
% -------------------------------------------------------------------------    



% HYPERBOLIC TAN (Tanh):
% -------------------------------------------------------------------------
elseif(isequal(act_fn,'hyp_tan'))
     y = tanh(z);       % Activation
    dy = sech(z).^2;    % Derivative


    
% RECTIFIED LINEAR UNIT (ReLU):
% -------------------------------------------------------------------------
elseif(isequal(act_fn,'ReLU'))
    % Activation.
    y = max(0,z);   

    % Derivative.
    dy = zeros(size(z)); 
    dy(z>=0) = 1;

    
    
% LEAKY RELU:
% -------------------------------------------------------------------------
elseif(isequal(act_fn,'leaky_ReLU'))
    % Slope parameter.
    s = 0.05;   
    
    % Activation.
    y = zeros(size(z));
    y(z>=0) = z(z>=0);
    y(z<0)  = s*z(z<0);
     
    % Derivative.
    dy = zeros(size(z));
    dy(z>=0) = 1;
    dy(z<0)  = s;
    
    
        
% EXPONENTIAL LINEAR UNIT (ELU):
% -------------------------------------------------------------------------
elseif(isequal(act_fn,'ELU'))
    % Slope parameter.
    s = 0.05;   
    
    % Activation.
    y = zeros(size(z));
    y(z>=0) = z(z>=0);
    y(z<0)  = s*(exp(z(z<0))-1);
     
    % Derivative.
    dy = zeros(size(z));
    dy(z>=0) = 1;
    dy(z<0)  = s*exp(z(z<0));
    
    

% OTHER (NOT SUPPORTED):
% -------------------------------------------------------------------------
else
    error(['Activation function type ' act_fn ' is not supported.'])
end