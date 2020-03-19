function [ cost ] = cost_fn( x1, x2, y, act_fn, Wb, nlv )

%{
FUNCTION DESCRIPTION:
---------------------------------------------------------------------------
Compute (least-squares) cost function.
---------------------------------------------------------------------------

INPUTS:
---------------------------------------------------------------------------
  x1,x2 -- co-ordinates of training data {vectors},
      y -- labels for trianing data {vector},
 act_fn -- type of activation function being used {string},
     Wb -- weight matrices and bias vectors {cell array},
    nlv -- number of nodes in each layer {vector}.
---------------------------------------------------------------------------

OUTPUTS:
---------------------------------------------------------------------------
   cost -- value of the cost function {scalar}.
---------------------------------------------------------------------------

Written by: James Rynn
Last edited: 19/03/2020
%}
    

% Compute individual costs.
costvec = cost_vec(x1, x2, y, act_fn, Wb, nlv);


% Compute total cost from individual costs.
cost = norm(costvec)^2;

end