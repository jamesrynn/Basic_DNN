function [ costvec ] = cost_vec( x1, x2, y, act_fn, Wb, nlv )

%{
FUNCTION DESCRIPTION:
---------------------------------------------------------------------------
Return a vector whose two-norm squared is the cost function given the 
weights and biases.
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
 costvec -- individual costs for each example {vector}.
---------------------------------------------------------------------------

Written by: James Rynn
Last edited: 19/03/2020
%}
    
% Number of layers
L = length(nlv);

% Number of data examples.
nx = length(x1);

% Forward propogation.
costvec = zeros(nx,1);
for i = 1:nx
    a = [x1(i);x2(i)]; % input x_i
    for l = 2:L
        a = my_activate(a, Wb{l,1}, Wb{l,2}, act_fn); % activate layer l
    end
    costvec(i) = norm(y(:,i) - a); % error for example i
end

end