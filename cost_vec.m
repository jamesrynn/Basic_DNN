function [ costvec ] = cost_vec( x1, x2, y, act_fn, Wb, nlv )

%{
FUNCTION DESCRIPTION:
Compute terms in the sum of the cost function (least squares minimiser).

FUNCTION DESCRIPTION:
---------------------------------------------------------------------------
---------------------------------------------------------------------------

INPUTS:
---------------------------------------------------------------------------
---------------------------------------------------------------------------

OUTPUTS:
---------------------------------------------------------------------------
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
        a = activate(a, Wb{l,1}, Wb{l,2}, act_fn); % activate layer l
    end
    costvec(i) = norm(y(:,i) - a); % error for example i
end

end