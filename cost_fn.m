function [ cost ] = cost_fn( costvec )

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
Last edited: 18/03/2020
%}
    
% Compute total cost from individual costs.
cost = norm(costvec)^2;

end