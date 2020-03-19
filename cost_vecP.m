function costvec = cost_vecP( P, x1, x2, y, act_fn, nlv, numW, numB )
    %{
    FUNCTION DESCRIPTION:
    Return a vector whose two-norm squared is the cost function given the
    vector of parameters P containing the weights and biases.
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
    
    % Retrieve weights from parameters.
    Wb = params_to_weights(P, nlv, numW, numB);
    
    
    % Compute vector of costs.
    costvec = cost_vec(x1,x2,y,act_fn,Wb,nlv);
end