function Wb = params_to_weights( P, nlv, numW, numB )
%{
FUNCTION DESCRIPTION:
---------------------------------------------------------------------------
Convert parameter vector P containing the weights and biases back into cell 
array structure Wb.
---------------------------------------------------------------------------

INPUTS:
---------------------------------------------------------------------------
P --
---------------------------------------------------------------------------

OUTPUTS:
---------------------------------------------------------------------------
---------------------------------------------------------------------------

Written by: James Rynn
Last edited: 19/03/2020
%}


% Total number of weights.
nW = sum(numW);

% Number of layers.
L = length(numW);

% Initialise storage for weights and biases.
Wb = cell(L,2);

% Counters for stepping through parameter vector.
countW = 0;
countB = nW;


for n = 2:L
    % Weight matrix.
    Wb{n,1} = zeros(nlv(n),nlv(n-1));
    Wb{n,1}(:) = P(countW+1:countW+numW(n));

    % Bias vector.
    Wb{n,2} = P(countB+1:countB+numB(n));

    % Update counters.
    countW = countW + numW(n);
    countB = countB + numB(n);
end

end