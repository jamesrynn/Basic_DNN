
%{
FUNCTION DESCRIPTION:
---------------------------------------------------------------------------
Run the two examples in the paper 'Deep Learning: An Introduction for
Applied Mathematicians' (2018) by C F Higham and D J Higham - available on 
ArXiV: https://arxiv.org/abs/1801.05894

To produce Figures 4/8, run with:
data = 'paper10';
fn_type = 'sigmoid';
hid_layers = [3,4,5];
M = 1;
Niter = 1e6;
eta = 0.05;
plot_data = 1;
track_cost = 1;
RNG = 5000;

To produce Figures 5/9, run as above but with
data = 'paper11';
---------------------------------------------------------------------------

Written by: James Rynn
Last edited: 16/03/2020
%}



%% CHOOSE DATA:

ex_in = input('Please choose data set 1 (Figures 4/8) or 2 (Figures 5/9)? [1/2] (default: 1): \n');
if(isempty(ex_in) || (ex_in==1))
    data = 'paper10';
elseif(ex_in==2)
    data = 'paper11';
end
load([data '.mat'])


%% CHOOSE ACTIVATION FUNCTION:

fn_in = input('Choose an activation function: [1 (Linear), 2 (Sigmoid), 3 (ReLU), 4 (Leaky ReLU), 5 (ELU)] (default: 2): \n');
if(fn_in==1)
    fn_type = 'linear';
elseif(isempty(fn_in) || (fn_in==2))
    fn_type = 'sigmoid';
elseif(fn_in==3)
    fn_type = 'ReLU';
elseif(fn_in==4)
    fn_type = 'leaky_ReLU';
elseif(fn_in==5)
    fn_type = 'ELU';
else
    error('Please enter a valid option')
end



%% CHOOSE NETWORK STRUCTURE:

% Initialise vector giving number of nodes in each hidden layer.
hid_layers = [];


% First hidden layer.
nl1_in = input('Please enter number of nodes in the first hidden layer or enter 0 for default hidden layers {layer 1: 3 nodes, layer 2: 4 nodes, layer 3: 5} (default: 0) \n');
if(isempty(nl1_in) || (nl1_in==0))
    hid_layers = [3,4,5];
    layer = 0;
else
    hid_layers = [hid_layers,nl1_in];
    layer = 2;
end


% Further hidden layers.
while(layer>0)
    nl_in = input(['Please enter number of nodes in hidden layer ' num2str(layer) ', enter 0 to add no further layers (default: 0) \n']);
    
    % Add no new layer and break loop.
    if(isempty(nl_in) || (nl_in==0))
        layer = 0;
        
    % Add layer of requested size.
    else
        hid_layers = [hid_layers, nl_in];
        layer = layer + 1;
    end
end


% Output architecture.
nl = [2,hid_layers,size(y,1)];
disp(['Number of nodes in each layer is ' mat2str(nl)]); fprintf('\n')



%% BATCH SIZE:

M_in = input(['Please choose (stochastic gradient descent) batch size (min: 1, max: ' num2str(length(x1)) 'default: 1) \n']);
if(isempty(M_in))
    M = 1;
elseif((M<1) || (M>length(x1)))
    error(['Batch size M must be an integer between 1 and ' num2str(length(x1))])
else
    M = M_in;
end



%% NUMBER OF ITERATIONS:

Niter_in = input('Please choose number of gradient descent iterations (default: 100,000) \n');
if(isempty(Niter_in))
    Niter = 1e5;
else
    Niter = Niter_in;
end



%% LEARNING RATE:

eta_in = input('Please choose learning rate (default: 0.05) \n');
if(isempty(eta_in))
    eta = 0.05;
else
    eta = eta_in;
end



%% OTHER PARAMETERS:

% Plot data.
plot_data = 1;

% Track value of cost functions.
track_cost = 1;

% Seed random number generator.
RNG = 5000;



%% TRAIN NETWORK AND MAKE PREDICTIONS:

netbp_cell(data, fn_type, hid_layers, M, Niter, eta, plot_data, track_cost, RNG)