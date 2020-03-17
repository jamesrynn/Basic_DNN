
%{

TURN THIS INTO AN INPUT SCRIPT?

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

To produce Figures 5/9, run withas above but with
data = 'paper11';

Written by: James Rynn
Last edited: 16/03/2020
%}



% Choose example to run.
ex_in = input('Please choose data set 1 (Figures 4/8) or 2 (Figures 5/9)? [1/2] (default: 1): \n');
if(isempty(ex_in) || (ex_in==1))
    data = 'paper10';
elseif(ex_in==2)
    data = 'paper11';
end



% Choose activation function.
fn_in = input('Choose an activation function: [1 (Linear), 2 (Sigmoid), 3 (ReLU), 4 (Leaky ReLU), 5 (ELU)] (default: 2): \n');
if(fn_in==1)
    fn_type = 'linear';
elseif((fn_in==2) || isempty(fn_in))
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



%% Build vector describing number of nodes in ech hidden layer.

% Initialise vector.
hid_layers = [];


% First hidden layer.
nl1_in = input('Please enter number of nodes in the first hidden layer (default: 3) \n');
if(isempty(nl1_in))
    hid_layers = [hid_layers,3];
else
    hid_layers = [hid_layers,nl1_in];
end


% Further hidden layers.
layer = 2;
while(layer>0)
    nl_in = input(['Please enter number of nodes in hidden layer ' num2str(layer) ', enter 0 to add no further layers (default: 0) \n']);
    
    % Add no new layer and break loop.
    if(isempty(nl_in) || (nl_in==0))
        layer = 0;
        
    % Add layer of appropriate size.
    else
        hid_layers = [hid_layers, nl_in];
        layer = layer + 1;
    end
end


% % Output architecture.
% nl = [2,hid_layers,2];
% print(['Number of nodes in each layer is ' mat2str(nl)])
    
    
% Set experiment parameters.
M = 1;
Niter = 1e5;
eta = 0.05;
plot_data = 1;
track_cost = 1;
RNG = 5000;


% Train network and plot results.
netbp_cell(data, fn_type, hid_layers, M, Niter, eta, plot_data, track_cost, RNG)