%{
FUNCTION DESCRIPTION:
---------------------------------------------------------------------------
Run the two examples in the paper 'Deep Learning: An Introduction for
Applied Mathematicians' (2018) by C F Higham and D J Higham - available on 
ArXiV: https://arxiv.org/abs/1801.05894


Use the following inputs to replicate the stated figures. Here * indicates
any choice of parameter is sufficient.

Figure 4: 1,*,2,0,*,*,*,1,0
Figure 5: 2,*,2,0,*,*,*,1,0
Figures 7 & 8: 1,*,2,0,1,1e6,0.05,0,1,1
Figure 9: 2,*,2,0,1,1e6,0.05,0,1,*

NB: Subtle differences will occur in Figures 8 and 9 from these in the
paper as we use a different function to select our random indices during
stochastic gradient descent.
---------------------------------------------------------------------------

Written by: James Rynn
Last edited: 19/03/2020
%}



%% CHOOSE DATA:

ex_in = input('Please choose data set 1 (Figures 4/8) or 2 (Figures 5/9)? [1/2] (default: 1): \n');
if(isempty(ex_in) || (ex_in==1))
    data = 'paper10';
elseif(ex_in==2)
    data = 'paper11';
end
load([data '.mat'])



%% PLOT DATA:

plot_data = input('Plot data? [0 (No), 1 (Yes)] (default: 0) \n');
if(plot_data == 1)
    figure
    a1 = subplot(1,1,1);
    plot(x1(1:5),x2(1:5),'ro','MarkerSize',8,'LineWidth',1.5)
    hold on
    plot(x1(6:end),x2(6:end),'bx','MarkerSize',8,'LineWidth',1.5)
    a1.XTick = [0 1];
    a1.YTick = [0 1];
    a1.FontWeight = 'Bold';
    a1.FontSize = 16;
    xlim([0,1])
    ylim([0,1])
end



%% CHOOSE ACTIVATION FUNCTION:

act_fn_in = input('Choose an activation function: [1 (Linear), 2 (Sigmoid), 3 (ReLU), 4 (Leaky ReLU), 5 (ELU)] (default: 2): \n');
if(act_fn_in==1)
    act_fn = 'linear';
elseif(isempty(act_fn_in) || (act_fn_in==2))
    act_fn = 'sigmoid';
elseif(act_fn_in==3)
    act_fn = 'ReLU';
elseif(act_fn_in==4)
    act_fn = 'leaky_ReLU';
elseif(act_fn_in==5)
    act_fn = 'ELU';
else
    error('Please enter a valid option.')
end



%% CHOOSE NETWORK STRUCTURE:

% Initialise vector giving number of nodes in each hidden layer.
hid_layers = [];


% First hidden layer.
nl1_in = input('Please enter number of nodes in the first hidden layer or enter 0 for default net {layer 2: 2 nodes, layer 4: 3 nodes} (default: 0) \n');
if(isempty(nl1_in) || (nl1_in==0))
    hid_layers = [2,3];
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



%% DEFINE NET:

% N.B. Data (input layer) and output are each dimension 2.
nlv = [2,hid_layers,size(y,1)];

% Number of layers.
L = length(nlv);

% Output architecture (number of neurons at each layer).
disp(['Number of nodes in each layer is ' mat2str(nlv)]); fprintf('\n')



%% BATCH SIZE:

M_in = input(['Please choose (stochastic gradient descent) batch size (min: 1, max: ' num2str(length(x1)) ', default: 1) \n']);
if(isempty(M_in))
    M = 1;
elseif((M_in<1) || (M_in>length(x1)))
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

% Seed random number generator (for reproducibility).
RNG = 5000; % As used in Higham paper.
rng(RNG)



%% TRAIN NETWORK USING NON-LINEAR LEAST SQUARES:

lsq_in = input('Train using non-linear least squares? [0 (No), 1 (Yes)] (default: 0) \n');
if(lsq_in==1)
    % Train using least squares.
    [Wb_lsq, t_train_lsq] = train_lsq(x1, x2, y, act_fn, nlv);
    
    % Plot predictions/decision boundary.
    plot_boundary(x1, x2, Wb_lsq, act_fn)
end



%% TRAIN NETWORK USING BACK PROPAGATION:

bp_in = input('Train using back propogation? [0 (No), 1 (Yes)] (default: 1) \n');
if(bp_in==1)
    % Train using back propogation.
    [Wb_bp, t_train_bp] = train_bp(x1, x2, y, act_fn, nlv, M, Niter, eta);
    
    % Plot predictions/decision boundary.
    plot_boundary(x1, x2, Wb_bp, act_fn)
end



