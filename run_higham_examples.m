
%{
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
Last edited: 18/12/2019
%}


% Set experiment parameters.
data = 'paper11';
fn_type = 'hyp_tan';
hid_layers = [3,4,5];
M = 1;
Niter = 1e5;
eta = 0.05;
plot_data = 1;
track_cost = 1;
RNG = 5000;


% Train network and plot results.
netbp_cell(data, fn_type, hid_layers, M, Niter, eta, plot_data, track_cost, RNG)