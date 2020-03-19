function [] = plot_boundary( x1, x2, Wb, act_type )

%{
FUNCTION DESCRIPTION:
---------------------------------------------------------------------------
Plot the decision boundary predicted by the trained NN.
---------------------------------------------------------------------------

INPUTS:
---------------------------------------------------------------------------
    x1,x2 -- co-ordinates of training data {vectors},
       Wb -- trained weights and bias vectors {cell array},
 act_type -- type of activation function used {string}.
---------------------------------------------------------------------------


Written by: C F Higham and D J Higham, August 2017
Available at: https://arxiv.org/abs/1801.05894

Adapted by: James Rynn
Last edited: 18/03/2020
%}



%% CHOOSE EVALUATION POINTS:

% Number of layers.
L = size(Wb,1);


% Number of evaluation points.
NN = 500;

% Spacing between points.
Dx = 1/NN;
Dy = 1/NN;

% Evaluation points.
xvals = 0:Dx:1;
yvals = 0:Dy:1;


% Allocate storage for predictions.
Aval = zeros(NN+1);
Bval = zeros(NN+1);



%% EVALUATE PREDICTIONS:

% Loop through evaluation points.
for k1 = 1:NN+1
    xk = xvals(k1);
    for k2 = 1:NN+1
        yk = yvals(k2);
        a = [xk; yk];
        for l = 2:L
           a = activate(a, Wb{l,1}, Wb{l,2}, act_type);
        end
        
        Aval(k2,k1) = a(1);
        Bval(k2,k1) = a(2);
    end
end
[X,Y] = meshgrid(xvals,yvals);



%% PLOT PREDICTIONS:

figure
clf
a2 = subplot(1,1,1);
Mval = Aval>Bval;
contourf(X,Y,Mval,[0.5 0.5])
hold on
colormap([1 1 1; 0.8 0.8 0.8])
plot(x1(1:5),x2(1:5),'ro','MarkerSize',10,'LineWidth',1.5)
plot(x1(6:end),x2(6:end),'bx','MarkerSize',10,'LineWidth',1.5)
a2.XTick = [0 1];
a2.YTick = [0 1];
a2.FontWeight = 'Bold';
a2.FontSize = 16;
xlim([0,1])
ylim([0,1])