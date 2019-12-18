function [t_train] = netbp_cell(data, fn_type, hid_layers, M, Niter, eta, plot_data, track_cost, RNG)
%{
NETBP_CELL
---------------------------------------------------------------------------
Training of DNN using (batch method) back propogation stochastic gradient
descent approach. Weights and biases stored in cell arrays.
---------------------------------------------------------------------------

INPUTS:
---------------------------------------------------------------------------
      data -- indicate data set to be used {string},
hid_layers -- number of neurons in hidden layers {vector},
         M -- batch size {scalar},
     Niter -- number of iterations {scalar},
       eta -- learning rate {scalar},
 plot_data -- indicate whether to plot data {binary scalar},
track_cost -- indicate whether to track cost {binary scalar},
       RNG -- seed for random number generation {scalar}.
---------------------------------------------------------------------------

OUTPUTS:
---------------------------------------------------------------------------
   t_train -- time taken to train network {scalar}.
---------------------------------------------------------------------------
%}


% Seed random number generator.
rng(RNG)


% DATA:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data for Figure 4/8.
if(isequal(data,'paper10')==1)
    x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
    x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
     y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];

% Data for Figure 5/9.
elseif(isequal(data,'paper11')==1)
    x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7, 0.3];
    x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6, 0.7];
     y = [ones(1,5) zeros(1,6); zeros(1,5) ones(1,6)];
end

% Number of data points.
N  = length(x1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% DEFINE NET:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Number of neurons at each layer.
% N.B. data and output are each dimension 2.
nlv = [2, hid_layers, 2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PLOT DATA:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(plot_data == 1)
    figure
    a1 = subplot(1,1,1);
    plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
    hold on
    plot(x1(6:N),x2(6:N),'bx','MarkerSize',12,'LineWidth',4)
    a1.XTick = [0 1];
    a1.YTick = [0 1];
    a1.FontWeight = 'Bold';
    a1.FontSize = 16;
    xlim([0,1])
    ylim([0,1])
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% WEIGHTS AND BIASES:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of layers (including input).
L = length(nlv);

% Initialise weights and biases with random values.
Wb = cell(L,2);
for l = 2:L
    Wb{l,1} = 0.5*randn(nlv(l), nlv(l-1));
    Wb{l,2} = 0.5*randn(nlv(l), 1);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% FORWARD AND BACK PROPOGATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Vector of costs (for plotting)
if(track_cost == 1)
    savecost = zeros(Niter,1);
end

% Begin timing.
tic


for n = 1:Niter
    
    % Index of batch of training points.
    rkv = randperm(N);
      k = rkv(1:M);


    % Storage for gradient terms.
    grad_C = cell(L,M,2);
    
    
    % Batches of size M.
    for m = 1:M
        
        % Data point.
        x = [x1(k(m)); x2(k(m))];
    
        % Forward pass.
        a = cell(L,1);
        a{1} = x;
        D = cell(L,1);
        D{1} = eye(length(x)); % un-used
        for l = 2:L
            [act_y, act_dy] = activate(a{l-1}, Wb{l,1}, Wb{l,2}, fn_type);
            a{l} = act_y;
            D{l} = diag(act_dy);
        end
        
        % Backward pass.
        delta = cell(L,1);
        delta{L} = D{L}*(a{L}-y(:,k(m)));
        for l = L-1:-1:2
            delta{l} = D{l}*(Wb{l+1,1}'*delta{l+1});
        end
        
        % Store gradient for current data point x_k(m).
        for l = L:-1:2
            grad_C{l,m,1} = delta{l}*a{l-1}';
            grad_C{l,m,2} = delta{l};
        end
    end
    
    % Gradient step.
     sum_grad_C = cell(L,2);
    mean_grad_C = cell(L,2);
    for l = 2:L
        sum_grad_C{l,1} = zeros(nlv(l),nlv(l-1));
        sum_grad_C{l,2} = zeros(nlv(l),1);
        for m = 1:M
            sum_grad_C{l,1} = sum_grad_C{l,1} + grad_C{l,m,1};
            sum_grad_C{l,2} = sum_grad_C{l,2} + grad_C{l,m,2};            
        end
        mean_grad_C{l,1} = sum_grad_C{l,1}/M;
        mean_grad_C{l,2} = sum_grad_C{l,2}/M;
    end
    
    % Update weights and biases.
    for l = 2:L
        Wb{l,1} = Wb{l,1} - eta*mean_grad_C{l,1};
        Wb{l,2} = Wb{l,2} - eta*mean_grad_C{l,2};        
    end
    
    
    % Output progress message.
    if(floor(100*n/Niter)>floor(100*(n-1)/Niter))
        fprintf('Progess: %u / %u iterations (%g%%). Estimated time remaining: %.3g seconds.\n', n, Niter, round(100*n/Niter), toc*((Niter/n)-1));
    end
    
    % Track cost if required.
    if(track_cost == 1)
        if(mod(n,Niter/100)==0)
            newcost = cost(fn_type, Wb, nlv, x1,x2, y)   % Occaisonally display cost to screen.
        else
            newcost = cost(fn_type, Wb, nlv, x1,x2, y);
        end
        savecost(n) = newcost;
    end
end

% Time taken to train network.
t_train = toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% PLOT COST DECREASE:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(track_cost == 1)
    figure
    semilogy([1:1e4:Niter],savecost(1:1e4:Niter),'b-','LineWidth',2)
    xlabel('Iteration Number')
    ylabel('Value of cost function')
    set(gca,'FontWeight','Bold','FontSize',18)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% PLOT RESULTS:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NN = 500;
Dx = 1/NN;
Dy = 1/NN;
Aval = zeros(NN+1);
Bval = zeros(NN+1);
xvals = 0:Dx:1;
yvals = 0:Dy:1;
for k1 = 1:NN+1
    xk = xvals(k1);
    for k2 = 1:NN+1
        yk = yvals(k2);
        a = [xk; yk];
        for l = 2:L
            a = activate(a, Wb{l,1}, Wb{l,2}, fn_type);
        end
        
        Aval(k2,k1) = a(1);
        Bval(k2,k1) = a(2);
    end
end
[X,Y] = meshgrid(xvals,yvals);

figure(3)
clf
a2 = subplot(1,1,1);
Mval = Aval>Bval;
contourf(X,Y,Mval,[0.5 0.5])
hold on
colormap([1 1 1; 0.8 0.8 0.8])
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
plot(x1(6:N),x2(6:N),'bx','MarkerSize',12,'LineWidth',4)
a2.XTick = [0 1];
a2.YTick = [0 1];
a2.FontWeight = 'Bold';
a2.FontSize = 16;
xlim([0,1])
ylim([0,1])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end



% COST SUBFUNCTION:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [costval] = cost(fn_type, Wb, nlv, x1,x2, y)

% Number of layers
L = length(nlv);

% Number of data points
N = length(x1);

% Storage for costs for each data point.
costvec = zeros(size(x1,2),1);

for i = 1:N
    x =[x1(i);x2(i)];
    a = x;
    for l = 2:L
        a = activate(a, Wb{l,1}, Wb{l,2}, fn_type);
    end
    costvec(i) = norm(y(:,i) - a,2);
end
costval = norm(costvec,2)^2;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%