function [Wb, t_train] = train_bp( x1, x2, y, act_fn, nlv, M, Niter, eta )
%{
FUNCTION DESCRIPTION:
---------------------------------------------------------------------------
Train neural net using back propogation stochastic gradient descent with 
batches of stated size.
---------------------------------------------------------------------------

INPUTS:
---------------------------------------------------------------------------
   x1,x2 -- co-ordinates of training data {vectors},
       y -- labels for trianing data {vector},
  act_fn -- type of activation function being used {string},
     nlv -- number of nodes in each layer {vector},
       M -- batch size {scalar},
   Niter -- number of iterations {scalar},
     eta -- learning rate {scalar}.
---------------------------------------------------------------------------

OUTPUTS:
---------------------------------------------------------------------------
      Wb -- weight matrices and bias vectors {cell array},
 t_train -- time taken to train network {scalar}.
---------------------------------------------------------------------------


Written by: C F Higham and D J Higham, August 2017
Available at: https://www.maths.ed.ac.uk/~dhigham/algfiles.html

Adapted by: James Rynn
Last edited: 20/03/2020
%}



%% USEFUL PARAMETERS:

% Number of data points.
N  = length(x1);

% Number of layers (including input).
L = length(nlv);



%% INITIALISE WEIGHTS AND BIASES:

% Initialise weights and biases with random values.
Wb = cell(L,2);
for l = 2:L
    Wb{l,1} = 0.5*randn(nlv(l), nlv(l-1));
    Wb{l,2} = 0.5*randn(nlv(l), 1);
end



%% FORWARD AND BACK PROPOGATION:

% Initialise vector of costs (for plotting).
savecost = zeros(Niter,1);


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
            [act_y, act_dy] = my_activate(a{l-1}, Wb{l,1}, Wb{l,2}, act_fn);
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
        fprintf('Progess: %u / %u iterations (%g%%). Estimated time remaining: %.3g seconds. \n', n, Niter, round(100*n/Niter), toc*((Niter/n)-1));
    end
    
    
    % Compute cost.
    savecost(n) = cost_fn(x1, x2, y, act_fn, Wb, nlv);

    % Occaisionally display cost to screen.
    if(mod(n,Niter/100)==0)
        disp(['Cost = ' num2str(savecost(n))])
    end
end


% Time taken to train network.
t_train = toc;

% Output time.
fprintf('Time taken to train neural network: %.2g seconds. \n', t_train)



%% PLOT COST DECREASE:

plot_cost = input('Plot cost? [0 (No), 1 (Yes)] (default: 0) \n');
if(plot_cost == 1)
    figure
    semilogy([1:1e4:Niter],savecost(1:1e4:Niter),'b-','LineWidth',2)
    xlabel('Iteration Number')
    ylabel('Value of cost function')
    set(gca,'FontWeight','Bold','FontSize',18)
end


end

