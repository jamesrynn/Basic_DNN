%{
Neural Net with single hidden layer of one neuron to perform linear
regression using batches of varying size.

Linear activation or sigmoid function.

NOTE: CHOICE BETWEEN/FUNCTION OF ACTIVATION FUNCTION, LINEAR OR SIGMOID?
ADD IN RELU TOO?

%} 



%% INITIALISATION:

% True  - Adaptively decrease learning rate, eta.
% False - Fixed learning rate, eta.
adapt = 'True';


% Seed random number generator.
rng(080992)


% Initialise weight and bias.
W = rand(1);
b = rand(1);


% Parameters for changing learning rate.
if(isequal(adapt,'True'))
    e_R = 2; % factor by which to divide learning rate
    N_R = 4; % how often to reduce learning rate
end



%% DATA:

% Construct data.
n = 20;                             % number of data points
x = linspace(0,1,n);                % known inputs
y = x + 0.25*randn(1,length(x));    % noisy outputs


% Solve least squares problem.
A = [x', ones(length(x),1)];
param = A\y';


% Plot data if required.
plotting_data = 1;
if(plotting_data == 1)
    figure()
    plot(x, y, 'xk', 'linewidth',20)
    hold on
    plot(x, param(1)*x + param(2), 'r', 'linewidth',1.5)
    xlabel('$x$', 'interpreter','latex', 'fontsize',17)
    ylabel('$y$', 'interpreter','latex', 'fontsize',17)
    legend({'Data Points', 'Least Squares Fit'}, 'interpreter','latex', 'fontsize',17, 'location', 'best')
    title('Data and Least Squares Solution', 'interpreter','latex', 'fontsize',20)
    hold off
end



%% ACTIVATION FUNCTION:

%sigmoid = @(x, W, b) 1./(1+exp(-(W*x+b)));
%d_sigmoid = @(a) a*(1-a);

% Activation function - linear
linear = @(x, W, b) W*x+b;
d_linear = @(a) 1;



%%  METHOD 1 - Stochastic Gradient Descent:

Niter = 5e3;
eta = 0.05;

% Initialise weight and bias.
W_stochastic = zeros(Niter,1);  W_stochastic(1) = W; 
b_stochastic = zeros(Niter,1);  b_stochastic(1) = b;

for iterations = 2:Niter
    
    % Select training data point.
    i = randi(length(x), 1);
    
    % Forward pass
    a2 = linear(x(i), W_stochastic(iterations-1), b_stochastic(iterations-1));
    
    % Backward pass
    delta = d_linear(a2)*(a2-y(i));
    
    % Update weights
    W_stochastic(iterations) = W_stochastic(iterations-1) - eta*delta*x(i);
    b_stochastic(iterations) = b_stochastic(iterations-1) - eta*delta;
    
    if(mod(iterations, floor(Niter/N_R)) == 0)
        eta = eta/e_R;
    end
end


% Plot training path.
figure
plot(W_stochastic, b_stochastic)
hold on
plot(param(1), param(2), 'kx', 'markersize', 20, 'linewidth', 4)
hold off
title('Stochastic')
ylabel('bias, b')
xlabel('weight, W')



%% METHOD 2 - Mini Batch:

Niter = 5e3;
eta = 0.05;

% Initialise weight and bias.
W_mini = zeros(Niter,1);  W_mini(1) = W; 
b_mini = zeros(Niter,1);  b_mini(1) = b;

% Size of mini batch.
mini_batch = 6;

for iterations = 2:Niter
    
    % Select batch of data training points.
    perm = randperm(length(x));
    perm = perm(1:mini_batch);
    
    % Storage for update terms (for mean).
    delta_w = 0;
    delta_b = 0;
    
    for i = 1:mini_batch
        
        % Forward pass
        a2 = linear(x(perm(i)), W_mini(iterations-1), b_mini(iterations-1));
        
        % Backward pass
        delta = d_linear(a2)*(a2-y(perm(i)));
        
        % Sum update terms.
        delta_w = delta_w + delta*x(perm(i));
        delta_b = delta_b + delta;
    end
    
    % Take mean of update terms.
    delta_w = delta_w/mini_batch;
    delta_b = delta_b/mini_batch;
    
    % Update weights
    W_mini(iterations) = W_mini(iterations-1) - eta*delta_w;
    b_mini(iterations) = b_mini(iterations-1) - eta*delta_b;
    
    if(mod(iterations, floor(Niter/N_R)) == 0)
        eta = eta/e_R;
    end
end

figure
plot(W_mini, b_mini)
hold on
plot(param(1), param(2), 'kx', 'markersize', 20, 'linewidth', 4)
hold off
title('Mini Batch')
ylabel('bias, b')
xlabel('weight, W')



%% METHOD 3 - Full Batch:

Niter = 3e3;
eta = 0.05;

% Initalise weight and bias.
W_batch = zeros(Niter,1);  W_batch(1) = W; 
b_batch = zeros(Niter,1);  b_batch(1) = b;

for iterations = 2:Niter
    
    % Storage for update terms (for mean).
    delta_w = 0;
    delta_b = 0;
    
    % Loop through all data points.
    for i = 1:length(x)
        
        % Forward pass.
        a2 = linear(x(i), W_batch(iterations-1), b_batch(iterations-1));
        
        % Backward pass.
        delta = d_linear(a2)*(a2-y(i));
        
        % Sum update terms.
        delta_w = delta_w + delta*x(i);
        delta_b = delta_b + delta;
    end
    
    % Take mean of update terms.
    delta_w = delta_w/length(x);
    delta_b = delta_b/length(x);
    
    % Update weights.
    W_batch(iterations) = W_batch(iterations-1) - eta*delta_w;
    b_batch(iterations) = b_batch(iterations-1) - eta*delta_b;
    
    if(mod(iterations, floor(Niter/N_R)) == 0)
        eta = eta/e_R;
    end  
end

figure
plot(W_batch, b_batch)
hold on
plot(param(1), param(2), 'kx', 'markersize', 20, 'linewidth', 4)
hold off
title('Batch')
ylabel('bias, b')
xlabel('weight, W')



%% PLOT PATHS:

figure
plot(W_stochastic, b_stochastic)
hold on
plot(W_mini, b_mini)
plot(W_batch, b_batch)
plot(param(1), param(2), 'kx', 'markersize', 20, 'linewidth', 4)
legend('Stochastic', 'Mini Batch', '(Full) Batch')
ylabel('bias, b')
xlabel('weight, W')
