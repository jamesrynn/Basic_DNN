% Plot time taken to train versus number of iterations performed,
% indicating whether or not the solution is 'acceptable'.


% Numbers of iterations.
N1 = 1e4;
N2 = 5e4;
N3 = 7e4;
N4 = 1e5;
N5 = 1e6;



%% 10 DATA POINTS:

figure
loglog(N1, 0.86, 'rx', 'linewidth', 2)
hold on
loglog(N1, 1.75, 'bx', 'linewidth', 2)
loglog(N1, 3.12, 'gx', 'linewidth', 2)
loglog(N1, 8.46, 'bx', 'linewidth', 2)

loglog(N2, 4.27, 'rx', 'linewidth', 2)
loglog(N2, 7.26, 'bx', 'linewidth', 2)
loglog(N2, 11.1, 'gx', 'linewidth', 2)
loglog(N2, 15.8, 'bx', 'linewidth', 2)

loglog(N3, 5.62, 'rx', 'linewidth', 2)
loglog(N3, 9.30, 'bo', 'linewidth', 2)
loglog(N3, 14.1, 'go', 'linewidth', 2)
loglog(N3, 21.0, 'bo', 'linewidth', 2)

loglog(N4, 7.10, 'ro', 'linewidth', 2)
loglog(N4, 12.0, 'bo', 'linewidth', 2)
loglog(N4, 18.7, 'go', 'linewidth', 2)
loglog(N4, 49.9, 'bo', 'linewidth', 2)

loglog(N5, 44.7, 'ro', 'linewidth', 2)
loglog(N5, 92.1, 'bo', 'linewidth', 2)
loglog(N5, 161,  'go', 'linewidth', 2)
loglog(N5, 278,  'bo', 'linewidth', 2)
hold off

legend({'M=1 (Stochastic)', 'M=3 (Mini Batch)', 'M=6 (Mini Batch)', 'M=10 (Full Batch)'})
title('10 Data Points')
xlabel('Number of Iterations')
ylabel('Time / s')



%% 11 DATA POINTS:

figure
loglog(N1, 0.821, 'rx', 'linewidth', 2)
hold on
loglog(N1, 1.73, 'bx', 'linewidth', 2)
loglog(N1, 3.11, 'gx', 'linewidth', 2)
loglog(N1, 5.63, 'bx', 'linewidth', 2)

loglog(N4, 7.01, 'rx', 'linewidth', 2)
loglog(N4, 12.1, 'bx', 'linewidth', 2)
loglog(N4, 19.2, 'gx', 'linewidth', 2)
loglog(N4, 32.0, 'bx', 'linewidth', 2)

loglog(N5, 45.8, 'rx', 'linewidth', 2)
loglog(N5, 94.6, 'bo', 'linewidth', 2)
loglog(N5, 168,  'go', 'linewidth', 2)
loglog(N5, 291,  'bo', 'linewidth', 2)
hold off

legend({'M=1 (Stochastic)', 'M=3 (Mini Batch)', 'M=6 (Mini Batch)', 'M=11 (Full Batch)'})
title('11 Data Points')
xlabel('Number of Iterations')
ylabel('Time / s')


