%{
FUNCTION DESCRIPTION:
---------------------------------------------------------------------------
Here we plot the time taken to train the neural net described in Figure 3
of [1] against the number of iterations used for various batch sizes using
the sigmoid activation function. Circle marker indicates 'acceptable'
solution, where all points are correctly banded, an x marker indicates an
unnaceptable' solution.

[1] Higham, C.F. and Higham, D.J., 2019. Deep Learning: An Introduction for
Applied Mathematicians. SIAM Review, 61(4), pp.860-891.
---------------------------------------------------------------------------

Written by: James Rynn
Last edited: 20/03/2020
%}



%% USEFUL PARAMETERS:

% Numbers of iterations.
N1 = 1e4;
N2 = 5e4;
N3 = 7e4;
N4 = 1e5;
N5 = 1e6;



%% 10 DATA POINTS:

figure
loglog(N1, 0.86, 'rx', 'linewidth',2, 'markersize',10)
hold on
loglog(N1, 1.75, 'bx', 'linewidth',2, 'markersize',10)
loglog(N1, 3.12, 'gx', 'linewidth',2, 'markersize',10)
loglog(N1, 8.46, 'kx', 'linewidth',2, 'markersize',10)

loglog(N2, 4.27, 'rx', 'linewidth',2, 'markersize',10)
loglog(N2, 7.26, 'bx', 'linewidth',2, 'markersize',10)
loglog(N2, 11.1, 'gx', 'linewidth',2, 'markersize',10)
loglog(N2, 15.8, 'kx', 'linewidth',2, 'markersize',10)

loglog(N3, 5.62, 'rx', 'linewidth',2, 'markersize',10)
loglog(N3, 9.30, 'bo', 'linewidth',2, 'markersize',10)
loglog(N3, 14.1, 'go', 'linewidth',2, 'markersize',10)
loglog(N3, 21.0, 'ko', 'linewidth',2, 'markersize',10)

loglog(N4, 7.10, 'ro', 'linewidth',2, 'markersize',10)
loglog(N4, 12.0, 'bo', 'linewidth',2, 'markersize',10)
loglog(N4, 18.7, 'go', 'linewidth',2, 'markersize',10)
loglog(N4, 49.9, 'ko', 'linewidth',2, 'markersize',10)

loglog(N5, 44.7, 'ro', 'linewidth',2, 'markersize',10)
loglog(N5, 92.1, 'bo', 'linewidth',2, 'markersize',10)
loglog(N5, 161,  'go', 'linewidth',2, 'markersize',10)
loglog(N5, 278,  'ko', 'linewidth',2, 'markersize',10)
hold off

legend({'$M=1$ (Stochastic)', '$M=3$ (Mini Batch)', '$M=6$ (Mini Batch)', '$M=10$ (Full Batch)'}, 'fontsize',18, 'interpreter','latex', 'location','SE')
title('10 Data Points', 'fontsize',18, 'interpreter','latex')
xlabel('Number of Iterations', 'fontsize',18, 'interpreter','latex')
ylabel('Time to train (s)', 'fontsize',18, 'interpreter','latex')



%% 11 DATA POINTS:

figure
loglog(N1, 0.821, 'rx', 'linewidth',2, 'markersize',10)
hold on
loglog(N1, 1.73, 'bx', 'linewidth',2, 'markersize',10)
loglog(N1, 3.11, 'gx', 'linewidth',2, 'markersize',10)
loglog(N1, 5.63, 'kx', 'linewidth',2, 'markersize',10)

loglog(N4, 7.01, 'rx', 'linewidth',2, 'markersize',10)
loglog(N4, 12.1, 'bx', 'linewidth',2, 'markersize',10)
loglog(N4, 19.2, 'gx', 'linewidth',2, 'markersize',10)
loglog(N4, 32.0, 'kx', 'linewidth',2, 'markersize',10)

loglog(N5, 45.8, 'rx', 'linewidth',2, 'markersize',10)
loglog(N5, 94.6, 'bo', 'linewidth',2, 'markersize',10)
loglog(N5, 168,  'go', 'linewidth',2, 'markersize',10)
loglog(N5, 291,  'ko', 'linewidth',2, 'markersize',10)
hold off

legend({'$M=1$ (Stochastic)', '$M=3$ (Mini Batch)', '$M=6$ (Mini Batch)', '$M=11$ (Full Batch)'}, 'fontsize',18, 'interpreter','latex', 'location','SE')
title('11 Data Points', 'fontsize',18, 'interpreter','latex')
xlabel('Number of Iterations', 'fontsize',18, 'interpreter','latex')
ylabel('Time to train (s)', 'fontsize',18, 'interpreter','latex')


