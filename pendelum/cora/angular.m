% ------------------------------ BEGIN CODE -------------------------------
disp("Pendulum Environment")
% ----------------Starting state definition----------
starting_angle = pi;
thetadot = 0;
%--------------------------------------------------------------

R0 = interval([pi-0.05; thetadot], [pi+0.05; thetadot]);
params.tFinal = 5;
params.R0 = polyZonotope(R0);

% Parameters --------------------------------------------------------------

sampling_time = 0.01
g = 10.0 % gravity of the system
m = 1.0 % mass of the pendulum
l = 1.0 % lenght of the pendulum
max_speed = 8.0
min_speed = -max_speed
max_torque = 2.0
min_torque = -max_torque
% Reachability Settings ---------------------------------------------------
options.timeStep = 0.001;
options.alg = 'lin';
options.tensorOrder = 2; % Lower values reduce the computational burden
options.taylorTerms = 1; % Lower values reduce the computational burden
options.zonotopeOrder = 80; % Lowering the zonotope order reduces the number of cross terms and overall complexity of the zonotopes used in the analysis

% Parameters for NN evaluation --------------------------------------------
% TODO Splitting
evParams = struct;
evParams.poly_method = 'regression';
evParams.bound_approx = true;
evParams.reuse_bounds = false;
evParams.num_generators = 100000
% System Dynamics ---------------------------------------------------------
f = @(x, u) [
     x(2);
     ((3*g)/(2*l))*sin(x(1))+(3/(m*l^2)*u(1))
    ];
sys = nonlinearSys(f);
% load neural network controller
nn = neuralNetwork.readONNXNetwork('/home/benedikt/PycharmProjects/nn_verification/pendelum/cora/network.onnx');
nn.evaluate(params.R0, evParams);

nn.refine(2, "layer", "both", params.R0.c, true);


% construct neural network controlled system
sys = neurNetContrSys(sys, nn, sampling_time);
% Specification -----------------------------------------------------------
safeSet = interval([0;-8.0], [1;8.0]); % we want the angle to be in the upright position and don't care about velocity
unsafeSet = specification(interval([0.2;-inf], [2.9; inf]), 'unsafeSet');
spec = specification(safeSet, 'safeSet', interval(1, 2));
R = reach(sys, params, options, evParams);
params.x0 = [pi;0]; % needed for simulation
params.tStart = 0;% needed for simulation

[t, x] = simulate(sys, params);


% Plotting
figure;
hold on;
dim = 1
% Plot Simulation Results
plotOverTime(R, dim, 'DisplayName', 'Reachable set', 'Unify', true);
%x(:, 1) = mod(x(:, 1) + pi,2*pi) - pi;
plot(t, x(:, 1), 'b', 'DisplayName', 'Simulation');

% Plot Safe Set
yline(-0.1, '--g', 'DisplayName');
yline(0.1, '--g', 'DisplayName');

% Labels and Legend
xlabel('Time');
ylabel('Theta');
title('System Simulation and Reachability Analysis');
legend('show');
axis([0, params.tFinal, -pi, pi]); % Adjust the axis limits as needed

hold off;
