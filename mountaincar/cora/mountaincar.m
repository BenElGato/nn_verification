
% ------------------------------ BEGIN CODE -------------------------------
disp("Mountain car Environment")
R0 = interval([-0.4; 0], [-0.4; 0]);
params.tFinal = 0.8;
params.R0 = polyZonotope(R0);

% Parameters --------------------------------------------------------------
dt_sim = 0.001;
max_speed = 0.07
min_speed = -0.07

% Reachability Settings ---------------------------------------------------
options.timeStep = dt_sim;
options.alg = 'lin';
options.tensorOrder = 2; % Lower values reduce the computational burden
options.taylorTerms = 4; % Lower values reduce the computational burden
options.zonotopeOrder = 250; % Lowering the zonotope order reduces the number of cross terms and overall complexity of the zonotopes used in the analysis
% Parameters for NN evaluation --------------------------------------------
evParams = struct();
evParams.poly_method = "singh";
% System Dynamics ---------------------------------------------------------
f = @(x, u) [
      x(2) + (u(1) - 1) * 0.001 + cos(3 * x(1)) * (-0.0025);% change in x
     (u(1) - 1) * 0.001 + cos(3 * x(1)) * (-0.0025)
    ];
sys = nonlinearSys(f);
% load neural network controller
nn = neuralNetwork.readONNXNetwork('/home/benedikt/PycharmProjects/nn_verification/mountaincar/cora/actor_model_mountainCar.onnx');
% construct neural network controlled system
sys = neurNetContrSys(sys, nn, dt_sim);
% Specification -----------------------------------------------------------
safeSet = interval([0.45; -0.07], [0.6;0.07]); % we want the angle to be in the upright position and don't care about velocity
spec = specification(safeSet, 'safeSet', interval(0.8, 1));
R = reach(sys, params, options, evParams);
params.x0 = [-0.5; 0]; % needed for simulation
params.tStart = 0;% needed for simulation

[t, x] = simulate(sys, params);
res = verify(sys, spec, params, options, evParams)
% Check the result
if res == 'VERIFIED'
    disp('Verification successful: The pendulum stays within the safe set in the specified interval.');
else
    disp('Verification failed: The pendulum does not stay within the safe set in the specified interval.');
end



% Plotting
figure;
hold on;
dim = 1
% Plot Simulation Results
% plot(t, x(:, 2), 'b', 'DisplayName', 'Simulation');
plotOverTime(R, dim, 'DisplayName', 'Reachable set');

plot(t, x(:, dim), 'b', 'DisplayName', 'Simulation');

% Plot Safe Set
yline(0.45, '--g', 'DisplayName', 'Safe Set Upper Bound');
yline(0.6, '--g', 'DisplayName', 'Safe Set Lower Bound');

% Labels and Legend
xlabel('Time');
ylabel('X-values');
title('System Simulation and Reachability Analysis');
legend('show');
axis([0, params.tFinal, -1.2, 0.6]); % Adjust the axis limits as needed

hold off;