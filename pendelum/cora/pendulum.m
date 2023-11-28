
% ------------------------------ BEGIN CODE -------------------------------
disp("Pendulum Environment")
R0 = interval([-0.51; -0.515; 0], [-0.49; -0.485; 0.1]);
params.tFinal = 2;
params.R0 = polyZonotope(R0);

% Parameters --------------------------------------------------------------
dt_sim = 0.001;
hmax = 5;
g = 10.0 % gravity of the system
m = 1.0 % mass of the pendulum
l = 1.0 % lenght of the pendulum
max_speed = 8.0
min_speed = -max_speed
max_torque = 2.0
min_torque = -max_torque
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
     -x(2)*x(3); % change in x
     x(1)*x(3);
     ((3*g)/(2*l))*x(2)+(3/(m*l^2)*u(1))
    ];
sys = nonlinearSys(f);
% load neural network controller
nn = neuralNetwork.readONNXNetwork('/home/benedikt/PycharmProjects/nn_verification/pendelum/cora/actor_model_mountaincar.onnx');
% construct neural network controlled system
sys = neurNetContrSys(sys, nn, dt_sim);
% Specification -----------------------------------------------------------
safeSet = interval([-1; -0.2;-8.0], [1;0.2;8.0]); % we want the angle to be in the upright position and don't care about velocity
spec = specification(safeSet, 'safeSet', interval(1, 2));
R = reach(sys, params, options, evParams);
params.x0 = [-0.5; -0.5;0]; % needed for simulation
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
dim = 2
% Plot Simulation Results
% plot(t, x(:, 2), 'b', 'DisplayName', 'Simulation');
plotOverTime(R, dim, 'DisplayName', 'Reachable set');

plot(t, x(:, 2), 'b', 'DisplayName', 'Simulation');

% Plot Safe Set
yline(0.2, '--g', 'DisplayName', 'Safe Set Upper Bound');
yline(-0.2, '--g', 'DisplayName', 'Safe Set Lower Bound');

% Labels and Legend
xlabel('Time');
ylabel('Second Dimension of Observation');
title('System Simulation and Reachability Analysis');
legend('show');
axis([0, params.tFinal, -1, 1]); % Adjust the axis limits as needed

hold off;