
% ------------------------------ BEGIN CODE -------------------------------
disp("Pendulum Environment")
R0 = interval([-0.51; -0.51; 0], [-0.49; -0.49; 0.1]);
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
options.tensorOrder = 2;
options.taylorTerms = 4;
options.zonotopeOrder = 50;
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
theta_max = 0.2
theta_min = -theta_max
x_safe_min = cos(theta_min)
x_safe_max = cos(0)
y_safe_min = sin(theta_min)
y_safe_max = sin(theta_max)

safeSet = interval([x_safe_min; y_safe_min;-8.0], [x_safe_max;y_safe_max;8.0]); % we want the angle to be in the upright position and don't care about velocity
spec = specification(safeSet, 'safeSet', interval(0, 2));
R = reach(sys, params, options, evParams);
% Verification ------------------------------------------------------------
%t = tic;

%[res, R, simRes] = verify(sys, spec, params, options, evParams, true);
%tTotal = toc(t);
%disp(['Result: ' res])

% Visualization -----------------------------------------------------------
disp("Plotting..")
figure; hold on; box on;
dim = 2
% plot specifications
 plotOverTime(spec, dim, 'DisplayName', 'Safe set');

% plot reachable set
%useCORAcolors("CORA:contDynamics")
plotOverTime(R, dim, 'DisplayName', 'Reachable set');
%updateColorIndex(); % don't plot initial set
%plotOverTime(R(1).R0, 1, 'DisplayName', 'Initial set');
%display(R)
% plot simulations
%plotOverTime(simRes, dim, 'DisplayName', 'Simulations');

% labels and legend
xlabel('time');
ylabel('y-value');
legend()