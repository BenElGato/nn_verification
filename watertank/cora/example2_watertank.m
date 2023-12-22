% ------------------------------ BEGIN CODE -------------------------------
disp("Water Tank Environment")




R0 = interval([0.9; 0.8], [1.1; 1.2]);

params.tFinal = 10;
params.R0 = polyZonotope(R0);

% Reachability Settings ---------------------------------------------------

options.timeStep = 0.05;
options.alg = 'lin';
options.tensorOrder = 2;
options.taylorTerms = 4;
options.zonotopeOrder = 50;

% Parameters for NN evaluation --------------------------------------------

evParams = struct();
evParams.poly_method = "singh";

% System Dynamics ---------------------------------------------------------

% Parameters --------------------------------------------------------------
alpha = 1.0;
dt_sim = 0.05;
hmax = 5;

% open-loop system (u = win)
f = @(x, u) [-alpha * sqrt(x(2)) + u(1); 0];
sys = nonlinearSys(f);

% load neural network controller
nn = neuralNetwork.readONNXNetwork('/Users/rayenmhadhbi/PycharmProjects/pythonProject/watertankactor_modelv4.onnx');

% construct neural network controlled system
sys = neurNetContrSys(sys, nn, dt_sim);

% Specification -----------------------------------------------------------

unsafeSet = interval([0.0; 1.5], [0.6; hmax])
safeSet = interval([0.9; 0.7], [1.1; 1.35]);
spec = specification(safeSet, 'safeSet', interval(0, 10));
specUnsafe = specification(unsafeSet, 'unsafeSet', interval(0, 10));

% Verification ------------------------------------------------------------

t = tic;
[res, R, simRes] = verify(sys, specUnsafe, params, options, evParams, true);
tTotal = toc(t);
disp(['Result: ' res])

% Visualization -----------------------------------------------------------

disp("Plotting..")
figure; hold on; box on;

% plot specifications
plotOverTime(spec, 2, 'DisplayName', 'Safe set');
plotOverTime(specUnsafe, 2, 'DisplayName', 'Unsafe set');

% plot reachable set
useCORAcolors("CORA:contDynamics")
plotOverTime(R, 2, 'DisplayName', 'Reachable set');
updateColorIndex(); % don't plot initial set
% plotOverTime(R(1).R0, 1, 'DisplayName', 'Initial set');

% plot simulations
plotOverTime(simRes, 2, 'DisplayName', 'Simulations');

% labels and legend
xlabel('time');
ylabel('height');
legend()