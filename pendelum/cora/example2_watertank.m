% ------------------------------ BEGIN CODE -------------------------------
disp("Pendulum Environment")


% Starting range for the observation space
% For pendulum: x, y, angular velocity
% x = cos(theta), y = sin(theta), theta starts randomly in the range -1 to
% 1
R0 = interval([cos(-1); sin(-1); -1], [cos(0); sin(1); 1]);

% Time?
params.tFinal = 10;
params.R0 = polyZonotope(R0);


% Parameters --------------------------------------------------------------
alpha = 1.0;
dt_sim = 0.05;
hmax = 5;
dt = 0.05 % 
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


% theta = atan2(x, -y)
% open-loop system (u = win)

f = @(x, u) [
     -x(2)*x(3); % change in x
     x(1)*x(3);
     ((3*g)/(2*l))*x(2)+(3/(m*l^2)*u(1))
    ];
    %x = [-0.3, 0.7, 7];
    %u = [0.4]


% f = @(x, u) [
  %   (cos(atan2(x(1),x(2)) + clip(x(3)+(3*g/(2*l)*sin(atan2(x(1),x(2))+3.0/(m*l^2)* clip(u(1),min_speed, max_speed))*dt),min_speed,max_speed)) - x(1)) / dt; % change in x
   %  (sin(atan2(x(1),x(2)) + clip(x(3)+(3*g/(2*l)*sin(atan2(x(1),x(2))+3.0/(m*l^2)* clip(u(1),min_speed, max_speed))*dt),min_speed,max_speed)) - x(2)) / dt; % change in y
    % (clip(x(3)+(3*g/(2*l)*sin(atan2(x(1),x(2))+3.0/(m*l^2)* clip(u(1),min_speed, max_speed))*dt),min_speed,max_speed) - x(3)) / dt];
     sys = nonlinearSys(f);
% load neural network controller
nn = neuralNetwork.readONNXNetwork('/home/benedikt/PycharmProjects/nn_verification/pendelum/cora/actor_model_mountaincar.onnx');

% construct neural network controlled system
sys = neurNetContrSys(sys, nn, dt_sim);

% Specification -----------------------------------------------------------

%unsafeSet = interval([-pi; 1.5], [0.6; hmax])
theta_max = 0.2
theta_min = -theta_max
% x = cos(theta)
x_safe_min = cos(theta_min)
x_safe_max = cos(0)
y_safe_min = sin(theta_min)
y_safe_max = sin(theta_max)

% unsafeSet1 = interval([-1; -1;-8.0], [x_safe_min;y_safe_min;8.0]); % we want the angle to be in the upright position and don't care about velocity
% unsafeSet2 = interval([x_safe_max; y_safe_max;-8.0], [1;1;8.0]);
% unsafeSet = unsafeSet1 | unsafeSet2
safeSet = interval([x_safe_min; y_safe_min;-8.0], [x_safe_max;y_safe_max;8.0]); % we want the angle to be in the upright position and don't care about velocity
spec = specification(safeSet, 'safeSet', interval(0, 10));
% specUnsafe = specification(unsafeSet, 'unsafeSet', interval(0, 10));

% Verification ------------------------------------------------------------

t = tic;
[res, R, simRes] = verify(sys, spec, params, options, evParams, true);
tTotal = toc(t);
disp(['Result: ' res])

% Visualization -----------------------------------------------------------
disp("Plotting..")
figure; hold on; box on;
dim = 2
% plot specifications
plotOverTime(spec, dim, 'DisplayName', 'Safe set');
% plotOverTime(specUnsafe, 2, 'DisplayName', 'Unsafe set');

% plot reachable set
useCORAcolors("CORA:contDynamics")
plotOverTime(R, dim, 'DisplayName', 'Reachable set');
updateColorIndex(); % don't plot initial set
plotOverTime(R(1).R0, 1, 'DisplayName', 'Initial set');

% plot simulations
plotOverTime(simRes, dim, 'DisplayName', 'Simulations');

% labels and legend
xlabel('time');
ylabel('height');
legend()

function result = clip(values, min_val, max_val)
    % CLIP Function to limit the values of an array within the range [min_val, max_val].
    % 
    % Syntax: result = clip(values, min_val, max_val)
    %
    % Inputs:
    %    values - Array or value to be clipped.
    %    min_val - Minimum value of the range.
    %    max_val - Maximum value of the range.
    %
    % Outputs:
    %    result - Array with values clipped within the specified range.

    % Clip the values less than min_val
    result = max(values, min_val);

    % Clip the values greater than max_val
    result = min(result, max_val);
end
