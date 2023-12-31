% ------------------------------ BEGIN CODE -------------------------------
disp("ACC Environment")
% ----------------Starting state definition----------%
x_lead_min = 90
x_lead_max = 100

x_ego_min = 11
x_ego_max = 12


v_lead_min = 32
v_lead_max = 32.2

v_ego_min = 30
v_ego_max = 30.2

a_lead_min = 0.0
a_lead_max = 0.0

a_ego_min = 0.0
a_ego_max = 0.0

T_Gap = 1.4
D_Default = 10.0

a_c_lead = -2.0
R0 = interval([x_lead_min;x_ego_min;v_lead_min;v_ego_min;a_lead_min;a_ego_min;T_Gap;D_Default],[x_lead_max;x_ego_max;v_lead_max;v_ego_max;a_lead_max;a_ego_max;T_Gap;D_Default]);


params.tFinal = 5;
params.R0 = polyZonotope(R0);

sampling_time = 0.1
% Reachability Settings ---------------------------------------------------
options.timeStep = 0.01;
options.alg = 'lin';
options.tensorOrder = 2;
options.taylorTerms = 1; % Lower values reduce the computational burden
options.zonotopeOrder = 20; % Lowering the zonotope order reduces the number of cross terms and overall complexity of the zonotopes used in the analysis

% Parameters for NN evaluation --------------------------------------------
% TODO Splitting
evParams = struct;
evParams.poly_method = 'regression';
evParams.bound_approx = true;
evParams.reuse_bounds = false;
evParams.num_generators = 100000

% System Dynamics ---------------------------------------------------------
f = @(x, u) [
     x(3);
     x(4);
     x(5);
     x(6);
     -2*x(5)+2*a_c_lead - 0.0001*(x(3)^2);
     -2*x(6)+2*u(1) - 0.0001*(x(4)^2);
     0;
     0;
    ];
sys = nonlinearSys(f);
% load neural network controller
nn = neuralNetwork.readONNXNetwork('/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network11.onnx');
nn.evaluate(params.R0, evParams);
num_refinements = 6
%for i = 1:num_refinements
 %   nn.refine(2, "layer", "both", params.R0.c, true);
%end
%nn.refine(2, "layer", "both", params.R0.c, true);


% construct neural network controlled system
sys = neurNetContrSys(sys, nn, sampling_time);
params.x0 = [x_lead_min;x_ego_min;v_lead_min;v_ego_min;a_lead_min;a_ego_min;T_Gap;D_Default]; % needed for simulation
params.tStart = 0;% needed for simulation
[t, x] = simulate(sys, params);
opt = struct;
opt.points = 50;
simRes = simulateRandom(sys, params,opt);
figure;
hold on;

distance = x(:, 1) - x(:, 2);
target_distance = D_Default + T_Gap * x(:,4);
unsafeSet = specification(interval([0.0, 0.0], [params.tFinal, 20.0]), 'unsafeSet');

R = reach(sys, params, options, evParams);


C = [-1]

DM = [1 -1 0 0 0 0 0 0; %Real distance in first dimension
    0 0 0 T_Gap 0 0 0 0;    % Desired distance = T_Gap * v_ego
    0 0 1 0 0 0 0 0;
    0 0 0 1 0 0 0 0;
    0 0 0 0 1 0 0 0;
    0 0 0 0 0 1 0 0;
    0 0 0 0 0 0 1 0;
    0 0 0 0 0 0 0 1;
    ] 
Db = [0;D_Default;0;0;0;0;0;0]; % Moves second dimension up by D_default
R_distances = DM * R + Db;

r1 = plotOverTime(R_distances, 1, 'DisplayName', 'Distance', 'Unify', true);

r2 = plotOverTime(R_distances, 2, 'DisplayName', 'Desired distance', 'Unify', true);
isVeri = true;
for i = 1:length(R_distances)
    for j = 1:length(R_distances(i).timeInterval.set)
            % read distances
            R_ij = R_distances(i).timeInterval.set{j};
            distance = interval(project(R_ij, 1));
            safe_distance = interval(project(R_ij, 2));

            % check safety
            isVeri = isVeri && (infimum(distance) > supremum(safe_distance));
     end
end    
if isVeri
        res = 'VERIFIED';
    else
        res = 'UNKNOWN';
    end
% Plot simulations
for i = 1:length(simRes)
    simRes_i = simRes(i);
    distance = simRes_i.x{1,1}(:, 1) - simRes_i.x{1,1}(:, 2);
    target_distance = D_Default + T_Gap * simRes_i.x{1,1}(:,4);
    plot(simRes_i.t{1,1}, target_distance(:, 1), 'r');
    plot(simRes_i.t{1,1}, distance(:, 1), 'b');
    %plot(simRes_i.t{1,1}, simRes_i.x{1,1}(:, 6),'g')
end


xlabel('Time');
ylabel('Distance');
title('System Simulation and Reachability Analysis');
legend([r1, r2], "Reachable distance", "Reachable safe distance");
axis([0, params.tFinal, 0, 130]); % Adjust the axis limits as needed

hold off;