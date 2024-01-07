% ------------------------------ BEGIN CODE -------------------------------
disp("ACC Environment")
% ----------------Starting state definition----------%
x_lead_min = 90
x_lead_max = 100

x_ego_min = 0
x_ego_max = 1


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

% Adjust this to expand the tested x_egos
x_ego_maximum = 58


% Reachability Settings ---------------------------------------------------
options.timeStep = 0.01;
options.alg = 'lin';
options.tensorOrder = 3; % Lower values reduce the computational burden
options.errorOrder = 20;
options.intermediateOrder = 20;
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
network = 65
display(network)
nn = neuralNetwork.readONNXNetwork('/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network70.onnx');
isVeri = true;
while x_ego_max <= x_ego_maximum
    x_lead_min = 90
    x_lead_max = 100
    while x_lead_max <= 110
        R0 = interval([x_lead_min;x_ego_min;v_lead_min;v_ego_min;a_lead_min;a_ego_min;T_Gap;D_Default],[x_lead_max;x_ego_max;v_lead_max;v_ego_max;a_lead_max;a_ego_max;T_Gap;D_Default]);
        params.tFinal = 5;
        params.R0 = polyZonotope(R0);
        sampling_time = 0.1
        % load neural network controller
        nn.evaluate(params.R0, evParams);
        sys = nonlinearSys(f);
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
        axis([0, params.tFinal, 0, 130]); % Adjust the axis limits as needed
        
        hold off;
        x_lead_min = x_lead_min + 5
        x_lead_max = x_lead_max + 5
    end
    
    x_ego_min = x_ego_min + 1
    x_ego_max = x_ego_max + 1
end
