% ------------------------------ BEGIN CODE -------------------------------
disp("Pendulum Environment")
% ----------------Starting state definition----------
starting_angle = pi;
thetadot = 0;
%--------------------------------------------------------------

R0 = interval([starting_angle-0.05; thetadot], [starting_angle+0.05; thetadot]);
params.tFinal = 5;
params.R0 = polyZonotope(R0);

% Parameters --------------------------------------------------------------

sampling_time = 0.01
g = 10.0 % gravity of the system
m = 1.0 % mass of the pendulum
l = 1.0 % lenght of the pendulum
max_speed = 8.0
min_speed = -max_speed
allowed_angle = 0.1

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
network = 4
display(network)
nn = neuralNetwork.readONNXNetwork('/home/benedikt/PycharmProjects/nn_verification/pendelum/cora/network4.onnx');
isVeri = true;
while starting_angle >= -pi
    R0 = interval([starting_angle-0.05; thetadot], [starting_angle+0.05; thetadot]);
    params.tFinal = 5;
    params.R0 = polyZonotope(R0);
    sys = nonlinearSys(f);
    % load neural network controller
    
    nn.evaluate(params.R0, evParams);
    
    %nn.refine(2, "layer", "both", params.R0.c, true);
    
    
    % construct neural network controlled system
    sys = neurNetContrSys(sys, nn, sampling_time);
    % Specification -----------------------------------------------------------
    safeSet = interval([-allowed_angle;-8.0], [allowed_angle;8.0]); % we want the angle to be in the upright position and don't care about velocity
    spec = specification(safeSet, 'safeSet', interval(1, 2));
    R = reach(sys, params, options, evParams);
        
    for i = 1:length(R)
       for j = 1:length(R(i).timeInterval.set)
           R_ij = R(i).timeInterval.set{j};
           theta = interval(project(R_ij, 1));
           if supremum(R(i).timeInterval.time{j,1}) > 3.5
            isVeri = isVeri && (infimum(theta) > -allowed_angle) && (supremum(theta) < allowed_angle);
           end
           if ~ isVeri
               error("Stop")
           end
       end
    end
    display(isVeri)
    params.x0 = [pi;0]; % needed for simulation
    params.tStart = 0;% needed for simulation
    
    opt = struct;
    opt.points = 50;
    simRes = simulateRandom(sys, params,opt);
    
    
    % Plotting
    figure;
    hold on;
    dim = 1
    % Plot Simulation Results
    r2 = plot(spec, [2 1])
    
    for i = 1:length(simRes)
                simRes_i = simRes(i);
                theta = simRes_i.x{1,1}(:, 1);
                plot(simRes_i.t{1,1}, theta(:, 1), 'b');
    end
    r1 = plotOverTime(R, dim, 'DisplayName', 'Reachable set', 'Unify', true);
    % Labels and Legend
    xlabel('Time');
    ylabel('Theta');
    title('System Simulation and Reachability Analysis');
    axis([0, params.tFinal, -pi, pi]); % Adjust the axis limits as needed
    legend([r1, r2], "Reachable Angle", "Desired angle");
    
    hold off;
    starting_angle = starting_angle - (pi/32)
end
