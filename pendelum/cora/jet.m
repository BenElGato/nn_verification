disp("Jet Environment")
R0 = interval([0; -0.2; 0; -50.0; 0; 0;0], [0; 0.2; 0; 0; 50.0; 0;0]);
params.tFinal = 5;
params.R0 = polyZonotope(R0);
sampling_time = 0.01

options.timeStep = 0.1;
options.alg = 'lin';
options.tensorOrder = 2;
options.taylorTerms = 4;
options.zonotopeOrder = 20;


evParams = struct;
evParams.poly_method = 'regression';
evParams.bound_approx = true;
evParams.reuse_bounds = false;
evParams.num_generators = 100000


f = @(x, u) [
    u(1);
    x(1) + u(1); 
    cos(x(1) + u(1));
    sin(x(1) + u(1)); 
    0; 
    cos(x(7)); 
    sin(x(7))
    ];

sys = nonlinearSys(f);
nn = neuralNetwork.readONNXNetwork('/home/benedikt/PycharmProjects/nn_verification/jets/cora/network.onnx');
nn.evaluate(params.R0, evParams);
nn.refine(2, "layer", "both", params.R0.c, true);

params.x0 = [0;-1;0;0;50.0;0;0]; % needed for simulation
params.tStart = 0;
sys = neurNetContrSys(sys, nn, sampling_time);

[t, x] = simulate(sys, params);
delta_x = x(:,3) - x(:,5)
delta_y = x(:,4) - x(:,6)
distance = sqrt(delta_x(:,1).^2 + delta_y(:,1).^2) 
R = reach(sys, params, options, evParams);
projectedSet_x = project(R, [3]);   
projectedSet_y = project(R, [4]);   
projectedSet_x_jet = project(R, [5]); 
projectedSet_y_jet = project(R, [6]); 
distances = [];
for i = 1:length(R(1,1).timePoint.set)
    % Extract the zonotopes at the current time step
    Z_x = projectedSet_x(1,1).timePoint.set{i};
    Z_y = projectedSet_y(1,1).timePoint.set{i};
    Z_x_jet = projectedSet_x_jet(1,1).timePoint.set{i};
    Z_y_jet = projectedSet_y_jet(1,1).timePoint.set{i};

    distance = ?;
    distances = [distances; distance];
end

timePoints = R(1,1).timePoint.set{1}.timeInterval; % Assuming time points are the same for all projections
plot(timePoints, distances);
xlabel('Time');
ylabel('Distance');
title('Reachable Distance Over Time');


%plot(t, distance(:, 1), 'b', 'DisplayName', 'Simulation');
