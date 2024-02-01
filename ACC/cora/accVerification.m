% ------------------------------ BEGIN CODE -------------------------------
disp("ACC Environment")
x_ego_maximum = 18; % int | x_ego_min < x:ego_maximum < 90
x_ego_min = 17; % int
stepsize = 1;
network = 22;
veri(network, x_ego_maximum, x_ego_min, true, sprintf("test.csv"));

% Verification function for an adapted version of the ACC Benchmark from 
% the ARCH Competition
% Lopez, Diego Manzanas, et al. "ARCH-COMP23 Category Report: Artificial Intelligence 
% and Neural Network Control Systems (AINNCS) for Continuous and Hybrid Systems Plants." 
% Proceedings of 10th International Workshop on Applied. Vol. 96. 2023.
% The adaption to the benchmark is that the possible starting states of the
% RL controlled car can be verified with the input. The system checks
% whether the safe distance between the 2 cars get violated.
%Arguments
%--------------------------------------------------------------------------
%- network_number: int, verification will use network{network_number}.onnx
% and write results into csv_file

%- x_ego_minimum = float, Defines the minimal possible starting position of the RL controlled car

% %- x_ego_maximum = float, Defines the maximal possible starting position of the RL controlled car

% - do_plotting: Boolean, set true for plotting graphs. Simulations are
% just plotted and checked if set wasn not verifiable with reachability
% analysis to save computational ressources

%-allowed_angle: The function checks whether the network will bring the
%pendulum into an upright position, with fault tolerance of |desired_angle|
%--> If desired_angle = 0.1, the system will pass the tests if the angle is
%between (-0.1,0.1) after t=3.5

%-csv_file: Path to the csv file the results should be stored in
%--------------------------------------------------------------------------
% Results
%--------------------------------------------------------------------------
%- Results will be stored in csv file
%- The first cell will be the result for the first sub-initialset, the 2.
%one for the 2.,........
%- A 1 means it was verified with reachability analysis
%- A 0 means it was not verified with reachability analysis but the
%simulations showed no violations, indicating that this subset could be
%verified with less aproximation errors in the reachability analysis or
%exploding sets
%- A -1 means that the reachability analysis detected violations of the
%specifications
function veri(network_number,x_ego_maximum, x_ego_min,do_plotting, csv_file)
    x_lead_min = 90;
    x_lead_max = 100;
    x_ego_max = x_ego_min + 1;
    v_lead_min = 32;
    v_lead_max = 32.2;
    v_ego_min = 30;
    v_ego_max = 30.2;
    a_lead_min = 0.0;
    a_lead_max = 0.0;
    a_ego_min = 0.0;
    a_ego_max = 0.0;
    T_Gap = 1.4;
    D_Default = 10.0;
    a_c_lead = -2.0;
    %----------------------------------------------------------------------
    % Reachability Settings
    % ---------------------------------------------------------------------
    options.timeStep = 0.01;
    options.alg = 'lin';
    options.tensorOrder = 3; 
    options.errorOrder = 20;
    options.intermediateOrder = 20;
    options.taylorTerms = 1; 
    options.zonotopeOrder = 20; 
    % ---------------------------------------------------------------------
    % Parameters for NN evaluation 
    % ---------------------------------------------------------------------
    evParams = struct;
    evParams.poly_method = 'regression';
    evParams.bound_approx = true;
    evParams.reuse_bounds = false;
    evParams.num_generators = 100000;
    %----------------------------------------------------------------------
    % Options for simulations
    %----------------------------------------------------------------------
    opt = struct;
    opt.points = 50;
    % ---------------------------------------------------------------------
    % System Dynamics 
    % ---------------------------------------------------------------------
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
    p = sprintf('network%d.onnx', network_number)
    nn = neuralNetwork.readONNXNetwork(p);
    number_of_tests = 4*(x_ego_maximum-x_ego_min);
    results = ones(4,(x_ego_maximum-x_ego_min));
    results = -10 * results;
    verified_tests = 0;
    x_pos = 1;
    while x_ego_max <= x_ego_maximum
        x_lead_min = 90;
        x_lead_max = 95;
        pos = 1;
        while x_lead_max <= 110
            isVeri = true;
            simVeri = true;
            R0 = interval([x_lead_min;x_ego_min;v_lead_min;v_ego_min; ...
                a_lead_min;a_ego_min;T_Gap;D_Default], ...
                [x_lead_max;x_ego_max;v_lead_max;v_ego_max ...
                ;a_lead_max;a_ego_max;T_Gap;D_Default]);
            params.tFinal = 5;
            params.R0 = polyZonotope(R0);
            params.tStart = 0;
            sampling_time = 0.1;
            
            nn.evaluate(params.R0, evParams);
            sys = nonlinearSys(f);
            sys = neurNetContrSys(sys, nn, sampling_time);
            %params.x0 = [x_lead_min;x_ego_min;v_lead_min;v_ego_min;
             %   a_lead_min;a_ego_min;T_Gap;D_Default]; 
            
            
            [R,res] = reach(sys, params, options, evParams);
            C = [-1];
            
            DM = [1 -1 0 0 0 0 0 0; %Real distance in first dimension
                0 0 0 T_Gap 0 0 0 0;    % Desired distance = T_Gap * v_ego
                0 0 1 0 0 0 0 0;
                0 0 0 1 0 0 0 0;
                0 0 0 0 1 0 0 0;
                0 0 0 0 0 1 0 0;
                0 0 0 0 0 0 1 0;
                0 0 0 0 0 0 0 1;
                ];
            Db = [0;D_Default;0;0;0;0;0;0]; % Moves second dimension up by D_default
            R_distances = DM * R + Db;
        
            %--------------------------------------------------------------
            % Check reachable set if set didnt explode
            if res == 1
                for i = 1:length(R_distances)
                    for j = 1:length(R_distances(i).timeInterval.set)
                            R_ij = R_distances(i).timeInterval.set{j};
                            distance = interval(project(R_ij, 1));
                            safe_distance = interval(project(R_ij, 2));
                            tp = supremum(R_distances(i).timeInterval.time{j,1});
                            if supremum(R_distances(i).timeInterval.time{j,1}) > 3.5
                                isVeri = isVeri && (infimum(distance) > supremum(safe_distance));
                            end
                            if ~isVeri
                                break
                            end
                     end
                end
            else
                isVeri = false;
            end
            %--------------------------------------------------------------
            if do_plotting
                figure;
                hold on;
                r1 = plotOverTime(R_distances, 1, 'DisplayName', 'Distance', 'Unify', true);
                r2 = plotOverTime(R_distances, 2, 'DisplayName', 'Desired distance', 'Unify', true);
            end
            %-------------------------------------------------------------
            % Check simulations if set could not be verified 
            if ~isVeri
                simRes = simulateRandom(sys, params,opt);
                for i = 1:length(simRes)
                    simRes_i = simRes(i);
                    distance = simRes_i.x{1,1}(:, 1) - simRes_i.x{1,1}(:, 2);
                    target_distance = D_Default + T_Gap * simRes_i.x{1,1}(:,4);
                    if do_plotting
                        plot(simRes_i.t{1,1}, target_distance(:, 1), 'r');
                        plot(simRes_i.t{1,1}, distance(:, 1), 'b');
                    end
                    
                    for j = 1:length(simRes(i).t{1,1})
                        time = simRes_i.t{1,1}(j,1);
                        if simRes_i.t{1,1}(j,1) > 3.5
                            distance = simRes_i.x{1,1}(j, 1) - simRes_i.x{1,1}(j, 2);
                            target_distance = D_Default + T_Gap * simRes_i.x{1,1}(j,4);
                            simVeri = simVeri && distance >= target_distance;
                        end
                    end
                end
            end
            %--------------------------------------------------------------
            if do_plotting
                xlabel('Time');
                ylabel('Distance');
                str = sprintf('Ego min: %f, Ego max: %f, Lead min: %f, Lead max: %f, Verified: %f', x_ego_min, x_ego_max, x_lead_min, x_lead_max, isVeri);
                title(str);
                legend([r1, r2], "Reachable distance", "Reachable safe distance");
                axis([0, params.tFinal, 0, 130]);
                hold off;
            end
            if ~simVeri
                results(pos,x_pos ) = -1
            else
                if ~isVeri
                    results(pos, x_pos) = 0
                else
                    results(pos, x_pos) = 1
                    verified_tests = verified_tests + 1;
                end
            end
            
            x_lead_min = x_lead_min + 5;
            x_lead_max = x_lead_max + 5;
            pos = pos + 1;
        end
        
        x_ego_min = x_ego_min + 1;
        x_ego_max = x_ego_max + 1
        x_pos = x_pos + 1;
    end
    score = verified_tests / number_of_tests
    xlswrite(csv_file, results);
end




