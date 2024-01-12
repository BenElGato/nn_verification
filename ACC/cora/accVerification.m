% ------------------------------ BEGIN CODE -------------------------------
disp("ACC Environment")
veri(26);
veri(32);
veri(36);
veri(61);
veri(64);
veri(68);
veri(70);
% ----------------Starting state definition----------%
function veri(network_number)
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


    results = ones(4,x_ego_maximum);
    results = -10 * ones
    
    while x_ego_max <= x_ego_maximum
        x_lead_min = 90
        x_lead_max = 95
    
        pos = 1;
        while x_lead_max <= 110
            isVeri = true;
            simVeri = true;
            R0 = interval([x_lead_min;x_ego_min;v_lead_min;v_ego_min;a_lead_min;a_ego_min;T_Gap;D_Default],[x_lead_max;x_ego_max;v_lead_max;v_ego_max;a_lead_max;a_ego_max;T_Gap;D_Default]);
            params.tFinal = 5;
            params.R0 = polyZonotope(R0);
            sampling_time = 0.1
            % load neural network controller
            
            p = sprintf('/home/benedikt/PycharmProjects/nn_verification/ACC/cora/network%d.onnx', network_number)
            nn = neuralNetwork.readONNXNetwork(p);
            nn.evaluate(params.R0, evParams);
            sys = nonlinearSys(f);
            sys = neurNetContrSys(sys, nn, sampling_time);
            params.x0 = [x_lead_min;x_ego_min;v_lead_min;v_ego_min;a_lead_min;a_ego_min;T_Gap;D_Default]; % needed for simulation
            params.tStart = 0;% needed for simulation
            [t, x] = simulate(sys, params);
            opt = struct;
            opt.points = 50;
            simRes = simulateRandom(sys, params,opt);
            
            %figure;
            %hold on;
            
            
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
            
            %r1 = plotOverTime(R_distances, 1, 'DisplayName', 'Distance', 'Unify', true);
            
            %r2 = plotOverTime(R_distances, 2, 'DisplayName', 'Desired distance', 'Unify', true);
            
            % Plot simulations
            %for i = 1:length(simRes)
             %   simRes_i = simRes(i);
              %  distance = simRes_i.x{1,1}(:, 1) - simRes_i.x{1,1}(:, 2);
               % target_distance = D_Default + T_Gap * simRes_i.x{1,1}(:,4);
                %plot(simRes_i.t{1,1}, target_distance(:, 1), 'r');
                %plot(simRes_i.t{1,1}, distance(:, 1), 'b');
            %end
            
            
            for i = 1:length(R_distances)
                for j = 1:length(R_distances(i).timeInterval.set)
                        % read distances
                        R_ij = R_distances(i).timeInterval.set{j};
                        distance = interval(project(R_ij, 1));
                        safe_distance = interval(project(R_ij, 2));
            
                         %check safety
                        tp = supremum(R_distances(i).timeInterval.time{j,1});
                        if supremum(R_distances(i).timeInterval.time{j,1}) > 3.5
                            isVeri = isVeri && (infimum(distance) > supremum(safe_distance));
                        end
                        if ~ isVeri
                            display(supremum(R_distances(i).timeInterval.time{j,1}))
                            display(x_ego_min)
                            display(x_ego_max)
                            display(x_lead_min)
                           display(x_lead_min)
                            %error("Stop")
                        end
                 end
            end
            for i = 1:length(simRes)
                simRes_i = simRes(i);
                distance = simRes_i.x{1,1}(:, 1) - simRes_i.x{1,1}(:, 2);
                target_distance = D_Default + T_Gap * simRes_i.x{1,1}(:,4);
                %plot(simRes_i.t{1,1}, target_distance(:, 1), 'r');
                %plot(simRes_i.t{1,1}, distance(:, 1), 'b');
                for j = 1:length(simRes(i).t{1,1})
                    time = simRes_i.t{1,1}(j,1)
                    if simRes_i.t{1,1}(j,1) > 3.5
                        distance = simRes_i.x{1,1}(j, 1) - simRes_i.x{1,1}(j, 2);
                        target_distance = D_Default + T_Gap * simRes_i.x{1,1}(j,4);
                        simVeri = simVeri && distance >= target_distance;
                    end
                end
            end
            %xlabel('Time');
            %ylabel('Distance');
            %str = sprintf('Ego min: %f, Ego max: %f, Lead min: %f, Lead max: %f, Verified: %f', x_ego_min, x_ego_max, x_lead_min, x_lead_max, isVeri);
            %title(str);
            %legend([r1, r2], "Reachable distance", "Reachable safe distance");
            %axis([0, params.tFinal, 0, 130]); % Adjust the axis limits as needed
            
            %hold off;
            
            if ~simVeri
                results(pos,x_ego_max ) = -1
            else
                if ~isVeri
                    results(pos, x_ego_max) = 0
                else
                    results(pos, x_ego_max) = 1
                end
            end
            
            x_lead_min = x_lead_min + 5
            x_lead_max = x_lead_max + 5
            pos = pos + 1;
        end
        
        x_ego_min = x_ego_min + 1
        x_ego_max = x_ego_max + 1
    end
    filename = sprintf('/home/benedikt/PycharmProjects/nn_verification/ACC/cora/results%d.xlsx',network_number);
    xlswrite(filename, results);
end




