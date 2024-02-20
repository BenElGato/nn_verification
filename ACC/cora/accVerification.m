% ------------------------------ BEGIN CODE -------------------------------
% Adjust x_ego starting space (should be integers from the interval [0,58])
x_ego_maximum = 21; 
x_ego_min = 20;
network = 4;
display(veri(network, x_ego_maximum, x_ego_min, true));
% Verifies whether the neural network controlled car keeps a safe distance
% to the car in front and also does not slow down unnecessarily. The
% verification fails if the reachable distance can get bigger than the
% reachable pursuit distance or smaller than the reachable safe distance
% after t=3.5 seconds
%- network_number: int, verification will use network{network_number}.onnx
%in this folder

%- x_ego_minimum = integer, Defines the minimal possible starting position of the RL controlled car
% %- x_ego_maximum = integer, Defines the maximal possible starting position of the RL controlled car
% - do_plotting: Boolean, set true for plotting graphs. Will make a
% seperate plot for every initial x_ego interval [x,x+1].
% Result: Portion of search space that could be verified
function score = veri(network_number,x_ego_maximum, x_ego_min,do_plotting)
    isValid = validateInputs(x_ego_min, x_ego_maximum);
    if ~isValid
        error('Invalid inputs. Please select integer numbers so that the resulting interval lies in [0,58].');
    end
    disp('Inputs are valid. Continuing with main functionality...');
    x_lead_min = 90;
    x_lead_max = 95;
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
    evParams.num_generators = 1000;
    %----------------------------------------------------------------------
    % Options for simulations
    %----------------------------------------------------------------------
    opt = struct;
    opt.points = 5;
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
    number_of_tests = (x_ego_maximum-x_ego_min);
    verified_tests = 0;
    while x_ego_max <= x_ego_maximum
        isVeri = true;
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
        [R,res] = reach(sys, params, options, evParams);  
        DM = [1 -1 0 0 0 0 0 0; %Real distance in first dimension
            0 0 0 T_Gap 0 0 0 0;    % Desired distance = T_Gap * v_ego
            0 0 0 3*T_Gap 0 0 0 0; % Pursuit distance = 3 * T_Gap * v_ego
            0 0 0 1 0 0 0 0;
            0 0 0 0 1 0 0 0;
            0 0 0 0 0 1 0 0;
            0 0 0 0 0 0 1 0;
            0 0 0 0 0 0 0 1;
            ];
        Db = [0;D_Default;D_Default;0;0;0;0;0]; % Add D_Default in 2. and 3. Dimesnion
        R_distances = DM * R + Db; 
        %--------------------------------------------------------------
        % Check reachable set if set didnt explode
        if res == 1
            for i = 1:length(R_distances)
                for j = 1:length(R_distances(i).timeInterval.set)
                        R_ij = R_distances(i).timeInterval.set{j};
                        distance = interval(project(R_ij, 1));
                        safe_distance = interval(project(R_ij, 2));
                        pursuit_distance = interval(project(R_ij, 3));
                        if supremum(R_distances(i).timeInterval.time{j,1}) > 3.5
                            isVeri = isVeri && (infimum(distance) > supremum(safe_distance) && (supremum(distance) < infimum(pursuit_distance)));
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
            r1 = plotOverTime(R_distances, 1, 'DisplayName', 'Distance', 'Unify', true, 'FaceColor', CORAcolor("CORA:reachSet"));
            r2 = plotOverTime(R_distances, 2, 'DisplayName', 'Forbidden distance', 'Unify', true, 'FaceColor', CORAcolor("CORA:unsafe"));
            r3 = plotOverTime(R_distances, 3, 'DisplayName', 'Forbidden distance', 'Unify', true, 'Color','r');
            xlabel('Time');
            ylabel('Distance');
            simRes = simulateRandom(sys, params,opt);
                for i = 1:length(simRes)
                simRes_i = simRes(i);
                distance = simRes_i.x{1,1}(:, 1) - simRes_i.x{1,1}(:, 2);
                target_distance = D_Default + T_Gap * simRes_i.x{1,1}(:,4);
                pursuit_distance = D_Default + 3 * T_Gap * simRes_i.x{1,1}(:,4);
                s1 = plot(simRes_i.t{1,1}, target_distance(:, 1), 'Color',CORAcolor("CORA:simulations"));
                s2 = plot(simRes_i.t{1,1}, distance(:, 1),'Color',CORAcolor("CORA:simulations"));
                s3 = plot(simRes_i.t{1,1}, pursuit_distance(:, 1),'Color',CORAcolor("CORA:simulations"));
                end
                legend([r1, r2, s1,r3], "Reachable distance", "Reachable minimal safe distance", "Simulations", "Reachable pursuit distance");
            axis([0, params.tFinal, 0, 300]);
            hold off;
        end
        if isVeri
            verified_tests = verified_tests + 1;
        end
        x_ego_min = x_ego_min + 1;
        x_ego_max = x_ego_max + 1
    end
    score = verified_tests / number_of_tests; % Portion of the initial set that could be verified
end
function isValid = validateInputs(x_ego_minimum, x_ego_maximum)
    enclosing_interval_min = 0;
    enclosing_interval_max = 58;
    isInteger = @(x) x == floor(x);
    isValid = isInteger(x_ego_minimum) && isInteger(x_ego_maximum) && ...
              x_ego_minimum >= enclosing_interval_min && ...
              x_ego_maximum <= enclosing_interval_max;
end





