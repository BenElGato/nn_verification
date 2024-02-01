% ------------------------------ BEGIN CODE -------------------------------
max_angle =  pi ;
min_angle = -pi;
step_size = 2* pi;
allowed_angle = 0.1;
network = 8;
csv_file = sprintf("network%dq.csv", network)
veri(network, max_angle, min_angle, step_size, true, allowed_angle, csv_file)
% Verification function for an adapted version of the pendulum gymnasium
% environment. Obersvation space:[Theta, Thetadot]
%Arguments
%--------------------------------------------------------------------------
%- network_number: int, verification will use network{network_number}.onnx
% and write results into csv_file

%- starting_angle = float, Defines the angle at which the reachability analysis
%start (Default: Pi)

% - Min_angle: float, Starting angle until which the initial set should be
% verified

% - step_size: float, Defines how big the initial set is in every reachset
% calculation --> Initial set goes always from [angle-step_size, angle].
% Each subset is then calulated seperatly until the whole space between
% [min_angle,starting_angle] is checked

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
function veri(network_number, starting_angle,min_angle, step_size, do_plotting,allowed_angle,csv_file)
    disp("Pendulum Environment")
    thetadot = 0;
    R0 = interval([starting_angle-step_size; thetadot], [starting_angle; thetadot]);
    params.tFinal = 5;
    params.R0 = polyZonotope(R0);
    sampling_time = 0.01;
    g = 10.0; 
    m = 1.0; 
    l = 1.0; 
    %----------------------------------------------------------------------
    % Reachability Settings
    options.timeStep = 0.001;
    options.alg = 'lin';
    options.tensorOrder = 2; 
    options.taylorTerms = 1; 
    options.zonotopeOrder = 80;
%----------------------------------------------------------------
    % Parameters for NN evaluation
    evParams = struct;
    evParams.poly_method = 'regression';
    evParams.bound_approx = true;
    evParams.reuse_bounds = false;
    evParams.num_generators = 100000;
    %----------------------------------------------------------------------
    % System Dynamics 
    f = @(x, u) [
         x(2);
         ((3*g)/(2*l))*sin(x(1))+(3/(m*l^2)*u(1))
        ];  
    %----------------------------------------------------------------------
    number_of_tests = (starting_angle - min_angle) / step_size;
    number_of_tests = ceil(number_of_tests - 0.0001);
    verified_tests = 0;
    results = ones(1,number_of_tests);
    results = -10 * results;
    nn = neuralNetwork.readONNXNetwork(sprintf('network%d.onnx',network_number));
    pos = 1
    while pos <= number_of_tests
        isVeri = true;
        simVeri = true;
        R0 = interval([starting_angle-step_size; thetadot], [starting_angle; thetadot])
        params.tFinal = 5;
        params.R0 = polyZonotope(R0);
        sys = nonlinearSys(f);
        nn.evaluate(params.R0, evParams);
        sys = neurNetContrSys(sys, nn, sampling_time);
        safeSet = interval([-allowed_angle;-8.0], [allowed_angle;8.0]); 
        spec = specification(safeSet, 'safeSet', interval(1, 2));
        [R,res] = reach(sys, params, options, evParams);
       
        
        %------------------------------------------------------------------ 
        % Check reachability analysis results
        if res == 1
            for i = 1:length(R)
               for j = 1:length(R(i).timeInterval.set)
                   R_ij = R(i).timeInterval.set{j};
                   theta = interval(project(R_ij, 1));
                   if supremum(R(i).timeInterval.time{j,1}) > 3.5
                    isVeri = isVeri && (infimum(theta) > -allowed_angle) && (supremum(theta) < allowed_angle);
                    if ~isVeri
                         break
                    end
                   end
               end
            end
        else
            isVeri = false
        end
        %------------------------------------------------------------------
        %params.x0 = [pi;0]; 
        params.tStart = 0;
        
        opt = struct;
        opt.points = 50;
       
        dim = 1;
        if do_plotting
            figure;
            hold on;
            r2 = plot(spec, [2 1]);
            r1 = plotOverTime(R, dim, 'DisplayName', 'Reachable set', 'Unify', true);
            
        end
        % Check for violations in simulations
        %--------------------------------------------------------------
        if ~isVeri
            simRes = simulateRandom(sys, params,opt);
            for i = 1:length(simRes)
                        simRes_i = simRes(i);
                        theta = simRes_i.x{1,1}(:, 1);
                        if do_plotting
                            plot(simRes_i.t{1,1}, theta(:, 1), 'b');
                        end
                        
                        for j = 1:length(simRes(i).t{1,1})
                            time = simRes_i.t{1,1}(j,1);
                            if simRes_i.t{1,1}(j,1) > 3.5
                                theta = simRes_i.x{1,1}(j, 1);
                                simVeri = simVeri && (abs(theta) < abs(allowed_angle));
                                if ~simVeri
                                    break
                                end
                            end
                        end
            end
        end
        %--------------------------------------------------------------  
        if do_plotting
            xlabel('Time');
            ylabel('Theta');
            title('System Simulation and Reachability Analysis');
            axis([0, params.tFinal, -pi, pi]); 
            legend([r1, r2], "Reachable Angle", "Desired angle");
            %matlab2tikz();
            hold off;
        end
        if ~simVeri
                    results(1,pos) = -1;
                else
                    if ~isVeri
                        results(1, pos) = 0;
                    else
                        results(1, pos) = 1
                        verified_tests = verified_tests + 1;
                    end
        end
        
        starting_angle = starting_angle - step_size
        pos = pos + 1;
    end
    xlswrite(csv_file, results);
    score = verified_tests / number_of_tests
end
