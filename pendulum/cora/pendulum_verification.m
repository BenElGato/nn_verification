% Adjust starting angles as desired, the range must be dividable by the
% stepsize! Maximal starting range for thetas: [-pi,pi]
max_angle =  0.5*pi;
min_angle = 0.5*pi-(pi / 16);
step_size = (pi / 16);
allowed_angle = 0.1;
network = 8;
veri(network, max_angle, min_angle, step_size, true, allowed_angle)
% Verification function for an adapted version of the pendulum gymnasium
% environment. Obersvation space:[Theta, Thetadot]
%Arguments
%--------------------------------------------------------------------------
%- network: int, verification will use network{network_number}.onnx
%- starting_angle = float, Defines the angle at which the reachability analysis
%start: 

% - Min_angle: float, Starting angle until which the initial set should be
% verified

% - step_size: float, Defines how big the initial set is in every reachset
% calculation --> Initial set goes always from [angle-step_size, angle].
% Each subset is then calulated seperatly until the whole space between
% [min_angle,starting_angle] is checked

% - do_plotting: Boolean, set true for plotting graphs. Plots each
% subverification problem as a seperate graph.

%-allowed_angle: The function checks whether the network will bring the
%pendulum into an upright position, with fault tolerance of |allowed_angle|
%--> If allowed_angle = 0.1, the system will pass the tests if the angle is
%between (-0.1,0.1) after t=3.5
function score = veri(network_number, starting_angle,min_angle, step_size, do_plotting,allowed_angle)
    isValid = validateInputs(min_angle, starting_angle, step_size);
    if isValid
        disp('Inputs are valid.');
    else
        error('Inputs are invalid. Step size does not cover the entire interval. Please choose a stepsize that is a multiple of pi and smaller than the interval of starting angles!');
    end
    thetadot = 0;
    R0 = interval([starting_angle-step_size; thetadot], [starting_angle; thetadot]);
    params.tFinal = 1.2;
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
    nn = neuralNetwork.readONNXNetwork(sprintf('network%d.onnx',network_number));
    pos = 1
    while pos <= number_of_tests
        isVeri = true;
        R0 = interval([starting_angle-step_size; thetadot], [starting_angle; thetadot])
        params.R0 = polyZonotope(R0);
        sys = nonlinearSys(f);
        nn.evaluate(params.R0, evParams);
        sys = neurNetContrSys(sys, nn, sampling_time);
        safeSet = interval([-allowed_angle;-8.0], [allowed_angle;8.0]); 
        spec = specification(safeSet, 'safeSet', interval(1, 2));
        [R,res] = reach(sys, params, options, evParams); 
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
        params.tStart = 0;
        
        if do_plotting
            figure;
            hold on;
            r2 = plot(spec, [2 1], 'FaceColor', CORAcolor("CORA:safe"));
            r1 = plotOverTime(R, 1, 'DisplayName', 'Reachable set', 'Unify', true, 'FaceColor', CORAcolor("CORA:reachSet"));
            xlabel('Time');
            ylabel('Theta');
            axis([0, 5, -pi, pi]); 
            opt = struct;
            opt.points = 50; 
            params.tFinal = 5;
            simRes = simulateRandom(sys, params,opt);
            s1 = plotOverTime(simRes, 1, 'DisplayName', 'Simulations', 'Color', CORAcolor("CORA:simulations"));
            legend([r1, r2, s1], "Reachable Angle", "Desired angle","Simulations"); 
        end
        if isVeri         
           verified_tests = verified_tests + 1;           
        end
        
        starting_angle = starting_angle - step_size
        pos = pos + 1;
    end
    score = verified_tests / number_of_tests
end

function isValid = validateInputs(min_angle, max_angle, step_size)
    total_range = max_angle - min_angle;
    tolerance = 1e-10;
    isValid = abs(mod(total_range, step_size)) < tolerance;
end

