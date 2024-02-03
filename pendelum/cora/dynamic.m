allowed_angle = 0.1;
network = 1;
veri(network, false, allowed_angle)

function veri(network_number, do_plotting,allowed_angle)
    thetadot = 0;
    params.tFinal = 5;
    sampling_time = 0.01;
    g = 10.0; 
    m = 1.0; 
    l = 1.0; 
    %----------------------------------------------------------------------
    % Reachability Settings
    options.timeStep = 0.01;
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
    evParams.num_generators = 1000;
    %----------------------------------------------------------------------
    % System Dynamics 
    f = @(x, u) [
         x(2);
         ((3*g)/(2*l))*sin(x(1))+(3/(m*l^2)*u(1))
        ];  
    %----------------------------------------------------------------------
    smallest_size = (pi / 128)
    number_of_tests = (2 * pi) / smallest_size;
    number_of_tests = ceil(number_of_tests - 0.0001);
    verified_tests = 0;
    nn = neuralNetwork.readONNXNetwork(sprintf('network%d.onnx',network_number));
    starting_angle = pi
    min_angle = -pi
    while starting_angle > min_angle
        isVeri = true;
        res = 0
        step_size = 4 * pi
        
        while res ~= 1 & step_size >= smallest_size
            step_size = step_size / 2;
            while starting_angle - step_size < -pi
                step_size = step_size / 2
            end
            R0 = interval([starting_angle-step_size; thetadot], [starting_angle; thetadot])
            params.tFinal = 5;
            params.R0 = polyZonotope(R0);
            sys = nonlinearSys(f);
            nn.evaluate(params.R0, evParams);
            sys = neurNetContrSys(sys, nn, sampling_time);
            safeSet = interval([-allowed_angle;-8.0], [allowed_angle;8.0]); 
            spec = specification(safeSet, 'safeSet', interval(1, 2));
            [R,res] = reach(sys, params, options, evParams);
            
        end
        starting_angle = starting_angle - step_size 
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
        
        dim = 1;
        if do_plotting
            figure;
            hold on;
            r2 = plot(spec, [2 1]);
            r1 = plotOverTime(R, dim, 'DisplayName', 'Reachable set', 'Unify', true);
            
            xlabel('Time');
            ylabel('Theta');
            title('System Simulation and Reachability Analysis');
            axis([0, params.tFinal, -pi, pi]); 
            legend([r1, r2], "Reachable Angle", "Desired angle");
            %matlab2tikz();
            hold off;
        end
        
         if isVeri
             display("Verified size")
             display(step_size)
             display("Verified tests")
             display((step_size / smallest_size))
            verified_tests = verified_tests + (step_size / smallest_size)
         end
    end
    score = verified_tests / number_of_tests
end
