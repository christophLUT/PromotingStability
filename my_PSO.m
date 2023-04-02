function [solution, fitness, fitnessDev] = my_PSO(x, y, hyperparam)
% Particle Swarm Optimization (PSO) for feature selection
%
% INPUT
%   x: Input data (row observations, columns features) without the
%      dependent variable (here: class labels)
%   y: Dependent variable (here: vector of class labels)
%   hyperparam: contains all the hyperparameters and selections (e.g., for
%       methods and crossvalidation)
%
% OUTPUT
%   solution: best solution found until termination
%   fitness: fitness of the best solution found until termination
%   fitnessDev: best fitness values in each iteration until termination


%% Parameters
noP = hyperparam.methodPSOPopsize;
nVar = size(x,2);

% Objective function details
fobj = hyperparam.fobj;

% PSO paramters
Max_iteration = hyperparam.methodPSOMax_iteration; % number of iterations of the algorithm
Vmax = hyperparam.methodPSOVelocityMax; % 1 default [maximum Velocity]
wMax = hyperparam.methodPSOInertiaWeightMax; %0.9 default [maximum inertia weight]
wMin = hyperparam.methodPSOInertiaWeightMin; % 0.2 default [minmum inertia weight]
c1 = hyperparam.methodPSOc1; % 2 default [coefficient for personal / cognitive component]
c2 = hyperparam.methodPSOc2; % 2 default [coefficient for global / social component]

lb = 0 * ones(1,nVar); % lower limit for the Position of particles
ub = 1 * ones(1,nVar); % upper limit for the Position of particles

%% Random Initialization

% Initializations
for k = 1: noP
    Swarm.Particles(k).X = unifrnd(lb(1),ub(1),1,nVar);
    Swarm.Particles(k).V = (2*rand(ub(1)-lb(1),nVar)-(ub(1)-lb(1))); % according to Engelbrecht (2012) [-0.1, 0.1] % 0.1 * (2*rand(1,nVar)-1);
    Swarm.Particles(k).PBEST.X = Swarm.Particles(k).X;
    Swarm.Particles(k).PBEST.O = inf; % for minimization problems
end

Swarm.GBEST.X = zeros(1,nVar);
Swarm.GBEST.O = inf;

% Preprocess Fitness Results
fitnessDev = -1 * zeros(Max_iteration,3);

%% Iterations of PSO
for t = 1 : Max_iteration % main loop
    for k = 1 : noP
        %Calculate objective function for each particle
        Swarm.Particles(k).O = fobj(Swarm.Particles(k).X, x, y, hyperparam); % Particle, data (no class), class labels, method (e.g. 'KNN'), objtype (e.g. 1)
        
        % Update personal best (if better)
        if(Swarm.Particles(k).O < Swarm.Particles(k).PBEST.O)
            Swarm.Particles(k).PBEST.O = Swarm.Particles(k).O;
            Swarm.Particles(k).PBEST.X = Swarm.Particles(k).X;
        end
        
        % Update global best (if better)
        if(Swarm.Particles(k).O < Swarm.GBEST.O)
            Swarm.GBEST.O = Swarm.Particles(k).O;
            Swarm.GBEST.X = Swarm.Particles(k).X;
        end
    end
    
    % Update the inertia weight (associated with the Velocity)
    w=wMax-(t-1)*((wMax-wMin)/Max_iteration);
    
    % Update the Velocity and Position of all particles
    for k = 1 : noP
        
        % Update of the Velocity of the particle
        Swarm.Particles(k).V  = w .* Swarm.Particles(k).V + ...  % inertia
            c1 .* rand(1,nVar) .* (Swarm.Particles(k).PBEST.X - Swarm.Particles(k).X ) +  ...   % congnitive
            c2 .* rand(1,nVar) .* (Swarm.GBEST.X - Swarm.Particles(k).X) ;  % social
        
        % Correction velocity to be within upper and lower limit Velocity limits (-Vmax, Vmax)       
        Swarm.Particles(k).V = min(max(Swarm.Particles(k).V,-Vmax),Vmax);
        
        % Update of the Position of the particle (current Position + updated Velocity)
        Swarm.Particles(k).X = Swarm.Particles(k).X + Swarm.Particles(k).V;

        % Correct Position to be within the upper and lower bound (lb, ub)
        Swarm.Particles(k).X = min(max(Swarm.Particles(k).X,lb),ub);   
        
    end
    
    % Save development of best fitness
    fitnessDev(t,:) = [t, Swarm.GBEST.O, sum(round(Swarm.GBEST.X))];
    
    % OPTIONAL: information on the fitness
    if hyperparam.info == 1
        disp(horzcat('Iteration: ', num2str(t), ', Fitness: ', num2str(Swarm.GBEST.O), ', No Features: ', num2str(sum(round(Swarm.GBEST.X)))));
    end
end

%% Output
solution = find(round(Swarm.GBEST.X)); % best solution
fitness = Swarm.GBEST.O; % fitness of best solution
fitnessDev = fitnessDev(fitnessDev(:,2)>-1,:); % Retain only fitness values for which iteration was conducted

end