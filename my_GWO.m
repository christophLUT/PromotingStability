function [solution, fitness, fitnessDev] = my_GWO(x, y, hyperparam)
% Grey Wolf Optimization (GWO) for feature selection
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

%% Paramters of the GA algorithm
fobj = hyperparam.fobj;
noP = hyperparam.methodGWOPopsize; % number of chromosomes (candidate solutions)
nVar = size(x,2);  % number of genes (variables)
Max_iteration = hyperparam.methodGWOMax_iteration;

lb = 0; % lower bound
ub = 1; % upper bound

%%  Random Initialization
for i = 1 : noP
    population.wolf(i).position = unifrnd(lb(1),ub(1),1,nVar);
    population.wolf(i).fitness = fobj(population.wolf(i).position, x, y, hyperparam); % Individual, data (no class), class labels, method (e.g. 'KNN'), objtype (e.g. 1)
end


%% Determine initial alpha, beta, and delta_pos
[~, indx] = sort([ population.wolf(:).fitness ] , 'ascend'); % For MINIMIZATION

Alpha_score = population.wolf(indx(1)).fitness; % Update alpha
Alpha_pos = population.wolf(indx(1)).position;

Beta_score = population.wolf(indx(2)).fitness; % Update beta
Beta_pos = population.wolf(indx(2)).position;

Delta_score = population.wolf(indx(3)).fitness; % Update delta
Delta_pos = population.wolf(indx(3)).position;

% Preprocess Fitness Results
fitnessDev = -1 * zeros(Max_iteration,3);

%% Main loop
for t = 1 : Max_iteration
      
    % Update a coefficient
    a = 2 - (t-1) * ((2) / Max_iteration); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i = 1 : noP
        
        % Generate A and C coefficient matrices
        A = 2 * a * rand(3,nVar) - a; % rand() for r1 in [0,1]
        C = 2 * rand(3,nVar); % rand() for r2 in [0,1]

        D_alpha = abs(C(1,:) .* Alpha_pos - population.wolf(i).position);
        X1 = Alpha_pos - A(1,:) .* D_alpha;

        D_beta = abs(C(2,:) .* Beta_pos - population.wolf(i).position);
        X2 = Beta_pos - A(2,:) .* D_beta;

        D_delta = abs(C(3,:) .* Delta_pos - population.wolf(i).position);
        X3 = Delta_pos - A(3,:) .* D_delta; % Equation (3.5)-part 3   
        
        % Update Position of Omega wolves based on alpha, beta and delta wolves's distance to "prey"
        population.wolf(i).position = (X1 + X2 + X3) / 3;         
    end
    
    for i = 1 : noP
        
        % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub = population.wolf(i).position > ub;
        Flag4lb = population.wolf(i).position < lb;
        population.wolf(i).position = (population.wolf(i).position .* (~ (Flag4ub + Flag4lb) )) + ub .* Flag4ub + lb .* Flag4lb; 
        
        % Determine the fitness of the omega wolves
        population.wolf(i).fitness = fobj(population.wolf(i).position, x, y, hyperparam); % Individual, data (no class), class labels, method (e.g. 'KNN'), objtype (e.g. 1)
        
        % Update Alpha, Beta, and Delta
        if population.wolf(i).fitness < Alpha_score 
            Alpha_score = population.wolf(i).fitness; % Update alpha
            Alpha_pos = population.wolf(i).position;
        end
        
        if population.wolf(i).fitness > Alpha_score && population.wolf(i).fitness < Beta_score 
            Beta_score = population.wolf(i).fitness; % Update beta
            Beta_pos = population.wolf(i).position;
        end
        
        if population.wolf(i).fitness > Alpha_score && population.wolf(i).fitness > Beta_score && population.wolf(i).fitness < Delta_score 
            Delta_score = population.wolf(i).fitness; % Update delta
            Delta_pos = population.wolf(i).position;
        end
    end

    % Save Development of Best Individual's Fitness
    fitnessDev(t,:) = [t, Alpha_score, sum(round(Alpha_pos))]; % assign best individual for initial population

    % OPTIONAL: information on the fitness
    if hyperparam.info == 1
        disp(horzcat('Iteration: ', num2str(t), ', Best Fitness: ', num2str(Alpha_score),', Mean Fitness: ' , num2str(mean([population.wolf(:).fitness])),', No Features: ', num2str(sum(round(Alpha_pos)))));
    end
    

end

% Output
solution = find(round(Alpha_pos));
fitness = Alpha_score;
fitnessDev = fitnessDev(fitnessDev(:,2)>-1,:); % Retain only fitness values for which iteration was conducted

end