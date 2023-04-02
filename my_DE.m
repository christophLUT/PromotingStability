function [solution, fitness, fitnessDev] = my_DE(x, y, hyperparam)
% Differential Evolution (DE) for feature selection
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

%% Problem Definition
fobj = hyperparam.fobj;
noP = hyperparam.methodDEPopsize; % number of chromosomes (cadinate solutions)
nVar = size(x,2);  % number of genes (variables)
Max_iteration = hyperparam.methodDEMax_iteration; % Maximum Number of Iterations
Pc = hyperparam.methodDECrossoverProp; % Crossover Probability
beta_min=hyperparam.methodDEBetaMin; % Lower Bound of Scaling Factor
beta_max=hyperparam.methodDEBetaMax; % Upper Bound of Scaling Factor

lb = 0; % lower bound
ub = 1; % upper bound


%% Initialization

for i = 1 : noP
    population.Chromosomes(i).Gene = unifrnd(lb(1),ub(1),1,nVar);
end

% Calculate fitness values for initial population
for i = 1 : noP
    population.Chromosomes(i).fitness = fobj(population.Chromosomes(i).Gene, x, y, hyperparam); % Individual, data (no class), class labels, method (e.g. 'KNN'), objtype (e.g. 1)
end

% Find best inidividual
[~, indx] = sort([ population.Chromosomes(:).fitness ] , 'ascend'); % For MINIMIZATION

% Preprocess Fitness Results
fitnessDev = -1 * zeros(Max_iteration,3);
fitnessDev(1,:) = [1, population.Chromosomes(indx(1)).fitness, sum(round(population.Chromosomes(indx(1)).Gene))]; % assign best individual for initial population

% OPTIONAL: information on the fitness
if hyperparam.info == 1
    disp(horzcat('Iteration: ', num2str(1), ', Fitness: ', num2str(population.Chromosomes(indx(1)).fitness), ', No Features: ', num2str(sum(round(population.Chromosomes(indx(1)).Gene)))));
end

%% DE Main Loop

for t = 2 : Max_iteration
    
    % Generation of Offspring
    for k = 1 : noP
        % Selection
        parents = reshape([population.Chromosomes([datasample(setdiff(1:noP,k),3,'Replace',false)]).Gene],[nVar,3]).'; % uniform random
        
        % Mutation
        beta = unifrnd(beta_min,beta_max,[1 nVar]); % scaling factors
        trialvec = parents(1,:) + beta.*(parents(2,:)-parents(3,:)); % trial vector
        trialvec = min(max(trialvec, lb),ub); % ensure that within lower and upper bound

        % Crossover
        [child1] = crossover(population.Chromosomes(k).Gene, trialvec, Pc, hyperparam.methodDECrossover); % uniform crossover
        
        newPopulation.Chromosomes(k).Gene = child1;
    end
    
        % Calculate fitness of Offspring
    for i = 1 : noP
        newPopulation.Chromosomes(i).fitness = fobj(newPopulation.Chromosomes(i).Gene, x, y, hyperparam);
        
        % Elitism - Deterministic Elitist Replacement
        if newPopulation.Chromosomes(i).fitness < population.Chromosomes(i).fitness % MINIMIZATION
            population.Chromosomes(i).Gene = newPopulation.Chromosomes(i).Gene;
            population.Chromosomes(i).fitness = newPopulation.Chromosomes(i).fitness;
        end
    end
    
    % Save Development of Best Individual's Fitness
    [~, indx] = sort([ population.Chromosomes(:).fitness ] , 'ascend'); % For MINIMIZATION
    fitnessDev(t,:) = [t, population.Chromosomes(indx(1)).fitness, sum(round(population.Chromosomes(indx(1)).Gene))]; % assign best individual for initial population

    % OPTIONAL: information on the fitness
    if hyperparam.info == 1
        disp(horzcat('Iteration: ', num2str(t), ', Fitness: ', num2str(population.Chromosomes(indx(1)).fitness), ', No Features: ', num2str(sum(round(population.Chromosomes(indx(1)).Gene)))));
    end
  

end

% Output
solution = find(round(population.Chromosomes(indx(1)).Gene));
fitness = population.Chromosomes(indx(1)).fitness;
fitnessDev = fitnessDev(fitnessDev(:,2)>-1,:); % Retain only fitness values for which iteration was conducted

end