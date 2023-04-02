function [solution, fitness, fitnessDev]  = my_GA(x, y, hyperparam)
% Genetic Algorithm (GA)(for Minimization) for feature selection
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
noP = hyperparam.methodGAPopsize; % number of chromosomes (candidate solutions)
nVar = size(x,2);  % number of genes (variables)
Max_iteration = hyperparam.methodGAMax_iteration;
Pc = hyperparam.methodGACrossoverProp; % Crossover Probability
Pm = hyperparam.methodGAMutationProp; % Mutation Probability
Er = hyperparam.methodGAElitismShare;

lb = 0; % lower bound
ub = 1; % upper bound


%%  Random Initialization
for i = 1 : noP
    
    % Initialize genes [0,1]
    population.Chromosomes(i).Gene = unifrnd(lb(1),ub(1),1,nVar);
    
    % Calculate fitness values for initial population
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

%% Main loop
for t = 2 : Max_iteration
    
    % Generation of Offspring
    for k = 1: 2: noP
        % Selection
        [parent1, parent2] = selection(population); % for MINIMIZATION, fitness-proportional
        
        % Crossover
        [child1 , child2] = crossover(parent1 , parent2, Pc, hyperparam.methodGACrossover); % one-point or two-point crossover
        
        % Mutation
        [child1] = mutation(child1, Pm, hyperparam.methodGAMutation); % uniform mutation [real-valued]
        [child2] = mutation(child2, Pm, hyperparam.methodGAMutation); % uniform mutation [real-valued]
        
        newPopulation.Chromosomes(k).Gene = child1.Gene;
        newPopulation.Chromosomes(k+1).Gene = child2.Gene;
    end
 
    % Calculate fitness of Offspring
    for i = 1 : noP
        newPopulation.Chromosomes(i).fitness = fobj(newPopulation.Chromosomes(i).Gene, x, y, hyperparam);
    end
    
    % Elitism
    [ newPopulation ] = elitism(population, newPopulation, Er); % For MINIMIZATION
    population = newPopulation; % replace the previous population with the newly made
    
    % Save Development of Best Individual's Fitness
    [~, indx] = sort([ population.Chromosomes(:).fitness ] , 'ascend'); % For MINIMIZATION
    fitnessDev(t,:) = [t, population.Chromosomes(indx(1)).fitness, sum(round(population.Chromosomes(indx(1)).Gene))]; % assign best individual for initial population

    % OPTIONAL: information on the fitness
    if hyperparam.info == 1
        disp(horzcat('Iteration: ', num2str(t), ', Best Fitness: ', num2str(population.Chromosomes(indx(1)).fitness),', Mean Fitness: ' , num2str(mean([population.Chromosomes(:).fitness])),', No Features: ', num2str(sum(round(population.Chromosomes(indx(1)).Gene)))));
    end
end
   
% Output
solution = find(round(population.Chromosomes(indx(1)).Gene));
fitness = population.Chromosomes(indx(1)).fitness;
fitnessDev = fitnessDev(fitnessDev(:,2)>-1,:); % Retain only fitness values for which iteration was conducted
end