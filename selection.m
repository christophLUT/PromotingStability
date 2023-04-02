function [parent1, parent2] = selection(population)
M = length(population.Chromosomes(:));

% Fitness values normalized
if any([population.Chromosomes(:).fitness] < 0 ) 
    % Fitness scaling in case of negative values scaled(f) = a * f + b
    a = 1;
    b = abs( min(  [population.Chromosomes(:).fitness] )  );
    Scaled_fitness = a *  [population.Chromosomes(:).fitness] + b;
    
%     normalized_fitness = [Scaled_fitness] ./ sum([Scaled_fitness]); % for MAXIMIZATION
    normalized_fitness = [1./Scaled_fitness] ./ sum([1./Scaled_fitness]); % for MINIMIZATION
else
%     normalized_fitness = [population.Chromosomes(:).fitness] ./ sum([population.Chromosomes(:).fitness]); % for MAXIMIZATION
    normalized_fitness = (1./[population.Chromosomes(:).fitness]) ./ sum((1./[population.Chromosomes(:).fitness])); % for MINIMIZATION
end


% Ordered
[~ , sorted_idx] = sort(normalized_fitness , 'descend'); 

% Create (temporary) sorted population
for i = 1 : length(population.Chromosomes)
    temp_population.Chromosomes(i).Gene = population.Chromosomes(sorted_idx(i)).Gene;
    temp_population.Chromosomes(i).fitness = population.Chromosomes(sorted_idx(i)).fitness;
    temp_population.Chromosomes(i).normalized_fitness = normalized_fitness(sorted_idx(i));
end

% Fitness-proportional parent selection
cumsum = zeros(1 , M);
for i = 1 : M
    for j = i : M
        cumsum(i) = cumsum(i) + temp_population.Chromosomes(j).normalized_fitness;
    end
end
R = rand(); % in [0,1]

% Selection of first parent
parent1_idx = M;
for i = 1 : length(cumsum)
    if R > cumsum(i)
        parent1_idx = i - 1;
        break;
    end
end

% Selection of second parent
parent2_idx = parent1_idx;
while_loop_stop = 0; % to break the while loop in rare cases where we keep getting the same index
while parent2_idx == parent1_idx
    while_loop_stop = while_loop_stop + 1;
    R = rand(); % in [0,1]
    if while_loop_stop > 20
        break;
    end
    for i = 1: length(cumsum)
        if R > cumsum(i)
            parent2_idx = i - 1;
            break;
        end
    end
end

% Obtaining parents from (temporary) sorted population
parent1 = temp_population.Chromosomes(parent1_idx);
parent2 = temp_population.Chromosomes(parent2_idx);
end