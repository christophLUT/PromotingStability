function [O] = ObjectiveFunction (solution, x, y, hyperparam)

% Hyperparameters contained in "hyperparam" input

% Note: Assumes that the solution is a vector of elements {0,1} where 1
% represents a feature is selected and 0 that it was not selected

%% In case no feature selected
if sum(round(solution))< 1
    O = inf; % for MINIMIZATION
    
else  
    %% At least one feature selected
    
    % Convert to binary
    selfeats = round(solution); % selected features
    
    % Split of external training data into: (1) training data (internal) and (2) validation data (for model selection)
    excrossval = cvpartition(y,'KFold',hyperparam.foldsinternalCV,'Stratify',true);
    
    % Interval CV loop
    for ex = 1 : hyperparam.foldsinternalCV
        % Training Data (Internal CV)
        xTrain = x(excrossval.training(ex),:);
        yTrain = y(excrossval.training(ex),:);

        % Validation Data (Internal CV)
        xVal = x(excrossval.test(ex),:);
        yVal = y(excrossval.test(ex),:);
        
        % Pre-processing of fitness values
        O = zeros(hyperparam.foldsinternalCV,1);

        if hyperparam.method == 'KNN'

            % KNN classifier
            mdl = fitcknn(xTrain(:,find(selfeats)),yTrain,'Distance','euclidean','NumNeighbors',hyperparam.methodKNNkval); % [Training data]

            if hyperparam.objtype == 1 % Error since MINIMIZATION
                O(ex) = 1-mean(predict(mdl,xVal(:,find(selfeats)))==yVal); % [Validation data]

            elseif hyperparam.objtype == 2 % 0.99 * Error + 0.01 * No retained features
                O(ex) = 0.99 * ( 1-mean(predict(mdl,xVal(:,find(selfeats)))==yVal) ) + 0.01 * ( sum(selfeats)/length(selfeats) ); % [Validation data]
            end
        end % KNN end
        
        
        % Average inner CV result
        O = mean(O);
  
    end   

end

end